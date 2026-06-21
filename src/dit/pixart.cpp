// PixArtTransformer2DModel — PixArt-Sigma's DiT denoiser.
//
// See dit/pixart.h. A Denoiser for the PixArt-Sigma-XL-2 transformer (HF
// diffusers Transformer2DModel, norm_type="ada_norm_single"). Forward-only,
// N = 1. Structurally a simplification of the Sana DiT: standard softmax
// self-attention (not ReLU-linear), a plain GELU(tanh) feed-forward (not the
// GLU-MBConv Mix-FFN), no caption_norm, but with a real 2x2 patch convolution
// and an added 2D sin-cos positional embedding.
//
// Per-block forward (the exact diffusers BasicTransformerBlock order for
// ada_norm_single):
//   chunks = scale_shift_table[6,D] + temb6[6,D]  -> shift_msa, scale_msa,
//            gate_msa, shift_mlp, scale_mlp, gate_mlp
//   1. self-attn:  h += gate_msa * attn1(modulate(LN(h), scale_msa, shift_msa))
//   2. cross-attn: h += attn2(h, caption)               (NO norm, NO modulation)
//   3. feed-fwd:   h += gate_mlp * ff(modulate(LN(h), scale_mlp, shift_mlp))
//   norm_out: modulate(LN(h), scale, shift) with the top-level 2-row
//   scale_shift_table offset by embedded_timestep, then proj_out + unpatchify
//   (keeping the first in_channels of the 2*in_channels output — the learned
//   variance half is discarded).

#include "brodiffusion/dit/pixart.h"

#include "brodiffusion/dit/common.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::dit {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("dit::PixArtDenoiser: " + msg);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    const auto* v = f.find(key);
    if (!v) fail("missing tensor '" + key + "'");
    return *v;
}

// Upload a checkpoint tensor (F16/F32/BF16 source) at `dt`, validating count.
void up(const st::File& f, bt::Dtype dt, const std::string& key, int rows,
        int cols, bt::Tensor& dst) {
    const st::TensorView& v = need(f, key);
    if (v.dtype != st::Dtype::F16 && v.dtype != st::Dtype::F32 &&
        v.dtype != st::Dtype::BF16) {
        fail(key + ": expected F16/F32/BF16, got " + st::dtype_name(v.dtype));
    }
    const std::int64_t expected =
        static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(cols);
    if (v.numel() != expected) {
        fail(key + ": shape mismatch (expected " + std::to_string(rows) + "x" +
             std::to_string(cols) + ", got " + std::to_string(v.numel()) + ")");
    }
    st::upload_as(v, rows, cols, dt, dst);
}

// Prepared conditioning: caption context already projected through
// caption_projection (Linear -> GELU(tanh) -> Linear) for each CFG branch.
struct PixArtPrepared : public PreparedConditioning::Impl {
    bt::Tensor ctx_cond;     // (L_cond, inner)
    bt::Tensor ctx_uncond;   // (L_uncond, inner), empty when no uncond
    bool has_uncond = false;
};

// One 1D sin-cos embedding of width `dim` (even) for a scalar position, written
// into out[off .. off+dim): [sin(pos*omega_0..), cos(pos*omega_0..)] with
// omega_q = 1/10000^(q/(dim/2)). Matches diffusers get_1d_sincos_pos_embed.
void sincos_1d(float pos, int dim, float* out) {
    const int half = dim / 2;
    for (int q = 0; q < half; ++q) {
        const float omega =
            1.0f / std::pow(10000.0f, static_cast<float>(q) /
                                          static_cast<float>(half));
        const float v = pos * omega;
        out[q]        = std::sin(v);
        out[half + q] = std::cos(v);
    }
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

PixArtDenoiser::PixArtDenoiser(const PixArtConfig& cfg) : cfg_(cfg) {
    if (cfg_.inner_dim() != cfg_.num_attention_heads * cfg_.attention_head_dim) {
        fail("inner_dim mismatch");
    }
    if (cfg_.inner_dim() % cfg_.num_attention_heads != 0) {
        fail("inner_dim must be divisible by num_attention_heads");
    }
    if (cfg_.patch_size <= 0) fail("patch_size must be positive");
    if (cfg_.out_channels < cfg_.in_channels) {
        fail("out_channels must be >= in_channels");
    }
    blocks_.resize(static_cast<std::size_t>(cfg_.num_layers));
}

PixArtDenoiser::~PixArtDenoiser() = default;

brotensor::Dtype PixArtDenoiser::compute_dtype() const {
    // The RESIDUAL stream is FP32, on CPU and CUDA alike. PixArt's residual grows
    // to ~±300 over 28 blocks: storing the residual itself in FP16 overflows the
    // layernorm variance (sum of 1152 squares ≈ 1e8 > 65504 → NaN), and BF16's
    // 7-bit mantissa rounds away each block's O(1) sublayer contribution on top
    // of the O(300) residual (cosine vs the reference falls to ~0.89 → the
    // denoise diverges to noise). So the residual, layernorms, modulation and
    // gating all stay FP32 (cosine ~0.99).
    //
    // The compute-HEAVY per-sublayer GEMMs and attention are a different story:
    // their operands are bounded (post-layernorm activations are O(1); the
    // cross-attn query off the residual is ~±300, well within FP16 range), and
    // brotensor's FP16 WMMA / flash kernels accumulate in FP32 internally. So
    // those run at mm_dtype() (FP16 on CUDA) for tensor-core speed, casting the
    // result back into the FP32 residual. See mm_dtype() and the sublayers.
    return brotensor::Dtype::FP32;
}

brotensor::Dtype PixArtDenoiser::mm_dtype() const {
    return brotensor::default_device() == brotensor::Device::CUDA
               ? brotensor::Dtype::FP16
               : brotensor::Dtype::FP32;
}

// ─── load_weights ──────────────────────────────────────────────────────────

void PixArtDenoiser::load_weights(const st::File& f, const std::string& prefix) {
    const int D   = cfg_.inner_dim();           // 1152
    const int IC  = cfg_.in_channels;           // 4
    const int OC  = cfg_.out_channels;          // 8
    const int CAP = cfg_.caption_channels;      // 4096
    const int P   = cfg_.patch_size;            // 2
    const int FF  = 4 * D;                       // gelu MLP hidden (4608)
    const bt::Dtype cdt = compute_dtype();       // FP32 — light / residual-path
    const bt::Dtype mdt = mm_dtype();            // FP16 on CUDA — heavy GEMMs

    auto load_lin_dt = [&](bt::Dtype dt, const std::string& key, int out, int in,
                           Linear& lin) {
        up(f, dt, prefix + key + ".weight", out, in, lin.W);
        up(f, dt, prefix + key + ".bias",   out, 1,  lin.b);
    };
    auto load_lin = [&](const std::string& key, int out, int in, Linear& lin) {
        load_lin_dt(cdt, key, out, in, lin);
    };

    // pos_embed.proj: P x P stride-P conv (D, IC, P, P) loaded as (D, IC*P*P).
    load_lin("pos_embed.proj", D, IC * P * P, patch_proj_);

    // adaln_single: timestep_embedder (256->D, SiLU, D->D) then the 6D proj.
    load_lin("adaln_single.emb.timestep_embedder.linear_1", D, 256, te_l1_);
    load_lin("adaln_single.emb.timestep_embedder.linear_2", D, D,   te_l2_);
    load_lin("adaln_single.linear", 6 * D, D, adaln_proj_);

    // caption_projection (PixArt text MLP). No caption_norm.
    load_lin("caption_projection.linear_1", D, CAP, cap_l1_);
    load_lin("caption_projection.linear_2", D, D,   cap_l2_);

    // norm_out top-level scale_shift_table (2, D) flattened, + proj_out.
    up(f, cdt, prefix + "scale_shift_table", 1, 2 * D, norm_out_sst_);
    load_lin("proj_out", P * P * OC, D, proj_out_);

    for (int i = 0; i < cfg_.num_layers; ++i) {
        Block& B = blocks_[static_cast<std::size_t>(i)];
        const std::string p =
            prefix + "transformer_blocks." + std::to_string(i) + ".";
        up(f, cdt, p + "scale_shift_table", 1, 6 * D, B.scale_shift);
        // attn1 (self), attn2 (cross), ff: the per-block compute-heavy GEMMs —
        // loaded at mm_dtype() (FP16 on CUDA) so they hit the WMMA / flash path.
        load_lin_dt(mdt, p + "attn1.to_q", D, D, B.q1);
        load_lin_dt(mdt, p + "attn1.to_k", D, D, B.k1);
        load_lin_dt(mdt, p + "attn1.to_v", D, D, B.v1);
        load_lin_dt(mdt, p + "attn1.to_out.0", D, D, B.out1);
        load_lin_dt(mdt, p + "attn2.to_q", D, D, B.q2);
        load_lin_dt(mdt, p + "attn2.to_k", D, D, B.k2);
        load_lin_dt(mdt, p + "attn2.to_v", D, D, B.v2);
        load_lin_dt(mdt, p + "attn2.to_out.0", D, D, B.out2);
        // ff: net.0.proj (D -> 4D, GELU) then net.2 (4D -> D).
        load_lin_dt(mdt, p + "ff.net.0.proj", FF, D,  B.ff1);
        load_lin_dt(mdt, p + "ff.net.2",      D,  FF, B.ff2);
    }

    // AdaLN affine-free LayerNorm params: ones gamma, zeros beta (D,1) at cdt.
    {
        std::vector<float> ones(static_cast<std::size_t>(D), 1.0f);
        std::vector<float> zeros(static_cast<std::size_t>(D), 0.0f);
        bt::Tensor g = bt::Tensor::from_host(ones.data(), D, 1)
                           .to(bt::default_device());
        bt::Tensor b = bt::Tensor::from_host(zeros.data(), D, 1)
                           .to(bt::default_device());
        if (cdt != bt::Dtype::FP32) {
            bt::cast(g, ada_gamma_, cdt);
            bt::cast(b, ada_beta_, cdt);
        } else {
            ada_gamma_ = std::move(g);
            ada_beta_  = std::move(b);
        }
    }
}

void PixArtDenoiser::finalize_weights() { finalized_ = true; }

void PixArtDenoiser::lin_(const Linear& l, const bt::Tensor& X, bt::Tensor& Y) {
    detail::linear_batched(l.W, l.has_bias() ? &l.b : nullptr, X, Y);
}

// ─── 2D sin-cos positional embedding ───────────────────────────────────────

const bt::Tensor& PixArtDenoiser::pos_embed_for_(int hp, int wp) {
    if (pos_hp_ == hp && pos_wp_ == wp && pos_embed_.size() > 0) {
        return pos_embed_;
    }
    const int D    = cfg_.inner_dim();
    const int half = D / 2;
    const int N    = hp * wp;
    const float base = static_cast<float>(cfg_.sample_size) /
                       static_cast<float>(cfg_.patch_size);   // base grid (64)
    const float interp = static_cast<float>(cfg_.interpolation_scale);

    // grid coordinates: grid[k] = k / (g/base) / interp  (diffusers).
    std::vector<float> gh(static_cast<std::size_t>(hp));
    std::vector<float> gw(static_cast<std::size_t>(wp));
    for (int k = 0; k < hp; ++k) {
        gh[static_cast<std::size_t>(k)] =
            static_cast<float>(k) / (static_cast<float>(hp) / base) / interp;
    }
    for (int k = 0; k < wp; ++k) {
        gw[static_cast<std::size_t>(k)] =
            static_cast<float>(k) / (static_cast<float>(wp) / base) / interp;
    }

    // pos[tok][0:half]   = sincos(grid_w[j])   (width coordinate, "emb_h")
    // pos[tok][half:D]   = sincos(grid_h[i])   (height coordinate, "emb_w")
    std::vector<float> host(static_cast<std::size_t>(N) *
                            static_cast<std::size_t>(D));
    for (int i = 0; i < hp; ++i) {
        for (int j = 0; j < wp; ++j) {
            const int tok = i * wp + j;
            float* row = host.data() + static_cast<std::size_t>(tok) *
                                           static_cast<std::size_t>(D);
            sincos_1d(gw[static_cast<std::size_t>(j)], half, row);
            sincos_1d(gh[static_cast<std::size_t>(i)], half, row + half);
        }
    }

    // Upload at the denoiser compute dtype (FP32) so it adds cleanly to the FP32
    // hidden stream — not detail::upload_host, which targets the pipeline dtype.
    const bt::Dtype cdt = compute_dtype();
    bt::Tensor p = bt::Tensor::from_host(host.data(), N, D).to(bt::default_device());
    if (cdt != bt::Dtype::FP32) {
        bt::cast(p, pos_embed_, cdt);
    } else {
        pos_embed_ = std::move(p);
    }
    pos_hp_ = hp;
    pos_wp_ = wp;
    return pos_embed_;
}

// ─── attention ─────────────────────────────────────────────────────────────

void PixArtDenoiser::manual_core_(int nh, const bt::Tensor& Q,
                                  const bt::Tensor& K, const bt::Tensor& V) {
    const int D = cfg_.inner_dim();
    const int hd = D / nh;
    const int Lq = Q.rows, Lk = K.rows;
    const bt::Dtype dt = Q.dtype;            // FP32
    const bt::Device dev = Q.device;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    detail::resize_like(ca_ocat_, Lq, D, dt, dev);
    for (int h = 0; h < nh; ++h) {
        const int off = h * hd;
        detail::resize_like(ca_qh_, Lq, hd, dt, dev);
        detail::resize_like(ca_kh_, Lk, hd, dt, dev);
        detail::resize_like(ca_vh_, Lk, hd, dt, dev);
        // Gather head h: Xh[r, :] = X[r, h*hd : h*hd+hd] (row pitch D -> hd).
        bt::copy_d2d_strided(Q, off, D, ca_qh_, 0, hd, hd, Lq);
        bt::copy_d2d_strided(K, off, D, ca_kh_, 0, hd, hd, Lk);
        bt::copy_d2d_strided(V, off, D, ca_vh_, 0, hd, hd, Lk);
        // KhT (hd, Lk): view Kh as NCHW (1, Lk, 1, hd) and transpose to (hd, Lk).
        bt::Tensor kh_flat = bt::Tensor::view(
            dev, ca_kh_.data, 1, Lk * hd, dt);
        bt::nchw_to_sequence(kh_flat, 1, Lk, 1, hd, ca_kt_);   // (hd, Lk)
        bt::matmul(ca_qh_, ca_kt_, ca_sc_);                    // scores (Lq, Lk)
        bt::scale_inplace(ca_sc_, scale);
        bt::softmax_rows_forward(ca_sc_, ca_sc_, Lq, Lk);
        bt::matmul(ca_sc_, ca_vh_, ca_oh_);                    // (Lq, hd)
        // Scatter head h back: ocat[r, h*hd:] = Oh[r, :].
        bt::copy_d2d_strided(ca_oh_, 0, hd, ca_ocat_, off, D, hd, Lq);
    }
}

void PixArtDenoiser::manual_attention_(int nh, const bt::Tensor& Q,
                                       const bt::Tensor& K, const bt::Tensor& V,
                                       const Linear& out_proj, bt::Tensor& out) {
    manual_core_(nh, Q, K, V);
    lin_(out_proj, ca_ocat_, out);   // output projection (+ bias)
}

void PixArtDenoiser::self_attention_(const Block& blk, const bt::Tensor& x_mod,
                                     bt::Tensor& out) {
    const int nh = cfg_.num_attention_heads;
    if (mm_dtype() == bt::Dtype::FP32) {
        // CPU: fused FP32 multi-head self-attention (brotensor has no FP32 flash
        // kernel, and mha_forward composes the projections + attention + out-proj
        // in one op). Biases applied post-q/k/v and post-out1.
        bt::mha_forward(x_mod, blk.q1.W, blk.k1.W, blk.v1.W, blk.out1.W,
                        &blk.q1.b, &blk.k1.b, &blk.v1.b, &blk.out1.b,
                        /*d_mask=*/nullptr, nh,
                        mha_qh_, mha_kh_, mha_vh_, mha_attn_, mha_yc_, out);
        return;
    }
    // CUDA: FP16 projections (WMMA) + FP16 flash attention (FP32 softmax accum
    // internally) + FP16 out-proj, then cast back into the FP32 residual. The
    // operands are bounded — x_mod is a post-layernorm modulation, Q/K stay O(1)
    // and the attention output is convex over the value rows — so FP16's 10-bit
    // mantissa is accurate here (unlike a 16-bit residual stream).
    bt::cast(x_mod, mod16_, bt::Dtype::FP16);
    lin_(blk.q1, mod16_, q_);     // (N, D) FP16
    lin_(blk.k1, mod16_, k_);
    lin_(blk.v1, mod16_, v_);
    bt::flash_attention_forward(q_, k_, v_, /*d_mask=*/nullptr, nh,
                                /*causal=*/false, attn_o_);   // FP16
    lin_(blk.out1, attn_o_, sub16_);                          // (N, D) FP16
    bt::cast(sub16_, out, compute_dtype());                   // → FP32 residual
}

void PixArtDenoiser::cross_attention_(const Block& blk, const bt::Tensor& ctx,
                                      const bt::Tensor& hidden,
                                      bt::Tensor& out) {
    const int nh = cfg_.num_attention_heads;
    if (mm_dtype() == bt::Dtype::FP32) {
        // CPU: project (FP32, biased) then attend in FP32 via the manual
        // per-head path.
        lin_(blk.q2, hidden, q_);   // (Lq, D)
        lin_(blk.k2, ctx, k_);      // (Lk, D)
        lin_(blk.v2, ctx, v_);      // (Lk, D)
        manual_attention_(nh, q_, k_, v_, blk.out2, out);
        return;
    }
    // CUDA: FP16 projections on tensor cores (the dominant cross-attn cost — the
    // Lq×D Q and out projections), but an FP32 attention core. PixArt's head_dim
    // (72) isn't in flash's fused FP32-score path, so flash falls to the per-head
    // WMMA kernel which stores scores in the operand dtype; the cross query is the
    // unnormed ±300 residual, so FP16 scores overflow → NaN and BF16 is too
    // coarse. The core is cheap here (caption context ~hundreds of tokens), so it
    // stays FP32 while the big projections ride FP16. ctx is already FP16.
    bt::cast(hidden, hid16_, bt::Dtype::FP16);
    lin_(blk.q2, hid16_, q_);   // (Lq, D) FP16
    lin_(blk.k2, ctx, k_);      // (Lk, D) FP16
    lin_(blk.v2, ctx, v_);      // (Lk, D) FP16
    bt::cast(q_, qf32_, bt::Dtype::FP32);
    bt::cast(k_, kf32_, bt::Dtype::FP32);
    bt::cast(v_, vf32_, bt::Dtype::FP32);
    manual_core_(nh, qf32_, kf32_, vf32_);   // FP32 attention → ca_ocat_ (Lq, D)
    bt::cast(ca_ocat_, o16_, bt::Dtype::FP16);
    lin_(blk.out2, o16_, sub16_);            // (Lq, D) FP16 out-projection
    bt::cast(sub16_, out, compute_dtype());  // → FP32 residual
}

void PixArtDenoiser::feed_forward_(const Block& blk, const bt::Tensor& x_mod,
                                   bt::Tensor& out) {
    if (mm_dtype() == bt::Dtype::FP32) {
        lin_(blk.ff1, x_mod, ff_h_);     // (N, 4D)
        bt::gelu_forward(ff_h_, ff_h_);  // GELU(tanh)
        lin_(blk.ff2, ff_h_, out);       // (N, D)
        return;
    }
    // CUDA: FP16 WMMA GEMMs around a FP16 GELU, cast back into the FP32 residual.
    bt::cast(x_mod, mod16_, bt::Dtype::FP16);
    lin_(blk.ff1, mod16_, ff_h_);    // (N, 4D) FP16
    bt::gelu_forward(ff_h_, ff_h_);  // GELU(tanh)
    lin_(blk.ff2, ff_h_, sub16_);    // (N, D) FP16
    bt::cast(sub16_, out, compute_dtype());   // → FP32 residual
}

// ─── prepared conditioning ─────────────────────────────────────────────────

PreparedConditioning PixArtDenoiser::prepare(const Conditioning& cond) {
    if (cap_l1_.W.size() == 0) fail("prepare: weights not loaded");
    const bt::Dtype cdt = compute_dtype();   // FP32 caption projection
    const bt::Dtype mdt = mm_dtype();         // cross-attn K/V dtype (FP16 on CUDA)

    auto project = [&](const bt::Tensor& seq, bt::Tensor& out) {
        if (seq.cols != cfg_.caption_channels) {
            fail("prepare: text_embeddings width != caption_channels");
        }
        bt::Tensor seq_cd, h1, proj;
        const bt::Tensor* in = &seq;
        if (seq.dtype != cdt) { bt::cast(seq, seq_cd, cdt); in = &seq_cd; }
        lin_(cap_l1_, *in, h1);
        bt::gelu_forward(h1, h1);
        lin_(cap_l2_, h1, proj);              // (L, D) FP32
        // Store the context at mm_dtype so the per-block cross-attn K/V
        // projections (FP16 weights on CUDA) take it directly.
        if (mdt != cdt) { bt::cast(proj, out, mdt); }
        else            { out = proj.clone(); }
    };

    auto prep = std::make_unique<PixArtPrepared>();
    project(cond.text_embeddings, prep->ctx_cond);
    if (cond.has_uncond && cond.uncond_embeddings.size() > 0) {
        prep->has_uncond = true;
        project(cond.uncond_embeddings, prep->ctx_uncond);
    }
    return PreparedConditioning(std::move(prep));
}

// ─── forward ───────────────────────────────────────────────────────────────

void PixArtDenoiser::forward(const bt::Tensor& latent, int H_lat, int W_lat,
                             float timestep,
                             const PreparedConditioning& prepared,
                             Branch branch, bt::Tensor& out) {
    if (!prepared) fail("forward: prepared conditioning is empty");
    const auto* prep = dynamic_cast<const PixArtPrepared*>(prepared.get());
    if (!prep) fail("forward: prepared conditioning has the wrong type");
    if (patch_proj_.W.size() == 0) fail("forward: weights not loaded");
    if (H_lat <= 0 || W_lat <= 0) fail("forward: H_lat/W_lat must be positive");
    if ((H_lat % cfg_.patch_size) != 0 || (W_lat % cfg_.patch_size) != 0) {
        fail("forward: H_lat/W_lat must be divisible by patch_size");
    }

    const int D   = cfg_.inner_dim();
    const int IC  = cfg_.in_channels;
    const int OC  = cfg_.out_channels;
    const int P   = cfg_.patch_size;
    const int hp  = H_lat / P;
    const int wp  = W_lat / P;
    const int N   = hp * wp;                 // patch tokens
    const bt::Dtype cdt = compute_dtype();
    const bt::Device dev = bt::default_device();

    if (latent.rows != 1 || latent.cols != IC * H_lat * W_lat) {
        fail("forward: latent must be (1, in_channels*H_lat*W_lat)");
    }

    const bt::Tensor* ctx = &prep->ctx_cond;
    if (branch == Branch::Uncond) {
        if (!prep->has_uncond) fail("forward: uncond branch but no uncond ctx");
        ctx = &prep->ctx_uncond;
    }

    // ── timestep → AdaLN-single ───────────────────────────────────────────
    {
        std::vector<float> tval = {timestep};
        ts_ = bt::Tensor::from_host(tval.data(), 1, 1).to(dev);
        bt::timestep_embedding(ts_, /*dim=*/256, /*max_period=*/10000.0f, freq_);
    }
    const bt::Tensor* tin = &freq_;
    if (cdt != bt::Dtype::FP32) { bt::cast(freq_, freq_cd_, cdt); tin = &freq_cd_; }
    // NOTE: te_l2's output must NOT alias its input — a matmul that reads and
    // writes the same buffer corrupts. (te_l1 is safe: its output has a
    // different shape than the freq input, so it reallocates.)
    lin_(te_l1_, *tin, emb_);
    bt::silu_forward(emb_, emb_);          // elementwise in-place is fine
    lin_(te_l2_, emb_, emb_silu_);         // separate output buffer
    emb_ = emb_silu_.clone();              // embedded_timestep (1, D)

    bt::silu_forward(emb_, emb_silu_);
    lin_(adaln_proj_, emb_silu_, temb6_);
    temb6_ = temb6_.clone();                      // (1, 6D)

    // ── patch embed: conv 2x2 stride 2 → tokens, + positional embedding ────
    bt::Tensor lat_cd;
    const bt::Tensor* lat = &latent;
    if (latent.dtype != cdt) { bt::cast(latent, lat_cd, cdt); lat = &lat_cd; }
    bt::conv2d_forward(*lat, patch_proj_.W, &patch_proj_.b, 1, IC, H_lat, W_lat,
                       D, P, P, P, P, 0, 0, 1, 1, /*groups=*/1, patch_nchw_);
    bt::nchw_to_sequence(patch_nchw_, 1, D, hp, wp, hidden_);   // (N, D)
    bt::add_inplace(hidden_, pos_embed_for_(hp, wp));
    hidden_ = hidden_.clone();

    // ── transformer blocks ────────────────────────────────────────────────
    std::vector<bt::Tensor> ch;
    for (const Block& blk : blocks_) {
        mod_row_ = temb6_.clone();
        bt::add_inplace(mod_row_, blk.scale_shift);
        slice_modulation_chunks(mod_row_, D, 6, ch);
        const bt::Tensor& shift_msa = ch[0];
        const bt::Tensor& scale_msa = ch[1];
        const bt::Tensor& gate_msa  = ch[2];
        const bt::Tensor& shift_mlp = ch[3];
        const bt::Tensor& scale_mlp = ch[4];
        const bt::Tensor& gate_mlp  = ch[5];

        // 1. self-attn: h += gate_msa * attn1(modulate(LN(h)))
        detail::layernorm_batched(hidden_, ada_gamma_, ada_beta_, ln_,
                                  cfg_.norm_eps);
        bt::modulate(ln_, scale_msa, shift_msa, mod_);
        self_attention_(blk, mod_, sub_out_);
        bt::broadcast_mul(sub_out_, gate_msa, gated_);
        bt::add_inplace(hidden_, gated_);

        // 2. cross-attn: h += attn2(h, caption)   (no norm, no modulation)
        cross_attention_(blk, *ctx, hidden_, sub_out_);
        bt::add_inplace(hidden_, sub_out_);

        // 3. feed-forward: h += gate_mlp * ff(modulate(LN(h)))
        detail::layernorm_batched(hidden_, ada_gamma_, ada_beta_, ln_,
                                  cfg_.norm_eps);
        bt::modulate(ln_, scale_mlp, shift_mlp, mod_);
        feed_forward_(blk, mod_, sub_out_);
        bt::broadcast_mul(sub_out_, gate_mlp, gated_);
        bt::add_inplace(hidden_, gated_);
    }

    // ── norm_out → proj_out ───────────────────────────────────────────────
    std::vector<bt::Tensor> no;
    slice_modulation_chunks(norm_out_sst_, D, 2, no);   // shift, scale
    bt::Tensor shift = no[0].clone();
    bt::Tensor scale = no[1].clone();
    bt::add_inplace(shift, emb_);                       // + embedded_timestep
    bt::add_inplace(scale, emb_);
    detail::layernorm_batched(hidden_, ada_gamma_, ada_beta_, ln_,
                              cfg_.norm_eps);
    bt::modulate(ln_, scale, shift, mod_);
    lin_(proj_out_, mod_, proj_);                       // (N, P*P*OC)

    // ── unpatchify, keeping the first in_channels (drop the variance half) ──
    // proj_[tok][(py*P+px)*OC + c]  ->  out[c, P*i+py, P*j+px], for c < IC.
    bt::sync_all();
    const std::size_t proj_n =
        static_cast<std::size_t>(proj_.rows) * static_cast<std::size_t>(proj_.cols);
    std::vector<float> proj_host(proj_n);
    if (proj_.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(proj_n);
        proj_.copy_to_host_fp16(bits.data());
        bt::sync_all();
        for (std::size_t k = 0; k < proj_n; ++k) {
            proj_host[k] = bt::fp16_bits_to_fp32(bits[k]);
        }
    } else if (proj_.dtype == bt::Dtype::BF16) {
        std::vector<std::uint16_t> bits(proj_n);
        proj_.copy_to_host_bf16(bits.data());
        bt::sync_all();
        for (std::size_t k = 0; k < proj_n; ++k) {
            proj_host[k] = bt::bf16_bits_to_fp32(bits[k]);
        }
    } else {
        proj_host = proj_.to_host_vector();
    }

    const int HW = H_lat * W_lat;
    std::vector<float> host_out(static_cast<std::size_t>(IC) *
                                static_cast<std::size_t>(HW));
    const int pp_oc = P * P * OC;
    for (int i = 0; i < hp; ++i) {
        for (int j = 0; j < wp; ++j) {
            const int tok = i * wp + j;
            const float* tv =
                proj_host.data() + static_cast<std::size_t>(tok) *
                                       static_cast<std::size_t>(pp_oc);
            for (int py = 0; py < P; ++py) {
                for (int px = 0; px < P; ++px) {
                    const int y = i * P + py;
                    const int x = j * P + px;
                    const int blk = (py * P + px) * OC;
                    for (int c = 0; c < IC; ++c) {
                        host_out[static_cast<std::size_t>(c) * HW +
                                 static_cast<std::size_t>(y) * W_lat + x] =
                            tv[blk + c];
                    }
                }
            }
        }
    }
    out = detail::upload_host(host_out.data(), 1, IC * HW);
    (void)N; (void)dev;
}

}  // namespace brodiffusion::dit
