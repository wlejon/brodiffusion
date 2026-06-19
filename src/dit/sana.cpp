// SanaTransformer2DModel — NVIDIA Sana's linear-attention DiT denoiser.
//
// See dit/sana.h. A Denoiser for the Sana 0.6B rectified-flow transformer
// (HF diffusers SanaTransformer2DModel). Forward-only, batch size N = 1.
//
// The model is natively FP16 (the checkpoint ships fp16, and Sana was designed
// for FP16 inference). Unlike Flux — whose residual stream overflows FP16, so
// it runs BF16 internally — Sana runs at the pipeline compute dtype (FP16 on a
// GPU backend, FP32 on CPU), exactly like its sibling DC-AE decoder. That is
// also forced by brotensor: matmul, rms_norm, and conv2d (all of which the
// linear self-attention and the GLU-MBConv Mix-FFN need) dispatch FP32/FP16
// only — BF16 has no kernel. The ReLU linear-attention core is evaluated in
// FP32 for range, matching diffusers' float32 upcast of that path.
//
// SanaTransformerBlock forward ordering (the exact diffusers order):
//   1. AdaLN-single: (scale_shift_table[6,D] + temb6[6,D]) split into
//      shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
//   2. self-attn:  h += gate_msa * attn1(modulate(LN(h), scale_msa, shift_msa))
//   3. cross-attn: h += attn2(h, caption)            (NO norm, NO modulation)
//   4. mix-ffn:    h += gate_mlp * ff(modulate(LN(h), scale_mlp, shift_mlp))
//      where ff is the GLU-MBConv over the (H,W) spatial reshape.
//   norm_out: modulate(LN(h), scale, shift) with the top-level 2-row
//   scale_shift_table offset by embedded_timestep, then proj_out + unpatchify.

#include "brodiffusion/dit/sana.h"

#include "brodiffusion/dit/common.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::dit {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

constexpr float kRmsEps     = 1e-5f;    // caption_norm RMSNorm eps
constexpr float kAttnEps    = 1e-15f;   // SanaLinearAttnProcessor denom eps
constexpr int   kSelfHeadDim = 32;      // attention_head_dim

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("dit::SanaDenoiser: " + msg);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    const auto* v = f.find(key);
    if (!v) fail("missing tensor '" + key + "'");
    return *v;
}

// Upload a checkpoint tensor (F16/F32/BF16 source) at `dt`, validating its
// element count.
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

// Non-owning (rows, cols) view of `count`=rows*cols elements of `t` from
// element offset `off`. Lets a contiguous buffer be reinterpreted with no copy.
bt::Tensor sub_view(const bt::Tensor& t, std::int64_t off, int rows, int cols) {
    char* p = static_cast<char*>(t.data) + off * bt::dtype_size_bytes(t.dtype);
    return bt::Tensor::view(t.device, p, rows, cols, t.dtype);
}

// Prepared conditioning: caption context already projected through
// caption_projection + caption_norm for each CFG branch.
struct SanaPrepared : public PreparedConditioning::Impl {
    bt::Tensor ctx_cond;     // (L_cond, inner)
    bt::Tensor ctx_uncond;   // (L_uncond, inner), empty when no uncond
    bool has_uncond = false;
};

// Per-op profiler, enabled with BRODIFFUSION_FLOW_PROFILE=1 (shared with the
// triposplat flow DiT). Syncs around each category, accumulates ms across all
// forwards, dumps once on demand. The per-op syncs serialise the stream, so the
// absolute total runs a little hot — the split, not the total, is the signal.
struct SanaProf {
    static bool enabled() {
        static const bool on = [] {
            const char* e = std::getenv("BRODIFFUSION_FLOW_PROFILE");
            return e && *e && *e != '0';
        }();
        return on;
    }
    std::vector<std::pair<const char*, double>> cats;  // insertion-ordered
    int forwards = 0;
    double& at(const char* n) {
        for (auto& p : cats)
            if (p.first == n || std::string(p.first) == n) return p.second;
        cats.emplace_back(n, 0.0);
        return cats.back().second;
    }
    void dump() {
        double total = 0.0;
        for (auto& p : cats) total += p.second;
        std::fprintf(stderr, "[sana] ── %d forwards ──\n", forwards);
        for (auto& p : cats)
            std::fprintf(stderr, "[sana] %-16s %9.1f ms  %4.1f%%  (%6.2f ms/fwd)\n",
                         p.first, p.second, total > 0 ? 100.0 * p.second / total : 0.0,
                         forwards > 0 ? p.second / forwards : 0.0);
        std::fprintf(stderr, "[sana] %-16s %9.1f ms        (%6.2f ms/fwd)\n",
                     "TOTAL", total, forwards > 0 ? total / forwards : 0.0);
    }
};
SanaProf g_sana_prof;

void sana_dump_profile_impl() {
    if (SanaProf::enabled()) g_sana_prof.dump();
}

// Register the atexit dump exactly once (a function-local static in a
// non-template function — the templated prof() would otherwise register one per
// distinct lambda type, dumping N times).
inline void prof_register_atexit() {
    static const bool registered = [] {
        std::atexit(+[] { sana_dump_profile_impl(); });
        return true;
    }();
    (void)registered;
}

template <class F>
inline void prof(const char* name, F&& f) {
    if (!SanaProf::enabled()) { f(); return; }
    prof_register_atexit();
    bt::sync_all();
    const auto t0 = std::chrono::steady_clock::now();
    f();
    bt::sync_all();
    g_sana_prof.at(name) +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

SanaDenoiser::SanaDenoiser(const SanaConfig& cfg) : cfg_(cfg) {
    if (cfg_.inner_dim() != cfg_.num_attention_heads * cfg_.attention_head_dim) {
        fail("inner_dim mismatch");
    }
    if (cfg_.attention_head_dim != kSelfHeadDim) {
        fail("attention_head_dim must be 32");
    }
    if (cfg_.inner_dim() % cfg_.num_cross_attention_heads != 0) {
        fail("inner_dim must be divisible by num_cross_attention_heads");
    }
    blocks_.resize(static_cast<std::size_t>(cfg_.num_layers));
}

SanaDenoiser::~SanaDenoiser() = default;

brotensor::Dtype SanaDenoiser::compute_dtype() const {
    // BF16 on CUDA, FP32 on CPU. Sana needs BF16/FP32 dynamic range, not FP16:
    // the residual stream grows into the hundreds, the GLU-MBConv FFN's GLU
    // product and conv_point sum push intermediates past the FP16 finite range
    // (~65504), and the cross-attention to_q of the raw residual overflows too
    // — the well-known "Sana in FP16 → NaN" failure. BF16 carries FP32's 8-bit
    // exponent, so none of those overflow, while halving weight/activation
    // memory traffic and putting every linear projection on Ada's BF16 tensor
    // cores (linear_forward_batched_fp16's WMMA path). brotensor's matmul,
    // rms_norm, layernorm, conv2d and flash attention all have BF16 kernels.
    // The CPU backend is FP32-only, so fall back to FP32 there.
    return brotensor::default_device() == brotensor::Device::CUDA
               ? brotensor::Dtype::BF16
               : brotensor::Dtype::FP32;
}

// ─── load_weights ──────────────────────────────────────────────────────────

void SanaDenoiser::load_weights(const st::File& f, const std::string& prefix) {
    const int D       = cfg_.inner_dim();           // 1152
    const int IC      = cfg_.in_channels;           // 32
    const int OC      = cfg_.out_channels;          // 32
    const int CAP     = cfg_.caption_channels;      // 2304
    const int hidden  = static_cast<int>(cfg_.mlp_ratio * D);  // 2880
    const int inv     = 2 * hidden;                 // 5760
    const bt::Dtype cdt = compute_dtype();          // BF16 on CUDA, FP32 on CPU

    auto load_lin = [&](const std::string& key, int out, int in, Linear& lin,
                        bool bias) {
        up(f, cdt, prefix + key + ".weight", out, in, lin.W);
        if (bias) up(f, cdt, prefix + key + ".bias", out, 1, lin.b);
    };

    // patch_embed: 1x1 conv (D, IC, 1, 1) loaded as a (D, IC) linear.
    load_lin("patch_embed.proj", D, IC, patch_embed_, /*bias=*/true);

    // AdaLayerNormSingle: timestep_embedder (256->D->D) then linear (D->6D).
    load_lin("time_embed.emb.timestep_embedder.linear_1", D, 256, te_l1_, true);
    load_lin("time_embed.emb.timestep_embedder.linear_2", D, D, te_l2_, true);
    load_lin("time_embed.linear", 6 * D, D, te_proj_, true);

    // caption_projection (PixArt text MLP) + caption_norm (RMSNorm).
    load_lin("caption_projection.linear_1", D, CAP, cap_l1_, true);
    load_lin("caption_projection.linear_2", D, D, cap_l2_, true);
    up(f, cdt, prefix + "caption_norm.weight", D, 1, caption_norm_g_);

    // norm_out top-level scale_shift_table (2, D) flattened, + proj_out.
    up(f, cdt, prefix + "scale_shift_table", 1, 2 * D, norm_out_sst_);
    load_lin("proj_out", cfg_.patch_size * cfg_.patch_size * OC, D, proj_out_,
             true);

    for (int i = 0; i < cfg_.num_layers; ++i) {
        Block& B = blocks_[static_cast<std::size_t>(i)];
        const std::string p =
            prefix + "transformer_blocks." + std::to_string(i) + ".";
        up(f, cdt, p + "scale_shift_table", 1, 6 * D, B.scale_shift);
        // attn1 (self, ReLU linear): q/k/v bias-free; to_out.0 biased.
        up(f, cdt, p + "attn1.to_q.weight", D, D, B.q1.W);
        up(f, cdt, p + "attn1.to_k.weight", D, D, B.k1.W);
        up(f, cdt, p + "attn1.to_v.weight", D, D, B.v1.W);
        up(f, cdt, p + "attn1.to_out.0.weight", D, D, B.out1.W);
        up(f, cdt, p + "attn1.to_out.0.bias",   D, 1, B.out1.b);
        // attn2 (cross, softmax MHA): all biased.
        up(f, cdt, p + "attn2.to_q.weight", D, D, B.q2.W);
        up(f, cdt, p + "attn2.to_q.bias",   D, 1, B.q2.b);
        up(f, cdt, p + "attn2.to_k.weight", D, D, B.k2.W);
        up(f, cdt, p + "attn2.to_k.bias",   D, 1, B.k2.b);
        up(f, cdt, p + "attn2.to_v.weight", D, D, B.v2.W);
        up(f, cdt, p + "attn2.to_v.bias",   D, 1, B.v2.b);
        up(f, cdt, p + "attn2.to_out.0.weight", D, D, B.out2.W);
        up(f, cdt, p + "attn2.to_out.0.bias",   D, 1, B.out2.b);
        // GLU-MBConv ff.
        up(f, cdt, p + "ff.conv_inverted.weight", inv, D, B.ff_inv_W);
        up(f, cdt, p + "ff.conv_inverted.bias",   inv, 1, B.ff_inv_b);
        up(f, cdt, p + "ff.conv_depth.weight", inv, 3 * 3, B.ff_depth_W);
        up(f, cdt, p + "ff.conv_depth.bias",   inv, 1, B.ff_depth_b);
        up(f, cdt, p + "ff.conv_point.weight", D, hidden, B.ff_point_W);
    }

    // AdaLN affine-free LayerNorm params: ones gamma, zeros beta (D,1). The
    // layernorm kernel requires gamma/beta to share the activation dtype, so
    // these follow cdt (BF16 on CUDA).
    {
        std::vector<float> ones(static_cast<std::size_t>(D), 1.0f);
        std::vector<float> zeros(static_cast<std::size_t>(D), 0.0f);
        bt::Tensor g = bt::Tensor::from_host(ones.data(), D, 1).to(bt::default_device());
        bt::Tensor b = bt::Tensor::from_host(zeros.data(), D, 1).to(bt::default_device());
        if (cdt != bt::Dtype::FP32) {
            bt::cast(g, ada_gamma_, cdt);
            bt::cast(b, ada_beta_, cdt);
        } else {
            ada_gamma_ = std::move(g);
            ada_beta_  = std::move(b);
        }
    }
}

void SanaDenoiser::finalize_weights() { finalized_ = true; }

void SanaDenoiser::lin_(const Linear& l, const bt::Tensor& X, bt::Tensor& Y) {
    detail::linear_batched(l.W, l.has_bias() ? &l.b : nullptr, X, Y);
}

void SanaDenoiser::ensure_ones_(int n) {
    if (ones_.rows == 1 && ones_.cols == n && ones_.dtype == bt::Dtype::FP32 &&
        ones_.data != nullptr) {
        return;
    }
    std::vector<float> h(static_cast<std::size_t>(n), 1.0f);
    ones_ = bt::Tensor::from_host(h.data(), 1, n);
    if (ones_.dtype != bt::Dtype::FP32) {
        bt::Tensor c;
        bt::cast(ones_, c, bt::Dtype::FP32);
        ones_ = std::move(c);
    }
}

// ─── prepared conditioning ─────────────────────────────────────────────────

PreparedConditioning SanaDenoiser::prepare(const Conditioning& cond) {
    if (patch_embed_.W.size() == 0) fail("prepare: weights not loaded");
    const int D = cfg_.inner_dim();

    // Project a caption sequence (L, caption_channels) through caption_projection
    // (Linear -> GELU(tanh) -> Linear) then caption_norm (RMSNorm).
    auto project = [&](const bt::Tensor& seq, bt::Tensor& out) {
        if (seq.cols != cfg_.caption_channels) {
            fail("prepare: text_embeddings width != caption_channels");
        }
        bt::Tensor seq_cd, h1, h2;
        const bt::Tensor* in = &seq;
        if (seq.dtype != compute_dtype()) {
            bt::cast(seq, seq_cd, compute_dtype());
            in = &seq_cd;
        }
        lin_(cap_l1_, *in, h1);
        bt::gelu_forward(h1, h1);
        lin_(cap_l2_, h1, h2);
        bt::rms_norm_forward(h2, caption_norm_g_, kRmsEps, out);
        out = out.clone();
    };

    auto prep = std::make_unique<SanaPrepared>();
    project(cond.text_embeddings, prep->ctx_cond);
    if (cond.has_uncond && cond.uncond_embeddings.size() > 0) {
        prep->has_uncond = true;
        project(cond.uncond_embeddings, prep->ctx_uncond);
    }
    (void)D;
    return PreparedConditioning(std::move(prep));
}

// ─── self-attention (SanaLinearAttnProcessor2_0) ───────────────────────────
//
// Single-scale ReLU linear attention. Per head: relu(q), relu(k), then
// out = (q · (kᵀ v)) / (q · (kᵀ 1) + eps). The core runs in FP32 channel-major,
// mirroring the DC-AE decoder's apply_attn_ (minus the multiscale aggregation).

void SanaDenoiser::self_attention_(const Block& blk, int N, int H, int W,
                                   const bt::Tensor& x_mod, bt::Tensor& out) {
    const int D  = cfg_.inner_dim();
    const int hd = kSelfHeadDim;
    const int nh = cfg_.num_attention_heads;
    const bt::Dtype cdt = compute_dtype();
    const bt::Device dev = x_mod.device;

    // To channel-major (D, N): Xcm[c,n] = x_mod[n,c].
    prof("sa_proj", [&] {
    // q/k/v = W @ x_modᵀ → (D, N) channel-major (bias-free). On CUDA this is a
    // single WMMA A@Bᵀ per projection: x_mod (N,D) is consumed transposed for
    // free (no sequence_to_nchw), W (D,D) is A, output is (D,N). On the FP32 CPU
    // backend matmul_abt is unavailable, so fall back to the naive channel-major
    // matmul (transpose x_mod, then W @ Xcm).
    if (cdt == bt::Dtype::FP32) {
        bt::sequence_to_nchw(x_mod, 1, D, H, W, xcm_);   // (1, D*N)
        bt::Tensor Xcm = sub_view(xcm_, 0, D, N);
        bt::matmul(blk.q1.W, Xcm, q_);
        bt::matmul(blk.k1.W, Xcm, k_);
        bt::matmul(blk.v1.W, Xcm, v_);
    } else {
        detail::resize_like(q_, D, N, cdt, dev);
        detail::resize_like(k_, D, N, cdt, dev);
        detail::resize_like(v_, D, N, cdt, dev);
        bt::matmul_abt(blk.q1.W, x_mod, q_, 1, D, N, D, 0, 0, 0, nullptr, 0);
        bt::matmul_abt(blk.k1.W, x_mod, k_, 1, D, N, D, 0, 0, 0, nullptr, 0);
        bt::matmul_abt(blk.v1.W, x_mod, v_, 1, D, N, D, 0, 0, 0, nullptr, 0);
    }
    });

    // The ReLU linear-attention core, per head h (channel-major (hd,N) blocks of
    // the (D,N) q/k/v):  relu(q), relu(k); then with Vaug = [V; 1]:
    //   out_h = (relu(q_h)ᵀ · (relu(k_h) · Vᵀ))   normalised by
    //           (relu(q_h)ᵀ · Σ_n relu(k_h)).
    // The result lands in attn_c_ (D,N) at the compute dtype.
    prof("sa_core", [&] {
    if (cdt == bt::Dtype::FP32) {
        // ── FP32 per-head reference (CPU backend; matmul_abt is 16-bit) ──
        detail::resize_like(attn_c_, D, N, bt::Dtype::FP32, dev);
        ensure_ones_(N);
        for (int h = 0; h < nh; ++h) {
            const int base = h * hd;
            bt::Tensor Qh = sub_view(q_, static_cast<std::int64_t>(base) * N, hd, N);
            bt::Tensor Kh = sub_view(k_, static_cast<std::int64_t>(base) * N, hd, N);
            bt::relu_forward(Qh, Qh);
            bt::relu_forward(Kh, Kh);
            // Vp = [V_h; ones] → (hd+1, N).
            detail::resize_like(vp_, hd + 1, N, bt::Dtype::FP32, dev);
            bt::copy_d2d(v_, static_cast<std::int64_t>(base) * N, vp_, 0, hd * N);
            bt::copy_d2d(ones_, 0, vp_, hd * N, N);
            // Kt = Khᵀ → (N, hd).
            bt::Tensor KhN = sub_view(k_, static_cast<std::int64_t>(base) * N,
                                      1, hd * N);
            bt::nchw_to_sequence(KhN, 1, hd, H, W, kt_);  // (N, hd)
            bt::matmul(vp_, kt_, scores_);                // (hd+1, hd)
            bt::matmul(scores_, Qh, hid_);                // (hd+1, N)
            detail::resize_like(recip_, 1, N, bt::Dtype::FP32, dev);
            bt::copy_d2d(hid_, hd * N, recip_, 0, N);
            bt::add_scalar_inplace(recip_, kAttnEps);
            bt::rsqrt_forward(recip_, recip_);
            bt::mul_inplace(recip_, recip_);              // 1/(denom+eps)
            bt::Tensor num  = sub_view(hid_, 0, hd, N);
            bt::Tensor outh = sub_view(attn_c_,
                                       static_cast<std::int64_t>(base) * N, hd, N);
            bt::broadcast_mul(num, recip_, outh);
        }
    } else {
        // ── Batched WMMA path (CUDA, BF16) ──────────────────────────────
        // All heads at once: 2 elementwise relus, 1 transpose, 4 batched A@Bᵀ
        // GEMMs and a per-head broadcast divide — instead of ~10 ops × nh heads.
        const std::int64_t hdN = static_cast<std::int64_t>(hd) * N;
        const std::int64_t hdhd = static_cast<std::int64_t>(hd) * hd;
        const std::int64_t Nhd = static_cast<std::int64_t>(N) * hd;
        bt::relu_forward(q_, q_);                          // relu(Q) (D,N)
        bt::relu_forward(k_, k_);                          // relu(K) (D,N)
        // S[h] = V[h] @ relu(K)[h]ᵀ → (hd, hd), batched over heads.
        detail::resize_like(sa_S_, nh * hd, hd, cdt, dev);
        bt::matmul_abt(v_, k_, sa_S_, nh, hd, hd, N, hdN, hdN, hdhd, nullptr, 0);
        // z[r] = Σ_n relu(K)[r,n] → (D,1) via relu(K) @ onesᵀ.
        if (ones_bf_.cols != N || ones_bf_.dtype != cdt ||
            ones_bf_.data == nullptr) {
            ensure_ones_(N);
            bt::cast(ones_, ones_bf_, cdt);
        }
        detail::resize_like(sa_z_, D, 1, cdt, dev);
        bt::matmul_abt(k_, ones_bf_, sa_z_, 1, D, 1, N, 0, 0, 0, nullptr, 0);
        // Qrᵀ: (nh,hd,N) → (nh,N,hd) channel→token transpose, batched over heads.
        bt::nchw_to_sequence(q_, nh, hd, H, W, sa_qt_);    // (nh*N, hd)
        // num[h] = S[h] @ Qrᵀ[h]ᵀ → (hd, N), batched.
        detail::resize_like(sa_num_, D, N, cdt, dev);
        bt::matmul_abt(sa_S_, sa_qt_, sa_num_, nh, hd, N, hd, hdhd, Nhd, hdN,
                       nullptr, 0);
        // den[h] = z[h] @ Qrᵀ[h]ᵀ → (1, N), batched (z[h] is a (1,hd) row).
        detail::resize_like(sa_den_, nh, N, cdt, dev);
        bt::matmul_abt(sa_z_, sa_qt_, sa_den_, nh, 1, N, hd, hd, Nhd,
                       static_cast<std::int64_t>(N), nullptr, 0);
        // recip = 1/(den+eps), computed in FP32 for a stable reciprocal.
        bt::cast(sa_den_, sa_denf_, bt::Dtype::FP32);
        bt::add_scalar_inplace(sa_denf_, kAttnEps);
        bt::rsqrt_forward(sa_denf_, sa_denf_);
        bt::mul_inplace(sa_denf_, sa_denf_);               // 1/(den+eps)
        bt::cast(sa_denf_, sa_recip_, cdt);                // (nh, N)
        // out[h] = num[h] * recip[h] (broadcast the (1,N) row over hd).
        detail::resize_like(attn_c_, D, N, cdt, dev);
        for (int h = 0; h < nh; ++h) {
            const std::int64_t base = static_cast<std::int64_t>(h) * hd;
            bt::Tensor num_h = sub_view(sa_num_, base * N, hd, N);
            bt::Tensor rec_h = sub_view(sa_recip_,
                                        static_cast<std::int64_t>(h) * N, 1, N);
            bt::Tensor out_h = sub_view(attn_c_, base * N, hd, N);
            bt::broadcast_mul(num_h, rec_h, out_h);
        }
    }
    });

    prof("sa_out", [&] {
    // attn_c_ (D,N) at the compute dtype → sequence (N,D), then to_out (biased).
    bt::Tensor attn_nchw = sub_view(attn_c_, 0, 1, D * N);
    bt::nchw_to_sequence(attn_nchw, 1, D, H, W, mod_);   // (N, D); reuse mod_
    lin_(blk.out1, mod_, out);                           // (N, D)
    });
}

// ─── cross-attention (softmax MHA to caption) ──────────────────────────────

void SanaDenoiser::cross_attention_(const Block& blk, const bt::Tensor& ctx,
                                    const bt::Tensor& hidden, bt::Tensor& out) {
    const int nch = cfg_.num_cross_attention_heads;
    const bt::Dtype cdt = compute_dtype();   // BF16 on CUDA, FP32 on CPU
    lin_(blk.q2, hidden, ca_q_);   // (N, D) at cdt
    lin_(blk.k2, ctx, ca_k_);      // (L, D) at cdt
    lin_(blk.v2, ctx, ca_v_);      // (L, D) at cdt
    // brotensor's flash attention runs at the flash dtype: BF16 on CUDA (full
    // FP32 exponent range, so the large to_q of the raw residual never
    // overflows), FP32 on CPU. With the DiT already at cdt these usually match,
    // so cast only on a genuine mismatch.
    const bt::Dtype fdt = flux_compute_dtype();
    const bt::Tensor* Q = &ca_q_;
    const bt::Tensor* K = &ca_k_;
    const bt::Tensor* V = &ca_v_;
    if (ca_q_.dtype != fdt) { bt::cast(ca_q_, ca_qf_, fdt); Q = &ca_qf_; }
    if (ca_k_.dtype != fdt) { bt::cast(ca_k_, ca_kf_, fdt); K = &ca_kf_; }
    if (ca_v_.dtype != fdt) { bt::cast(ca_v_, ca_vf_, fdt); V = &ca_vf_; }
    bt::flash_attention_forward(*Q, *K, *V, /*d_mask=*/nullptr, nch,
                                /*causal=*/false, ca_o_);
    // to_out consumes the attention output at the weight dtype (cdt). The
    // output is a convex combination of the bounded value rows, so the cast is
    // lossless in range.
    const bt::Tensor* O = &ca_o_;
    if (ca_o_.dtype != cdt) { bt::cast(ca_o_, ca_of_, cdt); O = &ca_of_; }
    lin_(blk.out2, *O, out);   // (N, D)
}

// ─── Mix-FFN (GLU-MBConv over the spatial reshape) ─────────────────────────

void SanaDenoiser::mix_ffn_(const Block& blk, int H, int W,
                            const bt::Tensor& x_mod, bt::Tensor& out) {
    const int D      = cfg_.inner_dim();
    const int hidden = static_cast<int>(cfg_.mlp_ratio * D);   // 2880
    const int inv    = 2 * hidden;                             // 5760
    const int HW     = H * W;

    // (N, D) sequence → (1, D, H, W) channel-major NCHW.
    bt::sequence_to_nchw(x_mod, 1, D, H, W, ff_spatial_);
    // conv_inverted 1x1 (D -> 2*hidden, biased) → SiLU.
    bt::conv2d_forward(ff_spatial_, blk.ff_inv_W, &blk.ff_inv_b, 1, D, H, W,
                       inv, 1, 1, 1, 1, 0, 0, 1, 1, 1, ff_t1_);
    bt::silu_forward(ff_t1_, ff_t1_);
    // conv_depth 3x3 depthwise (2*hidden, groups=2*hidden, biased).
    bt::conv2d_forward(ff_t1_, blk.ff_depth_W, &blk.ff_depth_b, 1, inv, H, W,
                       inv, 3, 3, 1, 1, 1, 1, 1, 1, inv, ff_t2_);
    // GLU: split channels (h, gate); h *= SiLU(gate).
    bt::Tensor hpart = sub_view(ff_t2_, 0, hidden, HW);
    bt::Tensor gpart = sub_view(ff_t2_, static_cast<std::int64_t>(hidden) * HW,
                                hidden, HW);
    bt::silu_forward(gpart, gpart);
    bt::mul_inplace(hpart, gpart);
    // conv_point 1x1 (hidden -> D, bias-free).
    bt::Tensor hflat = sub_view(ff_t2_, 0, 1, hidden * HW);
    bt::conv2d_forward(hflat, blk.ff_point_W, nullptr, 1, hidden, H, W,
                       D, 1, 1, 1, 1, 0, 0, 1, 1, 1, ff_out_);
    // Back to sequence (N, D). No internal norm or residual.
    bt::nchw_to_sequence(ff_out_, 1, D, H, W, out);
}

// ─── forward ───────────────────────────────────────────────────────────────

void SanaDenoiser::forward(const bt::Tensor& latent, int H_lat, int W_lat,
                           float timestep, const PreparedConditioning& prepared,
                           Branch branch, bt::Tensor& out) {
    if (!prepared) fail("forward: prepared conditioning is empty");
    const auto* prep = dynamic_cast<const SanaPrepared*>(prepared.get());
    if (!prep) fail("forward: prepared conditioning has the wrong type");
    if (patch_embed_.W.size() == 0) fail("forward: weights not loaded");
    if (H_lat <= 0 || W_lat <= 0) fail("forward: H_lat/W_lat must be positive");

    const int D   = cfg_.inner_dim();
    const int IC  = cfg_.in_channels;
    const int OC  = cfg_.out_channels;
    const int N   = H_lat * W_lat;
    const bt::Dtype cdt = compute_dtype();
    const bt::Device dev = bt::default_device();

    if (latent.rows != 1 || latent.cols != IC * N) {
        fail("forward: latent must be (1, in_channels*H_lat*W_lat)");
    }

    // Caption context for this CFG branch.
    const bt::Tensor* ctx = &prep->ctx_cond;
    if (branch == Branch::Uncond) {
        if (!prep->has_uncond) fail("forward: uncond branch but no uncond ctx");
        ctx = &prep->ctx_uncond;
    }

    // ── timestep → AdaLayerNormSingle ─────────────────────────────────────
    // embedded_timestep = timestep_embedder(time_proj(t));  (1, D)
    // temb6 = time_embed.linear(silu(embedded_timestep));   (1, 6D)
    {
        std::vector<float> tval = {timestep};
        ts_ = bt::Tensor::from_host(tval.data(), 1, 1).to(dev);
        bt::timestep_embedding(ts_, /*dim=*/256, /*max_period=*/10000.0f, freq_);
    }
    const bt::Tensor* tin = &freq_;
    if (cdt != bt::Dtype::FP32) { bt::cast(freq_, freq_cd_, cdt); tin = &freq_cd_; }
    lin_(te_l1_, *tin, emb_);
    bt::silu_forward(emb_, emb_);
    lin_(te_l2_, emb_, emb_);
    emb_ = emb_.clone();                         // embedded_timestep (1, D)
    bt::silu_forward(emb_, emb_silu_);
    lin_(te_proj_, emb_silu_, temb6_);
    temb6_ = temb6_.clone();                     // (1, 6D)

    // ── patch_embed: latent (1, IC*N) NCHW → tokens (N, D) ────────────────
    bt::Tensor lat_cd;
    const bt::Tensor* lat = &latent;
    if (latent.dtype != cdt) { bt::cast(latent, lat_cd, cdt); lat = &lat_cd; }
    bt::nchw_to_sequence(*lat, 1, IC, H_lat, W_lat, mod_);   // (N, IC)
    lin_(patch_embed_, mod_, hidden_);                       // (N, D)
    hidden_ = hidden_.clone();

    // ── 28 SanaTransformerBlocks ──────────────────────────────────────────
    std::vector<bt::Tensor> ch;
    for (const Block& blk : blocks_) {
        // AdaLN-single: (scale_shift_table + temb6) → 6 chunks.
        std::vector<bt::Tensor> c;
        prof("adaln", [&] {
            mod_row_ = temb6_.clone();
            bt::add_inplace(mod_row_, blk.scale_shift);
            slice_modulation_chunks(mod_row_, D, 6, ch);
            c.reserve(6);
            for (auto& t : ch) c.push_back(t.clone());
        });
        const bt::Tensor& shift_msa = c[0];
        const bt::Tensor& scale_msa = c[1];
        const bt::Tensor& gate_msa  = c[2];
        const bt::Tensor& shift_mlp = c[3];
        const bt::Tensor& scale_mlp = c[4];
        const bt::Tensor& gate_mlp  = c[5];

        // self-attn: h += gate_msa * attn1(modulate(LN(h)))
        prof("self_attn", [&] {
            detail::layernorm_batched(hidden_, ada_gamma_, ada_beta_, ln_,
                                      cfg_.norm_eps);
            bt::modulate(ln_, scale_msa, shift_msa, mod_);
            self_attention_(blk, N, H_lat, W_lat, mod_, sub_out_);
            bt::broadcast_mul(sub_out_, gate_msa, gated_);
            bt::add_inplace(hidden_, gated_);
        });

        // cross-attn: h += attn2(h, caption)  (no norm, no modulation)
        prof("cross_attn", [&] {
            cross_attention_(blk, *ctx, hidden_, sub_out_);
            bt::add_inplace(hidden_, sub_out_);
        });

        // mix-ffn: h += gate_mlp * ff(modulate(LN(h)))
        prof("mix_ffn", [&] {
            detail::layernorm_batched(hidden_, ada_gamma_, ada_beta_, ln_,
                                      cfg_.norm_eps);
            bt::modulate(ln_, scale_mlp, shift_mlp, mod_);
            mix_ffn_(blk, H_lat, W_lat, mod_, sub_out_);
            bt::broadcast_mul(sub_out_, gate_mlp, gated_);
            bt::add_inplace(hidden_, gated_);
        });
    }
    if (SanaProf::enabled()) ++g_sana_prof.forwards;

    // ── norm_out (SanaModulatedNorm) → proj_out → unpatchify ──────────────
    std::vector<bt::Tensor> no;
    slice_modulation_chunks(norm_out_sst_, D, 2, no);   // shift, scale
    bt::Tensor shift = no[0].clone();
    bt::Tensor scale = no[1].clone();
    bt::add_inplace(shift, emb_);                       // + embedded_timestep
    bt::add_inplace(scale, emb_);
    detail::layernorm_batched(hidden_, ada_gamma_, ada_beta_, ln_,
                              cfg_.norm_eps);
    bt::modulate(ln_, scale, shift, mod_);
    lin_(proj_out_, mod_, proj_);                       // (N, patch^2*OC)
    // patch_size == 1 → unpatchify is sequence_to_nchw with OC channels. The
    // DiT computes in cdt (BF16 on CUDA); the velocity is returned FP32 so the
    // sampler integrates the latent at full precision.
    if (cdt != bt::Dtype::FP32) {
        bt::sequence_to_nchw(proj_, 1, OC, H_lat, W_lat, out_cd_);
        bt::cast(out_cd_, out, bt::Dtype::FP32);
    } else {
        bt::sequence_to_nchw(proj_, 1, OC, H_lat, W_lat, out);  // (1, OC*N)
    }
}

}  // namespace brodiffusion::dit
