#include "brodiffusion/triposplat/flow_model.h"

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/dit/common.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::triposplat {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

constexpr float kPi = 3.14159265358979323846f;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("triposplat::FlowDiT: " + msg);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    if (const auto* v = f.find(key)) return *v;
    fail("missing tensor '" + key + "'");
}

// Read a small parameter (freqs / gamma) into host fp32, widening F16 if needed.
std::vector<float> read_host_f32(const st::TensorView& v) {
    std::vector<float> out(static_cast<std::size_t>(v.numel()));
    if (v.dtype == st::Dtype::F32) {
        const float* p = reinterpret_cast<const float*>(v.data);
        out.assign(p, p + out.size());
    } else if (v.dtype == st::Dtype::F16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(v.data);
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = bt::fp16_bits_to_fp32(p[i]);
    } else {
        fail("read_host_f32: unsupported dtype");
    }
    return out;
}

// Download a compute-dtype tensor to host fp32.
std::vector<float> download_f32(const bt::Tensor& t) {
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<uint16_t> bits = t.to_host_vector_fp16();
        std::vector<float> out(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) out[i] = bt::fp16_bits_to_fp32(bits[i]);
        return out;
    }
    return t.to_host_vector();
}

}  // namespace

FlowDiT::FlowDiT() : FlowDiT(FlowModelConfig{}) {}
FlowDiT::FlowDiT(const FlowModelConfig& cfg) : cfg_(cfg) {}
FlowDiT::~FlowDiT() = default;

// ─── weight loading ─────────────────────────────────────────────────────────

void FlowDiT::load_repo(const st::File& f, const std::string& p, Repo& r) {
    const int D = cfg_.model_channels;
    const int H = cfg_.num_heads;
    const int hidden = D / 8;                       // repo_hidden_ratio = 0.125
    st::upload_compute_checked(need(f, p + ".norm.weight"), D, 1, r.norm_g, p);
    st::upload_compute_checked(need(f, p + ".norm.bias"),   D, 1, r.norm_b, p);
    st::upload_compute_checked(need(f, p + ".gate_map.weight"),    hidden, D, r.gate_map.W, p);
    st::upload_compute_checked(need(f, p + ".content_map.weight"), hidden, D, r.content_map.W, p);
    st::upload_compute_checked(need(f, p + ".final_map.weight"),   3 * H, hidden, r.final_map.W, p);
    r.freqs0 = read_host_f32(need(f, p + ".freqs_0"));
    r.freqs1 = read_host_f32(need(f, p + ".freqs_1"));
    r.freqs2 = read_host_f32(need(f, p + ".freqs_2"));
}

void FlowDiT::load_block(const st::File& f, const std::string& p, bool modulated,
                         Block& blk) {
    const int D = cfg_.model_channels;
    const int FF = D * cfg_.mlp_ratio;
    const int H = cfg_.num_heads, hd = cfg_.head_dim;
    blk.modulated = modulated;

    // Fused qkv -> split into q/k/v (contiguous row ranges of the (3D,D) weight).
    bt::Tensor qkv_W, qkv_b;
    st::upload_compute_checked(need(f, p + ".attn.qkv.weight"), 3 * D, D, qkv_W, p);
    st::upload_compute_checked(need(f, p + ".attn.qkv.bias"),   3 * D, 1, qkv_b, p);
    auto split = [&](int idx, Linear& lin) {
        lin.W = bt::Tensor::zeros_on(qkv_W.device, D, D, qkv_W.dtype);
        lin.b = bt::Tensor::zeros_on(qkv_b.device, D, 1, qkv_b.dtype);
        bt::copy_d2d(qkv_W, static_cast<std::size_t>(idx) * D * D, lin.W, 0,
                     static_cast<std::size_t>(D) * D);
        bt::copy_d2d(qkv_b, static_cast<std::size_t>(idx) * D, lin.b, 0, D);
    };
    split(0, blk.q);
    split(1, blk.k);
    split(2, blk.v);

    st::upload_compute_checked(need(f, p + ".attn.out.weight"), D, D, blk.out.W, p);
    st::upload_compute_checked(need(f, p + ".attn.out.bias"),   D, 1, blk.out.b, p);

    // MultiHeadRMSNorm gamma (H, hd) folded with the *sqrt(head_dim) scale and
    // flattened to (1, D) so it drops straight into broadcast_mul.
    auto load_gamma = [&](const std::string& key, bt::Tensor& g) {
        std::vector<float> raw = read_host_f32(need(f, key));   // (H*hd,)
        const float s = std::sqrt(static_cast<float>(hd));
        for (float& x : raw) x *= s;
        g = detail::upload_host(raw.data(), 1, H * hd);
    };
    load_gamma(p + ".attn.q_norm.gamma", blk.q_gamma);
    load_gamma(p + ".attn.k_norm.gamma", blk.k_gamma);

    st::upload_compute_checked(need(f, p + ".mlp.mlp.0.weight"), FF, D, blk.ff0.W, p);
    st::upload_compute_checked(need(f, p + ".mlp.mlp.0.bias"),   FF, 1, blk.ff0.b, p);
    st::upload_compute_checked(need(f, p + ".mlp.mlp.2.weight"), D, FF, blk.ff2.W, p);
    st::upload_compute_checked(need(f, p + ".mlp.mlp.2.bias"),   D, 1, blk.ff2.b, p);

    if (modulated) {
        st::upload_compute_checked(need(f, p + ".shift_table"), 1, 6 * D, blk.shift_table, p);
    } else {
        st::upload_compute_checked(need(f, p + ".norm1.weight"), D, 1, blk.norm1_g, p);
        st::upload_compute_checked(need(f, p + ".norm1.bias"),   D, 1, blk.norm1_b, p);
        st::upload_compute_checked(need(f, p + ".norm2.weight"), D, 1, blk.norm2_g, p);
        st::upload_compute_checked(need(f, p + ".norm2.bias"),   D, 1, blk.norm2_b, p);
    }
}

void FlowDiT::load_weights(const st::File& f) {
    const int D = cfg_.model_channels;
    const int H = cfg_.num_heads, hd = cfg_.head_dim;

    st::upload_compute_checked(need(f, "t_embedder.mlp.0.weight"), D, 256, t_emb0_.W, "t");
    st::upload_compute_checked(need(f, "t_embedder.mlp.0.bias"),   D, 1, t_emb0_.b, "t");
    st::upload_compute_checked(need(f, "t_embedder.mlp.2.weight"), D, D, t_emb1_.W, "t");
    st::upload_compute_checked(need(f, "t_embedder.mlp.2.bias"),   D, 1, t_emb1_.b, "t");

    st::upload_compute_checked(need(f, "adaLN_modulation.1.weight"), 6 * D, D, adaLN_.W, "adaLN");
    st::upload_compute_checked(need(f, "adaLN_modulation.1.bias"),   6 * D, 1, adaLN_.b, "adaLN");

    st::upload_compute_checked(need(f, "input_layer.weight"), D, cfg_.in_channels, input_layer_.W, "in");
    st::upload_compute_checked(need(f, "input_layer.bias"),   D, 1, input_layer_.b, "in");
    st::upload_compute_checked(need(f, "cond_embedder.weight"), D, cfg_.cond_channels, cond_embedder_.W, "c1");
    st::upload_compute_checked(need(f, "cond_embedder.bias"),   D, 1, cond_embedder_.b, "c1");
    st::upload_compute_checked(need(f, "cond_embedder2.weight"), D, cfg_.cond2_channels, cond_embedder2_.W, "c2");
    st::upload_compute_checked(need(f, "cond_embedder2.bias"),   D, 1, cond_embedder2_.b, "c2");

    st::upload_compute_checked(need(f, "cam_refiner.mlp.0.weight"), D, cfg_.cam_channels, cam0_.W, "cam");
    st::upload_compute_checked(need(f, "cam_refiner.mlp.0.bias"),   D, 1, cam0_.b, "cam");
    st::upload_compute_checked(need(f, "cam_refiner.mlp.2.weight"), D, D, cam2_.W, "cam");
    st::upload_compute_checked(need(f, "cam_refiner.mlp.2.bias"),   D, 1, cam2_.b, "cam");

    st::upload_compute_checked(need(f, "shift_table"), 1, 2 * D, shift_table_, "shift");
    st::upload_compute_checked(need(f, "out_layer.weight"), cfg_.out_channels, D, out_layer_.W, "out");
    st::upload_compute_checked(need(f, "out_layer.bias"),   cfg_.out_channels, 1, out_layer_.b, "out");
    st::upload_compute_checked(need(f, "cam_out_layer.weight"), cfg_.cam_channels, D, cam_out_layer_.W, "cout");
    st::upload_compute_checked(need(f, "cam_out_layer.bias"),   cfg_.cam_channels, 1, cam_out_layer_.b, "cout");

    noise_repo_.resize(cfg_.num_refiner_blocks);
    context_repo_.resize(cfg_.num_refiner_blocks);
    noise_refiner_.resize(cfg_.num_refiner_blocks);
    context_refiner_.resize(cfg_.num_refiner_blocks);
    for (int i = 0; i < cfg_.num_refiner_blocks; ++i) {
        load_repo(f, "noise_repo_layers." + std::to_string(i), noise_repo_[i]);
        load_repo(f, "context_repo_layers." + std::to_string(i), context_repo_[i]);
        load_block(f, "noise_refiner." + std::to_string(i), /*modulated=*/true, noise_refiner_[i]);
        load_block(f, "context_refiner." + std::to_string(i), /*modulated=*/false, context_refiner_[i]);
    }
    repo_.resize(cfg_.num_blocks);
    blocks_.resize(cfg_.num_blocks);
    for (int i = 0; i < cfg_.num_blocks; ++i) {
        load_repo(f, "repo_layers." + std::to_string(i), repo_[i]);
        load_block(f, "blocks." + std::to_string(i), /*modulated=*/true, blocks_[i]);
    }

    // Affine-free LayerNorm params (ones / zeros) reused by every modulated norm.
    {
        std::vector<float> ones(D, 1.0f), zeros(D, 0.0f);
        ada_gamma_ = detail::upload_host(ones.data(), D, 1);
        ada_beta_  = detail::upload_host(zeros.data(), D, 1);
    }

    // Precompute the absolute position embedding pos_embedder(sobol_pe). It
    // depends only on the baked Sobol table, so build the (L, D) result once in
    // FP32 (the embedder's frequencies reach 2^16 — fp16 intermediates corrupt
    // it) and store at the compute dtype (the sin/cos result lies in [-1,1]).
    {
        const int L = cfg_.q_token_length;
        const int in_ch = 3, max_res = 16;
        const int freq_dim = D / in_ch / 2;          // 170
        std::vector<float> freqs(freq_dim);
        const int res_dim = std::max(0, freq_dim - max_res);
        for (int i = 0; i < freq_dim; ++i) {
            float e;
            if (i < max_res) e = static_cast<float>(i);
            else e = static_cast<float>(i - max_res) / std::max(res_dim, 1) * max_res;
            freqs[i] = std::pow(2.0f, e);
        }
        std::vector<float> emb(static_cast<std::size_t>(L) * D, 0.0f);
        for (int n = 0; n < L; ++n) {
            for (int a = 0; a < in_ch; ++a) {
                const float x = kFlowSobolPosPE[(static_cast<std::size_t>(n) * in_ch) + a];
                float* row = &emb[(static_cast<std::size_t>(n) * D) + a * (2 * freq_dim)];
                for (int j = 0; j < freq_dim; ++j) {
                    const float arg = x * freqs[j] * 2.0f * kPi;
                    row[j] = std::sin(arg);
                    row[freq_dim + j] = std::cos(arg);
                }
            }
        }
        pos_emb_ = detail::upload_host(emb.data(), L, D);
    }

    loaded_ = true;
}

// ─── content-dependent per-head 3D rotary tables ────────────────────────────

void FlowDiT::build_rope(const bt::Tensor& hidden, const Repo& r,
                         bt::Tensor& cos_out, bt::Tensor& sin_out) {
    const int D = cfg_.model_channels, H = cfg_.num_heads, hd = cfg_.head_dim;
    const int L = hidden.rows;
    const int half = hd / 2;                        // 32
    const int f0 = static_cast<int>(r.freqs0.size());  // 10
    const int f1 = static_cast<int>(r.freqs1.size());  // 10
    const int f2 = static_cast<int>(r.freqs2.size());  // 12

    // norm -> silu(gate)*content -> final_map  =>  delta_pos (L, 3H)
    bt::Tensor h, g, c, dp;
    detail::layernorm_batched(hidden, r.norm_g, r.norm_b, h, 1e-5f);
    detail::linear_batched(r.gate_map.W, nullptr, h, g);
    bt::silu_forward(g, g);
    detail::linear_batched(r.content_map.W, nullptr, h, c);
    bt::mul_inplace(g, c);
    detail::linear_batched(r.final_map.W, nullptr, g, dp);

    // The rotary is FP32 in the reference; build cos/sin host-side in FP32
    // (also satisfies rope_apply_perhead's FP32-table requirement). delta_pos
    // is small (L*3H); the tables are L*H*half each. A device sincos kernel can
    // replace this host round-trip later (cf. dit/common.h's pack/unpack note).
    bt::sync_all();
    std::vector<float> d = download_f32(dp);        // (L, 3H), [head, axis]
    std::vector<float> cosv(static_cast<std::size_t>(L) * H * half);
    std::vector<float> sinv(cosv.size());
    for (int l = 0; l < L; ++l) {
        for (int hh = 0; hh < H; ++hh) {
            const float d0 = d[(static_cast<std::size_t>(l) * H + hh) * 3 + 0];
            const float d1 = d[(static_cast<std::size_t>(l) * H + hh) * 3 + 1];
            const float d2 = d[(static_cast<std::size_t>(l) * H + hh) * 3 + 2];
            float* cr = &cosv[(static_cast<std::size_t>(l) * H + hh) * half];
            float* sr = &sinv[(static_cast<std::size_t>(l) * H + hh) * half];
            int j = 0;
            for (int i = 0; i < f0; ++i, ++j) { float a = d0 * r.freqs0[i] * kPi; cr[j] = std::cos(a); sr[j] = std::sin(a); }
            for (int i = 0; i < f1; ++i, ++j) { float a = d1 * r.freqs1[i] * kPi; cr[j] = std::cos(a); sr[j] = std::sin(a); }
            for (int i = 0; i < f2; ++i, ++j) { float a = d2 * r.freqs2[i] * kPi; cr[j] = std::cos(a); sr[j] = std::sin(a); }
        }
    }
    cos_out = bt::Tensor::from_host_on(hidden.device, cosv.data(), L * H, half);
    sin_out = bt::Tensor::from_host_on(hidden.device, sinv.data(), L * H, half);
    (void)D;
}

// ─── self-attention (per-head rotary, then per-head Q/K RMSNorm) ─────────────

void FlowDiT::attention(const bt::Tensor& x, const Block& blk,
                        const bt::Tensor& cos, const bt::Tensor& sin,
                        bt::Tensor& out) {
    const int H = cfg_.num_heads, hd = cfg_.head_dim;

    bt::Tensor q, k, v, qr, kr, attn;
    detail::linear_batched(blk.q.W, &blk.q.b, x, q);
    detail::linear_batched(blk.k.W, &blk.k.b, x, k);
    detail::linear_batched(blk.v.W, &blk.v.b, x, v);

    // Rotary first, then qk_rms_norm (reference order).
    bt::rope_apply_perhead(q, cos, sin, hd, H, qr);
    bt::rope_apply_perhead(k, cos, sin, hd, H, kr);

    // MultiHeadRMSNorm = F.normalize(per head) * (gamma*sqrt(hd)).
    bt::l2_norm_forward(qr, hd, H, 1e-12f, qr);
    bt::broadcast_mul(qr, blk.q_gamma, qr);
    bt::l2_norm_forward(kr, hd, H, 1e-12f, kr);
    bt::broadcast_mul(kr, blk.k_gamma, kr);

    bt::flash_attention_forward(qr, kr, v, /*d_mask=*/nullptr, H, /*causal=*/false, attn);
    detail::linear_batched(blk.out.W, &blk.out.b, attn, out);
}

// ─── one UnifiedTransformerBlock in place ────────────────────────────────────

void FlowDiT::run_block(bt::Tensor& x, const Block& blk, const bt::Tensor* t_mod,
                        const bt::Tensor& cos, const bt::Tensor& sin) {
    const int D = cfg_.model_channels;
    bt::Tensor ln, h, attn, ff_mid, ff_out, gated;

    if (blk.modulated) {
        // mod = t_mod + shift_table; chunk into 6 (1,D) modulation vectors.
        bt::Tensor mod = t_mod->clone();
        bt::add_inplace(mod, blk.shift_table);
        std::vector<bt::Tensor> ch;
        dit::slice_modulation_chunks(mod, D, 6, ch);
        bt::Tensor& shift_msa = ch[0]; bt::Tensor& scale_msa = ch[1]; bt::Tensor& gate_msa = ch[2];
        bt::Tensor& shift_mlp = ch[3]; bt::Tensor& scale_mlp = ch[4]; bt::Tensor& gate_mlp = ch[5];

        detail::layernorm_batched(x, ada_gamma_, ada_beta_, ln, 1e-6f);
        bt::modulate(ln, scale_msa, shift_msa, h);
        attention(h, blk, cos, sin, attn);
        bt::broadcast_mul(attn, gate_msa, gated);
        bt::add_inplace(x, gated);

        detail::layernorm_batched(x, ada_gamma_, ada_beta_, ln, 1e-6f);
        bt::modulate(ln, scale_mlp, shift_mlp, h);
        detail::linear_batched(blk.ff0.W, &blk.ff0.b, h, ff_mid);
        bt::gelu_forward(ff_mid, ff_mid);
        detail::linear_batched(blk.ff2.W, &blk.ff2.b, ff_mid, ff_out);
        bt::broadcast_mul(ff_out, gate_mlp, gated);
        bt::add_inplace(x, gated);
    } else {
        detail::layernorm_batched(x, blk.norm1_g, blk.norm1_b, ln, 1e-6f);
        attention(ln, blk, cos, sin, attn);
        bt::add_inplace(x, attn);

        detail::layernorm_batched(x, blk.norm2_g, blk.norm2_b, ln, 1e-6f);
        detail::linear_batched(blk.ff0.W, &blk.ff0.b, ln, ff_mid);
        bt::gelu_forward(ff_mid, ff_mid);
        detail::linear_batched(blk.ff2.W, &blk.ff2.b, ff_mid, ff_out);
        bt::add_inplace(x, ff_out);
    }
}

// ─── forward ─────────────────────────────────────────────────────────────────

void FlowDiT::forward(const bt::Tensor& latent, const bt::Tensor& camera,
                      const bt::Tensor& feature1, const bt::Tensor& feature2,
                      float t,
                      bt::Tensor& out_latent, bt::Tensor& out_camera) {
    if (!loaded_) fail("forward: weights not loaded");
    const int D = cfg_.model_channels;
    const int L = latent.rows;
    if (L != cfg_.q_token_length) fail("forward: latent token count must equal q_token_length");

    // t_emb = t_embedder(t); timestep_embedding built host-side (one token).
    bt::Tensor t_emb;
    {
        const int fe = 256, half = fe / 2;
        std::vector<float> e(fe);
        for (int i = 0; i < half; ++i) {
            const float freq = std::exp(-std::log(10000.0f) * static_cast<float>(i) / half);
            const float arg = t * freq;
            e[i] = std::cos(arg);
            e[half + i] = std::sin(arg);
        }
        bt::Tensor temb = detail::upload_host(e.data(), 1, fe);
        bt::Tensor h0;
        detail::linear_batched(t_emb0_.W, &t_emb0_.b, temb, h0);
        bt::silu_forward(h0, h0);
        detail::linear_batched(t_emb1_.W, &t_emb1_.b, h0, t_emb);
    }
    // t_mod = adaLN_modulation(t_emb) = Linear(silu(t_emb))  [share_mod].
    bt::Tensor t_mod, silu_t;
    silu_t = t_emb.clone();
    bt::silu_forward(silu_t, silu_t);
    detail::linear_batched(adaLN_.W, &adaLN_.b, silu_t, t_mod);

    // h_x = input_layer(z) + pos_embedder(sobol_pe)
    bt::Tensor h_x;
    detail::linear_batched(input_layer_.W, &input_layer_.b, latent, h_x);
    bt::add_inplace(h_x, pos_emb_);

    // h_cond = cond_embedder(feature1) + cond_embedder2(feature2)
    bt::Tensor h_cond, c2;
    detail::linear_batched(cond_embedder_.W, &cond_embedder_.b, feature1, h_cond);
    detail::linear_batched(cond_embedder2_.W, &cond_embedder2_.b, feature2, c2);
    bt::add_inplace(h_cond, c2);

    bt::Tensor cos, sin;
    for (int i = 0; i < cfg_.num_refiner_blocks; ++i) {
        build_rope(h_x, noise_repo_[i], cos, sin);
        run_block(h_x, noise_refiner_[i], &t_mod, cos, sin);
    }
    for (int i = 0; i < cfg_.num_refiner_blocks; ++i) {
        build_rope(h_cond, context_repo_[i], cos, sin);
        run_block(h_cond, context_refiner_[i], /*t_mod=*/nullptr, cos, sin);
    }

    // h_cam = cam_refiner(camera)  [MLP: Linear, GELU(tanh), Linear]
    bt::Tensor h_cam, cam_mid;
    detail::linear_batched(cam0_.W, &cam0_.b, camera, cam_mid);
    bt::gelu_forward(cam_mid, cam_mid);
    detail::linear_batched(cam2_.W, &cam2_.b, cam_mid, h_cam);

    // h = concat([h_x, h_cond, h_cam])
    const int Kc = h_cond.rows;
    const int Lh = L + Kc + 1;
    bt::Tensor h = bt::Tensor::zeros_on(h_x.device, Lh, D, h_x.dtype);
    bt::copy_d2d(h_x,   0, h, 0,                                static_cast<std::size_t>(L) * D);
    bt::copy_d2d(h_cond, 0, h, static_cast<std::size_t>(L) * D, static_cast<std::size_t>(Kc) * D);
    bt::copy_d2d(h_cam,  0, h, static_cast<std::size_t>(L + Kc) * D, static_cast<std::size_t>(1) * D);

    for (int i = 0; i < cfg_.num_blocks; ++i) {
        build_rope(h, repo_[i], cos, sin);
        run_block(h, blocks_[i], &t_mod, cos, sin);
    }

    // Final affine-free LayerNorm over the latent and camera slices, then the
    // shift_table+t_emb gate.  shift = shift_table[:,0]+t_emb, scale = [:,1]+t_emb.
    bt::Tensor hx, hcam;
    {
        bt::Tensor hx_slice = bt::Tensor::zeros_on(h.device, L, D, h.dtype);
        bt::copy_d2d(h, 0, hx_slice, 0, static_cast<std::size_t>(L) * D);
        detail::layernorm_batched(hx_slice, ada_gamma_, ada_beta_, hx, 1e-5f);

        bt::Tensor cam_slice = bt::Tensor::zeros_on(h.device, 1, D, h.dtype);
        bt::copy_d2d(h, static_cast<std::size_t>(L + Kc) * D, cam_slice, 0, static_cast<std::size_t>(D));
        detail::layernorm_batched(cam_slice, ada_gamma_, ada_beta_, hcam, 1e-5f);
    }
    // shift / scale: each (1, D).
    bt::Tensor shift = bt::Tensor::zeros_on(h.device, 1, D, h.dtype);
    bt::Tensor scale = bt::Tensor::zeros_on(h.device, 1, D, h.dtype);
    bt::copy_d2d(shift_table_, 0, shift, 0, D);
    bt::copy_d2d(shift_table_, D, scale, 0, D);
    bt::add_inplace(shift, t_emb);
    bt::add_inplace(scale, t_emb);

    bt::Tensor hx_mod, hcam_mod;
    bt::modulate(hx, scale, shift, hx_mod);            // hx*(1+scale)+shift
    bt::modulate(hcam, scale, shift, hcam_mod);

    detail::linear_batched(out_layer_.W, &out_layer_.b, hx_mod, out_latent);
    detail::linear_batched(cam_out_layer_.W, &cam_out_layer_.b, hcam_mod, out_camera);
}

}  // namespace brodiffusion::triposplat
