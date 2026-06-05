#include "brodiffusion/triposplat/octree_decoder.h"

#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::triposplat {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

constexpr float kPi = 3.14159265358979323846f;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("triposplat::OctreeGaussianDecoder: " + msg);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    if (const auto* v = f.find(key)) return *v;
    fail("missing tensor '" + key + "'");
}

std::vector<float> read_host_f32(const st::TensorView& v) {
    std::vector<float> out(static_cast<std::size_t>(v.numel()));
    if (v.dtype == st::Dtype::F32) {
        const float* p = reinterpret_cast<const float*>(v.data);
        out.assign(p, p + out.size());
    } else if (v.dtype == st::Dtype::F16) {
        const std::uint16_t* p = reinterpret_cast<const std::uint16_t*>(v.data);
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = bt::fp16_bits_to_fp32(p[i]);
    } else {
        fail("read_host_f32: unsupported dtype");
    }
    return out;
}

std::vector<float> download_f32(const bt::Tensor& t) {
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits = t.to_host_vector_fp16();
        std::vector<float> out(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) out[i] = bt::fp16_bits_to_fp32(bits[i]);
        return out;
    }
    return t.to_host_vector();
}

// softplus / its inverse, matching torch F.softplus (no beta) and the
// Gaussian wrapper's inverse_scaling_activation for the softplus branch.
inline float softplus(float x) {
    // numerically stable log(1+exp(x))
    return x > 20.0f ? x : std::log1p(std::exp(x));
}
inline float inverse_softplus(float x) {
    // x + log(-expm1(-x)) == x + log(1 - exp(-x))  (reference lambda)
    return x + std::log(-std::expm1(-x));
}

}  // namespace

OctreeGaussianDecoder::OctreeGaussianDecoder()
    : OctreeGaussianDecoder(OctreeDecoderConfig{}) {}
OctreeGaussianDecoder::OctreeGaussianDecoder(const OctreeDecoderConfig& cfg) : cfg_(cfg) {}
OctreeGaussianDecoder::~OctreeGaussianDecoder() = default;

// ─── weight loading ─────────────────────────────────────────────────────────

void OctreeGaussianDecoder::load_linear(const st::File& f, const std::string& key,
                                        int rows, int cols, Linear& lin) {
    st::upload_compute_checked(need(f, key + ".weight"), rows, cols, lin.W, key);
    st::upload_compute_checked(need(f, key + ".bias"),   rows, 1, lin.b, key);
}

bt::Tensor OctreeGaussianDecoder::load_gamma(const st::File& f, const std::string& key) {
    // MultiHeadRMSNorm gamma (H, head_dim) pre-multiplied by sqrt(head_dim) and
    // flattened to (1, D) so it drops straight into broadcast_mul (matches the
    // flow DiT's fold).
    const int H = cfg_.num_heads, hd = cfg_.head_dim;
    std::vector<float> raw = read_host_f32(need(f, key));   // (H*hd,)
    const float s = std::sqrt(static_cast<float>(hd));
    for (float& x : raw) x *= s;
    return detail::upload_host(raw.data(), 1, H * hd);
}

void OctreeGaussianDecoder::load_cross(const st::File& f, const std::string& p, CrossAttn& a) {
    const int D = cfg_.model_channels, C = cfg_.cond_channels;
    load_linear(f, p + ".to_q", D, D, a.to_q);

    // Fused to_kv (2D, cond_ch) -> k (rows[0:D]), v (rows[D:2D]).
    bt::Tensor kv_W, kv_b;
    st::upload_compute_checked(need(f, p + ".to_kv.weight"), 2 * D, C, kv_W, p);
    st::upload_compute_checked(need(f, p + ".to_kv.bias"),   2 * D, 1, kv_b, p);
    auto split = [&](int idx, Linear& lin) {
        lin.W = bt::Tensor::zeros_on(kv_W.device, D, C, kv_W.dtype);
        lin.b = bt::Tensor::zeros_on(kv_b.device, D, 1, kv_b.dtype);
        bt::copy_d2d(kv_W, static_cast<std::size_t>(idx) * D * C, lin.W, 0,
                     static_cast<std::size_t>(D) * C);
        bt::copy_d2d(kv_b, static_cast<std::size_t>(idx) * D, lin.b, 0, D);
    };
    split(0, a.to_k);
    split(1, a.to_v);

    a.q_gamma = load_gamma(f, p + ".q_rms_norm.gamma");
    a.k_gamma = load_gamma(f, p + ".k_rms_norm.gamma");
    load_linear(f, p + ".to_out", D, D, a.to_out);
}

void OctreeGaussianDecoder::load_self(const st::File& f, const std::string& p, SelfAttn& a) {
    const int D = cfg_.model_channels;
    bt::Tensor qkv_W, qkv_b;
    st::upload_compute_checked(need(f, p + ".to_qkv.weight"), 3 * D, D, qkv_W, p);
    st::upload_compute_checked(need(f, p + ".to_qkv.bias"),   3 * D, 1, qkv_b, p);
    auto split = [&](int idx, Linear& lin) {
        lin.W = bt::Tensor::zeros_on(qkv_W.device, D, D, qkv_W.dtype);
        lin.b = bt::Tensor::zeros_on(qkv_b.device, D, 1, qkv_b.dtype);
        bt::copy_d2d(qkv_W, static_cast<std::size_t>(idx) * D * D, lin.W, 0,
                     static_cast<std::size_t>(D) * D);
        bt::copy_d2d(qkv_b, static_cast<std::size_t>(idx) * D, lin.b, 0, D);
    };
    split(0, a.q);
    split(1, a.k);
    split(2, a.v);
    a.q_gamma = load_gamma(f, p + ".q_rms_norm.gamma");
    a.k_gamma = load_gamma(f, p + ".k_rms_norm.gamma");
    load_linear(f, p + ".to_out", D, D, a.to_out);
}

void OctreeGaussianDecoder::load_weights(const st::File& f) {
    const int D = cfg_.model_channels, FF = D * cfg_.mlp_ratio;

    // ── octree ──
    load_linear(f, "octree.in_proj", D, 3, oct_in_proj_);
    load_linear(f, "octree.input_layer", D, D, oct_input_layer_);
    load_linear(f, "octree.out_proj", 8, D, oct_out_proj_);
    load_linear(f, "octree.l_embedder.mlp.0", D, 256, oct_l_emb0_);
    load_linear(f, "octree.l_embedder.mlp.2", D, D, oct_l_emb1_);
    load_linear(f, "octree.adaLN_modulation.1", 6 * D, D, oct_adaLN_);
    oct_blocks_.resize(cfg_.octree_num_blocks);
    for (int i = 0; i < cfg_.octree_num_blocks; ++i) {
        const std::string p = "octree.blocks." + std::to_string(i);
        load_cross(f, p + ".cross_attn", oct_blocks_[i].cross);
        load_linear(f, p + ".mlp.mlp.0", FF, D, oct_blocks_[i].ff0);
        load_linear(f, p + ".mlp.mlp.2", D, FF, oct_blocks_[i].ff2);
    }

    // ── gs ──
    load_linear(f, "gs.in_proj", D, 3, gs_in_proj_);
    load_linear(f, "gs.input_layer", D, D, gs_input_layer_);
    load_linear(f, "gs.out_proj", cfg_.gs_out_channels, D, gs_out_proj_);
    gs_blocks_.resize(cfg_.gs_num_blocks);
    for (int i = 0; i < cfg_.gs_num_blocks; ++i) {
        const std::string p = "gs.blocks." + std::to_string(i);
        load_self(f, p + ".self_attn", gs_blocks_[i].self_attn);
        load_cross(f, p + ".cross_attn", gs_blocks_[i].cross);
        load_linear(f, p + ".mlp.mlp.0", FF, D, gs_blocks_[i].ff0);
        load_linear(f, p + ".mlp.mlp.2", D, FF, gs_blocks_[i].ff2);
        st::upload_compute_checked(need(f, p + ".norm2.weight"), D, 1,
                                   gs_blocks_[i].norm2_g, p);
        st::upload_compute_checked(need(f, p + ".norm2.bias"), D, 1,
                                   gs_blocks_[i].norm2_b, p);
    }

    // gaussian-assembly constants (host)
    base_offset_scale_ = read_host_f32(need(f, "gs.base_offset_scale"))[0];
    perturb_ = read_host_f32(need(f, "gs.points_offset_perturbation"));  // (ng,3)

    // affine-free LayerNorm params (ones / zeros)
    std::vector<float> ones(D, 1.0f), zeros(D, 0.0f);
    ada_gamma_ = detail::upload_host(ones.data(), D, 1);
    ada_beta_  = detail::upload_host(zeros.data(), D, 1);

    loaded_ = true;
}

// ─── absolute position embedding (PcdAbsolutePositionEmbedderV2) ──────────────

bt::Tensor OctreeGaussianDecoder::pos_embed_v2(const std::vector<float>& pts, int n) const {
    const int D = cfg_.model_channels;
    const int in_ch = 3, max_res = 10;
    const int freq_dim = D / in_ch / 2;             // 170
    std::vector<float> freqs(freq_dim);
    for (int j = 0; j < freq_dim; ++j) {
        const float lg = static_cast<float>(max_res) * j / std::max(freq_dim - 1, 1);
        freqs[j] = std::pow(2.0f, lg);
    }
    // row layout: [axis0 sin(170), axis0 cos(170), axis1 ..., axis2 ...], pad to D.
    std::vector<float> emb(static_cast<std::size_t>(n) * D, 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < in_ch; ++a) {
            const float x = pts[(static_cast<std::size_t>(i) * in_ch) + a];
            float* row = &emb[(static_cast<std::size_t>(i) * D) + a * (2 * freq_dim)];
            for (int j = 0; j < freq_dim; ++j) {
                const float ang = x * freqs[j] * kPi;
                row[j] = std::sin(ang);
                row[freq_dim + j] = std::cos(ang);
            }
        }
    }
    return detail::upload_host(emb.data(), n, D);
}

bt::Tensor OctreeGaussianDecoder::embed_points(const std::vector<float>& pts, int n,
                                               const Linear& in_proj,
                                               const Linear& input_layer) const {
    // in_proj(points): build the (n,3) point tensor at the compute dtype.
    bt::Tensor p = detail::upload_host(pts.data(), n, 3);
    bt::Tensor h, pe = pos_embed_v2(pts, n);
    detail::linear_batched(in_proj.W, &in_proj.b, p, h);
    bt::add_inplace(h, pe);
    bt::Tensor out;
    detail::linear_batched(input_layer.W, &input_layer.b, h, out);
    return out;
}

// ─── attention ────────────────────────────────────────────────────────────────

void OctreeGaussianDecoder::cross_attention(const bt::Tensor& x, const CrossAttn& a,
                                            const bt::Tensor& cond, bt::Tensor& out) {
    const int H = cfg_.num_heads, hd = cfg_.head_dim;
    bt::Tensor q, k, v, attn;
    detail::linear_batched(a.to_q.W, &a.to_q.b, x, q);
    detail::linear_batched(a.to_k.W, &a.to_k.b, cond, k);
    detail::linear_batched(a.to_v.W, &a.to_v.b, cond, v);

    // MultiHeadRMSNorm on Q and K (no rotary in the decoder).
    bt::l2_norm_forward(q, hd, H, 1e-12f, q);
    bt::broadcast_mul(q, a.q_gamma, q);
    bt::l2_norm_forward(k, hd, H, 1e-12f, k);
    bt::broadcast_mul(k, a.k_gamma, k);

    bt::flash_attention_forward(q, k, v, /*d_mask=*/nullptr, H, /*causal=*/false, attn);
    detail::linear_batched(a.to_out.W, &a.to_out.b, attn, out);
}

void OctreeGaussianDecoder::self_attention(const bt::Tensor& x, const SelfAttn& a,
                                           bt::Tensor& out) {
    const int H = cfg_.num_heads, hd = cfg_.head_dim;
    bt::Tensor q, k, v, attn;
    detail::linear_batched(a.q.W, &a.q.b, x, q);
    detail::linear_batched(a.k.W, &a.k.b, x, k);
    detail::linear_batched(a.v.W, &a.v.b, x, v);
    bt::l2_norm_forward(q, hd, H, 1e-12f, q);
    bt::broadcast_mul(q, a.q_gamma, q);
    bt::l2_norm_forward(k, hd, H, 1e-12f, k);
    bt::broadcast_mul(k, a.k_gamma, k);
    bt::flash_attention_forward(q, k, v, /*d_mask=*/nullptr, H, /*causal=*/false, attn);
    detail::linear_batched(a.to_out.W, &a.to_out.b, attn, out);
}

// ─── octree forward ───────────────────────────────────────────────────────────

void OctreeGaussianDecoder::octree_logits(const std::vector<float>& coords, int n,
                                          int res, const bt::Tensor& cond,
                                          bt::Tensor& logits_out) {
    if (!loaded_) fail("octree_logits: weights not loaded");
    const int D = cfg_.model_channels;

    bt::Tensor h = embed_points(coords, n, oct_in_proj_, oct_input_layer_);

    // l_embedder(res): LevelEmbedder (freq 256, max_period 1024), cos-then-sin,
    // *2pi; then mlp(Linear, SiLU, Linear). share_mod adaLN -> (1, 6D).
    bt::Tensor l_mod;
    {
        const int fe = 256, half = fe / 2;
        const float max_period = 1024.0f;
        std::vector<float> e(fe);
        for (int i = 0; i < half; ++i) {
            const float freq = std::exp(-std::log(max_period) * static_cast<float>(i) / half);
            const float arg = static_cast<float>(res) * freq * 2.0f * kPi;
            e[i] = std::cos(arg);
            e[half + i] = std::sin(arg);
        }
        bt::Tensor temb = detail::upload_host(e.data(), 1, fe);
        bt::Tensor l0, l_emb, silu_l;
        detail::linear_batched(oct_l_emb0_.W, &oct_l_emb0_.b, temb, l0);
        bt::silu_forward(l0, l0);
        detail::linear_batched(oct_l_emb1_.W, &oct_l_emb1_.b, l0, l_emb);
        silu_l = l_emb.clone();
        bt::silu_forward(silu_l, silu_l);
        detail::linear_batched(oct_adaLN_.W, &oct_adaLN_.b, silu_l, l_mod);
    }
    std::vector<bt::Tensor> ch;
    bt::Tensor mod = l_mod;  // shared across blocks
    {
        // slice 6 chunks once
        ch.resize(6);
        for (int c = 0; c < 6; ++c) {
            ch[c] = bt::Tensor::zeros_on(mod.device, 1, D, mod.dtype);
            bt::copy_d2d(mod, static_cast<std::size_t>(c) * D, ch[c], 0, D);
        }
    }
    bt::Tensor& shift_msa = ch[0]; bt::Tensor& scale_msa = ch[1]; bt::Tensor& gate_msa = ch[2];
    bt::Tensor& shift_mlp = ch[3]; bt::Tensor& scale_mlp = ch[4]; bt::Tensor& gate_mlp = ch[5];

    bt::Tensor ln, hm, attn, ff_mid, ff_out, gated;
    for (auto& blk : oct_blocks_) {
        detail::layernorm_batched(h, ada_gamma_, ada_beta_, ln, 1e-6f);
        bt::modulate(ln, scale_msa, shift_msa, hm);
        cross_attention(hm, blk.cross, cond, attn);
        bt::broadcast_mul(attn, gate_msa, gated);
        bt::add_inplace(h, gated);

        detail::layernorm_batched(h, ada_gamma_, ada_beta_, ln, 1e-6f);
        bt::modulate(ln, scale_mlp, shift_mlp, hm);
        detail::linear_batched(blk.ff0.W, &blk.ff0.b, hm, ff_mid);
        bt::gelu_forward(ff_mid, ff_mid);
        detail::linear_batched(blk.ff2.W, &blk.ff2.b, ff_mid, ff_out);
        bt::broadcast_mul(ff_out, gate_mlp, gated);
        bt::add_inplace(h, gated);
    }

    // final affine-free LayerNorm (default eps 1e-5), out_proj -> (N, 8).
    bt::Tensor hn;
    detail::layernorm_batched(h, ada_gamma_, ada_beta_, hn, 1e-5f);
    detail::linear_batched(oct_out_proj_.W, &oct_out_proj_.b, hn, logits_out);
}

// ─── gs forward ───────────────────────────────────────────────────────────────

void OctreeGaussianDecoder::gs_features(const std::vector<float>& points, int n,
                                        const bt::Tensor& cond, bt::Tensor& feats_out) {
    if (!loaded_) fail("gs_features: weights not loaded");

    bt::Tensor h = embed_points(points, n, gs_in_proj_, gs_input_layer_);

    bt::Tensor ln, attn, cattn, ff_mid, ff_out;
    for (auto& blk : gs_blocks_) {
        // self-attn (norm1 affine-free)
        detail::layernorm_batched(h, ada_gamma_, ada_beta_, ln, 1e-6f);
        self_attention(ln, blk.self_attn, attn);
        bt::add_inplace(h, attn);
        // cross-attn (norm2 affine)
        detail::layernorm_batched(h, blk.norm2_g, blk.norm2_b, ln, 1e-6f);
        cross_attention(ln, blk.cross, cond, cattn);
        bt::add_inplace(h, cattn);
        // FFN (norm3 affine-free)
        detail::layernorm_batched(h, ada_gamma_, ada_beta_, ln, 1e-6f);
        detail::linear_batched(blk.ff0.W, &blk.ff0.b, ln, ff_mid);
        bt::gelu_forward(ff_mid, ff_mid);
        detail::linear_batched(blk.ff2.W, &blk.ff2.b, ff_mid, ff_out);
        bt::add_inplace(h, ff_out);
    }

    // final affine-free LayerNorm (default eps 1e-5), out_proj -> (M, 480).
    bt::Tensor hn;
    detail::layernorm_batched(h, ada_gamma_, ada_beta_, hn, 1e-5f);
    detail::linear_batched(gs_out_proj_.W, &gs_out_proj_.b, hn, feats_out);
}

// ─── gaussian assembly ────────────────────────────────────────────────────────

GaussianSplats OctreeGaussianDecoder::build_gaussians(
    const std::vector<float>& points, int n, const bt::Tensor& feats) {
    const int ng = cfg_.num_gaussians;
    const int OC = cfg_.gs_out_channels;

    // layout ranges within each 480-vector (ng=32):
    //   _xyz[0:96] _features_dc[96:192] _scaling[192:288]
    //   _rotation[288:416] _opacity[416:448] _offset_scale[448:480]
    const int R_XYZ = 0, R_FDC = ng * 3, R_SCALE = ng * 6, R_ROT = ng * 9;
    const int R_OPAC = R_ROT + ng * 4, R_OFFS = R_OPAC + ng;

    const float scale_bias = inverse_softplus(0.004f);     // scaling_bias
    const float min_kernel = 0.0009f;                       // filter_kernel_size_3d
    const float opacity_bias = std::log(0.1f / 0.9f);       // logit(opacity_bias)
    const float perturbe_size = 1.5f;
    const float rot_lr = 0.1f;

    std::vector<float> h = download_f32(feats);             // (n, OC)

    GaussianSplats cloud;
    const std::size_t total = static_cast<std::size_t>(n) * ng;
    cloud.shDegree = 0;
    cloud.positions.resize(total * 3);
    cloud.scales.resize(total * 3);
    cloud.rotations.resize(total * 4);
    cloud.opacities.resize(total);
    cloud.sh.resize(total * 3);                             // shStride() = 3

    for (int m = 0; m < n; ++m) {
        const float* f = &h[static_cast<std::size_t>(m) * OC];
        const float px = points[static_cast<std::size_t>(m) * 3 + 0];
        const float py = points[static_cast<std::size_t>(m) * 3 + 1];
        const float pz = points[static_cast<std::size_t>(m) * 3 + 2];
        for (int g = 0; g < ng; ++g) {
            const std::size_t o = (static_cast<std::size_t>(m) * ng + g);

            // offset scale (softplus(raw + base))
            const float offset_scale = softplus(f[R_OFFS + g] + base_offset_scale_);

            // xyz offset: (raw + perturbation) -> tanh*0.5*perturbe_size * scale
            float off[3];
            for (int a = 0; a < 3; ++a) {
                float v = f[R_XYZ + g * 3 + a] + perturb_[g * 3 + a];
                v = std::tanh(v) * 0.5f * perturbe_size * offset_scale;
                off[a] = v;
            }
            // position: point + offset, then aabb [-0.5,-0.5,-0.5, 1,1,1] -> -0.5
            cloud.positions[o * 3 + 0] = (px + off[0]) - 0.5f;
            cloud.positions[o * 3 + 1] = (py + off[1]) - 0.5f;
            cloud.positions[o * 3 + 2] = (pz + off[2]) - 0.5f;

            // SH DC color (features_dc, lr 1.0)
            cloud.sh[o * 3 + 0] = f[R_FDC + g * 3 + 0];
            cloud.sh[o * 3 + 1] = f[R_FDC + g * 3 + 1];
            cloud.sh[o * 3 + 2] = f[R_FDC + g * 3 + 2];

            // scaling: sqrt(softplus(raw + scale_bias)^2 + min_kernel^2)
            for (int a = 0; a < 3; ++a) {
                const float s = softplus(f[R_SCALE + g * 3 + a] + scale_bias);
                cloud.scales[o * 3 + a] = std::sqrt(s * s + min_kernel * min_kernel);
            }

            // rotation: raw*lr (wxyz), normalized, re-emitted xyzw
            const float w = f[R_ROT + g * 4 + 0] * rot_lr;
            const float x = f[R_ROT + g * 4 + 1] * rot_lr;
            const float y = f[R_ROT + g * 4 + 2] * rot_lr;
            const float z = f[R_ROT + g * 4 + 3] * rot_lr;
            float len = std::sqrt(w * w + x * x + y * y + z * z);
            if (len < 1e-20f) { len = 1.0f; }
            cloud.rotations[o * 4 + 0] = x / len;
            cloud.rotations[o * 4 + 1] = y / len;
            cloud.rotations[o * 4 + 2] = z / len;
            cloud.rotations[o * 4 + 3] = w / len;

            // opacity: sigmoid(raw + opacity_bias)
            const float op = f[R_OPAC + g];
            cloud.opacities[o] = 1.0f / (1.0f + std::exp(-(op + opacity_bias)));
        }
    }
    return cloud;
}

// ─── octree sampler (systematic resampling) ───────────────────────────────────

namespace {

// Systematic resampling of `count` draws over an 8-way probability row, matching
// model.sample_probs(algo="systematic") for a single row: normalize, build the
// CDF, place `count` stratified samples us[g] = (u0 + g)/count and bucket them by
// searchsorted(cdf). Writes 8 integer counts summing to `count` into `out8`.
void sample_probs_row(const float* logits8, std::int64_t count,
                      std::mt19937_64& rng, std::int64_t* out8) {
    for (int i = 0; i < 8; ++i) out8[i] = 0;
    if (count <= 0) return;

    // softmax(logits) (temperature 1).
    float mx = logits8[0];
    for (int i = 1; i < 8; ++i) mx = std::max(mx, logits8[i]);
    float p[8], sum = 0.0f;
    for (int i = 0; i < 8; ++i) { p[i] = std::exp(logits8[i] - mx); sum += p[i]; }
    if (sum <= 0.0f) { for (int i = 0; i < 8; ++i) p[i] = 1.0f / 8.0f; }
    else { for (int i = 0; i < 8; ++i) p[i] /= sum; }

    float cdf[8];
    float acc = 0.0f;
    for (int i = 0; i < 8; ++i) {
        acc += p[i];
        cdf[i] = std::min(acc, 1.0f - 1e-12f);
    }
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    const float u0 = uni(rng) / static_cast<float>(count);
    for (std::int64_t gi = 0; gi < count; ++gi) {
        float u = u0 + static_cast<float>(gi) / static_cast<float>(count);
        u = std::min(u, 1.0f - 1e-12f);
        // searchsorted(cdf, u): first index with cdf[idx] >= u (right side of
        // torch.searchsorted default 'left' on a strictly-increasing CDF).
        int idx = 0;
        while (idx < 7 && cdf[idx] < u) ++idx;
        ++out8[idx];
    }
}

}  // namespace

GaussianSplats OctreeGaussianDecoder::decode(const bt::Tensor& latent,
                                             int num_gaussians,
                                             std::uint64_t seed) {
    if (!loaded_) fail("decode: weights not loaded");
    const int ng = cfg_.num_gaussians;
    const int num_points = std::max(1, num_gaussians / ng);
    std::mt19937_64 rng(seed);

    // child octant offsets: x=c&1, y=(c>>1)&1, z=(c>>2)&1  (reference order).
    auto child = [](int c, int axis) -> int { return (c >> axis) & 1; };

    // active voxels: integer coords (level-local) + point counts.
    std::vector<std::array<std::int64_t, 3>> coords{{0, 0, 0}};
    std::vector<std::int64_t> counts{num_points};

    for (int lv = 1; lv <= cfg_.max_level; ++lv) {
        const int res_p = 1 << (lv - 1);
        const int res = 1 << lv;
        const int n = static_cast<int>(coords.size());

        // parent_coords_norm = (coord + 0.5) / res_p
        std::vector<float> cn(static_cast<std::size_t>(n) * 3);
        for (int i = 0; i < n; ++i)
            for (int a = 0; a < 3; ++a)
                cn[static_cast<std::size_t>(i) * 3 + a] =
                    (static_cast<float>(coords[i][a]) + 0.5f) / static_cast<float>(res_p);

        bt::Tensor logits_t;
        octree_logits(cn, n, res, latent, logits_t);
        bt::sync_all();
        std::vector<float> logits = download_f32(logits_t);   // (n, 8)

        std::vector<std::array<std::int64_t, 3>> next_coords;
        std::vector<std::int64_t> next_counts;
        next_coords.reserve(coords.size() * 2);
        next_counts.reserve(coords.size() * 2);
        std::int64_t sampled[8];
        for (int i = 0; i < n; ++i) {
            sample_probs_row(&logits[static_cast<std::size_t>(i) * 8], counts[i], rng, sampled);
            for (int c = 0; c < 8; ++c) {
                if (sampled[c] <= 0) continue;
                next_coords.push_back({coords[i][0] * 2 + child(c, 0),
                                       coords[i][1] * 2 + child(c, 1),
                                       coords[i][2] * 2 + child(c, 2)});
                next_counts.push_back(sampled[c]);
            }
        }
        coords.swap(next_coords);
        counts.swap(next_counts);
    }

    // expand surviving voxels into num_points jittered points.
    const int res = 1 << cfg_.max_level;
    std::vector<float> points;
    points.reserve(static_cast<std::size_t>(num_points) * 3);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    for (std::size_t v = 0; v < coords.size(); ++v) {
        for (std::int64_t k = 0; k < counts[v]; ++k) {
            for (int a = 0; a < 3; ++a) {
                const float jit = uni(rng);
                points.push_back((static_cast<float>(coords[v][a]) + jit) / static_cast<float>(res));
            }
        }
    }
    const int m = static_cast<int>(points.size() / 3);

    bt::Tensor feats;
    gs_features(points, m, latent, feats);
    bt::sync_all();
    return build_gaussians(points, m, feats);
}

}  // namespace brodiffusion::triposplat
