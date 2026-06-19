// AutoencoderDC (DC-AE f32c32) decoder.
//
// See vae_dcae.h. Inference-only decoder branch of NVIDIA Sana's deep-
// compression VAE. Architecture (HF diffusers AutoencoderDC,
// dc-ae-f32c32-sana-1.0): conv_in (+ input shortcut via channel
// repeat_interleave) → up_blocks (deep→shallow, ResBlock or EfficientViTBlock
// stages, "interpolate" DCUpBlock2d upsamplers between stages) → RMSNorm →
// ReLU → conv_out. No GroupNorm — every norm is a channel-wise RMSNorm; the
// linear-attention transformer stages use Sana's multiscale ReLU linear
// attention plus a GLU-MBConv. The decoder runs at the pipeline compute dtype
// (FP32 on CPU, FP16 on a GPU backend); the linear-attention core is evaluated
// in FP32 for range (matching diffusers' float32 upcast of that path).

#include "brodiffusion/vae_dcae.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"
#include "brotensor/safetensors.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::dcae {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

constexpr float kRmsEps  = 1e-5f;    // diffusers RMSNorm eps for DC-AE
constexpr float kAttnEps = 1e-15f;   // SanaMultiscaleLinearAttention eps
constexpr int   kHeadDim = 32;
constexpr int   kMsKernel = 5;

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("dcae::Decoder: " + msg);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    const auto* v = f.find(key);
    if (!v) fail("missing tensor '" + key + "'");
    return *v;
}

// Upload a checkpoint tensor (F16/F32/BF16 source) at the compute dtype,
// validating its element count.
void up(const st::File& f, const std::string& key, int rows, int cols,
        bt::Tensor& dst) {
    const st::TensorView& v = need(f, key);
    if (v.dtype != st::Dtype::F16 && v.dtype != st::Dtype::F32 &&
        v.dtype != st::Dtype::BF16) {
        fail(key + ": expected F16/F32/BF16, got " + st::dtype_name(v.dtype));
    }
    const int64_t expected = static_cast<int64_t>(rows) * cols;
    if (v.numel() != expected) {
        fail(key + ": shape mismatch (expected " + std::to_string(rows) + "x" +
             std::to_string(cols) + ", got " + std::to_string(v.numel()) + ")");
    }
    st::upload_as(v, rows, cols, brodiffusion::compute_dtype(), dst);
}

// Non-owning view of `count` elements of `t` starting at element offset `off`,
// reshaped to (rows, cols) (rows*cols == count). Lets a contiguous activation
// buffer be reinterpreted (NCHW ↔ (C,N), per-group slices) without a copy.
bt::Tensor sub_view(const bt::Tensor& t, int64_t off, int rows, int cols) {
    char* p = static_cast<char*>(t.data) +
              off * bt::dtype_size_bytes(t.dtype);
    return bt::Tensor::view(t.device, p, rows, cols, t.dtype);
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

Decoder::Decoder(const DecoderConfig& cfg) : cfg_(cfg) {
    const std::size_t nb = cfg_.block_out_channels.size();
    if (nb == 0) fail("block_out_channels must be non-empty");
    if (cfg_.layers_per_block.size() != nb ||
        cfg_.is_attention.size() != nb ||
        cfg_.qkv_multiscales.size() != nb) {
        fail("per-stage config vectors must match block_out_channels length");
    }
    up_blocks_.resize(nb);
}

Decoder::~Decoder() = default;

// ─── load_weights ──────────────────────────────────────────────────────────

void Decoder::load_weights(const st::File& f, const std::string& prefix) {
    const int nb      = static_cast<int>(cfg_.block_out_channels.size());
    const int latentC = cfg_.latent_channels;
    const int deepC   = cfg_.block_out_channels.back();
    const int firstC  = cfg_.block_out_channels.front();

    // conv_in: latent_channels → deepest channels, 3x3.
    up(f, prefix + "conv_in.weight", deepC, latentC * 3 * 3, conv_in_W_);
    up(f, prefix + "conv_in.bias",   deepC, 1, conv_in_b_);

    // Input shortcut: repeat_interleave(latent, deepC/latentC) added to conv_in.
    in_shortcut_repeats_ = deepC / latentC;
    if (in_shortcut_repeats_ * latentC != deepC) in_shortcut_repeats_ = 0;

    for (int i = 0; i < nb; ++i) {
        UpBlock& ub = up_blocks_[static_cast<std::size_t>(i)];
        const int C = cfg_.block_out_channels[static_cast<std::size_t>(i)];
        ub.C = C;
        ub.is_attention = cfg_.is_attention[static_cast<std::size_t>(i)];
        ub.has_upsampler = (i < nb - 1);   // every stage but the deepest upsamples
        const std::string bp =
            prefix + "up_blocks." + std::to_string(i) + ".";
        int sub = 0;
        if (ub.has_upsampler) {
            ub.up_Cin  = cfg_.block_out_channels[static_cast<std::size_t>(i + 1)];
            ub.up_Cout = C;
            up(f, bp + "0.conv.weight", ub.up_Cout, ub.up_Cin * 3 * 3, ub.up_W);
            up(f, bp + "0.conv.bias",   ub.up_Cout, 1, ub.up_b);
            sub = 1;
        }
        const int layers = cfg_.layers_per_block[static_cast<std::size_t>(i)];
        for (int m = 0; m < layers; ++m) {
            const std::string sp = bp + std::to_string(sub + m) + ".";
            if (ub.is_attention) {
                EViT ev;
                Attn& a = ev.attn;
                a.C = C;
                a.inner = C;                 // num_heads*head_dim == C here
                a.num_heads = C / kHeadDim;
                const int innr = a.inner;
                up(f, sp + "attn.to_q.weight", innr, C, a.Wq);
                up(f, sp + "attn.to_k.weight", innr, C, a.Wk);
                up(f, sp + "attn.to_v.weight", innr, C, a.Wv);
                up(f, sp + "attn.to_qkv_multiscale.0.proj_in.weight",
                   3 * innr, kMsKernel * kMsKernel, a.ms_in_W);
                up(f, sp + "attn.to_qkv_multiscale.0.proj_out.weight",
                   3 * innr, kHeadDim, a.ms_out_W);
                up(f, sp + "attn.to_out.weight", C, 2 * innr, a.Wo);
                up(f, sp + "attn.norm_out.weight", C, 1, a.norm_g);
                up(f, sp + "attn.norm_out.bias",   C, 1, a.norm_b);

                GLUMB& g = ev.glu;
                g.C = C;
                up(f, sp + "conv_out.conv_inverted.weight", 8 * C, C, g.inv_W);
                up(f, sp + "conv_out.conv_inverted.bias",   8 * C, 1, g.inv_b);
                up(f, sp + "conv_out.conv_depth.weight", 8 * C, 3 * 3, g.depth_W);
                up(f, sp + "conv_out.conv_depth.bias",   8 * C, 1, g.depth_b);
                up(f, sp + "conv_out.conv_point.weight", C, 4 * C, g.point_W);
                up(f, sp + "conv_out.norm.weight", C, 1, g.norm_g);
                up(f, sp + "conv_out.norm.bias",   C, 1, g.norm_b);
                ub.evitblocks.push_back(std::move(ev));
            } else {
                ResBlock r;
                r.C = C;
                up(f, sp + "conv1.weight", C, C * 3 * 3, r.conv1_W);
                up(f, sp + "conv1.bias",   C, 1, r.conv1_b);
                up(f, sp + "conv2.weight", C, C * 3 * 3, r.conv2_W);
                up(f, sp + "norm.weight",  C, 1, r.norm_g);
                up(f, sp + "norm.bias",    C, 1, r.norm_b);
                ub.resblocks.push_back(std::move(r));
            }
        }
    }

    up(f, prefix + "norm_out.weight", firstC, 1, norm_out_g_);
    up(f, prefix + "norm_out.bias",   firstC, 1, norm_out_b_);
    up(f, prefix + "conv_out.weight", cfg_.image_channels, firstC * 3 * 3,
       conv_out_W_);
    up(f, prefix + "conv_out.bias",   cfg_.image_channels, 1, conv_out_b_);
}

// ─── forward helpers ───────────────────────────────────────────────────────

void Decoder::ensure_ones_(int n) {
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

// Channel-wise RMSNorm (weight + bias) on a channel-major NCHW tensor:
// move channels last, RMSNorm over C, move back, add per-channel bias.
// Reads `x` (1, C*H*W), writes `out` (same shape). x is not modified.
void Decoder::apply_rmsnorm_(const bt::Tensor& g, const bt::Tensor& b,
                             int C, int H, int W,
                             bt::Tensor& x, bt::Tensor& out) {
    bt::nchw_to_sequence(x, 1, C, H, W, seq_);     // (H*W, C)
    bt::rms_norm_forward(seq_, g, kRmsEps, seq2_); // normalize over C
    bt::sequence_to_nchw(seq2_, 1, C, H, W, out);  // (1, C*H*W)
    bt::add_channel_bias_inplace(out, b, C, H * W);
}

void Decoder::apply_resblock_(const ResBlock& r, int H, int W, bt::Tensor& x) {
    const int C = r.C;
    // conv1 (3x3) → SiLU → conv2 (3x3, no bias)
    bt::conv2d_forward(x, r.conv1_W, &r.conv1_b, 1, C, H, W, C, 3, 3,
                       1, 1, 1, 1, 1, 1, t1_);
    bt::silu_forward(t1_, t1_);
    bt::conv2d_forward(t1_, r.conv2_W, nullptr, 1, C, H, W, C, 3, 3,
                       1, 1, 1, 1, 1, 1, t2_);
    // RMSNorm(channels) then residual add of the block input.
    apply_rmsnorm_(r.norm_g, r.norm_b, C, H, W, t2_, t1_);
    bt::add_inplace(t1_, x);
    std::swap(x, t1_);
}

void Decoder::apply_glumb_(const GLUMB& m, int H, int W, bt::Tensor& x) {
    const int C  = m.C;
    const int h2 = 8 * C;       // inverted/depth channels
    const int hg = 4 * C;       // gated channels
    const int HW = H * W;
    // 1x1 inverted (C→8C) → SiLU → 3x3 depthwise (8C, groups=8C)
    bt::conv2d_forward(x, m.inv_W, &m.inv_b, 1, C, H, W, h2, 1, 1,
                       1, 1, 0, 0, 1, 1, 1, t1_);
    bt::silu_forward(t1_, t1_);
    bt::conv2d_forward(t1_, m.depth_W, &m.depth_b, 1, h2, H, W, h2, 3, 3,
                       1, 1, 1, 1, 1, 1, h2, t2_);
    // GLU: split channels (h, gate); h *= SiLU(gate).
    bt::Tensor hpart  = sub_view(t2_, 0, hg, HW);
    bt::Tensor gpart  = sub_view(t2_, static_cast<int64_t>(hg) * HW, hg, HW);
    bt::silu_forward(gpart, gpart);
    bt::mul_inplace(hpart, gpart);
    // 1x1 point (4C→C, no bias)
    bt::Tensor hflat = sub_view(t2_, 0, 1, hg * HW);
    bt::conv2d_forward(hflat, m.point_W, nullptr, 1, hg, H, W, C, 1, 1,
                       1, 1, 0, 0, 1, 1, 1, t1_);
    // RMSNorm(channels) then residual.
    apply_rmsnorm_(m.norm_g, m.norm_b, C, H, W, t1_, t3_);
    bt::add_inplace(t3_, x);
    std::swap(x, t3_);
}

void Decoder::apply_attn_(const Attn& a, int H, int W, bt::Tensor& x) {
    const int C    = a.C;
    const int innr = a.inner;
    const int N    = H * W;
    const int hd   = kHeadDim;

    // q/k/v projections (bias-free Linear over channels): q = Wq @ Xcm,
    // computed channel-major directly so no transpose is needed.
    bt::Tensor Xcm = sub_view(x, 0, C, N);
    bt::matmul(a.Wq, Xcm, t1_);   // (inner, N)
    bt::matmul(a.Wk, Xcm, t2_);
    bt::matmul(a.Wv, Xcm, t3_);
    // qkv_ = concat([q, k, v]) along channels → (3*inner, N) NCHW.
    const bt::Dtype cdt = brodiffusion::compute_dtype();
    detail::resize_like(qkv_, 3 * innr, N, cdt, x.device);
    bt::copy_d2d(t1_, 0, qkv_, 0,                    innr * N);
    bt::copy_d2d(t2_, 0, qkv_, static_cast<int64_t>(innr) * N, innr * N);
    bt::copy_d2d(t3_, 0, qkv_, static_cast<int64_t>(2) * innr * N, innr * N);

    // Multiscale aggregation: proj_out(proj_in(qkv)). proj_in is a 5x5
    // depthwise conv (groups=3*inner); proj_out a grouped 1x1 (groups=3*heads).
    bt::Tensor qkv_nchw = sub_view(qkv_, 0, 1, 3 * innr * N);
    bt::conv2d_forward(qkv_nchw, a.ms_in_W, nullptr, 1, 3 * innr, H, W,
                       3 * innr, kMsKernel, kMsKernel, 1, 1, 2, 2, 1, 1,
                       3 * innr, ms_);
    bt::conv2d_forward(ms_, a.ms_out_W, nullptr, 1, 3 * innr, H, W,
                       3 * innr, 1, 1, 1, 1, 0, 0, 1, 1, 3 * a.num_heads, t1_);
    // hs = concat([qkv, multiscale]) along channels → (6*inner, N).
    detail::resize_like(hs_, 6 * innr, N, cdt, x.device);
    bt::copy_d2d(qkv_, 0, hs_, 0,                       3 * innr * N);
    bt::copy_d2d(t1_,  0, hs_, static_cast<int64_t>(3) * innr * N,
                 3 * innr * N);
    // Linear-attention core runs in FP32 (matches diffusers' float32 upcast).
    bt::cast(hs_, hsf_, bt::Dtype::FP32);

    const int G = (6 * innr) / (3 * hd);   // (1 + num_scales) * num_heads groups
    detail::resize_like(attn_f_, G * hd, N, bt::Dtype::FP32, x.device);
    ensure_ones_(N);
    for (int gi = 0; gi < G; ++gi) {
        const int64_t base = static_cast<int64_t>(gi) * 3 * hd;  // row offset
        bt::Tensor Qg = sub_view(hsf_, base * N, hd, N);
        bt::Tensor Kg = sub_view(hsf_, (base + hd) * N, hd, N);
        // ReLU(query), ReLU(key); value is left untouched.
        bt::relu_forward(Qg, Qg);
        bt::relu_forward(Kg, Kg);
        // Vp = pad(value, +1 row of ones) → (head_dim+1, N).
        detail::resize_like(vp_, hd + 1, N, bt::Dtype::FP32, x.device);
        bt::copy_d2d(hsf_, (base + 2 * hd) * N, vp_, 0,
                     static_cast<int64_t>(hd) * N);
        bt::copy_d2d(ones_, 0, vp_, static_cast<int64_t>(hd) * N, N);
        // Kt = Kgᵀ via NCHW→sequence on the (1, head_dim, H, W) key block.
        bt::Tensor KgN = sub_view(hsf_, (base + hd) * N, 1, hd * N);
        bt::nchw_to_sequence(KgN, 1, hd, H, W, kt_);     // (N, head_dim)
        bt::matmul(vp_, kt_, scores_);                   // (head_dim+1, head_dim)
        bt::matmul(scores_, Qg, hidden_);                // (head_dim+1, N)
        // out = hidden[:head_dim] / (hidden[head_dim] + eps).
        detail::resize_like(recip_, 1, N, bt::Dtype::FP32, x.device);
        bt::copy_d2d(hidden_, static_cast<int64_t>(hd) * N, recip_, 0, N);
        bt::add_scalar_inplace(recip_, kAttnEps);
        bt::rsqrt_forward(recip_, recip_);
        bt::mul_inplace(recip_, recip_);                 // 1/(denom+eps)
        bt::Tensor num  = sub_view(hidden_, 0, hd, N);
        bt::Tensor outg = sub_view(attn_f_, static_cast<int64_t>(gi) * hd * N,
                                   hd, N);
        bt::broadcast_mul(num, recip_, outg);
    }

    // Back to compute dtype, then to_out (bias-free Linear over channels).
    bt::cast(attn_f_, attn_c_, cdt);                     // (2*inner, N)
    bt::matmul(a.Wo, attn_c_, toout_);                   // (C, N)
    bt::Tensor toout_nchw = sub_view(toout_, 0, 1, C * N);
    apply_rmsnorm_(a.norm_g, a.norm_b, C, H, W, toout_nchw, t1_);
    bt::add_inplace(t1_, x);                             // residual
    std::swap(x, t1_);
}

void Decoder::apply_upsample_(const UpBlock& u, int H, int W, bt::Tensor& x) {
    const int Cin  = u.up_Cin;
    const int Cout = u.up_Cout;
    // Interpolate path: nearest-2x → 3x3 conv (Cin→Cout).
    bt::upsample_nearest_2x(x, 1, Cin, H, W, up_t_);
    bt::conv2d_forward(up_t_, u.up_W, &u.up_b, 1, Cin, 2 * H, 2 * W,
                       Cout, 3, 3, 1, 1, 1, 1, 1, 1, t1_);
    // Shortcut path: channel repeat_interleave + pixel-shuffle (fused).
    bt::pixel_shuffle_upsample_2x_forward(x, 1, Cin, H, W, Cout, short_t_);
    bt::add_inplace(t1_, short_t_);
    std::swap(x, t1_);
}

// ─── decode ────────────────────────────────────────────────────────────────

void Decoder::decode(const bt::Tensor& latent, int H_lat, int W_lat,
                     bt::Tensor& out) {
    if (conv_in_W_.size() == 0) fail("decode: weights not loaded");
    if (H_lat <= 0 || W_lat <= 0) fail("decode: H_lat and W_lat must be positive");
    if (latent.rows != 1 ||
        latent.cols != cfg_.latent_channels * H_lat * W_lat) {
        fail("decode: latent must be (1, latent_channels*H_lat*W_lat)");
    }

    const int nb      = static_cast<int>(cfg_.block_out_channels.size());
    const int latentC = cfg_.latent_channels;
    const int deepC   = cfg_.block_out_channels.back();
    const int firstC  = cfg_.block_out_channels.front();
    const bt::Dtype cdt = brodiffusion::compute_dtype();

    // 1. latent / scaling_factor at the compute dtype.
    if (latent.dtype != cdt) bt::cast(latent, x_, cdt);
    else                     x_ = latent.clone();
    if (cfg_.scaling_factor != 1.0f) {
        bt::scale_inplace(x_, 1.0f / cfg_.scaling_factor);
    }

    // 2. conv_in (latent→deepC) + optional input shortcut.
    bt::conv2d_forward(x_, conv_in_W_, &conv_in_b_, 1, latentC, H_lat, W_lat,
                       deepC, 3, 3, 1, 1, 1, 1, 1, 1, y_);
    if (in_shortcut_repeats_ > 0) {
        const int N = H_lat * W_lat;
        detail::resize_like(repbuf_, deepC, N, cdt, x_.device);
        for (int j = 0; j < in_shortcut_repeats_; ++j) {
            bt::copy_d2d_strided(x_, 0, N, repbuf_,
                                 static_cast<int64_t>(j) * N,
                                 in_shortcut_repeats_ * N, N, latentC);
        }
        bt::add_inplace(y_, repbuf_);
    }
    std::swap(x_, y_);

    // 3. up_blocks, deep → shallow (reversed). Upsamplers run at the start of
    //    each stage but the deepest, so the running resolution doubles 5×.
    int H = H_lat, W = W_lat;
    for (int i = nb - 1; i >= 0; --i) {
        UpBlock& ub = up_blocks_[static_cast<std::size_t>(i)];
        if (ub.has_upsampler) {
            apply_upsample_(ub, H, W, x_);
            H *= 2;
            W *= 2;
        }
        if (ub.is_attention) {
            for (auto& ev : ub.evitblocks) {
                apply_attn_(ev.attn, H, W, x_);
                apply_glumb_(ev.glu, H, W, x_);
            }
        } else {
            for (auto& r : ub.resblocks) apply_resblock_(r, H, W, x_);
        }
    }

    // 4. norm_out (RMSNorm) → ReLU → conv_out.
    apply_rmsnorm_(norm_out_g_, norm_out_b_, firstC, H, W, x_, y_);
    bt::relu_forward(y_, y_);
    bt::conv2d_forward(y_, conv_out_W_, &conv_out_b_, 1, firstC, H, W,
                       cfg_.image_channels, 3, 3, 1, 1, 1, 1, 1, 1, out);
}

}  // namespace brodiffusion::dcae
