#include "brodiffusion/vae_qwenimage.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"
#include "brotensor/safetensors.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::vae_qwenimage {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

// PyTorch F.normalize's default eps (QwenImageRMS_norm never passes an
// explicit one) — see the header comment on why rms_norm_forward's
// sqrt(mean+eps) is numerically equivalent to F.normalize's max(norm,eps)
// for any non-degenerate (non-all-zero) channel vector.
constexpr float kRmsEps = 1e-12f;

[[noreturn]] void fail(const char* who, const std::string& msg) {
    throw std::runtime_error(std::string("vae_qwenimage::") + who + ": " + msg);
}

const st::TensorView& need(const st::File& f, const std::string& key, const char* who) {
    const auto* v = f.find(key);
    if (!v) fail(who, "missing tensor '" + key + "'");
    return *v;
}

bt::Dtype arith_dtype_for(bool force_upcast) {
    if (force_upcast && brotensor::default_device() == brotensor::Device::CUDA) {
        return bt::Dtype::BF16;
    }
    return brodiffusion::compute_dtype();
}

std::vector<float> view_to_float(const st::TensorView& v, const char* who,
                                 const std::string& key) {
    const int64_t n = v.numel();
    std::vector<float> out(static_cast<std::size_t>(n));
    if (v.dtype == st::Dtype::F32) {
        std::memcpy(out.data(), v.data, static_cast<std::size_t>(n) * sizeof(float));
    } else if (v.dtype == st::Dtype::F16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(v.data);
        for (int64_t i = 0; i < n; ++i) {
            out[static_cast<std::size_t>(i)] = bt::fp16_bits_to_fp32(p[static_cast<std::size_t>(i)]);
        }
    } else if (v.dtype == st::Dtype::BF16) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(v.data);
        for (int64_t i = 0; i < n; ++i) {
            out[static_cast<std::size_t>(i)] = bt::bf16_bits_to_fp32(p[static_cast<std::size_t>(i)]);
        }
    } else {
        fail(who, key + ": expected F16/F32/BF16, got " + st::dtype_name(v.dtype));
    }
    return out;
}

bt::Tensor upload_at(const std::vector<float>& host, int rows, int cols, bt::Dtype want) {
    const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (want == bt::Dtype::FP16) {
        std::vector<uint16_t> bits(n);
        for (std::size_t i = 0; i < n; ++i) bits[i] = bt::fp32_to_fp16_bits(host[i]);
        return bt::Tensor::from_host_fp16(bits.data(), rows, cols);
    }
    if (want == bt::Dtype::BF16) {
        std::vector<uint16_t> bits(n);
        for (std::size_t i = 0; i < n; ++i) bits[i] = bt::fp32_to_bf16_bits(host[i]);
        return bt::Tensor::from_host_bf16(bits.data(), rows, cols);
    }
    return bt::Tensor::from_host(host.data(), rows, cols);
}

// Plain (no temporal axis) tensor: a norm gamma / conv bias (numel==rows*cols)
// or a 2D conv weight (attention 1x1s, resample 3x3s) already shaped
// [Cout,Cin,kh,kw] with no leading kT axis to slice.
void load_plain(const st::File& f, const std::string& key, int rows, int cols,
                bt::Dtype want, bt::Tensor& dst, const char* who) {
    const st::TensorView& v = need(f, key, who);
    const int64_t expected = static_cast<int64_t>(rows) * cols;
    if (v.numel() != expected) {
        fail(who, key + ": shape mismatch (expected " + std::to_string(rows) + "x" +
             std::to_string(cols) + ", got " + std::to_string(v.numel()) + ")");
    }
    std::vector<float> host = view_to_float(v, who, key);
    dst = upload_at(host, rows, cols, want);
}

// Causal-Conv3d weight [Cout,Cin,kT,kH,kW]: slice out the LAST temporal tap
// (the num_frames=1 reduction — see vae_qwenimage.h) into an ordinary 2D
// (Cout, Cin*kH*kW) filter.
void load_conv_lasttap(const st::File& f, const std::string& key,
                       int Cout, int Cin, int kT, int kHW,
                       bt::Dtype want, bt::Tensor& dst, const char* who) {
    const st::TensorView& v = need(f, key, who);
    const int64_t expected = static_cast<int64_t>(Cout) * Cin * kT * kHW;
    if (v.numel() != expected) {
        fail(who, key + ": shape mismatch (expected Cout*Cin*kT*kHW=" +
             std::to_string(expected) + ", got " + std::to_string(v.numel()) + ")");
    }
    std::vector<float> host = view_to_float(v, who, key);
    std::vector<float> sliced(static_cast<std::size_t>(Cout) * static_cast<std::size_t>(Cin) *
                              static_cast<std::size_t>(kHW));
    for (int64_t oc = 0; oc < Cout; ++oc) {
        for (int64_t ic = 0; ic < Cin; ++ic) {
            const std::size_t src_off =
                (static_cast<std::size_t>(oc) * static_cast<std::size_t>(Cin) +
                 static_cast<std::size_t>(ic)) * static_cast<std::size_t>(kT) * static_cast<std::size_t>(kHW) +
                static_cast<std::size_t>(kT - 1) * static_cast<std::size_t>(kHW);
            const std::size_t dst_off =
                (static_cast<std::size_t>(oc) * static_cast<std::size_t>(Cin) +
                 static_cast<std::size_t>(ic)) * static_cast<std::size_t>(kHW);
            std::memcpy(&sliced[dst_off], &host[src_off],
                       static_cast<std::size_t>(kHW) * sizeof(float));
        }
    }
    dst = upload_at(sliced, Cout, Cin * kHW, want);
}

// Non-owning reinterpretation of `count` elements of `t` at element offset
// `off`, reshaped to (rows,cols). Lets q/k/v be sliced out of a fused to_qkv
// weight without a copy (mirrors vae_dcae.cpp's sub_view).
bt::Tensor sub_view(const bt::Tensor& t, int64_t off, int rows, int cols) {
    char* p = static_cast<char*>(t.data) + off * bt::dtype_size_bytes(t.dtype);
    return bt::Tensor::view(t.device, p, rows, cols, t.dtype);
}

// Host download of a device tensor to FP32, regardless of its (FP32/FP16/
// BF16) storage dtype. Used for the per-channel latents_mean/latents_std
// affine, which has no ready-made broadcast op in brotensor's op set (z_dim
// is tiny — 16 channels — so a host round-trip is not perf-sensitive).
std::vector<float> download_f32(const bt::Tensor& t) {
    bt::sync_all();
    if (t.dtype == bt::Dtype::FP16) {
        auto bits = t.to_host_vector_fp16();
        std::vector<float> out(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) out[i] = bt::fp16_bits_to_fp32(bits[i]);
        return out;
    }
    if (t.dtype == bt::Dtype::BF16) {
        auto bits = t.to_host_vector_bf16();
        std::vector<float> out(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) out[i] = bt::bf16_bits_to_fp32(bits[i]);
        return out;
    }
    return t.to_host_vector();
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════
// Decoder
// ═══════════════════════════════════════════════════════════════════════

Decoder::Decoder(const Config& cfg) : cfg_(cfg) {
    if (cfg_.dim_mult.empty()) fail("Decoder", "dim_mult must be non-empty");
    if (cfg_.num_res_blocks <= 0) fail("Decoder", "num_res_blocks must be positive");
    if (!cfg_.attn_scales.empty()) {
        fail("Decoder", "non-empty attn_scales not supported (no known Krea 2 / "
                        "Qwen-Image checkpoint uses one; down/up-block attention "
                        "is not implemented)");
    }
    if (static_cast<int>(cfg_.latents_mean.size()) != cfg_.z_dim ||
        static_cast<int>(cfg_.latents_std.size()) != cfg_.z_dim) {
        fail("Decoder", "latents_mean/latents_std must have z_dim entries");
    }
    up_blocks_.resize(cfg_.dim_mult.size());
}

Decoder::~Decoder() = default;

void Decoder::load_resnet_(const st::File& f, const std::string& p,
                           int C_in, int C_out, Resnet& r) {
    load_plain(f, p + "norm1.gamma", C_in, 1, arith_dtype_, r.norm1_g, "Decoder");
    load_conv_lasttap(f, p + "conv1.weight", C_out, C_in, 3, 9, arith_dtype_, r.conv1_W, "Decoder");
    load_plain(f, p + "conv1.bias", C_out, 1, arith_dtype_, r.conv1_b, "Decoder");
    load_plain(f, p + "norm2.gamma", C_out, 1, arith_dtype_, r.norm2_g, "Decoder");
    load_conv_lasttap(f, p + "conv2.weight", C_out, C_out, 3, 9, arith_dtype_, r.conv2_W, "Decoder");
    load_plain(f, p + "conv2.bias", C_out, 1, arith_dtype_, r.conv2_b, "Decoder");

    r.C_in = C_in;
    r.C_out = C_out;
    r.has_shortcut = (C_in != C_out);
    if (r.has_shortcut) {
        load_conv_lasttap(f, p + "conv_shortcut.weight", C_out, C_in, 1, 1,
                          arith_dtype_, r.short_W, "Decoder");
        load_plain(f, p + "conv_shortcut.bias", C_out, 1, arith_dtype_, r.short_b, "Decoder");
    }
}

void Decoder::load_weights(const st::File& f, const std::string& prefix) {
    arith_dtype_ = arith_dtype_for(cfg_.force_upcast);
    const int nb = static_cast<int>(cfg_.dim_mult.size());
    const int z_dim = cfg_.z_dim;

    // dims = base_dim * [dim_mult.back(), dim_mult[nb-1], ..., dim_mult[0]]
    // (matches diffusers' `[dim*u for u in [dim_mult[-1]] + dim_mult[::-1]]`).
    std::vector<int> dims;
    dims.reserve(static_cast<std::size_t>(nb) + 1);
    dims.push_back(cfg_.base_dim * cfg_.dim_mult.back());
    for (int i = nb - 1; i >= 0; --i) dims.push_back(cfg_.base_dim * cfg_.dim_mult[i]);

    load_plain(f, prefix + "post_quant_conv.weight", z_dim, z_dim, arith_dtype_,
              post_quant_W_, "Decoder");
    load_plain(f, prefix + "post_quant_conv.bias", z_dim, 1, arith_dtype_,
              post_quant_b_, "Decoder");

    load_conv_lasttap(f, prefix + "decoder.conv_in.weight", dims[0], z_dim, 3, 9,
                      arith_dtype_, conv_in_W_, "Decoder");
    load_plain(f, prefix + "decoder.conv_in.bias", dims[0], 1, arith_dtype_, conv_in_b_, "Decoder");

    load_resnet_(f, prefix + "decoder.mid_block.resnets.0.", dims[0], dims[0], mid_res0_);
    load_resnet_(f, prefix + "decoder.mid_block.resnets.1.", dims[0], dims[0], mid_res1_);
    {
        const std::string ap = prefix + "decoder.mid_block.attentions.0.";
        load_plain(f, ap + "norm.gamma", dims[0], 1, arith_dtype_, mid_attn_.norm_g, "Decoder");
        load_plain(f, ap + "to_qkv.weight", 3 * dims[0], dims[0], arith_dtype_, mid_attn_.qkv_W, "Decoder");
        load_plain(f, ap + "to_qkv.bias", 3 * dims[0], 1, arith_dtype_, mid_attn_.qkv_b, "Decoder");
        load_plain(f, ap + "proj.weight", dims[0], dims[0], arith_dtype_, mid_attn_.proj_W, "Decoder");
        load_plain(f, ap + "proj.bias", dims[0], 1, arith_dtype_, mid_attn_.proj_b, "Decoder");
        mid_attn_.C = dims[0];
    }

    for (int i = 0; i < nb; ++i) {
        int in_dim = dims[static_cast<std::size_t>(i)];
        const int out_dim = dims[static_cast<std::size_t>(i) + 1];
        if (i > 0) in_dim /= 2;

        UpBlock& ub = up_blocks_[static_cast<std::size_t>(i)];
        ub.C_out = out_dim;
        ub.resnets.resize(static_cast<std::size_t>(cfg_.num_res_blocks) + 1);
        for (int j = 0; j <= cfg_.num_res_blocks; ++j) {
            const int Ci = (j == 0) ? in_dim : out_dim;
            const std::string rp = prefix + "decoder.up_blocks." + std::to_string(i) +
                                   ".resnets." + std::to_string(j) + ".";
            load_resnet_(f, rp, Ci, out_dim, ub.resnets[static_cast<std::size_t>(j)]);
        }

        ub.has_upsampler = (i < nb - 1);
        if (ub.has_upsampler) {
            const std::string up = prefix + "decoder.up_blocks." + std::to_string(i) +
                                   ".upsamplers.0.resample.1.";
            load_plain(f, up + "weight", out_dim / 2, out_dim * 9, arith_dtype_, ub.up_W, "Decoder");
            load_plain(f, up + "bias", out_dim / 2, 1, arith_dtype_, ub.up_b, "Decoder");
        }
    }

    const int firstC = dims.back();
    load_plain(f, prefix + "decoder.norm_out.gamma", firstC, 1, arith_dtype_, norm_out_g_, "Decoder");
    load_conv_lasttap(f, prefix + "decoder.conv_out.weight", cfg_.input_channels, firstC, 3, 9,
                      arith_dtype_, conv_out_W_, "Decoder");
    load_plain(f, prefix + "decoder.conv_out.bias", cfg_.input_channels, 1, arith_dtype_,
              conv_out_b_, "Decoder");
}

void Decoder::apply_rmsnorm_(const bt::Tensor& gamma, int C, int H, int W,
                             const bt::Tensor& x, bt::Tensor& out) {
    bt::nchw_to_sequence(x, 1, C, H, W, seq_);
    bt::rms_norm_forward(seq_, gamma, kRmsEps, seq2_);
    bt::sequence_to_nchw(seq2_, 1, C, H, W, out);
}

void Decoder::apply_resnet_(const Resnet& r, int H, int W, bt::Tensor& x) {
    if (r.has_shortcut) {
        bt::conv2d_forward(x, r.short_W, &r.short_b, 1, r.C_in, H, W, r.C_out,
                           1, 1, 1, 1, 0, 0, 1, 1, h_);
    } else {
        h_ = x.clone();
    }
    apply_rmsnorm_(r.norm1_g, r.C_in, H, W, x, n1_);
    bt::silu_forward(n1_, n1_);
    bt::conv2d_forward(n1_, r.conv1_W, &r.conv1_b, 1, r.C_in, H, W, r.C_out,
                       3, 3, 1, 1, 1, 1, 1, 1, y_);
    apply_rmsnorm_(r.norm2_g, r.C_out, H, W, y_, n2_);
    bt::silu_forward(n2_, n2_);
    bt::conv2d_forward(n2_, r.conv2_W, &r.conv2_b, 1, r.C_out, H, W, r.C_out,
                       3, 3, 1, 1, 1, 1, 1, 1, y_);
    bt::add_inplace(y_, h_);
    std::swap(x, y_);
}

void Decoder::apply_attention_(const Attention& a, int H, int W, bt::Tensor& x) {
    apply_rmsnorm_(a.norm_g, a.C, H, W, x, n1_);
    bt::Tensor Wq = sub_view(a.qkv_W, 0, a.C, a.C);
    bt::Tensor Wk = sub_view(a.qkv_W, static_cast<int64_t>(a.C) * a.C, a.C, a.C);
    bt::Tensor Wv = sub_view(a.qkv_W, static_cast<int64_t>(2) * a.C * a.C, a.C, a.C);
    bt::Tensor bq = sub_view(a.qkv_b, 0, a.C, 1);
    bt::Tensor bk = sub_view(a.qkv_b, a.C, a.C, 1);
    bt::Tensor bv = sub_view(a.qkv_b, static_cast<int64_t>(2) * a.C, a.C, 1);

    bt::nchw_to_sequence(n1_, 1, a.C, H, W, seq_);
    bt::flash_attention_qkvo_forward(
        seq_, /*Ctx=*/nullptr,
        Wq, &bq, Wk, &bk, Wv, &bv, a.proj_W, &a.proj_b,
        /*d_mask=*/nullptr, /*num_heads=*/1, /*causal=*/false,
        seq2_);
    bt::sequence_to_nchw(seq2_, 1, a.C, H, W, n2_);
    bt::add_inplace(x, n2_);
}

void Decoder::apply_upsample_(const UpBlock& u, int H, int W, bt::Tensor& x) {
    bt::upsample_nearest_2x(x, 1, u.C_out, H, W, up_t_);
    bt::conv2d_forward(up_t_, u.up_W, &u.up_b, 1, u.C_out, 2 * H, 2 * W,
                       u.C_out / 2, 3, 3, 1, 1, 1, 1, 1, 1, y_);
    std::swap(x, y_);
}

void Decoder::decode(const bt::Tensor& latent, int H_lat, int W_lat, bt::Tensor& out) {
    if (conv_in_W_.size() == 0) fail("Decoder", "decode: weights not loaded");
    if (H_lat <= 0 || W_lat <= 0) fail("Decoder", "decode: H_lat and W_lat must be positive");
    const int z_dim = cfg_.z_dim;
    if (latent.rows != 1 || latent.cols != z_dim * H_lat * W_lat) {
        fail("Decoder", "decode: latent must be (1, z_dim*H_lat*W_lat)");
    }

    const int spatial = H_lat * W_lat;

    // Per-channel denormalize: latent = latent*latents_std + latents_mean
    // (the diffusers pipeline applies this BEFORE calling vae.decode(); see
    // vae_qwenimage.h). Host-side affine — z_dim (16) is tiny, no
    // per-channel broadcast op exists in brotensor, and this runs once per
    // decode call.
    std::vector<float> lat_h = download_f32(latent);
    for (int c = 0; c < z_dim; ++c) {
        const float mean = cfg_.latents_mean[static_cast<std::size_t>(c)];
        const float std_ = cfg_.latents_std[static_cast<std::size_t>(c)];
        float* row = &lat_h[static_cast<std::size_t>(c) * static_cast<std::size_t>(spatial)];
        for (int i = 0; i < spatial; ++i) row[i] = row[i] * std_ + mean;
    }
    x_ = upload_at(lat_h, 1, z_dim * spatial, arith_dtype_);

    // post_quant_conv (1x1).
    bt::conv2d_forward(x_, post_quant_W_, &post_quant_b_, 1, z_dim, H_lat, W_lat,
                       z_dim, 1, 1, 1, 1, 0, 0, 1, 1, y_);
    std::swap(x_, y_);

    // conv_in.
    const int mid_C = conv_in_W_.rows;
    bt::conv2d_forward(x_, conv_in_W_, &conv_in_b_, 1, z_dim, H_lat, W_lat,
                       mid_C, 3, 3, 1, 1, 1, 1, 1, 1, y_);
    std::swap(x_, y_);

    // mid_block: resnet -> attention -> resnet.
    apply_resnet_(mid_res0_, H_lat, W_lat, x_);
    apply_attention_(mid_attn_, H_lat, W_lat, x_);
    apply_resnet_(mid_res1_, H_lat, W_lat, x_);

    // up_blocks.
    int H = H_lat, W = W_lat;
    for (auto& ub : up_blocks_) {
        for (auto& r : ub.resnets) apply_resnet_(r, H, W, x_);
        if (ub.has_upsampler) {
            apply_upsample_(ub, H, W, x_);
            H *= 2;
            W *= 2;
        }
    }

    // norm_out + SiLU + conv_out.
    const int firstC = norm_out_g_.rows;
    apply_rmsnorm_(norm_out_g_, firstC, H, W, x_, y_);
    bt::silu_forward(y_, y_);
    bt::conv2d_forward(y_, conv_out_W_, &conv_out_b_, 1, firstC, H, W,
                       cfg_.input_channels, 3, 3, 1, 1, 1, 1, 1, 1, out);
}

// ═══════════════════════════════════════════════════════════════════════
// Encoder
// ═══════════════════════════════════════════════════════════════════════

Encoder::Encoder(const Config& cfg) : cfg_(cfg) {
    if (cfg_.dim_mult.empty()) fail("Encoder", "dim_mult must be non-empty");
    if (cfg_.num_res_blocks <= 0) fail("Encoder", "num_res_blocks must be positive");
    if (!cfg_.attn_scales.empty()) {
        fail("Encoder", "non-empty attn_scales not supported (no known Krea 2 / "
                        "Qwen-Image checkpoint uses one; down/up-block attention "
                        "is not implemented)");
    }
    if (static_cast<int>(cfg_.latents_mean.size()) != cfg_.z_dim ||
        static_cast<int>(cfg_.latents_std.size()) != cfg_.z_dim) {
        fail("Encoder", "latents_mean/latents_std must have z_dim entries");
    }
    down_blocks_.resize(cfg_.dim_mult.size());
}

Encoder::~Encoder() = default;

void Encoder::load_resnet_(const st::File& f, const std::string& p,
                           int C_in, int C_out, Resnet& r) {
    load_plain(f, p + "norm1.gamma", C_in, 1, arith_dtype_, r.norm1_g, "Encoder");
    load_conv_lasttap(f, p + "conv1.weight", C_out, C_in, 3, 9, arith_dtype_, r.conv1_W, "Encoder");
    load_plain(f, p + "conv1.bias", C_out, 1, arith_dtype_, r.conv1_b, "Encoder");
    load_plain(f, p + "norm2.gamma", C_out, 1, arith_dtype_, r.norm2_g, "Encoder");
    load_conv_lasttap(f, p + "conv2.weight", C_out, C_out, 3, 9, arith_dtype_, r.conv2_W, "Encoder");
    load_plain(f, p + "conv2.bias", C_out, 1, arith_dtype_, r.conv2_b, "Encoder");

    r.C_in = C_in;
    r.C_out = C_out;
    r.has_shortcut = (C_in != C_out);
    if (r.has_shortcut) {
        load_conv_lasttap(f, p + "conv_shortcut.weight", C_out, C_in, 1, 1,
                          arith_dtype_, r.short_W, "Encoder");
        load_plain(f, p + "conv_shortcut.bias", C_out, 1, arith_dtype_, r.short_b, "Encoder");
    }
}

void Encoder::load_weights(const st::File& f, const std::string& prefix) {
    arith_dtype_ = arith_dtype_for(cfg_.force_upcast);
    const int nb = static_cast<int>(cfg_.dim_mult.size());

    // dims = base_dim * ([1] + dim_mult) — matches diffusers'
    // `[dim*u for u in [1] + dim_mult]`.
    std::vector<int> dims;
    dims.reserve(static_cast<std::size_t>(nb) + 1);
    dims.push_back(cfg_.base_dim);
    for (int i = 0; i < nb; ++i) dims.push_back(cfg_.base_dim * cfg_.dim_mult[static_cast<std::size_t>(i)]);

    load_conv_lasttap(f, prefix + "encoder.conv_in.weight", dims[0], cfg_.input_channels, 3, 9,
                      arith_dtype_, conv_in_W_, "Encoder");
    load_plain(f, prefix + "encoder.conv_in.bias", dims[0], 1, arith_dtype_, conv_in_b_, "Encoder");

    // The checkpoint stores every resnet AND every downsampler in encoder.
    // down_blocks as one running FLAT index (unlike the decoder's nested
    // up_blocks.{i}.resnets.{j}) — see vae_qwenimage.h.
    int flat_idx = 0;
    for (int i = 0; i < nb; ++i) {
        const int in_dim = dims[static_cast<std::size_t>(i)];
        const int out_dim = dims[static_cast<std::size_t>(i) + 1];

        DownBlock& db = down_blocks_[static_cast<std::size_t>(i)];
        db.C_out = out_dim;
        db.resnets.resize(static_cast<std::size_t>(cfg_.num_res_blocks));
        for (int j = 0; j < cfg_.num_res_blocks; ++j) {
            const int Ci = (j == 0) ? in_dim : out_dim;
            const std::string rp = prefix + "encoder.down_blocks." + std::to_string(flat_idx) + ".";
            load_resnet_(f, rp, Ci, out_dim, db.resnets[static_cast<std::size_t>(j)]);
            ++flat_idx;
        }

        db.has_downsampler = (i < nb - 1);
        if (db.has_downsampler) {
            const std::string dp = prefix + "encoder.down_blocks." + std::to_string(flat_idx) +
                                   ".resample.1.";
            load_plain(f, dp + "weight", out_dim, out_dim * 9, arith_dtype_, db.down_W, "Encoder");
            load_plain(f, dp + "bias", out_dim, 1, arith_dtype_, db.down_b, "Encoder");
            ++flat_idx;
        }
    }

    const int mid_C = dims.back();
    load_resnet_(f, prefix + "encoder.mid_block.resnets.0.", mid_C, mid_C, mid_res0_);
    load_resnet_(f, prefix + "encoder.mid_block.resnets.1.", mid_C, mid_C, mid_res1_);
    {
        const std::string ap = prefix + "encoder.mid_block.attentions.0.";
        load_plain(f, ap + "norm.gamma", mid_C, 1, arith_dtype_, mid_attn_.norm_g, "Encoder");
        load_plain(f, ap + "to_qkv.weight", 3 * mid_C, mid_C, arith_dtype_, mid_attn_.qkv_W, "Encoder");
        load_plain(f, ap + "to_qkv.bias", 3 * mid_C, 1, arith_dtype_, mid_attn_.qkv_b, "Encoder");
        load_plain(f, ap + "proj.weight", mid_C, mid_C, arith_dtype_, mid_attn_.proj_W, "Encoder");
        load_plain(f, ap + "proj.bias", mid_C, 1, arith_dtype_, mid_attn_.proj_b, "Encoder");
        mid_attn_.C = mid_C;
    }

    load_plain(f, prefix + "encoder.norm_out.gamma", mid_C, 1, arith_dtype_, norm_out_g_, "Encoder");
    const int twoZ = 2 * cfg_.z_dim;
    load_conv_lasttap(f, prefix + "encoder.conv_out.weight", twoZ, mid_C, 3, 9,
                      arith_dtype_, conv_out_W_, "Encoder");
    load_plain(f, prefix + "encoder.conv_out.bias", twoZ, 1, arith_dtype_, conv_out_b_, "Encoder");

    load_plain(f, prefix + "quant_conv.weight", twoZ, twoZ, arith_dtype_, quant_W_, "Encoder");
    load_plain(f, prefix + "quant_conv.bias", twoZ, 1, arith_dtype_, quant_b_, "Encoder");
}

void Encoder::apply_rmsnorm_(const bt::Tensor& gamma, int C, int H, int W,
                             const bt::Tensor& x, bt::Tensor& out) {
    bt::nchw_to_sequence(x, 1, C, H, W, seq_);
    bt::rms_norm_forward(seq_, gamma, kRmsEps, seq2_);
    bt::sequence_to_nchw(seq2_, 1, C, H, W, out);
}

void Encoder::apply_resnet_(const Resnet& r, int H, int W, bt::Tensor& x) {
    if (r.has_shortcut) {
        bt::conv2d_forward(x, r.short_W, &r.short_b, 1, r.C_in, H, W, r.C_out,
                           1, 1, 1, 1, 0, 0, 1, 1, h_);
    } else {
        h_ = x.clone();
    }
    apply_rmsnorm_(r.norm1_g, r.C_in, H, W, x, n1_);
    bt::silu_forward(n1_, n1_);
    bt::conv2d_forward(n1_, r.conv1_W, &r.conv1_b, 1, r.C_in, H, W, r.C_out,
                       3, 3, 1, 1, 1, 1, 1, 1, y_);
    apply_rmsnorm_(r.norm2_g, r.C_out, H, W, y_, n2_);
    bt::silu_forward(n2_, n2_);
    bt::conv2d_forward(n2_, r.conv2_W, &r.conv2_b, 1, r.C_out, H, W, r.C_out,
                       3, 3, 1, 1, 1, 1, 1, 1, y_);
    bt::add_inplace(y_, h_);
    std::swap(x, y_);
}

void Encoder::apply_attention_(const Attention& a, int H, int W, bt::Tensor& x) {
    apply_rmsnorm_(a.norm_g, a.C, H, W, x, n1_);
    bt::Tensor Wq = sub_view(a.qkv_W, 0, a.C, a.C);
    bt::Tensor Wk = sub_view(a.qkv_W, static_cast<int64_t>(a.C) * a.C, a.C, a.C);
    bt::Tensor Wv = sub_view(a.qkv_W, static_cast<int64_t>(2) * a.C * a.C, a.C, a.C);
    bt::Tensor bq = sub_view(a.qkv_b, 0, a.C, 1);
    bt::Tensor bk = sub_view(a.qkv_b, a.C, a.C, 1);
    bt::Tensor bv = sub_view(a.qkv_b, static_cast<int64_t>(2) * a.C, a.C, 1);

    bt::nchw_to_sequence(n1_, 1, a.C, H, W, seq_);
    bt::flash_attention_qkvo_forward(
        seq_, /*Ctx=*/nullptr,
        Wq, &bq, Wk, &bk, Wv, &bv, a.proj_W, &a.proj_b,
        /*d_mask=*/nullptr, /*num_heads=*/1, /*causal=*/false,
        seq2_);
    bt::sequence_to_nchw(seq2_, 1, a.C, H, W, n2_);
    bt::add_inplace(x, n2_);
}

void Encoder::apply_downsample_(const DownBlock& d, int H, int W, bt::Tensor& x) {
    // Diffusers Downsample2D-equivalent: F.pad(x,(0,1,0,1)) then stride-2 3x3
    // conv, pad=0 (matches vae.cpp's Encoder::apply_downsample_).
    bt::pad2d_forward(x, 1, d.C_out, H, W, /*pad_top=*/0, /*pad_bottom=*/1,
                      /*pad_left=*/0, /*pad_right=*/1, /*mode=*/0, pad_);
    bt::conv2d_forward(pad_, d.down_W, &d.down_b, 1, d.C_out, H + 1, W + 1,
                       d.C_out, 3, 3, 2, 2, 0, 0, 1, 1, y_);
    std::swap(x, y_);
}

void Encoder::encode(const bt::Tensor& image, int H, int W,
                     const bt::Tensor* eps, bt::Tensor& out) {
    if (conv_in_W_.size() == 0) fail("Encoder", "encode: weights not loaded");
    if (H <= 0 || W <= 0) fail("Encoder", "encode: H and W must be positive");
    const int total_ds = 1 << (static_cast<int>(cfg_.dim_mult.size()) - 1);
    if (H % total_ds != 0 || W % total_ds != 0) {
        fail("Encoder", "encode: H and W must be multiples of " + std::to_string(total_ds));
    }
    if (image.rows != 1 || image.cols != cfg_.input_channels * H * W) {
        fail("Encoder", "encode: image must be (1, input_channels*H*W)");
    }

    const int H_lat = H / total_ds, W_lat = W / total_ds;
    const int z_dim = cfg_.z_dim;
    const int spatial_lat = H_lat * W_lat;

    if (image.dtype != arith_dtype_) {
        bt::cast(image, x_, arith_dtype_);
    } else {
        x_ = image.clone();
    }

    const int dims0 = conv_in_W_.rows;
    bt::conv2d_forward(x_, conv_in_W_, &conv_in_b_, 1, cfg_.input_channels, H, W,
                       dims0, 3, 3, 1, 1, 1, 1, 1, 1, y_);
    std::swap(x_, y_);

    int Hc = H, Wc = W;
    for (auto& db : down_blocks_) {
        for (auto& r : db.resnets) apply_resnet_(r, Hc, Wc, x_);
        if (db.has_downsampler) {
            apply_downsample_(db, Hc, Wc, x_);
            Hc /= 2;
            Wc /= 2;
        }
    }

    apply_resnet_(mid_res0_, Hc, Wc, x_);
    apply_attention_(mid_attn_, Hc, Wc, x_);
    apply_resnet_(mid_res1_, Hc, Wc, x_);

    const int mid_C = norm_out_g_.rows;
    apply_rmsnorm_(norm_out_g_, mid_C, Hc, Wc, x_, y_);
    bt::silu_forward(y_, y_);
    const int twoZ = 2 * z_dim;
    bt::conv2d_forward(y_, conv_out_W_, &conv_out_b_, 1, mid_C, Hc, Wc,
                       twoZ, 3, 3, 1, 1, 1, 1, 1, 1, x_);

    bt::conv2d_forward(x_, quant_W_, &quant_b_, 1, twoZ, H_lat, W_lat,
                       twoZ, 1, 1, 1, 1, 0, 0, 1, 1, moments_);

    // Split (mean, logvar) along the channel axis — channel-major NCHW, so
    // mean occupies the first half of channels, logvar the second (matches
    // torch.chunk(parameters, 2, dim=1)).
    const int half = z_dim * spatial_lat;
    detail::resize_like(out, 1, half, brodiffusion::compute_dtype(), moments_.device);
    bt::copy_d2d(moments_, 0, out, 0, half);

    if (eps != nullptr) {
        if (eps->rows != 1 || eps->cols != half) {
            fail("Encoder", "encode: eps must be (1, z_dim*H_lat*W_lat)");
        }
        detail::resize_like(logvar_, 1, half, brodiffusion::compute_dtype(), moments_.device);
        bt::copy_d2d(moments_, half, logvar_, 0, half);

        std::vector<float> lv_host = download_f32(logvar_);
        for (float& v : lv_host) v = std::exp(0.5f * v);
        bt::Tensor std_dev = detail::upload_host(lv_host.data(), 1, half);
        if (std_dev.dtype != out.dtype) {
            bt::Tensor c;
            bt::cast(std_dev, c, out.dtype);
            std_dev = std::move(c);
        }
        bt::mul_inplace(std_dev, *eps);
        bt::add_inplace(out, std_dev);
    }

    // out = (sample - latents_mean) / latents_std, per z_dim channel. Host-
    // side affine (see Decoder::decode's comment).
    std::vector<float> out_h = download_f32(out);
    for (int c = 0; c < z_dim; ++c) {
        const float mean = cfg_.latents_mean[static_cast<std::size_t>(c)];
        const float std_ = cfg_.latents_std[static_cast<std::size_t>(c)];
        float* row = &out_h[static_cast<std::size_t>(c) * static_cast<std::size_t>(spatial_lat)];
        for (int i = 0; i < spatial_lat; ++i) row[i] = (row[i] - mean) / std_;
    }
    out = upload_at(out_h, 1, half, brodiffusion::compute_dtype());
}

}  // namespace brodiffusion::vae_qwenimage
