#include "brodiffusion/vae_qwenimage21.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"
#include "brodiffusion/detail/jit_fusion.h"
#include "brotensor/safetensors.h"

#include "brotensor/jit/trace.h"
#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::vae_qwenimage21 {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

// F.normalize's default eps — see vae_qwenimage.h.
constexpr float kRmsEps = 1e-12f;

[[noreturn]] void fail(const char* who, const std::string& msg) {
    throw std::runtime_error(std::string("vae_qwenimage21::") + who + ": " + msg);
}

const st::TensorView& need(const st::File& f, const std::string& key, const char* who) {
    const auto* v = f.find(key);
    if (!v) fail(who, "missing tensor '" + key + "'");
    return *v;
}

// ── RMSNorm + SiLU, one kernel ─────────────────────────────────────────────
//
// Every norm in this graph except the attention block's is immediately
// followed by a SiLU. Eagerly that is two full passes over the feature map:
// rms_norm_forward writes it and silu_forward reads it straight back. The
// widest map in a 1024x1024 decode reaches the row kernel as 1048576x96 —
// 384 MB written and 384 MB read for a value that never needed to land.
// Traced, it is one kernel and the normalised map never exists.
//
// gamma is stored as (C, 1). The trace needs it as the (1, C) broadcast row
// it actually is, or the compiler sees an operand whose element count does
// not match the trace's (H*W, C) shape and refuses to fuse. The view is
// non-owning and costs nothing.
//
// The binding is identified by the three buffers plus both extents: the
// scratch tensors are shared across every norm in the graph and grow as the
// decode walks up the resolutions, so addresses alone would replay a kernel
// compiled for one feature-map size against another.
bool fuse_rmsnorm_silu(brodiffusion::detail::JitSite& site, const bt::Tensor& seq,
                       const bt::Tensor& gamma, bt::Tensor& dst) {
    const bt::Tensor g_row =
        bt::Tensor::view(gamma.device, gamma.data, 1, gamma.rows * gamma.cols,
                         gamma.dtype);
    return brodiffusion::detail::try_fused(
        site,
        {seq.data, gamma.data, dst.data,
         brodiffusion::detail::token_of_extent(static_cast<std::size_t>(seq.rows)),
         brodiffusion::detail::token_of_extent(static_cast<std::size_t>(seq.cols))},
        [&] { bt::store(dst, bt::silu(bt::rms_norm(seq, g_row, kRmsEps))); });
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

// Every checkpoint tensor is a plain 2D conv filter [Cout,Cin,kh,kw], a
// bias, or a norm gamma — numel must equal rows*cols.
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

bt::Tensor sub_view(const bt::Tensor& t, int64_t off, int rows, int cols) {
    char* p = static_cast<char*>(t.data) + off * bt::dtype_size_bytes(t.dtype);
    return bt::Tensor::view(t.device, p, rows, cols, t.dtype);
}

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

// Parity debugging: when BRODIFFUSION_VAE_DUMP names a directory, every
// stage boundary of encode()/decode() is written there as raw FP32
// (<dir>/qi21_<enc|dec>_<name>.f32) for diffing against the reference
// hooks in scripts/qwenimage21_vae_ref.py.
void maybe_dump(const char* name, const bt::Tensor& t) {
    const char* dir = std::getenv("BRODIFFUSION_VAE_DUMP");
    if (!dir || !dir[0]) return;
    std::vector<float> v = download_f32(t);
    const std::string path = std::string(dir) + "/qi21_" + name + ".f32";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
}

void check_config(const Config& cfg, const char* who) {
    if (cfg.dim_mult.empty()) fail(who, "dim_mult must be non-empty");
    if (cfg.num_res_blocks <= 0) fail(who, "num_res_blocks must be positive");
    if (!cfg.attn_scales.empty()) fail(who, "non-empty attn_scales is not supported");
    // One flag per stage TRANSITION (diffusers indexes it only for
    // i != last), so nb-1 entries; longer lists are tolerated.
    if (cfg.temperal_downsample.size() + 1 < cfg.dim_mult.size()) {
        fail(who, "temperal_downsample needs at least dim_mult.size()-1 entries");
    }
    if (static_cast<int>(cfg.latents_mean.size()) != cfg.z_dim ||
        static_cast<int>(cfg.latents_std.size()) != cfg.z_dim) {
        fail(who, "latents_mean/latents_std must have z_dim entries");
    }
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════
// Decoder
// ═══════════════════════════════════════════════════════════════════════

Decoder::Decoder(const Config& cfg) : cfg_(cfg) {
    check_config(cfg_, "Decoder");
    up_blocks_.resize(cfg_.dim_mult.size());
}

Decoder::~Decoder() = default;

void Decoder::load_resnet_(const st::File& f, const std::string& p,
                           int C_in, int C_out, Resnet& r) {
    load_plain(f, p + "norm1.gamma", C_in, 1, arith_dtype_, r.norm1_g, "Decoder");
    load_plain(f, p + "conv1.weight", C_out, C_in * 9, arith_dtype_, r.conv1_W, "Decoder");
    load_plain(f, p + "conv1.bias", C_out, 1, arith_dtype_, r.conv1_b, "Decoder");
    load_plain(f, p + "norm2.gamma", C_out, 1, arith_dtype_, r.norm2_g, "Decoder");
    load_plain(f, p + "conv2.weight", C_out, C_out * 9, arith_dtype_, r.conv2_W, "Decoder");
    load_plain(f, p + "conv2.bias", C_out, 1, arith_dtype_, r.conv2_b, "Decoder");

    r.C_in = C_in;
    r.C_out = C_out;
    r.has_shortcut = (C_in != C_out);
    if (r.has_shortcut) {
        load_plain(f, p + "conv_shortcut.weight", C_out, C_in, arith_dtype_, r.short_W, "Decoder");
        load_plain(f, p + "conv_shortcut.bias", C_out, 1, arith_dtype_, r.short_b, "Decoder");
    }
}

void Decoder::load_weights(const st::File& f, const std::string& prefix) {
    arith_dtype_ = arith_dtype_for(cfg_.force_upcast);
    const int nb = static_cast<int>(cfg_.dim_mult.size());
    const int z_dim = cfg_.z_dim;

    // dims = decoder_base_dim * [dim_mult.back()] + reversed(dim_mult).
    std::vector<int> dims;
    dims.reserve(static_cast<std::size_t>(nb) + 1);
    dims.push_back(cfg_.decoder_base_dim * cfg_.dim_mult.back());
    for (int i = nb - 1; i >= 0; --i) {
        dims.push_back(cfg_.decoder_base_dim * cfg_.dim_mult[static_cast<std::size_t>(i)]);
    }

    load_plain(f, prefix + "post_quant_conv.weight", z_dim, z_dim, arith_dtype_,
               post_quant_W_, "Decoder");
    load_plain(f, prefix + "post_quant_conv.bias", z_dim, 1, arith_dtype_,
               post_quant_b_, "Decoder");

    load_plain(f, prefix + "decoder.conv_in.weight", dims[0], z_dim * 9,
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

    // temperal_upsample = reversed(temperal_downsample[:nb-1]); decoder
    // stage i (< nb-1) reads entry nb-2-i.
    for (int i = 0; i < nb; ++i) {
        const int in_dim = dims[static_cast<std::size_t>(i)];
        const int out_dim = dims[static_cast<std::size_t>(i) + 1];

        UpBlock& ub = up_blocks_[static_cast<std::size_t>(i)];
        ub.C_in = in_dim;
        ub.C_out = out_dim;
        ub.resnets.resize(static_cast<std::size_t>(cfg_.num_res_blocks) + 1);
        for (int j = 0; j <= cfg_.num_res_blocks; ++j) {
            const int Ci = (j == 0) ? in_dim : out_dim;
            const std::string rp = prefix + "decoder.up_blocks." + std::to_string(i) +
                                   ".resnets." + std::to_string(j) + ".";
            load_resnet_(f, rp, Ci, out_dim, ub.resnets[static_cast<std::size_t>(j)]);
        }

        ub.has_upsampler = (i < nb - 1);
        ub.temporal = ub.has_upsampler &&
                      cfg_.temperal_downsample[static_cast<std::size_t>(nb - 2 - i)];
        if (ub.has_upsampler) {
            const std::string up = prefix + "decoder.up_blocks." + std::to_string(i) +
                                   ".upsampler.resample.1.";
            load_plain(f, up + "weight", out_dim, out_dim * 9, arith_dtype_, ub.up_W, "Decoder");
            load_plain(f, up + "bias", out_dim, 1, arith_dtype_, ub.up_b, "Decoder");
            // upsampler.time_conv.* (upsample3d) is never executed for a
            // single frame — intentionally not read.

            // Validate the DupUp3D reduction this stage needs (see header).
            const int factor = (ub.temporal ? 2 : 1) * 4;
            if ((static_cast<int64_t>(out_dim) * factor) % in_dim != 0) {
                fail("Decoder", "up_block " + std::to_string(i) +
                     ": DupUp3D requires out*factor % in == 0");
            }
            const int repeats = out_dim * factor / in_dim;
            if (ub.temporal && repeats != 8 && repeats != 4) {
                fail("Decoder", "up_block " + std::to_string(i) +
                     ": unsupported temporal DupUp3D repeats=" + std::to_string(repeats));
            }
        }
    }

    const int firstC = dims.back();
    load_plain(f, prefix + "decoder.norm_out.gamma", firstC, 1, arith_dtype_, norm_out_g_, "Decoder");
    load_plain(f, prefix + "decoder.conv_out.weight", cfg_.out_channels, firstC * 9,
               arith_dtype_, conv_out_W_, "Decoder");
    load_plain(f, prefix + "decoder.conv_out.bias", cfg_.out_channels, 1, arith_dtype_,
               conv_out_b_, "Decoder");
}

void Decoder::apply_rmsnorm_(const bt::Tensor& gamma, int C, int H, int W,
                             const bt::Tensor& x, bt::Tensor& out) {
    bt::nchw_to_sequence(x, 1, C, H, W, seq_);
    bt::rms_norm_forward(seq_, gamma, kRmsEps, seq2_);
    bt::sequence_to_nchw(seq2_, 1, C, H, W, out);
}

void Decoder::apply_rmsnorm_silu_(detail::JitSite& site, const bt::Tensor& gamma,
                                  int C, int H, int W, const bt::Tensor& x,
                                  bt::Tensor& out) {
    bt::nchw_to_sequence(x, 1, C, H, W, seq_);
    // store() writes into a buffer the caller owns, so seq2_ has to be the
    // right shape before the expression is traced.
    detail::resize_like(seq2_, seq_.rows, seq_.cols, seq_.dtype, seq_.device);
    if (!fuse_rmsnorm_silu(site, seq_, gamma, seq2_)) {
        bt::rms_norm_forward(seq_, gamma, kRmsEps, seq2_);
        bt::silu_forward(seq2_, seq2_);
    }
    bt::sequence_to_nchw(seq2_, 1, C, H, W, out);
}

void Decoder::apply_resnet_(const Resnet& r, int H, int W, bt::Tensor& x) {
    if (r.has_shortcut) {
        bt::conv2d_forward(x, r.short_W, &r.short_b, 1, r.C_in, H, W, r.C_out,
                           1, 1, 1, 1, 0, 0, 1, 1, h_);
    } else {
        h_ = x.clone();
    }
    apply_rmsnorm_silu_(jit_norm1_, r.norm1_g, r.C_in, H, W, x, n1_);
    bt::conv2d_forward(n1_, r.conv1_W, &r.conv1_b, 1, r.C_in, H, W, r.C_out,
                       3, 3, 1, 1, 1, 1, 1, 1, y_);
    apply_rmsnorm_silu_(jit_norm2_, r.norm2_g, r.C_out, H, W, y_, n2_);
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
    // nearest-exact 2x at an integer factor is plain nearest; conv keeps
    // out_dim -> out_dim (upsample_out_dim=out_dim).
    bt::upsample_nearest_2x(x, 1, u.C_out, H, W, up_t_);
    bt::conv2d_forward(up_t_, u.up_W, &u.up_b, 1, u.C_out, 2 * H, 2 * W,
                       u.C_out, 3, 3, 1, 1, 1, 1, 1, 1, y_);
    std::swap(x, y_);
}

void Decoder::apply_dup_shortcut_(const UpBlock& u, int H, int W,
                                  const bt::Tensor& x_copy, bt::Tensor& out) {
    const int HW = H * W;
    if (!u.temporal) {
        // factor_t=1: repeat_interleave(4*out/in) + pixel_shuffle(2).
        bt::pixel_shuffle_upsample_2x_forward(x_copy, 1, u.C_in, H, W, u.C_out, out);
        return;
    }
    const int repeats = u.C_out * 8 / u.C_in;
    if (repeats == 8) {
        // in == out: every kept slot reads the same channel -> nearest 2x.
        bt::upsample_nearest_2x(x_copy, 1, u.C_in, H, W, out);
        return;
    }
    // repeats == 4 (in == 2*out): kept frame slot maps output channel o to
    // input channel 2o+1 -> gather odd channels, then nearest 2x.
    detail::resize_like(gather_, 1, u.C_out * HW, x_copy.dtype, x_copy.device);
    bt::copy_d2d_strided(x_copy, HW, 2 * HW, gather_, 0, HW, HW, u.C_out);
    bt::upsample_nearest_2x(gather_, 1, u.C_out, H, W, out);
}

void Decoder::decode(const bt::Tensor& latent, int H_lat, int W_lat, bt::Tensor& out) {
    if (conv_in_W_.size() == 0) fail("Decoder", "decode: weights not loaded");
    if (H_lat <= 0 || W_lat <= 0) fail("Decoder", "decode: H_lat and W_lat must be positive");
    const int z_dim = cfg_.z_dim;
    if (latent.rows != 1 || latent.cols != z_dim * H_lat * W_lat) {
        fail("Decoder", "decode: latent must be (1, z_dim*H_lat*W_lat)");
    }

    const int spatial = H_lat * W_lat;

    // Per-channel denormalise (pipeline-side in diffusers).
    std::vector<float> lat_h = download_f32(latent);
    for (int c = 0; c < z_dim; ++c) {
        const float mean = cfg_.latents_mean[static_cast<std::size_t>(c)];
        const float std_ = cfg_.latents_std[static_cast<std::size_t>(c)];
        float* row = &lat_h[static_cast<std::size_t>(c) * static_cast<std::size_t>(spatial)];
        for (int i = 0; i < spatial; ++i) row[i] = row[i] * std_ + mean;
    }
    x_ = upload_at(lat_h, 1, z_dim * spatial, arith_dtype_);

    bt::conv2d_forward(x_, post_quant_W_, &post_quant_b_, 1, z_dim, H_lat, W_lat,
                       z_dim, 1, 1, 1, 1, 0, 0, 1, 1, y_);
    std::swap(x_, y_);

    const int mid_C = conv_in_W_.rows;
    bt::conv2d_forward(x_, conv_in_W_, &conv_in_b_, 1, z_dim, H_lat, W_lat,
                       mid_C, 3, 3, 1, 1, 1, 1, 1, 1, y_);
    std::swap(x_, y_);
    maybe_dump("dec_conv_in", x_);

    apply_resnet_(mid_res0_, H_lat, W_lat, x_);
    apply_attention_(mid_attn_, H_lat, W_lat, x_);
    apply_resnet_(mid_res1_, H_lat, W_lat, x_);
    maybe_dump("dec_mid_block", x_);

    int H = H_lat, W = W_lat;
    int ub_idx = 0;
    for (auto& ub : up_blocks_) {
        if (ub.has_upsampler) {
            // Residual up-block: main = upsample(resnets(x)); out = main +
            // DupUp3D(x). Keep x for the shortcut.
            x_copy_ = x_.clone();
        }
        for (auto& r : ub.resnets) apply_resnet_(r, H, W, x_);
        if (ub.has_upsampler) {
            apply_upsample_(ub, H, W, x_);
            apply_dup_shortcut_(ub, H, W, x_copy_, short_);
            bt::add_inplace(x_, short_);
            H *= 2;
            W *= 2;
        }
        maybe_dump(("dec_up" + std::to_string(ub_idx++)).c_str(), x_);
    }

    const int firstC = norm_out_g_.rows;
    apply_rmsnorm_silu_(jit_norm_out_, norm_out_g_, firstC, H, W, x_, y_);
    bt::conv2d_forward(y_, conv_out_W_, &conv_out_b_, 1, firstC, H, W,
                       cfg_.out_channels, 3, 3, 1, 1, 1, 1, 1, 1, out);
    maybe_dump("dec_conv_out", out);
    bt::clamp(out, -1.0f, 1.0f);
}

// ═══════════════════════════════════════════════════════════════════════
// Encoder
// ═══════════════════════════════════════════════════════════════════════

Encoder::Encoder(const Config& cfg) : cfg_(cfg) {
    check_config(cfg_, "Encoder");
    down_blocks_.resize(cfg_.dim_mult.size());
}

Encoder::~Encoder() = default;

void Encoder::load_resnet_(const st::File& f, const std::string& p,
                           int C_in, int C_out, Resnet& r) {
    load_plain(f, p + "norm1.gamma", C_in, 1, arith_dtype_, r.norm1_g, "Encoder");
    load_plain(f, p + "conv1.weight", C_out, C_in * 9, arith_dtype_, r.conv1_W, "Encoder");
    load_plain(f, p + "conv1.bias", C_out, 1, arith_dtype_, r.conv1_b, "Encoder");
    load_plain(f, p + "norm2.gamma", C_out, 1, arith_dtype_, r.norm2_g, "Encoder");
    load_plain(f, p + "conv2.weight", C_out, C_out * 9, arith_dtype_, r.conv2_W, "Encoder");
    load_plain(f, p + "conv2.bias", C_out, 1, arith_dtype_, r.conv2_b, "Encoder");

    r.C_in = C_in;
    r.C_out = C_out;
    r.has_shortcut = (C_in != C_out);
    if (r.has_shortcut) {
        load_plain(f, p + "conv_shortcut.weight", C_out, C_in, arith_dtype_, r.short_W, "Encoder");
        load_plain(f, p + "conv_shortcut.bias", C_out, 1, arith_dtype_, r.short_b, "Encoder");
    }
}

void Encoder::load_weights(const st::File& f, const std::string& prefix) {
    arith_dtype_ = arith_dtype_for(cfg_.force_upcast);
    const int nb = static_cast<int>(cfg_.dim_mult.size());

    // dims = base_dim * ([1] + dim_mult).
    std::vector<int> dims;
    dims.reserve(static_cast<std::size_t>(nb) + 1);
    dims.push_back(cfg_.base_dim);
    for (int i = 0; i < nb; ++i) dims.push_back(cfg_.base_dim * cfg_.dim_mult[static_cast<std::size_t>(i)]);

    load_plain(f, prefix + "encoder.conv_in.weight", dims[0], cfg_.in_channels * 9,
               arith_dtype_, conv_in_W_, "Encoder");
    load_plain(f, prefix + "encoder.conv_in.bias", dims[0], 1, arith_dtype_, conv_in_b_, "Encoder");

    for (int i = 0; i < nb; ++i) {
        const int in_dim = dims[static_cast<std::size_t>(i)];
        const int out_dim = dims[static_cast<std::size_t>(i) + 1];

        DownBlock& db = down_blocks_[static_cast<std::size_t>(i)];
        db.C_in = in_dim;
        db.C_out = out_dim;
        db.resnets.resize(static_cast<std::size_t>(cfg_.num_res_blocks));
        for (int j = 0; j < cfg_.num_res_blocks; ++j) {
            const int Ci = (j == 0) ? in_dim : out_dim;
            const std::string rp = prefix + "encoder.down_blocks." + std::to_string(i) +
                                   ".resnets." + std::to_string(j) + ".";
            load_resnet_(f, rp, Ci, out_dim, db.resnets[static_cast<std::size_t>(j)]);
        }

        db.has_downsampler = (i < nb - 1);
        db.temporal = db.has_downsampler &&
                      cfg_.temperal_downsample[static_cast<std::size_t>(i)];
        if (db.has_downsampler) {
            const std::string dp = prefix + "encoder.down_blocks." + std::to_string(i) +
                                   ".downsampler.resample.1.";
            load_plain(f, dp + "weight", out_dim, out_dim * 9, arith_dtype_, db.down_W, "Encoder");
            load_plain(f, dp + "bias", out_dim, 1, arith_dtype_, db.down_b, "Encoder");
            // downsampler.time_conv.* (downsample3d) never executes for one
            // frame — intentionally not read.
        }

        // Validate the AvgDown3D reduction this stage needs (see header).
        const int factor = (db.temporal ? 2 : 1) * (db.has_downsampler ? 4 : 1);
        if ((static_cast<int64_t>(in_dim) * factor) % out_dim != 0) {
            fail("Encoder", "down_block " + std::to_string(i) +
                 ": AvgDown3D requires in*factor % out == 0");
        }
        const int group = in_dim * factor / out_dim;
        if (db.has_downsampler && group != 4) {
            fail("Encoder", "down_block " + std::to_string(i) +
                 ": unsupported AvgDown3D group_size=" + std::to_string(group));
        }
        if (!db.has_downsampler && group != 1) {
            fail("Encoder", "down_block " + std::to_string(i) +
                 ": non-identity AvgDown3D on the last stage is unsupported");
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
    load_plain(f, prefix + "encoder.conv_out.weight", twoZ, mid_C * 9,
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

void Encoder::apply_rmsnorm_silu_(detail::JitSite& site, const bt::Tensor& gamma,
                                  int C, int H, int W, const bt::Tensor& x,
                                  bt::Tensor& out) {
    bt::nchw_to_sequence(x, 1, C, H, W, seq_);
    detail::resize_like(seq2_, seq_.rows, seq_.cols, seq_.dtype, seq_.device);
    if (!fuse_rmsnorm_silu(site, seq_, gamma, seq2_)) {
        bt::rms_norm_forward(seq_, gamma, kRmsEps, seq2_);
        bt::silu_forward(seq2_, seq2_);
    }
    bt::sequence_to_nchw(seq2_, 1, C, H, W, out);
}

void Encoder::apply_resnet_(const Resnet& r, int H, int W, bt::Tensor& x) {
    if (r.has_shortcut) {
        bt::conv2d_forward(x, r.short_W, &r.short_b, 1, r.C_in, H, W, r.C_out,
                           1, 1, 1, 1, 0, 0, 1, 1, h_);
    } else {
        h_ = x.clone();
    }
    apply_rmsnorm_silu_(jit_norm1_, r.norm1_g, r.C_in, H, W, x, n1_);
    bt::conv2d_forward(n1_, r.conv1_W, &r.conv1_b, 1, r.C_in, H, W, r.C_out,
                       3, 3, 1, 1, 1, 1, 1, 1, y_);
    apply_rmsnorm_silu_(jit_norm2_, r.norm2_g, r.C_out, H, W, y_, n2_);
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
    // ZeroPad2d((0,1,0,1)) then stride-2 3x3 conv, pad 0.
    bt::pad2d_forward(x, 1, d.C_out, H, W, /*pad_top=*/0, /*pad_bottom=*/1,
                      /*pad_left=*/0, /*pad_right=*/1, /*mode=*/0, pad_);
    bt::conv2d_forward(pad_, d.down_W, &d.down_b, 1, d.C_out, H + 1, W + 1,
                       d.C_out, 3, 3, 2, 2, 0, 0, 1, 1, y_);
    std::swap(x, y_);
}

void Encoder::apply_avg_shortcut_(const DownBlock& d, int H, int W,
                                  const bt::Tensor& x_copy, bt::Tensor& out) {
    if (!d.has_downsampler) {
        // factor 1, in == out: identity.
        out = x_copy.clone();
        return;
    }
    if (!d.temporal) {
        // factor_t=1, factor_s=2, group_size 4 (in == out): plain 2x2 avg-pool.
        bt::downsample_avg_2x(x_copy, 1, d.C_in, H, W, out);
        return;
    }
    // factor_t=2, factor_s=2, group_size 4 (out == 2*in): the zero frame
    // padded at the front of T fills the even output channels; odd channel
    // o is the 2x2 avg-pool of input channel o/2.
    const int HWq = (H / 2) * (W / 2);
    bt::downsample_avg_2x(x_copy, 1, d.C_in, H, W, pool_);
    out = bt::Tensor::zeros_on(x_copy.device, 1, d.C_out * HWq, x_copy.dtype);
    bt::copy_d2d_strided(pool_, 0, HWq, out, HWq, 2 * HWq, HWq, d.C_in);
}

void Encoder::encode(const bt::Tensor& image, int H, int W,
                     const bt::Tensor* eps, bt::Tensor& out) {
    if (conv_in_W_.size() == 0) fail("Encoder", "encode: weights not loaded");
    if (H <= 0 || W <= 0) fail("Encoder", "encode: H and W must be positive");
    const int total_ds = cfg_.spatial_scale();
    if (H % total_ds != 0 || W % total_ds != 0) {
        fail("Encoder", "encode: H and W must be multiples of " + std::to_string(total_ds));
    }
    if (image.rows != 1 || image.cols != cfg_.in_channels * H * W) {
        fail("Encoder", "encode: image must be (1, in_channels*H*W)");
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
    bt::conv2d_forward(x_, conv_in_W_, &conv_in_b_, 1, cfg_.in_channels, H, W,
                       dims0, 3, 3, 1, 1, 1, 1, 1, 1, y_);
    std::swap(x_, y_);
    maybe_dump("enc_conv_in", x_);

    int Hc = H, Wc = W;
    int db_idx = 0;
    for (auto& db : down_blocks_) {
        x_copy_ = x_.clone();
        for (auto& r : db.resnets) apply_resnet_(r, Hc, Wc, x_);
        if (db.has_downsampler) apply_downsample_(db, Hc, Wc, x_);
        apply_avg_shortcut_(db, Hc, Wc, x_copy_, short_);
        bt::add_inplace(x_, short_);
        if (db.has_downsampler) {
            Hc /= 2;
            Wc /= 2;
        }
        maybe_dump(("enc_down" + std::to_string(db_idx++)).c_str(), x_);
    }

    apply_resnet_(mid_res0_, Hc, Wc, x_);
    apply_attention_(mid_attn_, Hc, Wc, x_);
    apply_resnet_(mid_res1_, Hc, Wc, x_);
    maybe_dump("enc_mid_block", x_);

    const int mid_C = norm_out_g_.rows;
    apply_rmsnorm_silu_(jit_norm_out_, norm_out_g_, mid_C, Hc, Wc, x_, y_);
    const int twoZ = 2 * z_dim;
    bt::conv2d_forward(y_, conv_out_W_, &conv_out_b_, 1, mid_C, Hc, Wc,
                       twoZ, 3, 3, 1, 1, 1, 1, 1, 1, x_);
    maybe_dump("enc_conv_out", x_);

    bt::conv2d_forward(x_, quant_W_, &quant_b_, 1, twoZ, H_lat, W_lat,
                       twoZ, 1, 1, 1, 1, 0, 0, 1, 1, moments_);
    maybe_dump("enc_quant", moments_);

    // Split (mean, logvar) along the channel axis. copy_d2d is a raw byte
    // copy, so the halves must stay at moments_'s (arithmetic) dtype — the
    // caller-facing compute dtype is restored by the affine re-upload below.
    const int half = z_dim * spatial_lat;
    detail::resize_like(out, 1, half, moments_.dtype, moments_.device);
    bt::copy_d2d(moments_, 0, out, 0, half);

    if (eps != nullptr) {
        if (eps->rows != 1 || eps->cols != half) {
            fail("Encoder", "encode: eps must be (1, z_dim*H_lat*W_lat)");
        }
        detail::resize_like(logvar_, 1, half, moments_.dtype, moments_.device);
        bt::copy_d2d(moments_, half, logvar_, 0, half);

        std::vector<float> lv_host = download_f32(logvar_);
        for (float& v : lv_host) v = std::exp(0.5f * v);
        bt::Tensor std_dev = upload_at(lv_host, 1, half, out.dtype);
        bt::Tensor eps_c;
        if (eps->dtype != out.dtype) {
            bt::cast(*eps, eps_c, out.dtype);
        } else {
            eps_c = *eps;
        }
        bt::mul_inplace(std_dev, eps_c);
        bt::add_inplace(out, std_dev);
    }

    std::vector<float> out_h = download_f32(out);
    for (int c = 0; c < z_dim; ++c) {
        const float mean = cfg_.latents_mean[static_cast<std::size_t>(c)];
        const float std_ = cfg_.latents_std[static_cast<std::size_t>(c)];
        float* row = &out_h[static_cast<std::size_t>(c) * static_cast<std::size_t>(spatial_lat)];
        for (int i = 0; i < spatial_lat; ++i) row[i] = (row[i] - mean) / std_;
    }
    out = upload_at(out_h, 1, half, brodiffusion::compute_dtype());
}

}  // namespace brodiffusion::vae_qwenimage21
