#include "brodiffusion/vae.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"
#include "brotensor/safetensors.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::vae {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("vae::Decoder: " + msg);
}
[[noreturn]] void fail_enc(const std::string& msg) {
    throw std::runtime_error("vae::Encoder: " + msg);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    const auto* v = f.find(key);
    if (!v) throw std::runtime_error("vae::Decoder: missing tensor '" + key + "'");
    return *v;
}
const st::TensorView& need_enc(const st::File& f, const std::string& key) {
    const auto* v = f.find(key);
    if (!v) throw std::runtime_error("vae::Encoder: missing tensor '" + key + "'");
    return *v;
}

// The arithmetic dtype for a VAE: BF16 when force_upcast is set on a CUDA
// backend, otherwise the pipeline compute dtype. SDXL/Flux-class VAEs
// overflow FP16 internally (the Flux 16-channel VAE NaNs every output
// pixel); what they need is FP32's RANGE, which BF16 provides at half the
// bandwidth — and the CUDA backend's fused resblock has FP16/BF16 slots but
// no FP32 one.
bt::Dtype vae_arith_dtype(bool force_upcast) {
    if (force_upcast &&
        brotensor::default_device() == brotensor::Device::CUDA) {
        return bt::Dtype::BF16;
    }
    return brodiffusion::compute_dtype();
}

// Target dtype for the upload helpers below. Set by each load_weights()
// before its upload sequence (single-threaded loads; the helpers are free
// functions and a parameter would touch every one of ~60 call sites).
bt::Dtype g_vae_upload_dtype = bt::Dtype::FP32;

// Accepts F16 / F32 / BF16 (converted host-side as needed); SD1.5 full
// checkpoints ship as F32, Flux-family VAEs as BF16. Uploads at
// g_vae_upload_dtype (the owning module's arithmetic dtype).
void upload_compute_checked(const st::TensorView& v, int rows, int cols,
                            bt::Tensor& dst, const char* name) {
    if (v.dtype != st::Dtype::F16 && v.dtype != st::Dtype::F32 &&
        v.dtype != st::Dtype::BF16) {
        fail(std::string(name) + ": expected F16/F32/BF16, got " + st::dtype_name(v.dtype));
    }
    int64_t expected = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    if (v.numel() != expected) {
        fail(std::string(name) + ": shape mismatch (expected " +
             std::to_string(rows) + "x" + std::to_string(cols) + ", got " +
             std::to_string(v.numel()) + ")");
    }
    st::upload_as(v, rows, cols, g_vae_upload_dtype, dst);
}
void upload_compute_checked_enc(const st::TensorView& v, int rows, int cols,
                                bt::Tensor& dst, const char* name) {
    if (v.dtype != st::Dtype::F16 && v.dtype != st::Dtype::F32 &&
        v.dtype != st::Dtype::BF16) {
        fail_enc(std::string(name) + ": expected F16/F32/BF16, got " + st::dtype_name(v.dtype));
    }
    int64_t expected = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    if (v.numel() != expected) {
        fail_enc(std::string(name) + ": shape mismatch (expected " +
                 std::to_string(rows) + "x" + std::to_string(cols) + ", got " +
                 std::to_string(v.numel()) + ")");
    }
    st::upload_as(v, rows, cols, g_vae_upload_dtype, dst);
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

Decoder::Decoder(const DecoderConfig& cfg) : cfg_(cfg) {
    if (cfg_.block_out_channels.empty()) fail("block_out_channels must be non-empty");
    if (cfg_.layers_per_block <= 0) fail("layers_per_block must be positive");
    for (int c : cfg_.block_out_channels) {
        if (c <= 0 || c % cfg_.norm_num_groups != 0) {
            fail("each block_out_channels entry must be a positive multiple of norm_num_groups");
        }
    }
    up_blocks_.resize(cfg_.block_out_channels.size());
}

Decoder::~Decoder() = default;

// ─── load_weights ──────────────────────────────────────────────────────────

void Decoder::load_resnet_(const st::File& f, const std::string& p,
                           int C_in, int C_out, Resnet& r) {
    upload_compute_checked(need(f, p + "norm1.weight"), C_in, 1, r.norm1_g, "norm1.weight");
    upload_compute_checked(need(f, p + "norm1.bias"),   C_in, 1, r.norm1_b, "norm1.bias");
    // conv1: (C_out, C_in * 3 * 3) — flattened as 2D for safetensors-upload.
    upload_compute_checked(need(f, p + "conv1.weight"), C_out, C_in * 3 * 3, r.conv1_W, "conv1.weight");
    upload_compute_checked(need(f, p + "conv1.bias"),   C_out, 1, r.conv1_b, "conv1.bias");
    upload_compute_checked(need(f, p + "norm2.weight"), C_out, 1, r.norm2_g, "norm2.weight");
    upload_compute_checked(need(f, p + "norm2.bias"),   C_out, 1, r.norm2_b, "norm2.bias");
    upload_compute_checked(need(f, p + "conv2.weight"), C_out, C_out * 3 * 3, r.conv2_W, "conv2.weight");
    upload_compute_checked(need(f, p + "conv2.bias"),   C_out, 1, r.conv2_b, "conv2.bias");

    r.C_in = C_in;
    r.C_out = C_out;
    r.has_shortcut = (C_in != C_out);
    if (r.has_shortcut) {
        // conv_shortcut: 1x1 conv (C_out, C_in, 1, 1) — flattened to (C_out, C_in).
        upload_compute_checked(need(f, p + "conv_shortcut.weight"),
                            C_out, C_in, r.short_W, "conv_shortcut.weight");
        upload_compute_checked(need(f, p + "conv_shortcut.bias"),
                            C_out, 1, r.short_b, "conv_shortcut.bias");
    }
}

void Decoder::load_weights(const st::File& f, const std::string& prefix) {
    g_vae_upload_dtype = vae_arith_dtype(cfg_.force_upcast);
    const int mid_C   = cfg_.block_out_channels.back();
    const int first_C = cfg_.block_out_channels.front();

    // post_quant_conv lives at the sibling level of "decoder." (e.g. plain
    // "post_quant_conv.{weight,bias}" for a standalone diffusers VAE export, or
    // "first_stage_model.post_quant_conv.{weight,bias}" for the SD1.5 single-
    // file checkpoint). Derive the parent prefix by stripping a trailing
    // "decoder." if present, then look up the tensors. If they're missing we
    // assume the checkpoint omits this layer (rare; not standard SD1.5).
    std::string parent = prefix;
    {
        const std::string tail = "decoder.";
        if (parent.size() >= tail.size() &&
            parent.compare(parent.size() - tail.size(), tail.size(), tail) == 0) {
            parent.erase(parent.size() - tail.size());
        }
    }
    const auto* pqw = f.find(parent + "post_quant_conv.weight");
    const auto* pqb = f.find(parent + "post_quant_conv.bias");
    has_post_quant_conv_ = (pqw != nullptr && pqb != nullptr);
    if (has_post_quant_conv_) {
        // 1x1 conv (in_channels, in_channels, 1, 1) -> 2D (in_channels, in_channels).
        upload_compute_checked(*pqw, cfg_.in_channels, cfg_.in_channels,
                            post_quant_W_, "post_quant_conv.weight");
        upload_compute_checked(*pqb, cfg_.in_channels, 1,
                            post_quant_b_, "post_quant_conv.bias");
    }

    // conv_in: (mid_C, in_channels, 3, 3) → 2D (mid_C, in_channels * 9)
    upload_compute_checked(need(f, prefix + "conv_in.weight"),
                        mid_C, cfg_.in_channels * 3 * 3, conv_in_W_, "conv_in.weight");
    upload_compute_checked(need(f, prefix + "conv_in.bias"),
                        mid_C, 1, conv_in_b_, "conv_in.bias");

    // mid_block
    load_resnet_(f, prefix + "mid_block.resnets.0.", mid_C, mid_C, mid_res0_);
    load_resnet_(f, prefix + "mid_block.resnets.1.", mid_C, mid_C, mid_res1_);

    const std::string ap = prefix + "mid_block.attentions.0.";
    upload_compute_checked(need(f, ap + "group_norm.weight"), mid_C, 1, mid_attn_.gn_g, "attn.gn.w");
    upload_compute_checked(need(f, ap + "group_norm.bias"),   mid_C, 1, mid_attn_.gn_b, "attn.gn.b");
    // Diffusers >=0.20 renamed the VAE mid-block attention tensors to match
    // the BasicTransformerBlock naming (to_q/to_k/to_v/to_out.0). The older
    // query/key/value/proj_attn names live on in legacy checkpoints — accept
    // either.
    auto need_alt = [&](const char* primary, const char* legacy) -> const st::TensorView& {
        const auto* v = f.find(ap + primary);
        if (v) return *v;
        v = f.find(ap + legacy);
        if (v) return *v;
        fail("missing attention tensor '" + ap + primary + "' (also tried '" + ap + legacy + "')");
    };
    upload_compute_checked(need_alt("to_q.weight",     "query.weight"),     mid_C, mid_C, mid_attn_.Wq, "attn.q.w");
    upload_compute_checked(need_alt("to_q.bias",       "query.bias"),       mid_C, 1, mid_attn_.bq, "attn.q.b");
    upload_compute_checked(need_alt("to_k.weight",     "key.weight"),       mid_C, mid_C, mid_attn_.Wk, "attn.k.w");
    upload_compute_checked(need_alt("to_k.bias",       "key.bias"),         mid_C, 1, mid_attn_.bk, "attn.k.b");
    upload_compute_checked(need_alt("to_v.weight",     "value.weight"),     mid_C, mid_C, mid_attn_.Wv, "attn.v.w");
    upload_compute_checked(need_alt("to_v.bias",       "value.bias"),       mid_C, 1, mid_attn_.bv, "attn.v.b");
    upload_compute_checked(need_alt("to_out.0.weight", "proj_attn.weight"), mid_C, mid_C, mid_attn_.Wo, "attn.o.w");
    upload_compute_checked(need_alt("to_out.0.bias",   "proj_attn.bias"),   mid_C, 1, mid_attn_.bo, "attn.o.b");
    mid_attn_.C = mid_C;

    // up_blocks. Channel order is reversed: up_blocks[0] takes block_out_channels.back().
    int C_prev = mid_C;
    const int nb = static_cast<int>(cfg_.block_out_channels.size());
    for (int i = 0; i < nb; ++i) {
        // HF stores up_blocks[i] keyed by the forward (block_out_channels) order
        // reversed at construction. i=0 corresponds to block_out_channels[-1].
        int C_block = cfg_.block_out_channels[static_cast<std::size_t>(nb - 1 - i)];
        UpBlock& ub = up_blocks_[static_cast<std::size_t>(i)];
        ub.resnets.clear();
        ub.resnets.resize(static_cast<std::size_t>(cfg_.layers_per_block + 1));
        ub.C_out = C_block;
        for (int j = 0; j <= cfg_.layers_per_block; ++j) {
            const int Ci = (j == 0) ? C_prev : C_block;
            const std::string rp = prefix + "up_blocks." + std::to_string(i) +
                                   ".resnets." + std::to_string(j) + ".";
            load_resnet_(f, rp, Ci, C_block,
                         ub.resnets[static_cast<std::size_t>(j)]);
        }
        ub.has_upsampler = (i + 1 < nb);
        if (ub.has_upsampler) {
            const std::string up = prefix + "up_blocks." + std::to_string(i) + ".upsamplers.0.conv.";
            upload_compute_checked(need(f, up + "weight"),
                                C_block, C_block * 3 * 3, ub.upsampler.W,
                                "upsampler.conv.weight");
            upload_compute_checked(need(f, up + "bias"),
                                C_block, 1, ub.upsampler.b,
                                "upsampler.conv.bias");
        }
        C_prev = C_block;
    }

    upload_compute_checked(need(f, prefix + "conv_norm_out.weight"),
                        first_C, 1, norm_out_g_, "conv_norm_out.weight");
    upload_compute_checked(need(f, prefix + "conv_norm_out.bias"),
                        first_C, 1, norm_out_b_, "conv_norm_out.bias");

    // conv_out: (out_channels, first_C, 3, 3) → (out_channels, first_C * 9)
    upload_compute_checked(need(f, prefix + "conv_out.weight"),
                        cfg_.out_channels, first_C * 3 * 3, conv_out_W_, "conv_out.weight");
    upload_compute_checked(need(f, prefix + "conv_out.bias"),
                        cfg_.out_channels, 1, conv_out_b_, "conv_out.bias");
}

// ─── per-block forward helpers ─────────────────────────────────────────────

void Decoder::apply_resnet_(const Resnet& r, int H, int W,
                            bt::Tensor& x, bt::Tensor& tmp) {
    const bt::Tensor* skip_W = r.has_shortcut ? &r.short_W : nullptr;
    const bt::Tensor* skip_b = r.has_shortcut ? &r.short_b : nullptr;
    bt::resblock_forward(x,
                             r.norm1_g, r.norm1_b,
                             r.conv1_W, &r.conv1_b,
                             /*t_emb_shift=*/nullptr,
                             r.norm2_g, r.norm2_b,
                             r.conv2_W, &r.conv2_b,
                             skip_W, skip_b,
                             /*N=*/1, r.C_in, r.C_out, H, W,
                             cfg_.norm_num_groups, cfg_.eps,
                             tmp);
    std::swap(x, tmp);
}

void Decoder::apply_attention_(const Attention& a, int H, int W,
                               bt::Tensor& x) {
    // GroupNorm in NCHW.
    bt::group_norm_forward(x, a.gn_g, a.gn_b,
                               1, a.C, H, W, cfg_.norm_num_groups, cfg_.eps,
                               ln_nchw_);
    // NCHW → (H*W, C) sequence.
    bt::nchw_to_sequence(ln_nchw_, 1, a.C, H, W, seq_);
    // Fused Q/K/V/O projections + non-causal single-head self-attention.
    bt::flash_attention_qkvo_forward(
        seq_, /*Ctx=*/nullptr,
        a.Wq, &a.bq, a.Wk, &a.bk, a.Wv, &a.bv, a.Wo, &a.bo,
        /*d_mask=*/nullptr, cfg_.num_attention_heads, /*causal=*/false,
        proj_seq_);
    // Sequence → NCHW.
    bt::sequence_to_nchw(proj_seq_, 1, a.C, H, W, attn_nchw_);
    // Residual add.
    bt::add_inplace(x, attn_nchw_);
}

void Decoder::apply_upsample_(const UpsampleConv& u, int C, int H, int W,
                              bt::Tensor& x, bt::Tensor& tmp) {
    // 2x nearest-neighbour upsample, then 3x3 same-channel conv.
    bt::upsample_nearest_2x(x, 1, C, H, W, tmp);
    bt::conv2d_forward(tmp, u.W, &u.b,
                           /*N=*/1, /*C_in=*/C, /*H=*/2 * H, /*W=*/2 * W,
                           /*C_out=*/C, /*kH=*/3, /*kW=*/3,
                           /*stride=*/1, 1, /*pad=*/1, 1, /*dil=*/1, 1,
                           x);
}

void Decoder::apply_conv3x3_(const bt::Tensor& W, const bt::Tensor& b,
                             int C_in, int C_out, int H, int W_,
                             bt::Tensor& x_in, bt::Tensor& x_out) {
    bt::conv2d_forward(x_in, W, &b,
                           /*N=*/1, C_in, H, W_,
                           C_out, /*kH=*/3, /*kW=*/3,
                           1, 1, /*pad=*/1, 1, 1, 1,
                           x_out);
}

// ─── decode ────────────────────────────────────────────────────────────────

void Decoder::decode(const bt::Tensor& latent,
                     int H_lat, int W_lat,
                     bt::Tensor& out) {
    if (conv_in_W_.size() == 0) fail("decode: weights not loaded");
    if (H_lat <= 0 || W_lat <= 0) fail("decode: H_lat and W_lat must be positive");
    if (latent.rows != 1 || latent.cols != cfg_.in_channels * H_lat * W_lat) {
        fail("decode: latent must be (1, in_channels*H_lat*W_lat)");
    }

    const int first_C = cfg_.block_out_channels.front();
    const int mid_C   = cfg_.block_out_channels.back();

    // 1. Scale (latent /= scaling_factor), then optional shift. Under
    // force_upcast the decoder runs FP32 while the pipeline latent is FP16 —
    // cast at the boundary (decode output simply stays FP32; the pipeline's
    // host download handles either dtype).
    const bt::Dtype adt = vae_arith_dtype(cfg_.force_upcast);
    if (latent.dtype != adt) {
        bt::cast(latent, x_, adt);
    } else {
        x_ = latent.clone();
    }
    if (cfg_.scaling_factor != 1.0f) {
        bt::scale_inplace(x_, 1.0f / cfg_.scaling_factor);
    }
    if (cfg_.shift_factor != 0.0f) {
        bt::add_scalar_inplace(x_, cfg_.shift_factor);
    }

    // 1b. post_quant_conv: 1x1 conv (in_channels -> in_channels). Applied by
    // AutoencoderKL.decode() in diffusers BEFORE the decoder module proper. The
    // weights ship in the same safetensors file (loaded with prefix above the
    // "decoder." subtree), and skipping it produces images that match the
    // decoder output bit-for-bit but render with severe block/color artifacts.
    if (has_post_quant_conv_) {
        bt::conv2d_forward(x_, post_quant_W_, &post_quant_b_,
                               /*N=*/1, cfg_.in_channels, H_lat, W_lat,
                               cfg_.in_channels, /*kH=*/1, /*kW=*/1,
                               1, 1, /*pad=*/0, 0, 1, 1,
                               y_);
        std::swap(x_, y_);
    }

    // 2. conv_in: in_channels -> mid_C.
    apply_conv3x3_(conv_in_W_, conv_in_b_,
                   cfg_.in_channels, mid_C, H_lat, W_lat,
                   x_, y_);
    std::swap(x_, y_);

    // 3. mid_block: resnet -> attention -> resnet.
    apply_resnet_(mid_res0_, H_lat, W_lat, x_, y_);
    apply_attention_(mid_attn_, H_lat, W_lat, x_);
    apply_resnet_(mid_res1_, H_lat, W_lat, x_, y_);

    // 4. up_blocks.
    int H = H_lat, W = W_lat;
    for (auto& ub : up_blocks_) {
        for (auto& r : ub.resnets) {
            apply_resnet_(r, H, W, x_, y_);
        }
        if (ub.has_upsampler) {
            apply_upsample_(ub.upsampler, ub.C_out, H, W, x_, y_);
            H *= 2;
            W *= 2;
        }
    }

    // 5. conv_norm_out + SiLU.
    bt::group_norm_forward(x_, norm_out_g_, norm_out_b_,
                               1, first_C, H, W, cfg_.norm_num_groups, cfg_.eps,
                               y_);
    bt::silu_forward(y_, y_);

    // 6. conv_out: first_C -> out_channels.
    apply_conv3x3_(conv_out_W_, conv_out_b_,
                   first_C, cfg_.out_channels, H, W,
                   y_, out);
}

// ─── Encoder ───────────────────────────────────────────────────────────────

Encoder::Encoder(const EncoderConfig& cfg) : cfg_(cfg) {
    if (cfg_.block_out_channels.empty()) fail_enc("block_out_channels must be non-empty");
    if (cfg_.layers_per_block <= 0) fail_enc("layers_per_block must be positive");
    for (int c : cfg_.block_out_channels) {
        if (c <= 0 || c % cfg_.norm_num_groups != 0) {
            fail_enc("each block_out_channels entry must be a positive multiple of norm_num_groups");
        }
    }
    down_blocks_.resize(cfg_.block_out_channels.size());
}

Encoder::~Encoder() = default;

void Encoder::load_resnet_(const st::File& f, const std::string& p,
                           int C_in, int C_out, Resnet& r) {
    upload_compute_checked_enc(need_enc(f, p + "norm1.weight"), C_in, 1, r.norm1_g, "norm1.weight");
    upload_compute_checked_enc(need_enc(f, p + "norm1.bias"),   C_in, 1, r.norm1_b, "norm1.bias");
    upload_compute_checked_enc(need_enc(f, p + "conv1.weight"), C_out, C_in * 3 * 3, r.conv1_W, "conv1.weight");
    upload_compute_checked_enc(need_enc(f, p + "conv1.bias"),   C_out, 1, r.conv1_b, "conv1.bias");
    upload_compute_checked_enc(need_enc(f, p + "norm2.weight"), C_out, 1, r.norm2_g, "norm2.weight");
    upload_compute_checked_enc(need_enc(f, p + "norm2.bias"),   C_out, 1, r.norm2_b, "norm2.bias");
    upload_compute_checked_enc(need_enc(f, p + "conv2.weight"), C_out, C_out * 3 * 3, r.conv2_W, "conv2.weight");
    upload_compute_checked_enc(need_enc(f, p + "conv2.bias"),   C_out, 1, r.conv2_b, "conv2.bias");

    r.C_in = C_in;
    r.C_out = C_out;
    r.has_shortcut = (C_in != C_out);
    if (r.has_shortcut) {
        upload_compute_checked_enc(need_enc(f, p + "conv_shortcut.weight"),
                                   C_out, C_in, r.short_W, "conv_shortcut.weight");
        upload_compute_checked_enc(need_enc(f, p + "conv_shortcut.bias"),
                                   C_out, 1, r.short_b, "conv_shortcut.bias");
    }
}

void Encoder::load_weights(const st::File& f, const std::string& prefix) {
    g_vae_upload_dtype = vae_arith_dtype(cfg_.force_upcast);
    const int first_C = cfg_.block_out_channels.front();
    const int mid_C   = cfg_.block_out_channels.back();
    const int twoC_lat = 2 * cfg_.in_channels;

    // quant_conv sits above the "encoder." subtree (sibling of post_quant_conv).
    std::string parent = prefix;
    {
        const std::string tail = "encoder.";
        if (parent.size() >= tail.size() &&
            parent.compare(parent.size() - tail.size(), tail.size(), tail) == 0) {
            parent.erase(parent.size() - tail.size());
        }
    }
    const auto* qw = f.find(parent + "quant_conv.weight");
    const auto* qb = f.find(parent + "quant_conv.bias");
    has_quant_conv_ = (qw != nullptr && qb != nullptr);
    if (has_quant_conv_) {
        upload_compute_checked_enc(*qw, twoC_lat, twoC_lat, quant_W_, "quant_conv.weight");
        upload_compute_checked_enc(*qb, twoC_lat, 1, quant_b_, "quant_conv.bias");
    }

    // conv_in: (first_C, out_channels, 3, 3) -> 2D (first_C, out_channels * 9)
    upload_compute_checked_enc(need_enc(f, prefix + "conv_in.weight"),
                               first_C, cfg_.out_channels * 3 * 3,
                               conv_in_W_, "conv_in.weight");
    upload_compute_checked_enc(need_enc(f, prefix + "conv_in.bias"),
                               first_C, 1, conv_in_b_, "conv_in.bias");

    // down_blocks
    int C_prev = first_C;
    const int nb = static_cast<int>(cfg_.block_out_channels.size());
    for (int i = 0; i < nb; ++i) {
        const int C_block = cfg_.block_out_channels[static_cast<std::size_t>(i)];
        DownBlock& db = down_blocks_[static_cast<std::size_t>(i)];
        db.resnets.clear();
        db.resnets.resize(static_cast<std::size_t>(cfg_.layers_per_block));
        db.C_out = C_block;
        for (int j = 0; j < cfg_.layers_per_block; ++j) {
            const int Ci = (j == 0) ? C_prev : C_block;
            const std::string rp = prefix + "down_blocks." + std::to_string(i) +
                                   ".resnets." + std::to_string(j) + ".";
            load_resnet_(f, rp, Ci, C_block,
                         db.resnets[static_cast<std::size_t>(j)]);
        }
        db.has_downsampler = (i + 1 < nb);
        if (db.has_downsampler) {
            const std::string dp = prefix + "down_blocks." + std::to_string(i) +
                                   ".downsamplers.0.conv.";
            upload_compute_checked_enc(need_enc(f, dp + "weight"),
                                       C_block, C_block * 3 * 3, db.downsampler.W,
                                       "downsampler.conv.weight");
            upload_compute_checked_enc(need_enc(f, dp + "bias"),
                                       C_block, 1, db.downsampler.b,
                                       "downsampler.conv.bias");
        }
        C_prev = C_block;
    }

    // mid_block
    load_resnet_(f, prefix + "mid_block.resnets.0.", mid_C, mid_C, mid_res0_);
    load_resnet_(f, prefix + "mid_block.resnets.1.", mid_C, mid_C, mid_res1_);

    const std::string ap = prefix + "mid_block.attentions.0.";
    upload_compute_checked_enc(need_enc(f, ap + "group_norm.weight"), mid_C, 1, mid_attn_.gn_g, "attn.gn.w");
    upload_compute_checked_enc(need_enc(f, ap + "group_norm.bias"),   mid_C, 1, mid_attn_.gn_b, "attn.gn.b");
    auto need_alt = [&](const char* primary, const char* legacy) -> const st::TensorView& {
        const auto* v = f.find(ap + primary);
        if (v) return *v;
        v = f.find(ap + legacy);
        if (v) return *v;
        fail_enc("missing attention tensor '" + ap + primary + "' (also tried '" + ap + legacy + "')");
    };
    upload_compute_checked_enc(need_alt("to_q.weight",     "query.weight"),     mid_C, mid_C, mid_attn_.Wq, "attn.q.w");
    upload_compute_checked_enc(need_alt("to_q.bias",       "query.bias"),       mid_C, 1, mid_attn_.bq, "attn.q.b");
    upload_compute_checked_enc(need_alt("to_k.weight",     "key.weight"),       mid_C, mid_C, mid_attn_.Wk, "attn.k.w");
    upload_compute_checked_enc(need_alt("to_k.bias",       "key.bias"),         mid_C, 1, mid_attn_.bk, "attn.k.b");
    upload_compute_checked_enc(need_alt("to_v.weight",     "value.weight"),     mid_C, mid_C, mid_attn_.Wv, "attn.v.w");
    upload_compute_checked_enc(need_alt("to_v.bias",       "value.bias"),       mid_C, 1, mid_attn_.bv, "attn.v.b");
    upload_compute_checked_enc(need_alt("to_out.0.weight", "proj_attn.weight"), mid_C, mid_C, mid_attn_.Wo, "attn.o.w");
    upload_compute_checked_enc(need_alt("to_out.0.bias",   "proj_attn.bias"),   mid_C, 1, mid_attn_.bo, "attn.o.b");
    mid_attn_.C = mid_C;

    upload_compute_checked_enc(need_enc(f, prefix + "conv_norm_out.weight"),
                               mid_C, 1, norm_out_g_, "conv_norm_out.weight");
    upload_compute_checked_enc(need_enc(f, prefix + "conv_norm_out.bias"),
                               mid_C, 1, norm_out_b_, "conv_norm_out.bias");

    // conv_out: (2*in_channels, mid_C, 3, 3) -> 2D
    upload_compute_checked_enc(need_enc(f, prefix + "conv_out.weight"),
                               twoC_lat, mid_C * 3 * 3, conv_out_W_, "conv_out.weight");
    upload_compute_checked_enc(need_enc(f, prefix + "conv_out.bias"),
                               twoC_lat, 1, conv_out_b_, "conv_out.bias");
}

void Encoder::apply_resnet_(const Resnet& r, int H, int W,
                            bt::Tensor& x, bt::Tensor& tmp) {
    const bt::Tensor* skip_W = r.has_shortcut ? &r.short_W : nullptr;
    const bt::Tensor* skip_b = r.has_shortcut ? &r.short_b : nullptr;
    bt::resblock_forward(x,
                         r.norm1_g, r.norm1_b,
                         r.conv1_W, &r.conv1_b,
                         /*t_emb_shift=*/nullptr,
                         r.norm2_g, r.norm2_b,
                         r.conv2_W, &r.conv2_b,
                         skip_W, skip_b,
                         /*N=*/1, r.C_in, r.C_out, H, W,
                         cfg_.norm_num_groups, cfg_.eps,
                         tmp);
    std::swap(x, tmp);
}

void Encoder::apply_attention_(const Attention& a, int H, int W,
                               bt::Tensor& x) {
    bt::group_norm_forward(x, a.gn_g, a.gn_b,
                           1, a.C, H, W, cfg_.norm_num_groups, cfg_.eps,
                           ln_nchw_);
    bt::nchw_to_sequence(ln_nchw_, 1, a.C, H, W, seq_);
    bt::flash_attention_qkvo_forward(
        seq_, /*Ctx=*/nullptr,
        a.Wq, &a.bq, a.Wk, &a.bk, a.Wv, &a.bv, a.Wo, &a.bo,
        /*d_mask=*/nullptr, cfg_.num_attention_heads, /*causal=*/false,
        proj_seq_);
    bt::sequence_to_nchw(proj_seq_, 1, a.C, H, W, attn_nchw_);
    bt::add_inplace(x, attn_nchw_);
}

void Encoder::apply_downsample_(const DownsampleConv& d, int C, int H, int W,
                                bt::Tensor& x, bt::Tensor& tmp) {
    // Diffusers Downsample2D: F.pad(x, (0, 1, 0, 1)) then conv stride=2 pad=0.
    // pad2d_forward(... pad_top, pad_bottom, pad_left, pad_right, mode=0/zero)
    bt::pad2d_forward(x, /*N=*/1, C, H, W,
                      /*pad_top=*/0, /*pad_bottom=*/1,
                      /*pad_left=*/0, /*pad_right=*/1,
                      /*mode=*/0, pad_);
    bt::conv2d_forward(pad_, d.W, &d.b,
                       /*N=*/1, /*C_in=*/C, /*H=*/H + 1, /*W=*/W + 1,
                       /*C_out=*/C, /*kH=*/3, /*kW=*/3,
                       /*stride=*/2, 2, /*pad=*/0, 0, /*dil=*/1, 1,
                       tmp);
    std::swap(x, tmp);
}

void Encoder::apply_conv3x3_(const bt::Tensor& W, const bt::Tensor& b,
                             int C_in, int C_out, int H, int W_,
                             bt::Tensor& x_in, bt::Tensor& x_out) {
    bt::conv2d_forward(x_in, W, &b,
                       /*N=*/1, C_in, H, W_,
                       C_out, /*kH=*/3, /*kW=*/3,
                       1, 1, /*pad=*/1, 1, 1, 1,
                       x_out);
}

void Encoder::encode(const bt::Tensor& image, int H, int W,
                     const bt::Tensor* eps, bt::Tensor& out) {
    if (conv_in_W_.size() == 0) fail_enc("encode: weights not loaded");
    if (H <= 0 || W <= 0) fail_enc("encode: H and W must be positive");
    if (H % 8 != 0 || W % 8 != 0) fail_enc("encode: H and W must be multiples of 8");
    if (image.rows != 1 || image.cols != cfg_.out_channels * H * W) {
        fail_enc("encode: image must be (1, out_channels*H*W)");
    }

    const int first_C = cfg_.block_out_channels.front();
    const int mid_C   = cfg_.block_out_channels.back();
    const int H_lat   = H / 8;
    const int W_lat   = W / 8;
    const int spatial_lat = H_lat * W_lat;
    const int C_lat   = cfg_.in_channels;

    // 1. conv_in: out_channels (3) -> first_C. Under force_upcast the
    // encoder runs FP32 while the caller's image is at the pipeline dtype —
    // cast at the boundary (and back at the end).
    const bt::Dtype enc_adt = vae_arith_dtype(cfg_.force_upcast);
    if (image.dtype != enc_adt) {
        bt::cast(image, x_, enc_adt);
    } else {
        x_ = image.clone();
    }
    apply_conv3x3_(conv_in_W_, conv_in_b_,
                   cfg_.out_channels, first_C, H, W,
                   x_, y_);
    std::swap(x_, y_);

    // 2. down_blocks.
    int Hc = H, Wc = W;
    for (auto& db : down_blocks_) {
        for (auto& r : db.resnets) {
            apply_resnet_(r, Hc, Wc, x_, y_);
        }
        if (db.has_downsampler) {
            apply_downsample_(db.downsampler, db.C_out, Hc, Wc, x_, y_);
            Hc /= 2;
            Wc /= 2;
        }
    }

    // 3. mid_block: resnet -> attention -> resnet.
    apply_resnet_(mid_res0_, Hc, Wc, x_, y_);
    apply_attention_(mid_attn_, Hc, Wc, x_);
    apply_resnet_(mid_res1_, Hc, Wc, x_, y_);

    // 4. conv_norm_out + SiLU.
    bt::group_norm_forward(x_, norm_out_g_, norm_out_b_,
                           1, mid_C, Hc, Wc, cfg_.norm_num_groups, cfg_.eps,
                           y_);
    bt::silu_forward(y_, y_);

    // 5. conv_out: mid_C -> 2 * in_channels. Lands in moments_ if quant_conv
    //    follows, else directly into a temp for split.
    apply_conv3x3_(conv_out_W_, conv_out_b_,
                   mid_C, 2 * C_lat, Hc, Wc,
                   y_, x_);

    // 6. quant_conv (1x1 conv).
    if (has_quant_conv_) {
        bt::conv2d_forward(x_, quant_W_, &quant_b_,
                           /*N=*/1, /*C_in=*/2 * C_lat, H_lat, W_lat,
                           /*C_out=*/2 * C_lat, /*kH=*/1, /*kW=*/1,
                           1, 1, /*pad=*/0, 0, 1, 1,
                           moments_);
    } else {
        // Just alias via swap so the rest of the path reads from moments_.
        std::swap(x_, moments_);
    }

    // 7. Split (mean, logvar) along channel dim and reparameterize. The flat
    //    layout is NCHW with N=1, so channels are the slowest-varying axis:
    //    mean   = moments[0 .. C_lat*spatial_lat)
    //    logvar = moments[C_lat*spatial_lat .. 2*C_lat*spatial_lat)
    const int half = C_lat * spatial_lat;
    detail::resize_like(out, 1, half, compute_dtype(), moments_.device);
    // Copy mean -> out.
    bt::copy_d2d(moments_, /*src_off=*/0, out, /*dst_off=*/0, half);

    if (eps != nullptr) {
        if (eps->rows != 1 || eps->cols != half) {
            fail_enc("encode: eps must be (1, in_channels*H/8*W/8)");
        }
        if (eps->dtype != out.dtype) fail_enc("encode: eps dtype must match output");

        // Stage logvar in logvar_, compute exp(0.5 * logvar) host-side
        // (one-off per encode, not perf-critical), upload back, multiply
        // elementwise by eps, then add to out.
        detail::resize_like(logvar_, 1, half, compute_dtype(), moments_.device);
        bt::copy_d2d(moments_, /*src_off=*/half, logvar_, /*dst_off=*/0, half);

        // Host-side exp(0.5*lv). Use bd-style download then upload at compute
        // dtype.
        std::vector<float> lv_host;
        bt::sync_all();
        if (logvar_.dtype == bt::Dtype::FP16) {
            std::vector<uint16_t> bits = logvar_.to_host_vector_fp16();
            lv_host.resize(bits.size());
            for (std::size_t i = 0; i < bits.size(); ++i) {
                lv_host[i] = std::exp(0.5f * bt::fp16_bits_to_fp32(bits[i]));
            }
        } else {
            lv_host = logvar_.to_host_vector();
            for (float& v : lv_host) v = std::exp(0.5f * v);
        }
        bt::Tensor std_dev = detail::upload_host(lv_host.data(), 1, half);
        if (std_dev.dtype != out.dtype) {
            bt::Tensor cast_buf;
            bt::cast(std_dev, cast_buf, out.dtype);
            std_dev = std::move(cast_buf);
        }
        // std_dev *= eps  (mul elementwise)
        bt::mul_inplace(std_dev, *eps);
        bt::add_inplace(out, std_dev);
    }

    // 8. out = (sample - shift_factor) * scaling_factor.
    if (cfg_.shift_factor != 0.0f) {
        bt::add_scalar_inplace(out, -cfg_.shift_factor);
    }
    if (cfg_.scaling_factor != 1.0f) {
        bt::scale_inplace(out, cfg_.scaling_factor);
    }

    // Hand the latent back at the pipeline compute dtype: the scheduler and
    // denoiser operate there, not at the VAE's (possibly upcast) dtype.
    const bt::Dtype pdt = brodiffusion::compute_dtype();
    if (out.dtype != pdt) {
        bt::Tensor cast_buf;
        bt::cast(out, cast_buf, pdt);
        out = std::move(cast_buf);
    }
}

}  // namespace brodiffusion::vae
