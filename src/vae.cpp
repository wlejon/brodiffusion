#include "brodiffusion/vae.h"
#include "brotensor/safetensors.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

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

const st::TensorView& need(const st::File& f, const std::string& key) {
    const auto* v = f.find(key);
    if (!v) throw std::runtime_error("vae::Decoder: missing tensor '" + key + "'");
    return *v;
}

// Accepts F16 (used as-is) or F32 (converted host-side); SD1.5 full
// checkpoints ship as F32. Uploads at the pipeline compute dtype.
void upload_compute_checked(const st::TensorView& v, int rows, int cols,
                            bt::Tensor& dst, const char* name) {
    if (v.dtype != st::Dtype::F16 && v.dtype != st::Dtype::F32) {
        fail(std::string(name) + ": expected F16 or F32, got " + st::dtype_name(v.dtype));
    }
    int64_t expected = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    if (v.numel() != expected) {
        fail(std::string(name) + ": shape mismatch (expected " +
             std::to_string(rows) + "x" + std::to_string(cols) + ", got " +
             std::to_string(v.numel()) + ")");
    }
    st::upload_compute(v, rows, cols, dst);
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

    // 1. Scale (latent /= scaling_factor).
    x_ = latent.clone();
    if (cfg_.scaling_factor != 1.0f) {
        bt::scale_inplace(x_, 1.0f / cfg_.scaling_factor);
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

}  // namespace brodiffusion::vae
