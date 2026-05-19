#include "brodiffusion/ddpm_unet.h"
#include "brodiffusion/safetensors.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::ddpm_unet {

namespace bt = ::brotensor;
namespace st = ::brodiffusion::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("ddpm_unet::UNet: " + msg);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    const auto* v = f.find(key);
    if (!v) fail("missing tensor '" + key + "'");
    return *v;
}

void upload_fp16_checked(const st::TensorView& v, int rows, int cols,
                         bt::GpuTensor& dst, const char* name) {
    if (v.dtype != st::Dtype::F16 && v.dtype != st::Dtype::F32) {
        fail(std::string(name) + ": expected F16 or F32, got " + st::dtype_name(v.dtype));
    }
    const int64_t expected = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    if (v.numel() != expected) {
        fail(std::string(name) + ": shape mismatch (expected " +
             std::to_string(rows) + "x" + std::to_string(cols) + ", got " +
             std::to_string(v.numel()) + " elements)");
    }
    st::upload_fp16(v, rows, cols, dst);
}

// Ho-style sinusoidal time embedding (diffusers `get_timestep_embedding`
// with flip_sin_to_cos=False, downscale_freq_shift=1, max_period=10000).
// Output layout: [sin(t*freq_0..freq_{H-1}), cos(t*freq_0..freq_{H-1})].
// Exponent divisor is (half - 1), NOT half — this is the key DDPM/SD diff.
void compute_ddpm_time_emb_fp16(float t, int dim,
                                std::vector<uint16_t>& out) {
    const int half = dim / 2;
    out.resize(static_cast<std::size_t>(dim));
    const float log_period = std::log(10000.0f);
    const float denom = (half > 1) ? static_cast<float>(half - 1) : 1.0f;
    for (int i = 0; i < half; ++i) {
        const float freq = std::exp(-log_period * static_cast<float>(i) / denom);
        const float angle = t * freq;
        out[static_cast<std::size_t>(i)]        = bt::fp32_to_fp16_bits(std::sin(angle));
        out[static_cast<std::size_t>(i + half)] = bt::fp32_to_fp16_bits(std::cos(angle));
    }
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

UNet::UNet(const UNetConfig& cfg) : cfg_(cfg) {
    const int nb = static_cast<int>(cfg_.block_out_channels.size());
    if (nb < 2) fail("block_out_channels must have at least 2 entries");
    if (cfg_.layers_per_block <= 0) fail("layers_per_block must be positive");
    if (static_cast<int>(cfg_.down_block_types.size()) != nb) {
        fail("down_block_types must have one entry per block_out_channels");
    }
    if (static_cast<int>(cfg_.up_block_types.size()) != nb) {
        fail("up_block_types must have one entry per block_out_channels");
    }
    for (int c : cfg_.block_out_channels) {
        if (c <= 0 || c % cfg_.norm_num_groups != 0) {
            fail("each block_out_channels entry must be a positive multiple "
                 "of norm_num_groups");
        }
    }
    if (cfg_.downsample_padding != 0 && cfg_.downsample_padding != 1) {
        fail("downsample_padding must be 0 (asymmetric prepad) or 1 (symmetric)");
    }
    if (cfg_.attention_num_heads <= 0) fail("attention_num_heads must be positive");

    freq_dim_       = cfg_.block_out_channels.front();
    if (freq_dim_ % 2 != 0) fail("block_out_channels[0] must be even");
    time_embed_dim_ = freq_dim_ * cfg_.time_embed_dim_mult;

    // Down blocks: type-determined attention, all but last block downsample.
    down_blocks_.resize(static_cast<std::size_t>(nb));
    for (int i = 0; i < nb; ++i) {
        DownBlock& d = down_blocks_[static_cast<std::size_t>(i)];
        d.has_attention   = (cfg_.down_block_types[static_cast<std::size_t>(i)]
                             == DownBlockType::AttnDown);
        d.has_downsampler = (i < nb - 1);
        d.C_out = cfg_.block_out_channels[static_cast<std::size_t>(i)];
    }

    // Up blocks index 0 is the side closest to the mid-block (highest channel
    // count, lowest spatial). Diffusers stores up_block_types in that order
    // already (first entry consumes mid-block output).
    up_blocks_.resize(static_cast<std::size_t>(nb));
    for (int i = 0; i < nb; ++i) {
        UpBlock& u = up_blocks_[static_cast<std::size_t>(i)];
        u.has_attention = (cfg_.up_block_types[static_cast<std::size_t>(i)]
                           == UpBlockType::AttnUp);
        u.has_upsampler = (i < nb - 1);
        u.C_out = cfg_.block_out_channels[static_cast<std::size_t>(nb - 1 - i)];
    }
}

UNet::~UNet() = default;

// ─── weight loading ────────────────────────────────────────────────────────

void UNet::load_resnet_(const st::File& f, const std::string& p,
                        int C_in, int C_out, Resnet& r) {
    upload_fp16_checked(need(f, p + "norm1.weight"), C_in,  1, r.n1g, "resnet.norm1.weight");
    upload_fp16_checked(need(f, p + "norm1.bias"),   C_in,  1, r.n1b, "resnet.norm1.bias");
    upload_fp16_checked(need(f, p + "conv1.weight"), C_out, C_in * 3 * 3, r.W1, "resnet.conv1.weight");
    upload_fp16_checked(need(f, p + "conv1.bias"),   C_out, 1, r.b1, "resnet.conv1.bias");

    upload_fp16_checked(need(f, p + "time_emb_proj.weight"),
                        C_out, time_embed_dim_, r.temb_W, "resnet.time_emb_proj.weight");
    upload_fp16_checked(need(f, p + "time_emb_proj.bias"),
                        C_out, 1, r.temb_b, "resnet.time_emb_proj.bias");

    upload_fp16_checked(need(f, p + "norm2.weight"), C_out, 1, r.n2g, "resnet.norm2.weight");
    upload_fp16_checked(need(f, p + "norm2.bias"),   C_out, 1, r.n2b, "resnet.norm2.bias");
    upload_fp16_checked(need(f, p + "conv2.weight"), C_out, C_out * 3 * 3, r.W2, "resnet.conv2.weight");
    upload_fp16_checked(need(f, p + "conv2.bias"),   C_out, 1, r.b2, "resnet.conv2.bias");

    r.C_in = C_in;
    r.C_out = C_out;
    r.has_shortcut = (C_in != C_out);
    if (r.has_shortcut) {
        // 1x1 conv: stored as (C_out, C_in, 1, 1) → flatten to (C_out, C_in).
        upload_fp16_checked(need(f, p + "conv_shortcut.weight"),
                            C_out, C_in, r.Ws, "resnet.conv_shortcut.weight");
        upload_fp16_checked(need(f, p + "conv_shortcut.bias"),
                            C_out, 1, r.bs, "resnet.conv_shortcut.bias");
    }
}

void UNet::load_attn_(const st::File& f, const std::string& p, int C, AttnBlock& a) {
    a.C = C;
    upload_fp16_checked(need(f, p + "group_norm.weight"), C, 1, a.gn_g, "attn.group_norm.weight");
    upload_fp16_checked(need(f, p + "group_norm.bias"),   C, 1, a.gn_b, "attn.group_norm.bias");
    upload_fp16_checked(need(f, p + "query.weight"), C, C, a.Wq, "attn.query.weight");
    upload_fp16_checked(need(f, p + "query.bias"),   C, 1, a.bq, "attn.query.bias");
    upload_fp16_checked(need(f, p + "key.weight"),   C, C, a.Wk, "attn.key.weight");
    upload_fp16_checked(need(f, p + "key.bias"),     C, 1, a.bk, "attn.key.bias");
    upload_fp16_checked(need(f, p + "value.weight"), C, C, a.Wv, "attn.value.weight");
    upload_fp16_checked(need(f, p + "value.bias"),   C, 1, a.bv, "attn.value.bias");
    upload_fp16_checked(need(f, p + "proj_attn.weight"), C, C, a.Wo, "attn.proj_attn.weight");
    upload_fp16_checked(need(f, p + "proj_attn.bias"),   C, 1, a.bo, "attn.proj_attn.bias");
}

void UNet::load_weights(const st::File& f, const std::string& prefix) {
    const int nb       = static_cast<int>(cfg_.block_out_channels.size());
    const int first_C  = cfg_.block_out_channels.front();
    const int mid_C    = cfg_.block_out_channels.back();

    upload_fp16_checked(need(f, prefix + "conv_in.weight"),
                        first_C, cfg_.in_channels * 3 * 3, conv_in_W_, "conv_in.weight");
    upload_fp16_checked(need(f, prefix + "conv_in.bias"),
                        first_C, 1, conv_in_b_, "conv_in.bias");

    upload_fp16_checked(need(f, prefix + "time_embedding.linear_1.weight"),
                        time_embed_dim_, freq_dim_, te_l1_W_, "te.linear_1.weight");
    upload_fp16_checked(need(f, prefix + "time_embedding.linear_1.bias"),
                        time_embed_dim_, 1, te_l1_b_, "te.linear_1.bias");
    upload_fp16_checked(need(f, prefix + "time_embedding.linear_2.weight"),
                        time_embed_dim_, time_embed_dim_, te_l2_W_, "te.linear_2.weight");
    upload_fp16_checked(need(f, prefix + "time_embedding.linear_2.bias"),
                        time_embed_dim_, 1, te_l2_b_, "te.linear_2.bias");

    // ── down_blocks ────────────────────────────────────────────────────────
    int C_prev = first_C;
    for (int i = 0; i < nb; ++i) {
        DownBlock& d = down_blocks_[static_cast<std::size_t>(i)];
        const int C_out = d.C_out;

        d.resnets.clear();
        d.resnets.resize(static_cast<std::size_t>(cfg_.layers_per_block));
        d.attns.clear();
        if (d.has_attention) {
            d.attns.resize(static_cast<std::size_t>(cfg_.layers_per_block));
        }

        for (int j = 0; j < cfg_.layers_per_block; ++j) {
            const int Ci = (j == 0) ? C_prev : C_out;
            const std::string rp = prefix + "down_blocks." + std::to_string(i) +
                                   ".resnets." + std::to_string(j) + ".";
            load_resnet_(f, rp, Ci, C_out, d.resnets[static_cast<std::size_t>(j)]);

            if (d.has_attention) {
                const std::string ap = prefix + "down_blocks." + std::to_string(i) +
                                       ".attentions." + std::to_string(j) + ".";
                load_attn_(f, ap, C_out, d.attns[static_cast<std::size_t>(j)]);
            }
        }

        if (d.has_downsampler) {
            const std::string sp = prefix + "down_blocks." + std::to_string(i) +
                                   ".downsamplers.0.conv.";
            upload_fp16_checked(need(f, sp + "weight"),
                                C_out, C_out * 3 * 3, d.downsampler.W,
                                "downsampler.conv.weight");
            upload_fp16_checked(need(f, sp + "bias"),
                                C_out, 1, d.downsampler.b,
                                "downsampler.conv.bias");
        }

        C_prev = C_out;
    }

    // ── mid_block: resnet → attention → resnet ─────────────────────────────
    {
        const std::string mp = prefix + "mid_block.";
        load_resnet_(f, mp + "resnets.0.", mid_C, mid_C, mid_.r0);
        load_attn_(f, mp + "attentions.0.", mid_C, mid_.attn);
        load_resnet_(f, mp + "resnets.1.", mid_C, mid_C, mid_.r1);
    }

    // ── up_blocks ──────────────────────────────────────────────────────────
    // Walk the same skip stack the down side built, in reverse, to derive
    // per-layer skip channel counts. Matches the SD1.5 UNet loader pattern.
    std::vector<int> skip_stack;
    skip_stack.reserve(static_cast<std::size_t>(nb) *
                       static_cast<std::size_t>(cfg_.layers_per_block + 1));
    skip_stack.push_back(first_C);
    for (int i = 0; i < nb; ++i) {
        const int Cb = cfg_.block_out_channels[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg_.layers_per_block; ++j) skip_stack.push_back(Cb);
        if (down_blocks_[static_cast<std::size_t>(i)].has_downsampler) {
            skip_stack.push_back(Cb);
        }
    }

    int C_up_prev = mid_C;
    for (int i = 0; i < nb; ++i) {
        UpBlock& u = up_blocks_[static_cast<std::size_t>(i)];
        const int C_out = u.C_out;
        const int layers = cfg_.layers_per_block + 1;

        u.resnets.clear();
        u.resnets.resize(static_cast<std::size_t>(layers));
        u.attns.clear();
        if (u.has_attention) {
            u.attns.resize(static_cast<std::size_t>(layers));
        }

        for (int j = 0; j < layers; ++j) {
            if (skip_stack.empty()) fail("internal: skip stack underflow during weight load");
            const int Cskip = skip_stack.back();
            skip_stack.pop_back();
            const int C_h = (j == 0) ? C_up_prev : C_out;
            const int Ci  = C_h + Cskip;

            const std::string rp = prefix + "up_blocks." + std::to_string(i) +
                                   ".resnets." + std::to_string(j) + ".";
            load_resnet_(f, rp, Ci, C_out, u.resnets[static_cast<std::size_t>(j)]);

            if (u.has_attention) {
                const std::string ap = prefix + "up_blocks." + std::to_string(i) +
                                       ".attentions." + std::to_string(j) + ".";
                load_attn_(f, ap, C_out, u.attns[static_cast<std::size_t>(j)]);
            }
        }

        if (u.has_upsampler) {
            const std::string sp = prefix + "up_blocks." + std::to_string(i) +
                                   ".upsamplers.0.conv.";
            upload_fp16_checked(need(f, sp + "weight"),
                                C_out, C_out * 3 * 3, u.upsampler.W,
                                "upsampler.conv.weight");
            upload_fp16_checked(need(f, sp + "bias"),
                                C_out, 1, u.upsampler.b,
                                "upsampler.conv.bias");
        }

        C_up_prev = C_out;
    }

    if (!skip_stack.empty()) fail("internal: skip stack not drained during weight load");

    upload_fp16_checked(need(f, prefix + "conv_norm_out.weight"),
                        first_C, 1, norm_out_g_, "conv_norm_out.weight");
    upload_fp16_checked(need(f, prefix + "conv_norm_out.bias"),
                        first_C, 1, norm_out_b_, "conv_norm_out.bias");
    upload_fp16_checked(need(f, prefix + "conv_out.weight"),
                        cfg_.out_channels, first_C * 3 * 3, conv_out_W_, "conv_out.weight");
    upload_fp16_checked(need(f, prefix + "conv_out.bias"),
                        cfg_.out_channels, 1, conv_out_b_, "conv_out.bias");
}

// ─── forward helpers ───────────────────────────────────────────────────────

void UNet::apply_conv3x3_(const bt::GpuTensor& W, const bt::GpuTensor& b,
                          int C_in, int C_out, int H, int W_,
                          int stride, int pad,
                          const bt::GpuTensor& in, bt::GpuTensor& out) {
    bt::conv2d_forward_gpu(in, W, &b,
                           /*N=*/1, C_in, H, W_,
                           C_out, /*kH=*/3, /*kW=*/3,
                           stride, stride, pad, pad, /*dil=*/1, 1,
                           out);
}

void UNet::apply_downsample_(const SampleConv& d, int C, int H, int W,
                             bt::GpuTensor& in, bt::GpuTensor& out) {
    if (cfg_.downsample_padding == 0) {
        // Asymmetric pre-pad: (0,1,0,1) → conv with padding=0, stride=2.
        // Output spatial: (H+1)/2 if pre-padded then conv-3-stride-2-pad-0.
        // For our targets (even H), this matches diffusers exactly.
        brodiffusion::pad_right_bottom_zero_fp16(in, /*N=*/1, C, H, W, pad_buf_);
        bt::conv2d_forward_gpu(pad_buf_, d.W, &d.b,
                               /*N=*/1, C, H + 1, W + 1,
                               C, /*kH=*/3, /*kW=*/3,
                               /*stride=*/2, 2, /*pad=*/0, 0, /*dil=*/1, 1,
                               out);
    } else {
        apply_conv3x3_(d.W, d.b, C, C, H, W, /*stride=*/2, /*pad=*/1, in, out);
    }
}

void UNet::apply_resnet_(const Resnet& r, int H, int W,
                         bt::GpuTensor& x, bt::GpuTensor& tmp) {
    // 1) per-resblock temb projection: silu(temb) → Linear → (1, C_out).
    bt::linear_forward_batched_fp16_gpu(r.temb_W, &r.temb_b, temb_silu_, temb_proj_);

    // 2) Fused resblock (brotensor). Lays out exactly as DDPM ResnetBlock2D:
    //    GN(C_in) → SiLU → Conv3x3 → +temb → GN(C_out) → SiLU → Conv3x3 → +skip.
    const bt::GpuTensor* skip_W = r.has_shortcut ? &r.Ws : nullptr;
    const bt::GpuTensor* skip_b = r.has_shortcut ? &r.bs : nullptr;
    bt::resblock_forward_gpu(x,
                             r.n1g, r.n1b,
                             r.W1, &r.b1,
                             &temb_proj_,
                             r.n2g, r.n2b,
                             r.W2, &r.b2,
                             skip_W, skip_b,
                             /*N=*/1, r.C_in, r.C_out, H, W,
                             cfg_.norm_num_groups, cfg_.eps,
                             tmp);
    std::swap(x, tmp);
}

void UNet::apply_attn_(const AttnBlock& a, int H, int W, bt::GpuTensor& x) {
    const int C = a.C;
    // 1) residual is x itself; we add the post-attention output back.
    // 2) GroupNorm in NCHW. DDPM AttnBlock uses the same `norm_eps` as the
    //    surrounding model (1e-6 for google/ddpm-cifar10-32).
    bt::group_norm_forward_gpu(x, a.gn_g, a.gn_b,
                               1, C, H, W, cfg_.norm_num_groups, cfg_.eps, gn_);

    // 3) NCHW → (HW, C) token sequence.
    bt::nchw_to_sequence_gpu(gn_, 1, C, H, W, seq_);

    // 4) Q, K, V (biased linears in the DDPM AttnBlock — contrast with SD1.5
    //    Transformer2D, which uses bias-free Q/K/V).
    bt::linear_forward_batched_fp16_gpu(a.Wq, &a.bq, seq_, q_);
    bt::linear_forward_batched_fp16_gpu(a.Wk, &a.bk, seq_, k_);
    bt::linear_forward_batched_fp16_gpu(a.Wv, &a.bv, seq_, v_);

    // 5) Flash-attention. AttnBlock is unmasked, non-causal.
    bt::flash_attention_forward_gpu(q_, k_, v_,
                                    /*d_mask=*/nullptr,
                                    cfg_.attention_num_heads,
                                    /*causal=*/false,
                                    attn_out_);

    // 6) Output projection (with bias).
    bt::linear_forward_batched_fp16_gpu(a.Wo, &a.bo, attn_out_, proj_);

    // 7) seq → NCHW; residual add.
    bt::sequence_to_nchw_gpu(proj_, 1, C, H, W, proj_nchw_);
    bt::add_inplace_gpu(x, proj_nchw_);
}

// ─── forward ───────────────────────────────────────────────────────────────

void UNet::forward(const bt::GpuTensor& sample,
                   int H, int W, float timestep,
                   bt::GpuTensor& out) {
    if (conv_in_W_.size() == 0) fail("forward: weights not loaded");

    const int nb      = static_cast<int>(cfg_.block_out_channels.size());
    const int first_C = cfg_.block_out_channels.front();

    // 1) Build the time embedding (DDPM-style sinusoid, host-side).
    std::vector<uint16_t> sin_bits;
    compute_ddpm_time_emb_fp16(timestep, freq_dim_, sin_bits);
    bt::upload_fp16(sin_bits.data(), 1, freq_dim_, freq_emb_);

    bt::linear_forward_batched_fp16_gpu(te_l1_W_, &te_l1_b_, freq_emb_, temb_a_);
    bt::silu_forward_gpu(temb_a_, temb_a_);
    bt::linear_forward_batched_fp16_gpu(te_l2_W_, &te_l2_b_, temb_a_, temb_b_);
    bt::silu_forward_gpu(temb_b_, temb_silu_);

    // Skip stack: deep copies pushed at each resnet-output point + after
    // each downsampler.
    std::vector<bt::GpuTensor> skips;
    skips.reserve(static_cast<std::size_t>(nb) *
                  static_cast<std::size_t>(cfg_.layers_per_block + 1));
    int Hc = H, Wc = W;

    // 2) conv_in: in_channels → first_C.
    x_ = sample.clone();
    apply_conv3x3_(conv_in_W_, conv_in_b_, cfg_.in_channels, first_C, H, W,
                   /*stride=*/1, /*pad=*/1, x_, y_);
    std::swap(x_, y_);
    skips.push_back(x_.clone());

    // 3) down_blocks.
    for (int i = 0; i < nb; ++i) {
        DownBlock& d = down_blocks_[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg_.layers_per_block; ++j) {
            apply_resnet_(d.resnets[static_cast<std::size_t>(j)], Hc, Wc, x_, y_);
            if (d.has_attention) {
                apply_attn_(d.attns[static_cast<std::size_t>(j)], Hc, Wc, x_);
            }
            skips.push_back(x_.clone());
        }
        if (d.has_downsampler) {
            apply_downsample_(d.downsampler, d.C_out, Hc, Wc, x_, y_);
            std::swap(x_, y_);
            Hc /= 2;
            Wc /= 2;
            skips.push_back(x_.clone());
        }
    }

    // 4) mid_block: resnet → attention → resnet.
    apply_resnet_(mid_.r0, Hc, Wc, x_, y_);
    apply_attn_  (mid_.attn, Hc, Wc, x_);
    apply_resnet_(mid_.r1, Hc, Wc, x_, y_);

    // 5) up_blocks.
    for (int i = 0; i < nb; ++i) {
        UpBlock& u = up_blocks_[static_cast<std::size_t>(i)];
        const int layers = cfg_.layers_per_block + 1;
        for (int j = 0; j < layers; ++j) {
            // Channel-axis concat with the matching popped skip.
            bt::GpuTensor skip = std::move(skips.back());
            skips.pop_back();
            const int C_x_now    = x_.cols / (Hc * Wc);
            const int C_skip_now = skip.cols / (Hc * Wc);
            const std::vector<int> C_parts    = {C_x_now, C_skip_now};
            const std::vector<const bt::GpuTensor*> parts = {&x_, &skip};
            bt::concat_nchw_channels_gpu(parts, /*N=*/1, Hc, Wc, C_parts, cat_buf_);
            std::swap(x_, cat_buf_);

            apply_resnet_(u.resnets[static_cast<std::size_t>(j)], Hc, Wc, x_, y_);
            if (u.has_attention) {
                apply_attn_(u.attns[static_cast<std::size_t>(j)], Hc, Wc, x_);
            }
        }
        if (u.has_upsampler) {
            // Nearest-2x upsample, then 3x3 stride-1 conv.
            bt::upsample_nearest_2x_gpu(x_, 1, u.C_out, Hc, Wc, y_);
            apply_conv3x3_(u.upsampler.W, u.upsampler.b,
                           u.C_out, u.C_out, 2 * Hc, 2 * Wc,
                           /*stride=*/1, /*pad=*/1, y_, x_);
            Hc *= 2;
            Wc *= 2;
        }
    }

    // 6) conv_norm_out → SiLU → conv_out.
    bt::group_norm_forward_gpu(x_, norm_out_g_, norm_out_b_,
                               1, first_C, Hc, Wc, cfg_.norm_num_groups, cfg_.eps,
                               y_);
    bt::silu_forward_gpu(y_, y_);
    apply_conv3x3_(conv_out_W_, conv_out_b_,
                   first_C, cfg_.out_channels, Hc, Wc,
                   /*stride=*/1, /*pad=*/1, y_, out);
}

}  // namespace brodiffusion::ddpm_unet
