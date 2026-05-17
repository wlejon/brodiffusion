#include "brodiffusion/unet.h"
#include "brodiffusion/safetensors.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::unet {

namespace bt = ::brotensor;
namespace st = ::brodiffusion::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("unet::UNet: " + msg);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    const auto* v = f.find(key);
    if (!v) throw std::runtime_error("unet::UNet: missing tensor '" + key + "'");
    return *v;
}

void upload_fp16_checked(const st::TensorView& v, int rows, int cols,
                         bt::GpuTensor& dst, const char* name) {
    if (v.dtype != st::Dtype::F16) {
        fail(std::string(name) + ": expected FP16, got " + st::dtype_name(v.dtype));
    }
    int64_t expected = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    if (v.numel() != expected) {
        fail(std::string(name) + ": shape mismatch (expected " +
             std::to_string(rows) + "x" + std::to_string(cols) + ", got " +
             std::to_string(v.numel()) + " elements)");
    }
    st::upload(v, rows, cols, dst);
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

UNet::UNet(const UNetConfig& cfg) : cfg_(cfg) {
    const int nb = static_cast<int>(cfg_.block_out_channels.size());
    if (nb < 2) fail("block_out_channels must have at least 2 entries");
    if (cfg_.layers_per_block <= 0) fail("layers_per_block must be positive");
    if (cfg_.attention_head_dim <= 0) fail("attention_head_dim must be positive");
    if (cfg_.cross_attention_dim <= 0) fail("cross_attention_dim must be positive");
    if (cfg_.time_embed_dim_mult <= 0) fail("time_embed_dim_mult must be positive");
    for (int c : cfg_.block_out_channels) {
        if (c <= 0 || c % cfg_.norm_num_groups != 0) {
            fail("each block_out_channels entry must be a positive multiple of norm_num_groups");
        }
        if (c % cfg_.attention_head_dim != 0) {
            fail("each block_out_channels entry must be divisible by attention_head_dim");
        }
    }

    freq_dim_       = cfg_.block_out_channels.front();
    if (freq_dim_ % 2 != 0) fail("block_out_channels[0] must be even (sinusoidal embedding)");
    time_embed_dim_ = cfg_.block_out_channels.front() * cfg_.time_embed_dim_mult;

    down_blocks_.resize(static_cast<std::size_t>(nb));
    up_blocks_.resize(static_cast<std::size_t>(nb));
    for (int i = 0; i < nb; ++i) {
        // Down: all except the last are cross-attention + downsample.
        DownBlock& d = down_blocks_[static_cast<std::size_t>(i)];
        d.has_attention   = (i < nb - 1);
        d.has_downsampler = (i < nb - 1);
        d.C_out = cfg_.block_out_channels[static_cast<std::size_t>(i)];

        // Up (reversed): all except the first are cross-attention; all except
        // the last have an upsampler.
        UpBlock& u = up_blocks_[static_cast<std::size_t>(i)];
        u.has_attention = (i > 0);
        u.has_upsampler = (i < nb - 1);
        u.C_out = cfg_.block_out_channels[static_cast<std::size_t>(nb - 1 - i)];
    }
}

UNet::~UNet() = default;

// ─── weight-loading helpers ────────────────────────────────────────────────

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
        upload_fp16_checked(need(f, p + "conv_shortcut.weight"),
                            C_out, C_in, r.Ws, "resnet.conv_shortcut.weight");
        upload_fp16_checked(need(f, p + "conv_shortcut.bias"),
                            C_out, 1, r.bs, "resnet.conv_shortcut.bias");
    }
}

void UNet::load_transformer_(const st::File& f, const std::string& p,
                             int C, int num_heads, Transformer2D& t) {
    t.C = C;
    t.num_heads = num_heads;

    upload_fp16_checked(need(f, p + "norm.weight"), C, 1, t.gn_g, "tr.norm.weight");
    upload_fp16_checked(need(f, p + "norm.bias"),   C, 1, t.gn_b, "tr.norm.bias");

    // 1x1 convs flatten to (C, C). Stored as (C_out, C_in, 1, 1) in safetensors.
    upload_fp16_checked(need(f, p + "proj_in.weight"),  C, C, t.pi_W, "tr.proj_in.weight");
    upload_fp16_checked(need(f, p + "proj_in.bias"),    C, 1, t.pi_b, "tr.proj_in.bias");
    upload_fp16_checked(need(f, p + "proj_out.weight"), C, C, t.po_W, "tr.proj_out.weight");
    upload_fp16_checked(need(f, p + "proj_out.bias"),   C, 1, t.po_b, "tr.proj_out.bias");

    // SD1.5 always uses a single BasicTransformerBlock per Transformer2DModel.
    t.blocks.clear();
    t.blocks.resize(1);
    AttnFFN& blk = t.blocks[0];

    const std::string b = p + "transformer_blocks.0.";

    // norm1 + self-attention (no Q/K/V bias).
    upload_fp16_checked(need(f, b + "norm1.weight"), C, 1, blk.n1g, "tr.b.norm1.weight");
    upload_fp16_checked(need(f, b + "norm1.bias"),   C, 1, blk.n1b, "tr.b.norm1.bias");
    upload_fp16_checked(need(f, b + "attn1.to_q.weight"), C, C, blk.Wq1, "tr.b.attn1.to_q.weight");
    upload_fp16_checked(need(f, b + "attn1.to_k.weight"), C, C, blk.Wk1, "tr.b.attn1.to_k.weight");
    upload_fp16_checked(need(f, b + "attn1.to_v.weight"), C, C, blk.Wv1, "tr.b.attn1.to_v.weight");
    upload_fp16_checked(need(f, b + "attn1.to_out.0.weight"), C, C, blk.Wo1, "tr.b.attn1.to_out.weight");
    upload_fp16_checked(need(f, b + "attn1.to_out.0.bias"),   C, 1, blk.bo1, "tr.b.attn1.to_out.bias");

    // norm2 + cross-attention (no Q/K/V bias; K/V project from cross_attention_dim).
    upload_fp16_checked(need(f, b + "norm2.weight"), C, 1, blk.n2g, "tr.b.norm2.weight");
    upload_fp16_checked(need(f, b + "norm2.bias"),   C, 1, blk.n2b, "tr.b.norm2.bias");
    upload_fp16_checked(need(f, b + "attn2.to_q.weight"), C, C, blk.Wq2, "tr.b.attn2.to_q.weight");
    upload_fp16_checked(need(f, b + "attn2.to_k.weight"),
                        C, cfg_.cross_attention_dim, blk.Wk2, "tr.b.attn2.to_k.weight");
    upload_fp16_checked(need(f, b + "attn2.to_v.weight"),
                        C, cfg_.cross_attention_dim, blk.Wv2, "tr.b.attn2.to_v.weight");
    upload_fp16_checked(need(f, b + "attn2.to_out.0.weight"), C, C, blk.Wo2, "tr.b.attn2.to_out.weight");
    upload_fp16_checked(need(f, b + "attn2.to_out.0.bias"),   C, 1, blk.bo2, "tr.b.attn2.to_out.bias");

    // norm3 + FF (GEGLU): Linear(C, 8C) -> split-and-multiply -> Linear(4C, C).
    upload_fp16_checked(need(f, b + "norm3.weight"), C, 1, blk.n3g, "tr.b.norm3.weight");
    upload_fp16_checked(need(f, b + "norm3.bias"),   C, 1, blk.n3b, "tr.b.norm3.bias");
    const int ff_inner = 4 * C;
    upload_fp16_checked(need(f, b + "ff.net.0.proj.weight"),
                        2 * ff_inner, C, blk.ff1_W, "tr.b.ff.net.0.proj.weight");
    upload_fp16_checked(need(f, b + "ff.net.0.proj.bias"),
                        2 * ff_inner, 1, blk.ff1_b, "tr.b.ff.net.0.proj.bias");
    upload_fp16_checked(need(f, b + "ff.net.2.weight"),
                        C, ff_inner, blk.ff2_W, "tr.b.ff.net.2.weight");
    upload_fp16_checked(need(f, b + "ff.net.2.bias"),
                        C, 1, blk.ff2_b, "tr.b.ff.net.2.bias");
}

void UNet::load_weights(const st::File& f, const std::string& prefix) {
    const int nb       = static_cast<int>(cfg_.block_out_channels.size());
    const int first_C  = cfg_.block_out_channels.front();
    const int mid_C    = cfg_.block_out_channels.back();

    // conv_in: in_channels -> block_out_channels[0]
    upload_fp16_checked(need(f, prefix + "conv_in.weight"),
                        first_C, cfg_.in_channels * 3 * 3, conv_in_W_, "conv_in.weight");
    upload_fp16_checked(need(f, prefix + "conv_in.bias"),
                        first_C, 1, conv_in_b_, "conv_in.bias");

    // time_embedding: linear_1 (D, freq_dim) -> silu -> linear_2 (D, D).
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
        const int num_heads = C_out / cfg_.attention_head_dim;

        d.resnets.clear();
        d.resnets.resize(static_cast<std::size_t>(cfg_.layers_per_block));
        d.transformers.clear();
        if (d.has_attention) {
            d.transformers.resize(static_cast<std::size_t>(cfg_.layers_per_block));
        }

        for (int j = 0; j < cfg_.layers_per_block; ++j) {
            const int Ci = (j == 0) ? C_prev : C_out;
            const std::string rp = prefix + "down_blocks." + std::to_string(i) +
                                   ".resnets." + std::to_string(j) + ".";
            load_resnet_(f, rp, Ci, C_out, d.resnets[static_cast<std::size_t>(j)]);

            if (d.has_attention) {
                const std::string tp = prefix + "down_blocks." + std::to_string(i) +
                                       ".attentions." + std::to_string(j) + ".";
                load_transformer_(f, tp, C_out, num_heads,
                                  d.transformers[static_cast<std::size_t>(j)]);
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

    // ── mid_block ──────────────────────────────────────────────────────────
    {
        const std::string mp = prefix + "mid_block.";
        load_resnet_(f, mp + "resnets.0.", mid_C, mid_C, mid_.r0);
        load_transformer_(f, mp + "attentions.0.", mid_C,
                          mid_C / cfg_.attention_head_dim, mid_.t);
        load_resnet_(f, mp + "resnets.1.", mid_C, mid_C, mid_.r1);
    }

    // ── up_blocks ──────────────────────────────────────────────────────────
    // We replay the down-side skip stack to know per-layer skip channel counts.
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
        const int num_heads = C_out / cfg_.attention_head_dim;
        const int layers = cfg_.layers_per_block + 1;

        u.resnets.clear();
        u.resnets.resize(static_cast<std::size_t>(layers));
        u.transformers.clear();
        if (u.has_attention) {
            u.transformers.resize(static_cast<std::size_t>(layers));
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
                const std::string tp = prefix + "up_blocks." + std::to_string(i) +
                                       ".attentions." + std::to_string(j) + ".";
                load_transformer_(f, tp, C_out, num_heads,
                                  u.transformers[static_cast<std::size_t>(j)]);
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

// ─── per-block forward helpers ─────────────────────────────────────────────

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

void UNet::apply_resnet_(const Resnet& r, int H, int W,
                         bt::GpuTensor& x, bt::GpuTensor& tmp) {
    // Per-resblock time-emb projection: silu(temb) -> Linear -> (1, C_out).
    bt::linear_forward_batched_fp16_gpu(r.temb_W, &r.temb_b, temb_silu_, temb_proj_);

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

void UNet::apply_transformer_(const Transformer2D& t,
                              const bt::GpuTensor& ctx,
                              int H, int W, bt::GpuTensor& x) {
    const int C  = t.C;
    const int L  = H * W;
    const int H_heads = t.num_heads;

    // 1. residual is `x` itself; we add the post-transformer result back in.
    // 2. GroupNorm in NCHW.
    bt::group_norm_forward_gpu(x, t.gn_g, t.gn_b,
                               1, C, H, W, cfg_.norm_num_groups, cfg_.eps, gn_);

    // 3. NCHW -> (L, C) sequence.
    bt::nchw_to_sequence_gpu(gn_, 1, C, H, W, seq_);

    // 4. proj_in: 1x1 conv ≡ Linear over C.
    bt::linear_forward_batched_fp16_gpu(t.pi_W, &t.pi_b, seq_, proj_in_seq_);

    // 5. transformer blocks (always 1 for SD1.5).
    tseq_ = proj_in_seq_.clone();
    for (const AttnFFN& blk : t.blocks) {
        // ── self-attention ────────────────────────────────────────────────
        bt::layernorm_forward_inference_batched_fp16_gpu(
            tseq_, blk.n1g, blk.n1b, ln_, cfg_.eps);
        bt::linear_forward_batched_fp16_gpu(blk.Wq1, /*bias=*/nullptr, ln_, Q_);
        bt::linear_forward_batched_fp16_gpu(blk.Wk1, /*bias=*/nullptr, ln_, K_);
        bt::linear_forward_batched_fp16_gpu(blk.Wv1, /*bias=*/nullptr, ln_, V_);
        bt::flash_attention_forward_gpu(Q_, K_, V_, /*d_mask=*/nullptr,
                                        H_heads, /*causal=*/false, attn_out_);
        bt::linear_forward_batched_fp16_gpu(blk.Wo1, &blk.bo1, attn_out_, attn_proj_);
        bt::add_inplace_gpu(tseq_, attn_proj_);

        // ── cross-attention (K, V from `ctx`) ─────────────────────────────
        bt::layernorm_forward_inference_batched_fp16_gpu(
            tseq_, blk.n2g, blk.n2b, ln_, cfg_.eps);
        bt::linear_forward_batched_fp16_gpu(blk.Wq2, /*bias=*/nullptr, ln_, Q_);
        bt::linear_forward_batched_fp16_gpu(blk.Wk2, /*bias=*/nullptr, ctx, K_);
        bt::linear_forward_batched_fp16_gpu(blk.Wv2, /*bias=*/nullptr, ctx, V_);
        bt::flash_attention_forward_gpu(Q_, K_, V_, /*d_mask=*/nullptr,
                                        H_heads, /*causal=*/false, attn_out_);
        bt::linear_forward_batched_fp16_gpu(blk.Wo2, &blk.bo2, attn_out_, attn_proj_);
        bt::add_inplace_gpu(tseq_, attn_proj_);

        // ── feed-forward (GEGLU) ──────────────────────────────────────────
        bt::layernorm_forward_inference_batched_fp16_gpu(
            tseq_, blk.n3g, blk.n3b, ln_, cfg_.eps);
        bt::linear_forward_batched_fp16_gpu(blk.ff1_W, &blk.ff1_b, ln_, ff_mid_);
        bt::geglu_forward_gpu(ff_mid_, ff_act_);
        bt::linear_forward_batched_fp16_gpu(blk.ff2_W, &blk.ff2_b, ff_act_, ff_out_);
        bt::add_inplace_gpu(tseq_, ff_out_);
    }

    // 6. proj_out: 1x1 conv ≡ Linear.
    bt::linear_forward_batched_fp16_gpu(t.po_W, &t.po_b, tseq_, proj_out_seq_);

    // 7. seq -> NCHW.
    bt::sequence_to_nchw_gpu(proj_out_seq_, 1, C, H, W, proj_out_nchw_);

    // 8. residual add.
    bt::add_inplace_gpu(x, proj_out_nchw_);
    (void)L;
}

// ─── forward ───────────────────────────────────────────────────────────────

namespace {

// Sinusoidal timestep embedding matching diffusers `get_timestep_embedding`
// with flip_sin_to_cos=True, downscale_freq_shift=0, scale=1, max_period=10000.
// Output layout: [cos(t*freq_0), ..., cos(t*freq_{H-1}), sin(t*freq_0), ...].
void compute_sinusoidal_emb_fp16(float t, int dim,
                                 std::vector<uint16_t>& out) {
    const int half = dim / 2;
    out.resize(static_cast<std::size_t>(dim));
    const float log_period = std::log(10000.0f);
    for (int i = 0; i < half; ++i) {
        const float freq = std::exp(-log_period * static_cast<float>(i) /
                                     static_cast<float>(half));
        const float angle = t * freq;
        out[static_cast<std::size_t>(i)]        = bt::fp32_to_fp16_bits(std::cos(angle));
        out[static_cast<std::size_t>(i + half)] = bt::fp32_to_fp16_bits(std::sin(angle));
    }
}

}  // namespace

void UNet::forward(const bt::GpuTensor& sample,
                   int H, int W,
                   float timestep,
                   const bt::GpuTensor& encoder_hidden_states,
                   bt::GpuTensor& out) {
    if (conv_in_W_.size() == 0) fail("forward: weights not loaded");

    const int nb      = static_cast<int>(cfg_.block_out_channels.size());
    const int first_C = cfg_.block_out_channels.front();

    // ── 1. Build the time embedding ────────────────────────────────────────
    std::vector<uint16_t> sin_bits;
    compute_sinusoidal_emb_fp16(timestep, freq_dim_, sin_bits);
    bt::upload_fp16(sin_bits.data(), 1, freq_dim_, freq_emb_);

    bt::linear_forward_batched_fp16_gpu(te_l1_W_, &te_l1_b_, freq_emb_, temb_a_);
    bt::silu_forward_gpu(temb_a_, temb_a_);
    bt::linear_forward_batched_fp16_gpu(te_l2_W_, &te_l2_b_, temb_a_, temb_b_);
    // Master temb in temb_b_. Pre-compute SiLU once for reuse across resblocks.
    bt::silu_forward_gpu(temb_b_, temb_silu_);

    // ── 2. conv_in: in_channels -> first_C ─────────────────────────────────
    x_ = sample.clone();
    apply_conv3x3_(conv_in_W_, conv_in_b_, cfg_.in_channels, first_C, H, W,
                   /*stride=*/1, /*pad=*/1, x_, y_);
    std::swap(x_, y_);

    // Skip stack: each entry is a deep copy of the residual stream at push time.
    std::vector<bt::GpuTensor> skips;
    skips.reserve(static_cast<std::size_t>(nb) *
                  static_cast<std::size_t>(cfg_.layers_per_block + 1));
    skips.push_back(x_.clone());

    // ── 3. down_blocks ─────────────────────────────────────────────────────
    int Hc = H, Wc = W;
    for (int i = 0; i < nb; ++i) {
        DownBlock& d = down_blocks_[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg_.layers_per_block; ++j) {
            apply_resnet_(d.resnets[static_cast<std::size_t>(j)], Hc, Wc, x_, y_);
            if (d.has_attention) {
                apply_transformer_(d.transformers[static_cast<std::size_t>(j)],
                                   encoder_hidden_states, Hc, Wc, x_);
            }
            skips.push_back(x_.clone());
        }
        if (d.has_downsampler) {
            // 3x3 stride-2 conv same-channels.
            apply_conv3x3_(d.downsampler.W, d.downsampler.b,
                           d.C_out, d.C_out, Hc, Wc,
                           /*stride=*/2, /*pad=*/1, x_, y_);
            std::swap(x_, y_);
            Hc /= 2;
            Wc /= 2;
            skips.push_back(x_.clone());
        }
    }

    // ── 4. mid_block: resnet -> transformer -> resnet ──────────────────────
    apply_resnet_(mid_.r0, Hc, Wc, x_, y_);
    apply_transformer_(mid_.t, encoder_hidden_states, Hc, Wc, x_);
    apply_resnet_(mid_.r1, Hc, Wc, x_, y_);

    // ── 5. up_blocks ───────────────────────────────────────────────────────
    for (int i = 0; i < nb; ++i) {
        UpBlock& u = up_blocks_[static_cast<std::size_t>(i)];
        const int layers = cfg_.layers_per_block + 1;
        for (int j = 0; j < layers; ++j) {
            // Channel-axis concat with popped skip. For N=1, NCHW channel
            // concat is exactly a flat row-concat.
            bt::GpuTensor skip = std::move(skips.back());
            skips.pop_back();
            std::vector<const bt::GpuTensor*> parts = {&x_, &skip};
            bt::concat_rows_gpu(parts, cat_buf_);
            std::swap(x_, cat_buf_);

            apply_resnet_(u.resnets[static_cast<std::size_t>(j)], Hc, Wc, x_, y_);
            if (u.has_attention) {
                apply_transformer_(u.transformers[static_cast<std::size_t>(j)],
                                   encoder_hidden_states, Hc, Wc, x_);
            }
        }
        if (u.has_upsampler) {
            bt::upsample_nearest_2x_gpu(x_, 1, u.C_out, Hc, Wc, y_);
            apply_conv3x3_(u.upsampler.W, u.upsampler.b,
                           u.C_out, u.C_out, 2 * Hc, 2 * Wc,
                           /*stride=*/1, /*pad=*/1, y_, x_);
            Hc *= 2;
            Wc *= 2;
        }
    }

    // ── 6. conv_norm_out -> SiLU -> conv_out ───────────────────────────────
    bt::group_norm_forward_gpu(x_, norm_out_g_, norm_out_b_,
                               1, first_C, Hc, Wc, cfg_.norm_num_groups, cfg_.eps,
                               y_);
    bt::silu_forward_gpu(y_, y_);
    apply_conv3x3_(conv_out_W_, conv_out_b_,
                   first_C, cfg_.out_channels, Hc, Wc,
                   /*stride=*/1, /*pad=*/1, y_, out);
}

}  // namespace brodiffusion::unet
