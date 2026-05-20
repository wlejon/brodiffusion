#include "brodiffusion/unet.h"
#include "brodiffusion/safetensors.h"
#include "brodiffusion/fused_resblock.h"
#include "brodiffusion/fused_transformer.h"
#include "brodiffusion/detail/device.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef BROTENSOR_HAS_CUDA
#include <cuda_runtime.h>
#endif

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
             std::to_string(v.numel()) + " elements)");
    }
    st::upload_compute(v, rows, cols, dst);
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
    if (cfg_.time_cond_proj_dim < 0) fail("time_cond_proj_dim must be >= 0");
    if (cfg_.time_cond_proj_dim > 0 && (cfg_.time_cond_proj_dim % 2) != 0) {
        fail("time_cond_proj_dim must be even (sinusoidal embedding)");
    }
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
    upload_compute_checked(need(f, p + "norm1.weight"), C_in,  1, r.n1g, "resnet.norm1.weight");
    upload_compute_checked(need(f, p + "norm1.bias"),   C_in,  1, r.n1b, "resnet.norm1.bias");
    upload_compute_checked(need(f, p + "conv1.weight"), C_out, C_in * 3 * 3, r.W1, "resnet.conv1.weight");
    upload_compute_checked(need(f, p + "conv1.bias"),   C_out, 1, r.b1, "resnet.conv1.bias");

    upload_compute_checked(need(f, p + "time_emb_proj.weight"),
                        C_out, time_embed_dim_, r.temb_W, "resnet.time_emb_proj.weight");
    upload_compute_checked(need(f, p + "time_emb_proj.bias"),
                        C_out, 1, r.temb_b, "resnet.time_emb_proj.bias");

    upload_compute_checked(need(f, p + "norm2.weight"), C_out, 1, r.n2g, "resnet.norm2.weight");
    upload_compute_checked(need(f, p + "norm2.bias"),   C_out, 1, r.n2b, "resnet.norm2.bias");
    upload_compute_checked(need(f, p + "conv2.weight"), C_out, C_out * 3 * 3, r.W2, "resnet.conv2.weight");
    upload_compute_checked(need(f, p + "conv2.bias"),   C_out, 1, r.b2, "resnet.conv2.bias");

    r.C_in = C_in;
    r.C_out = C_out;
    r.has_shortcut = (C_in != C_out);
    if (r.has_shortcut) {
        upload_compute_checked(need(f, p + "conv_shortcut.weight"),
                            C_out, C_in, r.Ws, "resnet.conv_shortcut.weight");
        upload_compute_checked(need(f, p + "conv_shortcut.bias"),
                            C_out, 1, r.bs, "resnet.conv_shortcut.bias");
    }
}

void UNet::load_transformer_(const st::File& f, const std::string& p,
                             int C, int num_heads, Transformer2D& t) {
    t.C = C;
    t.num_heads = num_heads;

    upload_compute_checked(need(f, p + "norm.weight"), C, 1, t.gn_g, "tr.norm.weight");
    upload_compute_checked(need(f, p + "norm.bias"),   C, 1, t.gn_b, "tr.norm.bias");

    // 1x1 convs flatten to (C, C). Stored as (C_out, C_in, 1, 1) in safetensors.
    upload_compute_checked(need(f, p + "proj_in.weight"),  C, C, t.pi_W, "tr.proj_in.weight");
    upload_compute_checked(need(f, p + "proj_in.bias"),    C, 1, t.pi_b, "tr.proj_in.bias");
    upload_compute_checked(need(f, p + "proj_out.weight"), C, C, t.po_W, "tr.proj_out.weight");
    upload_compute_checked(need(f, p + "proj_out.bias"),   C, 1, t.po_b, "tr.proj_out.bias");

    // SD1.5 always uses a single BasicTransformerBlock per Transformer2DModel.
    t.blocks.clear();
    t.blocks.resize(1);
    AttnFFN& blk = t.blocks[0];

    const std::string b = p + "transformer_blocks.0.";

    // norm1 + self-attention (no Q/K/V bias).
    upload_compute_checked(need(f, b + "norm1.weight"), C, 1, blk.n1g, "tr.b.norm1.weight");
    upload_compute_checked(need(f, b + "norm1.bias"),   C, 1, blk.n1b, "tr.b.norm1.bias");
    upload_compute_checked(need(f, b + "attn1.to_q.weight"), C, C, blk.Wq1, "tr.b.attn1.to_q.weight");
    upload_compute_checked(need(f, b + "attn1.to_k.weight"), C, C, blk.Wk1, "tr.b.attn1.to_k.weight");
    upload_compute_checked(need(f, b + "attn1.to_v.weight"), C, C, blk.Wv1, "tr.b.attn1.to_v.weight");
    upload_compute_checked(need(f, b + "attn1.to_out.0.weight"), C, C, blk.Wo1, "tr.b.attn1.to_out.weight");
    upload_compute_checked(need(f, b + "attn1.to_out.0.bias"),   C, 1, blk.bo1, "tr.b.attn1.to_out.bias");

    // norm2 + cross-attention (no Q/K/V bias; K/V project from cross_attention_dim).
    upload_compute_checked(need(f, b + "norm2.weight"), C, 1, blk.n2g, "tr.b.norm2.weight");
    upload_compute_checked(need(f, b + "norm2.bias"),   C, 1, blk.n2b, "tr.b.norm2.bias");
    upload_compute_checked(need(f, b + "attn2.to_q.weight"), C, C, blk.Wq2, "tr.b.attn2.to_q.weight");
    upload_compute_checked(need(f, b + "attn2.to_k.weight"),
                        C, cfg_.cross_attention_dim, blk.Wk2, "tr.b.attn2.to_k.weight");
    upload_compute_checked(need(f, b + "attn2.to_v.weight"),
                        C, cfg_.cross_attention_dim, blk.Wv2, "tr.b.attn2.to_v.weight");
    upload_compute_checked(need(f, b + "attn2.to_out.0.weight"), C, C, blk.Wo2, "tr.b.attn2.to_out.weight");
    upload_compute_checked(need(f, b + "attn2.to_out.0.bias"),   C, 1, blk.bo2, "tr.b.attn2.to_out.bias");

    // norm3 + FF (GEGLU): Linear(C, 8C) -> split-and-multiply -> Linear(4C, C).
    upload_compute_checked(need(f, b + "norm3.weight"), C, 1, blk.n3g, "tr.b.norm3.weight");
    upload_compute_checked(need(f, b + "norm3.bias"),   C, 1, blk.n3b, "tr.b.norm3.bias");
    const int ff_inner = 4 * C;
    upload_compute_checked(need(f, b + "ff.net.0.proj.weight"),
                        2 * ff_inner, C, blk.ff1_W, "tr.b.ff.net.0.proj.weight");
    upload_compute_checked(need(f, b + "ff.net.0.proj.bias"),
                        2 * ff_inner, 1, blk.ff1_b, "tr.b.ff.net.0.proj.bias");
    upload_compute_checked(need(f, b + "ff.net.2.weight"),
                        C, ff_inner, blk.ff2_W, "tr.b.ff.net.2.weight");
    upload_compute_checked(need(f, b + "ff.net.2.bias"),
                        C, 1, blk.ff2_b, "tr.b.ff.net.2.bias");
}

void UNet::load_weights(const st::File& f, const std::string& prefix) {
    const int nb       = static_cast<int>(cfg_.block_out_channels.size());
    const int first_C  = cfg_.block_out_channels.front();
    const int mid_C    = cfg_.block_out_channels.back();

    // conv_in: in_channels -> block_out_channels[0]
    upload_compute_checked(need(f, prefix + "conv_in.weight"),
                        first_C, cfg_.in_channels * 3 * 3, conv_in_W_, "conv_in.weight");
    upload_compute_checked(need(f, prefix + "conv_in.bias"),
                        first_C, 1, conv_in_b_, "conv_in.bias");

    // time_embedding: linear_1 (D, freq_dim) -> silu -> linear_2 (D, D).
    upload_compute_checked(need(f, prefix + "time_embedding.linear_1.weight"),
                        time_embed_dim_, freq_dim_, te_l1_W_, "te.linear_1.weight");
    upload_compute_checked(need(f, prefix + "time_embedding.linear_1.bias"),
                        time_embed_dim_, 1, te_l1_b_, "te.linear_1.bias");
    upload_compute_checked(need(f, prefix + "time_embedding.linear_2.weight"),
                        time_embed_dim_, time_embed_dim_, te_l2_W_, "te.linear_2.weight");
    upload_compute_checked(need(f, prefix + "time_embedding.linear_2.bias"),
                        time_embed_dim_, 1, te_l2_b_, "te.linear_2.bias");

    // LCM cond_proj: only present in distilled checkpoints. No bias term
    // (diffusers' TimestepEmbedding constructs it as `nn.Linear(cond_proj_dim,
    // in_channels=freq_dim, bias=False)`, and adds its output to `sample`
    // *before* linear_1). Shape is (freq_dim_, cond_proj_dim), not
    // (time_embed_dim_, cond_proj_dim) — caught by Dreamshaper-7 LCM at load.
    if (cfg_.time_cond_proj_dim > 0) {
        upload_compute_checked(need(f, prefix + "time_embedding.cond_proj.weight"),
                            freq_dim_, cfg_.time_cond_proj_dim,
                            te_cond_W_, "te.cond_proj.weight");
    }

    // ── down_blocks ────────────────────────────────────────────────────────
    int C_prev = first_C;
    for (int i = 0; i < nb; ++i) {
        DownBlock& d = down_blocks_[static_cast<std::size_t>(i)];
        const int C_out = d.C_out;
        // NOTE: diffusers' SD1.5 config uses `attention_head_dim` as the
        // *number of heads* (a backwards-compat quirk documented in
        // UNet2DConditionModel.__init__). Head dim is therefore C_out/num_heads.
        const int num_heads = cfg_.attention_head_dim;

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
            upload_compute_checked(need(f, sp + "weight"),
                                C_out, C_out * 3 * 3, d.downsampler.W,
                                "downsampler.conv.weight");
            upload_compute_checked(need(f, sp + "bias"),
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
                          cfg_.attention_head_dim, mid_.t);
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
        const int num_heads = cfg_.attention_head_dim;
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
            upload_compute_checked(need(f, sp + "weight"),
                                C_out, C_out * 3 * 3, u.upsampler.W,
                                "upsampler.conv.weight");
            upload_compute_checked(need(f, sp + "bias"),
                                C_out, 1, u.upsampler.b,
                                "upsampler.conv.bias");
        }

        C_up_prev = C_out;
    }

    if (!skip_stack.empty()) fail("internal: skip stack not drained during weight load");

    upload_compute_checked(need(f, prefix + "conv_norm_out.weight"),
                        first_C, 1, norm_out_g_, "conv_norm_out.weight");
    upload_compute_checked(need(f, prefix + "conv_norm_out.bias"),
                        first_C, 1, norm_out_b_, "conv_norm_out.bias");
    upload_compute_checked(need(f, prefix + "conv_out.weight"),
                        cfg_.out_channels, first_C * 3 * 3, conv_out_W_, "conv_out.weight");
    upload_compute_checked(need(f, prefix + "conv_out.bias"),
                        cfg_.out_channels, 1, conv_out_b_, "conv_out.bias");
}

// ─── per-block forward helpers ─────────────────────────────────────────────

void UNet::apply_conv3x3_(const bt::Tensor& W, const bt::Tensor& b,
                          int C_in, int C_out, int H, int W_,
                          int stride, int pad,
                          const bt::Tensor& in, bt::Tensor& out) {
    bt::conv2d_forward(in, W, &b,
                           /*N=*/1, C_in, H, W_,
                           C_out, /*kH=*/3, /*kW=*/3,
                           stride, stride, pad, pad, /*dil=*/1, 1,
                           out);
}

void UNet::apply_conv3x3_q_(const QWeight& Wq, const bt::Tensor& b,
                            int C_in, int C_out, int H, int W_,
                            int stride, int pad,
                            const bt::Tensor& in, bt::Tensor& out) {
    bt::conv2d_int8w_fp16_forward(in, Wq.W_int8, Wq.scales, &b,
                                      /*N=*/1, C_in, H, W_,
                                      C_out, /*kH=*/3, /*kW=*/3,
                                      stride, stride, pad, pad, /*dil=*/1, 1,
                                      /*groups=*/1, out);
}

void UNet::apply_resnet_(const Resnet& r, int H, int W,
                         bt::Tensor& x, bt::Tensor& tmp) {
    // Per-resblock time-emb projection: silu(temb) -> Linear -> (1, C_out).
    detail::linear_batched(r.temb_W, &r.temb_b, temb_silu_, temb_proj_);

    if (r.W1_q.active()) {
        const QWeight* skip_q = r.has_shortcut ? &r.Ws_q : nullptr;
        const bt::Tensor* skip_b = r.has_shortcut ? &r.bs : nullptr;
        const bt::Tensor* skip_W_int8   = skip_q ? &skip_q->W_int8 : nullptr;
        const bt::Tensor* skip_W_scales = skip_q ? &skip_q->scales : nullptr;
        brodiffusion::fused_resblock_forward(x,
                                             r.n1g, r.n1b,
                                             r.W1_q.W_int8, r.W1_q.scales, r.b1,
                                             temb_proj_,
                                             r.n2g, r.n2b,
                                             r.W2_q.W_int8, r.W2_q.scales, r.b2,
                                             skip_W_int8, skip_W_scales, skip_b,
                                             r.C_in, r.C_out, H, W,
                                             cfg_.norm_num_groups, cfg_.eps,
                                             tmp);
    } else {
        const bt::Tensor* skip_W = r.has_shortcut ? &r.Ws : nullptr;
        const bt::Tensor* skip_b = r.has_shortcut ? &r.bs : nullptr;
        brodiffusion::fused_resblock_forward(x,
                                             r.n1g, r.n1b,
                                             r.W1, r.b1,
                                             temb_proj_,
                                             r.n2g, r.n2b,
                                             r.W2, r.b2,
                                             skip_W, skip_b,
                                             r.C_in, r.C_out, H, W,
                                             cfg_.norm_num_groups, cfg_.eps,
                                             tmp);
    }
    std::swap(x, tmp);
}

void UNet::apply_transformer_(const Transformer2D& t,
                              const bt::Tensor& ctx,
                              const CrossAttnKVCacheEntry* cache_entry,
                              int H, int W, bt::Tensor& x,
                              bt::Tensor* trace_out_entry,
                              const bt::Tensor* attn_logit_bias) {
    // Trace mode is incompatible with the cached K/V fast path because
    // brotensor's cross_attention_forward_with_attn reprojects K/V from
    // ctx every call. forward_trace() already ensures no cache is passed.
    if (trace_out_entry != nullptr && cache_entry != nullptr) {
        fail("apply_transformer_: trace_out_entry and cache_entry are "
             "mutually exclusive");
    }
    const int C  = t.C;
    const int H_heads = t.num_heads;

    // 1. residual is `x` itself; we add the post-transformer result back in.
    // 2. GroupNorm in NCHW. Diffusers' Transformer2DModel uses eps=1e-6 on
    //    this outer GroupNorm (distinct from the 1e-5 used on the ResNet
    //    GroupNorms); hard-code it here rather than relying on cfg_.eps.
    bt::group_norm_forward(x, t.gn_g, t.gn_b,
                               1, C, H, W, cfg_.norm_num_groups, 1e-6f, gn_);

    // 3. NCHW -> (L, C) sequence.
    bt::nchw_to_sequence(gn_, 1, C, H, W, seq_);

    // 4. proj_in: 1x1 conv ≡ Linear over C.
    if (t.pi_q.active()) {
        bt::linear_forward_batched_int8w_fp16(
            t.pi_q.W_int8, t.pi_q.scales, &t.pi_b, seq_, proj_in_seq_);
    } else {
        detail::linear_batched(t.pi_W, &t.pi_b, seq_, proj_in_seq_);
    }

    // 5. transformer blocks (always 1 for SD1.5).
    tseq_ = proj_in_seq_.clone();
    for (const AttnFFN& blk : t.blocks) {
        // ── self-attention (Q/K/V bias-less, Wo biased) ───────────────────
        detail::layernorm_batched(tseq_, blk.n1g, blk.n1b, ln_, cfg_.eps);
        if (blk.Wq1_q.active()) {
            bt::flash_attention_qkvo_int8w_fp16(
                ln_, /*Ctx=*/nullptr,
                blk.Wq1_q.W_int8, blk.Wq1_q.scales, /*bq=*/nullptr,
                blk.Wk1_q.W_int8, blk.Wk1_q.scales, /*bk=*/nullptr,
                blk.Wv1_q.W_int8, blk.Wv1_q.scales, /*bv=*/nullptr,
                blk.Wo1_q.W_int8, blk.Wo1_q.scales, &blk.bo1,
                /*d_mask=*/nullptr, H_heads, /*causal=*/false,
                attn_proj_);
        } else {
            bt::flash_attention_qkvo_forward(
                ln_, /*Ctx=*/nullptr,
                blk.Wq1, /*bq=*/nullptr,
                blk.Wk1, /*bk=*/nullptr,
                blk.Wv1, /*bv=*/nullptr,
                blk.Wo1, &blk.bo1,
                /*d_mask=*/nullptr, H_heads, /*causal=*/false,
                attn_proj_);
        }
        brodiffusion::add_inplace_vec(tseq_, attn_proj_);

        // ── cross-attention (K, V from `ctx` — possibly cached) ───────────
        detail::layernorm_batched(tseq_, blk.n2g, blk.n2b, ln_, cfg_.eps);
        if (trace_out_entry) {
            // Trace path: brotensor's cross_attention_forward_with_attn
            // writes the head-averaged softmax map to AttnAvg. No Wo
            // bias is supported by that op, so we manually add bo2 after.
            // (forward_trace already guards against INT8.)
            bt::cross_attention_forward_with_attn(
                ln_, ctx,
                blk.Wq2, blk.Wk2, blk.Wv2, blk.Wo2,
                /*d_mask=*/nullptr,
                attn_logit_bias,
                H_heads,
                attn_proj_, *trace_out_entry);
            // Add output bias bo2 (per-column broadcast across rows of attn_proj_).
            brodiffusion::add_inplace_row_bias(attn_proj_, blk.bo2);
        } else if (cache_entry) {
            // K/V already projected from `ctx` upstream — skip the two
            // per-step matmuls and feed the cached buffers straight in.
            if (blk.Wq2_q.active()) {
                bt::flash_attention_q_with_kv_cached_int8w_fp16(
                    ln_, cache_entry->K, cache_entry->V,
                    blk.Wq2_q.W_int8, blk.Wq2_q.scales, /*bq=*/nullptr,
                    blk.Wo2_q.W_int8, blk.Wo2_q.scales, &blk.bo2,
                    /*d_mask=*/nullptr, H_heads, /*causal=*/false,
                    attn_proj_);
            } else {
                bt::flash_attention_q_with_kv_cached_forward(
                    ln_, cache_entry->K, cache_entry->V,
                    blk.Wq2, /*bq=*/nullptr,
                    blk.Wo2, &blk.bo2,
                    /*d_mask=*/nullptr, H_heads, /*causal=*/false,
                    attn_proj_);
            }
        } else {
            if (blk.Wq2_q.active()) {
                bt::flash_attention_qkvo_int8w_fp16(
                    ln_, &ctx,
                    blk.Wq2_q.W_int8, blk.Wq2_q.scales, /*bq=*/nullptr,
                    blk.Wk2_q.W_int8, blk.Wk2_q.scales, /*bk=*/nullptr,
                    blk.Wv2_q.W_int8, blk.Wv2_q.scales, /*bv=*/nullptr,
                    blk.Wo2_q.W_int8, blk.Wo2_q.scales, &blk.bo2,
                    /*d_mask=*/nullptr, H_heads, /*causal=*/false,
                    attn_proj_);
            } else {
                bt::flash_attention_qkvo_forward(
                    ln_, &ctx,
                    blk.Wq2, /*bq=*/nullptr,
                    blk.Wk2, /*bk=*/nullptr,
                    blk.Wv2, /*bv=*/nullptr,
                    blk.Wo2, &blk.bo2,
                    /*d_mask=*/nullptr, H_heads, /*causal=*/false,
                    attn_proj_);
            }
        }
        brodiffusion::add_inplace_vec(tseq_, attn_proj_);

        // ── feed-forward (GEGLU) ──────────────────────────────────────────
        detail::layernorm_batched(tseq_, blk.n3g, blk.n3b, ln_, cfg_.eps);
        // Fused FF1 + exact-GEGLU: skips the (B, 2*D) intermediate of FF1.
        // SD1.5's BasicTransformerBlock uses F.gelu(approximate=False).
        if (blk.ff1_q.active()) {
            brodiffusion::fused_linear_geglu(
                ln_, blk.ff1_q.W_int8, blk.ff1_q.scales, blk.ff1_b, ff_act_);
        } else {
            brodiffusion::fused_linear_geglu(ln_, blk.ff1_W, blk.ff1_b, ff_act_);
        }
        if (blk.ff2_q.active()) {
            bt::linear_forward_batched_int8w_fp16(
                blk.ff2_q.W_int8, blk.ff2_q.scales, &blk.ff2_b, ff_act_, ff_out_);
        } else {
            detail::linear_batched(blk.ff2_W, &blk.ff2_b, ff_act_, ff_out_);
        }
        brodiffusion::add_inplace_vec(tseq_, ff_out_);
    }

    // 6. proj_out: 1x1 conv ≡ Linear.
    if (t.po_q.active()) {
        bt::linear_forward_batched_int8w_fp16(
            t.po_q.W_int8, t.po_q.scales, &t.po_b, tseq_, proj_out_seq_);
    } else {
        detail::linear_batched(t.po_W, &t.po_b, tseq_, proj_out_seq_);
    }

    // 7. seq -> NCHW.
    bt::sequence_to_nchw(proj_out_seq_, 1, C, H, W, proj_out_nchw_);

    // 8. residual add.
    bt::add_inplace(x, proj_out_nchw_);
}

// ─── forward ───────────────────────────────────────────────────────────────

namespace {

// Sinusoidal embedding matching diffusers' `get_guidance_scale_embedding`
// helper used by LCM. Two important differences from the timestep helper:
//   1. The frequency divisor is `(half - 1)`, not `half`.
//   2. The output layout is [sin..., cos...] (no flip_sin_to_cos).
// Callers should pre-multiply w by 1000 (diffusers does this before calling
// the helper).
//   Output layout: [sin(w*freq_0), ..., sin(w*freq_{H-1}), cos(w*freq_0), ...].
void compute_guidance_scale_emb(float w, int dim,
                                std::vector<float>& out) {
    const int half = dim / 2;
    out.resize(static_cast<std::size_t>(dim));
    const float log_period = std::log(10000.0f);
    const float denom = (half > 1) ? static_cast<float>(half - 1) : 1.0f;
    for (int i = 0; i < half; ++i) {
        const float freq = std::exp(-log_period * static_cast<float>(i) / denom);
        const float angle = w * freq;
        out[static_cast<std::size_t>(i)]        = std::sin(angle);
        out[static_cast<std::size_t>(i + half)] = std::cos(angle);
    }
}

// Sinusoidal timestep embedding matching diffusers `get_timestep_embedding`
// with flip_sin_to_cos=True, downscale_freq_shift=0, scale=1, max_period=10000.
// Output layout: [cos(t*freq_0), ..., cos(t*freq_{H-1}), sin(t*freq_0), ...].
void compute_sinusoidal_emb(float t, int dim,
                            std::vector<float>& out) {
    const int half = dim / 2;
    out.resize(static_cast<std::size_t>(dim));
    const float log_period = std::log(10000.0f);
    for (int i = 0; i < half; ++i) {
        const float freq = std::exp(-log_period * static_cast<float>(i) /
                                     static_cast<float>(half));
        const float angle = t * freq;
        out[static_cast<std::size_t>(i)]        = std::cos(angle);
        out[static_cast<std::size_t>(i + half)] = std::sin(angle);
    }
}

}  // namespace

void UNet::forward(const bt::Tensor& sample,
                   int H, int W,
                   float timestep,
                   const bt::Tensor& encoder_hidden_states,
                   bt::Tensor& out) {
    if (cfg_.time_cond_proj_dim > 0) {
        fail("forward: UNet built with time_cond_proj_dim>0 requires the "
             "guidance_scale_embedding overload");
    }
    forward_impl_(sample, H, W, timestep, /*gs_emb=*/nullptr,
                  encoder_hidden_states, /*xattn_cache=*/nullptr,
                  /*attn_logit_biases=*/nullptr, /*trace_out=*/nullptr, out);
}

void UNet::forward(const bt::Tensor& sample,
                   int H, int W,
                   float timestep,
                   const bt::Tensor& encoder_hidden_states,
                   const CrossAttnKVCache& xattn_cache,
                   bt::Tensor& out) {
    if (cfg_.time_cond_proj_dim > 0) {
        fail("forward: UNet built with time_cond_proj_dim>0 requires the "
             "guidance_scale_embedding overload");
    }
    if (static_cast<int>(xattn_cache.size()) != num_xattn_blocks()) {
        fail("forward: cross-attn KV cache has " +
             std::to_string(xattn_cache.size()) + " entries, expected " +
             std::to_string(num_xattn_blocks()));
    }
    forward_impl_(sample, H, W, timestep, /*gs_emb=*/nullptr,
                  encoder_hidden_states, &xattn_cache,
                  /*attn_logit_biases=*/nullptr, /*trace_out=*/nullptr, out);
}

void UNet::forward(const bt::Tensor& sample,
                   int H, int W,
                   float timestep,
                   float guidance_scale_embedding,
                   const bt::Tensor& encoder_hidden_states,
                   const CrossAttnKVCache& xattn_cache,
                   bt::Tensor& out) {
    if (cfg_.time_cond_proj_dim <= 0) {
        fail("forward(guidance_scale_embedding): UNet built with "
             "time_cond_proj_dim=0; LCM cond_proj path unavailable");
    }
    if (static_cast<int>(xattn_cache.size()) != num_xattn_blocks()) {
        fail("forward: cross-attn KV cache has " +
             std::to_string(xattn_cache.size()) + " entries, expected " +
             std::to_string(num_xattn_blocks()));
    }
    forward_impl_(sample, H, W, timestep, &guidance_scale_embedding,
                  encoder_hidden_states, &xattn_cache,
                  /*attn_logit_biases=*/nullptr, /*trace_out=*/nullptr, out);
}

void UNet::forward_trace(const bt::Tensor& sample,
                         int H, int W,
                         float timestep,
                         const bt::Tensor& encoder_hidden_states,
                         const std::vector<const bt::Tensor*>* attn_logit_biases,
                         CrossAttnTrace* trace_out,
                         bt::Tensor& out) {
    if (cfg_.quantize_weights) {
        fail("forward_trace: INT8 quantize_weights not yet supported in trace "
             "mode (brotensor::cross_attention_forward_with_attn is FP16 "
             "only)");
    }
    if (cfg_.time_cond_proj_dim > 0) {
        fail("forward_trace: LCM cond_proj path not yet supported in trace "
             "mode; use the vanilla SD1.5 path");
    }
    const int n = num_xattn_blocks();
    if (attn_logit_biases &&
        static_cast<int>(attn_logit_biases->size()) != n) {
        fail("forward_trace: attn_logit_biases has " +
             std::to_string(attn_logit_biases->size()) + " entries, expected " +
             std::to_string(n));
    }
    if (trace_out) {
        trace_out->resize(static_cast<std::size_t>(n));
    }
    forward_impl_(sample, H, W, timestep, /*gs_emb=*/nullptr,
                  encoder_hidden_states, /*xattn_cache=*/nullptr,
                  attn_logit_biases, trace_out, out);
}

// ─── LoRA merge ────────────────────────────────────────────────────────────
//
// Resolve a diffusers tensor path to the corresponding base weight
// pointer. Supports only the keys community LoRAs actually patch in SD1.5:
// the eight attention projections and the two GEGLU FF projections inside
// each Transformer2D's BasicTransformerBlock. Returns nullptr if the path
// is not recognized.

namespace {

// Local resolution: walks the UNet to find the Tensor* for a diffusers
// target path. Lives in the .cpp so it can touch private struct fields via
// the UNet method `lora_target_` below.

}  // namespace

namespace {

// Parse "<head>.<int>(.|$)" — consume the integer immediately after a literal
// head + dot. On success advances `pos` past the integer and returns true.
bool match_dotted_int(const std::string& s, std::size_t& pos,
                      const char* head, int& out) {
    std::size_t h = std::strlen(head);
    if (s.compare(pos, h, head) != 0) return false;
    pos += h;
    if (pos >= s.size() || !std::isdigit(static_cast<unsigned char>(s[pos]))) return false;
    int v = 0;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
        v = v * 10 + (s[pos] - '0');
        ++pos;
    }
    out = v;
    return true;
}

}  // namespace

bt::Tensor* UNet::resolve_transformer_target_(Transformer2D& tr,
                                                  const std::string& sub) {
    if (sub == "proj_in")  return &tr.pi_W;
    if (sub == "proj_out") return &tr.po_W;
    static const std::string tb = "transformer_blocks.0.";
    if (sub.rfind(tb, 0) != 0) return nullptr;
    if (tr.blocks.empty()) return nullptr;
    AttnFFN& blk = tr.blocks[0];
    const std::string tail = sub.substr(tb.size());
    if (tail == "attn1.to_q")       return &blk.Wq1;
    if (tail == "attn1.to_k")       return &blk.Wk1;
    if (tail == "attn1.to_v")       return &blk.Wv1;
    if (tail == "attn1.to_out.0")   return &blk.Wo1;
    if (tail == "attn2.to_q")       return &blk.Wq2;
    if (tail == "attn2.to_k")       return &blk.Wk2;
    if (tail == "attn2.to_v")       return &blk.Wv2;
    if (tail == "attn2.to_out.0")   return &blk.Wo2;
    if (tail == "ff.net.0.proj")    return &blk.ff1_W;
    if (tail == "ff.net.2")         return &blk.ff2_W;
    return nullptr;
}

bt::Tensor* UNet::resolve_resnet_target_(Resnet& r, const std::string& tail) {
    if (tail == "conv1")          return &r.W1;
    if (tail == "conv2")          return &r.W2;
    if (tail == "conv_shortcut")  return r.has_shortcut ? &r.Ws : nullptr;
    if (tail == "time_emb_proj")  return &r.temb_W;
    return nullptr;
}

bt::Tensor* UNet::lora_target_(const std::string& target_path) {
    // Branch on the top-level block segment.
    if (target_path.rfind("down_blocks.", 0) == 0) {
        std::size_t pos = 0;
        int i = 0;
        if (!match_dotted_int(target_path, pos, "down_blocks.", i)) return nullptr;
        if (i < 0 || i >= static_cast<int>(down_blocks_.size())) return nullptr;
        if (pos >= target_path.size() || target_path[pos] != '.') return nullptr;
        ++pos;
        DownBlock& d = down_blocks_[static_cast<std::size_t>(i)];
        // resnets.<k>.<tail>
        std::size_t p = pos;
        int k = 0;
        if (match_dotted_int(target_path, p, "resnets.", k)) {
            if (p >= target_path.size() || target_path[p] != '.') return nullptr;
            ++p;
            if (k < 0 || k >= static_cast<int>(d.resnets.size())) return nullptr;
            return resolve_resnet_target_(d.resnets[static_cast<std::size_t>(k)],
                                         target_path.substr(p));
        }
        // attentions.<j>.<sub>
        p = pos;
        int j = 0;
        if (match_dotted_int(target_path, p, "attentions.", j)) {
            if (p >= target_path.size() || target_path[p] != '.') return nullptr;
            ++p;
            if (!d.has_attention) return nullptr;
            if (j < 0 || j >= static_cast<int>(d.transformers.size())) return nullptr;
            return resolve_transformer_target_(
d.transformers[static_cast<std::size_t>(j)],
                       target_path.substr(p));
        }
        // downsamplers.0.conv
        if (target_path.compare(pos, std::string::npos, "downsamplers.0.conv") == 0) {
            return d.has_downsampler ? &d.downsampler.W : nullptr;
        }
        return nullptr;
    }
    if (target_path.rfind("up_blocks.", 0) == 0) {
        std::size_t pos = 0;
        int i = 0;
        if (!match_dotted_int(target_path, pos, "up_blocks.", i)) return nullptr;
        if (i < 0 || i >= static_cast<int>(up_blocks_.size())) return nullptr;
        if (pos >= target_path.size() || target_path[pos] != '.') return nullptr;
        ++pos;
        UpBlock& u = up_blocks_[static_cast<std::size_t>(i)];
        std::size_t p = pos;
        int k = 0;
        if (match_dotted_int(target_path, p, "resnets.", k)) {
            if (p >= target_path.size() || target_path[p] != '.') return nullptr;
            ++p;
            if (k < 0 || k >= static_cast<int>(u.resnets.size())) return nullptr;
            return resolve_resnet_target_(u.resnets[static_cast<std::size_t>(k)],
                                         target_path.substr(p));
        }
        p = pos;
        int j = 0;
        if (match_dotted_int(target_path, p, "attentions.", j)) {
            if (p >= target_path.size() || target_path[p] != '.') return nullptr;
            ++p;
            if (!u.has_attention) return nullptr;
            if (j < 0 || j >= static_cast<int>(u.transformers.size())) return nullptr;
            return resolve_transformer_target_(
u.transformers[static_cast<std::size_t>(j)],
                       target_path.substr(p));
        }
        if (target_path.compare(pos, std::string::npos, "upsamplers.0.conv") == 0) {
            return u.has_upsampler ? &u.upsampler.W : nullptr;
        }
        return nullptr;
    }
    // mid_block: "mid_block.resnets.{0,1}.<tail>" or
    //            "mid_block.attentions.0.<sub>".
    static const std::string mid_pfx = "mid_block.";
    if (target_path.rfind(mid_pfx, 0) == 0) {
        const std::string rest = target_path.substr(mid_pfx.size());
        std::size_t p = 0;
        int k = 0;
        if (match_dotted_int(rest, p, "resnets.", k)) {
            if (p >= rest.size() || rest[p] != '.') return nullptr;
            ++p;
            const std::string tail = rest.substr(p);
            if (k == 0) return resolve_resnet_target_(mid_.r0, tail);
            if (k == 1) return resolve_resnet_target_(mid_.r1, tail);
            return nullptr;
        }
        p = 0;
        int j = 0;
        if (match_dotted_int(rest, p, "attentions.", j)) {
            if (p >= rest.size() || rest[p] != '.') return nullptr;
            ++p;
            if (j != 0) return nullptr;
            return resolve_transformer_target_(mid_.t, rest.substr(p));
        }
        return nullptr;
    }
    return nullptr;
}

namespace {

// Upload an arbitrary 2-D safetensors view (F16 or F32) at the compute dtype
// as a Tensor of shape (rows, cols). Convolutional LoRA layouts
// (rank, C, kH, kW) flatten to (rank, C*kH*kW) — the caller is responsible
// for picking rows/cols.
void upload_view_compute(const st::TensorView& v, int rows, int cols,
                         bt::Tensor& dst, const std::string& tag) {
    if (v.dtype != st::Dtype::F16 && v.dtype != st::Dtype::F32) {
        throw std::runtime_error("unet::UNet: lora " + tag +
            ": expected F16 or F32, got " + st::dtype_name(v.dtype));
    }
    int64_t expected = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    if (v.numel() != expected) {
        throw std::runtime_error("unet::UNet: lora " + tag +
            ": shape numel mismatch (expected " + std::to_string(expected) +
            ", got " + std::to_string(v.numel()) + ")");
    }
    st::upload_compute(v, rows, cols, dst);
}

// Read a compute-dtype tensor back into host memory as fp32 (debug /
// numerical merge path uses this only in tests). Not used here; merge happens
// entirely on-device via matmul + scale_inplace + add_inplace.

// Compute the (rank, in_dim) and (out_dim, rank) shapes from the raw views.
// Validates against the base weight's (rows = out_dim, cols = in_dim) shape
// when the LoRA down is 2-D, and supports the conv-style 4-D layout
// (rank, C_in, kH, kW) by flattening to (rank, C_in*kH*kW).
struct LoraShape {
    int rank   = 0;
    int in_dim = 0;
    int out_dim = 0;
};
LoraShape resolve_lora_shape(const st::TensorView& down,
                             const st::TensorView& up,
                             int W_rows, int W_cols,
                             const std::string& tag) {
    if (down.shape.empty() || up.shape.empty()) {
        throw std::runtime_error("unet::UNet: lora " + tag + ": empty shapes");
    }
    LoraShape s;
    s.rank = static_cast<int>(down.shape[0]);
    if (s.rank <= 0) {
        throw std::runtime_error("unet::UNet: lora " + tag + ": rank <= 0");
    }
    // lora_up: (out_dim, rank). Always 2-D in practice.
    if (up.shape.size() == 2) {
        s.out_dim = static_cast<int>(up.shape[0]);
        if (static_cast<int>(up.shape[1]) != s.rank) {
            throw std::runtime_error("unet::UNet: lora " + tag +
                ": lora_up.shape[1] (" + std::to_string(up.shape[1]) +
                ") != rank (" + std::to_string(s.rank) + ")");
        }
    } else if (up.shape.size() == 4) {
        // (out_dim, rank, 1, 1) conv-shaped up.
        if (up.shape[2] != 1 || up.shape[3] != 1) {
            throw std::runtime_error("unet::UNet: lora " + tag +
                ": lora_up 4-D shape with kH/kW != 1 not supported");
        }
        s.out_dim = static_cast<int>(up.shape[0]);
        if (static_cast<int>(up.shape[1]) != s.rank) {
            throw std::runtime_error("unet::UNet: lora " + tag +
                ": lora_up 4-D shape[1] != rank");
        }
    } else {
        throw std::runtime_error("unet::UNet: lora " + tag +
            ": lora_up has unsupported rank " + std::to_string(up.shape.size()));
    }
    // lora_down: (rank, in_dim) 2-D, or (rank, C_in, kH, kW) 4-D.
    if (down.shape.size() == 2) {
        s.in_dim = static_cast<int>(down.shape[1]);
    } else if (down.shape.size() == 4) {
        if (down.shape[2] == 1 && down.shape[3] == 1) {
            s.in_dim = static_cast<int>(down.shape[1]);
        } else {
            // 3x3 conv LoRA (rare). Flatten only if it reduces cleanly to W_cols.
            int64_t flat = down.shape[1] * down.shape[2] * down.shape[3];
            if (flat == static_cast<int64_t>(W_cols)) {
                s.in_dim = static_cast<int>(flat);
            } else {
                throw std::runtime_error("unet::UNet: lora " + tag +
                    ": 4-D lora_down with kH*kW > 1 doesn't reduce to base "
                    "weight cols (" + std::to_string(W_cols) + ")");
            }
        }
    } else {
        throw std::runtime_error("unet::UNet: lora " + tag +
            ": lora_down has unsupported rank " + std::to_string(down.shape.size()));
    }
    if (s.out_dim != W_rows || s.in_dim != W_cols) {
        throw std::runtime_error("unet::UNet: lora " + tag +
            ": (out_dim, in_dim) = (" + std::to_string(s.out_dim) + ", " +
            std::to_string(s.in_dim) + ") does not match base weight (" +
            std::to_string(W_rows) + ", " + std::to_string(W_cols) + ")");
    }
    return s;
}

}  // namespace

void UNet::apply_lora_delta(const std::string& target_path,
                            const st::TensorView& lora_down,
                            const st::TensorView& lora_up,
                            float scale_total) {
    if (finalized_) {
        fail("apply_lora_delta: called after finalize_weights(); LoRA must be "
             "applied before quantisation/finalisation (the base weights "
             "are no longer in memory)");
    }
    bt::Tensor* W = lora_target_(target_path);
    if (!W) {
        fail("apply_lora_delta: unknown target '" + target_path + "'");
    }
    if (W->size() == 0) {
        fail("apply_lora_delta: target '" + target_path + "' has no weights "
             "loaded (call load_weights first)");
    }
    const int W_rows = W->rows;
    const int W_cols = W->cols;
    const LoraShape s = resolve_lora_shape(lora_down, lora_up, W_rows, W_cols,
                                           target_path);

    // Upload at the compute dtype. lora_down: (rank, in_dim);
    // lora_up: (out_dim, rank).
    bt::Tensor down_g, up_g;
    upload_view_compute(lora_down, s.rank,    s.in_dim, down_g, target_path + ".lora_down");
    upload_view_compute(lora_up,   s.out_dim, s.rank,   up_g,   target_path + ".lora_up");

    // delta = up @ down — shape (out_dim, in_dim) = W shape.
    bt::Tensor delta;
    bt::matmul(up_g, down_g, delta);
    bt::scale_inplace(delta, scale_total);
    bt::add_inplace(*W, delta);
}

void UNet::quantize_weight_inplace_(bt::Tensor& W_fp16, QWeight& q) {
    if (W_fp16.size() == 0) return;             // nothing to quantise
    if (W_fp16.dtype != bt::Dtype::FP16) {
        fail("quantize_weight_inplace_: weight is not FP16");
    }
    const int out = W_fp16.rows;
    const int in  = W_fp16.cols;

    std::vector<uint16_t> host_fp16(static_cast<std::size_t>(out) *
                                    static_cast<std::size_t>(in));
    W_fp16.copy_to_host_fp16(host_fp16.data());

    std::vector<int8_t> host_int8(host_fp16.size());
    std::vector<float>  host_scales(static_cast<std::size_t>(out));
    bt::quantize_int8_per_row_host(host_fp16.data(), out, in,
                                   host_int8.data(), host_scales.data());

    // Upload INT8 weights: brotensor has no from_host path for INT8, so
    // stage the quantised bytes on the host then migrate to W_fp16's device.
    {
        bt::Tensor cpu_int8 = bt::Tensor::empty_on(
            bt::Device::CPU, out, in, bt::Dtype::INT8);
        std::memcpy(cpu_int8.host_raw_mut(), host_int8.data(),
                    static_cast<std::size_t>(out) * static_cast<std::size_t>(in));
        q.W_int8 = cpu_int8.to(W_fp16.device);
    }
    // Upload FP32 scales onto the same device as the INT8 weights.
    q.scales = brotensor::Tensor::from_host_on(W_fp16.device,
                                               host_scales.data(), out, 1);

    // Free original FP16 storage by replacing with default-constructed tensor.
    W_fp16 = bt::Tensor();
}

void UNet::finalize_weights() {
    if (finalized_) return;
    finalized_ = true;
    if (!cfg_.quantize_weights) return;
    if (conv_in_W_.size() == 0) fail("finalize_weights: weights not loaded");
    // INT8 (W8A16) weight quantization is GPU-only — the W8A16 ops have no CPU
    // fallback. On the CPU backend, skip quantization so every QWeight stays
    // inactive and all INT8 branches remain dead.
    if (conv_in_W_.device == brotensor::Device::CPU) {
        std::fprintf(stderr,
            "brodiffusion: INT8 weight quantization is GPU-only; "
            "ignoring --quantize-unet on the CPU backend.\n");
        return;
    }

    auto quant_resnet = [&](Resnet& r) {
        quantize_weight_inplace_(r.W1, r.W1_q);
        quantize_weight_inplace_(r.W2, r.W2_q);
        if (r.has_shortcut) quantize_weight_inplace_(r.Ws, r.Ws_q);
    };
    auto quant_transformer = [&](Transformer2D& t) {
        quantize_weight_inplace_(t.pi_W, t.pi_q);
        quantize_weight_inplace_(t.po_W, t.po_q);
        for (AttnFFN& blk : t.blocks) {
            quantize_weight_inplace_(blk.Wq1, blk.Wq1_q);
            quantize_weight_inplace_(blk.Wk1, blk.Wk1_q);
            quantize_weight_inplace_(blk.Wv1, blk.Wv1_q);
            quantize_weight_inplace_(blk.Wo1, blk.Wo1_q);
            quantize_weight_inplace_(blk.Wq2, blk.Wq2_q);
            quantize_weight_inplace_(blk.Wk2, blk.Wk2_q);
            quantize_weight_inplace_(blk.Wv2, blk.Wv2_q);
            quantize_weight_inplace_(blk.Wo2, blk.Wo2_q);
            quantize_weight_inplace_(blk.ff1_W, blk.ff1_q);
            quantize_weight_inplace_(blk.ff2_W, blk.ff2_q);
        }
    };

    for (DownBlock& d : down_blocks_) {
        for (Resnet& r : d.resnets) quant_resnet(r);
        for (Transformer2D& t : d.transformers) quant_transformer(t);
        if (d.has_downsampler) {
            quantize_weight_inplace_(d.downsampler.W, d.downsampler.W_q);
        }
    }
    quant_resnet(mid_.r0);
    quant_transformer(mid_.t);
    quant_resnet(mid_.r1);
    for (UpBlock& u : up_blocks_) {
        for (Resnet& r : u.resnets) quant_resnet(r);
        for (Transformer2D& t : u.transformers) quant_transformer(t);
        if (u.has_upsampler) {
            quantize_weight_inplace_(u.upsampler.W, u.upsampler.W_q);
        }
    }
}

int UNet::num_xattn_blocks() const {
    int n = 0;
    for (const DownBlock& d : down_blocks_) {
        if (d.has_attention) n += static_cast<int>(d.transformers.size());
    }
    n += 1;  // mid block always has one Transformer2D
    for (const UpBlock& u : up_blocks_) {
        if (u.has_attention) n += static_cast<int>(u.transformers.size());
    }
    return n;
}

std::vector<int> UNet::layer_strides() const {
    // SD1.5 schedule: down 0/1/2 (2 transformers each), mid (1), up 1/2/3
    // (3 transformers each). Strides relative to the top latent resolution.
    return {1,1, 2,2, 4,4, 8, 4,4,4, 2,2,2, 1,1,1};
}

void UNet::prime_xattn_cache(const bt::Tensor& ctx,
                             CrossAttnKVCache& cache) {
    if (conv_in_W_.size() == 0) fail("prime_xattn_cache: weights not loaded");
    const int n = num_xattn_blocks();
    cache.resize(static_cast<std::size_t>(n));

    int idx = 0;
    auto project = [&](const Transformer2D& t) {
        if (t.blocks.empty()) fail("internal: Transformer2D has no inner blocks");
        const AttnFFN& blk = t.blocks[0];
        CrossAttnKVCacheEntry& e = cache[static_cast<std::size_t>(idx++)];
        if (blk.Wk2_q.active()) {
            bt::flash_attention_project_kv_int8w_fp16(
                ctx,
                blk.Wk2_q.W_int8, blk.Wk2_q.scales, /*bk=*/nullptr,
                blk.Wv2_q.W_int8, blk.Wv2_q.scales, /*bv=*/nullptr,
                e.K, e.V);
        } else {
            bt::flash_attention_project_kv(
                ctx,
                blk.Wk2, /*bk=*/nullptr,
                blk.Wv2, /*bv=*/nullptr,
                e.K, e.V);
        }
    };

    for (const DownBlock& d : down_blocks_) {
        if (!d.has_attention) continue;
        for (const Transformer2D& t : d.transformers) project(t);
    }
    project(mid_.t);
    for (const UpBlock& u : up_blocks_) {
        if (!u.has_attention) continue;
        for (const Transformer2D& t : u.transformers) project(t);
    }

    if (idx != n) fail("internal: prime_xattn_cache traversal mismatch");
}

void UNet::forward_impl_(const bt::Tensor& sample,
                         int H, int W,
                         float timestep,
                         const float* gs_emb,
                         const bt::Tensor& encoder_hidden_states,
                         const CrossAttnKVCache* xattn_cache,
                         const std::vector<const bt::Tensor*>* attn_logit_biases,
                         CrossAttnTrace* trace_out,
                         bt::Tensor& out) {
    if (conv_in_W_.size() == 0) fail("forward: weights not loaded");

    // ── Optional GPU profiling (env: UNET_PROF=1) ───────────────────────────
    // CUDA-event based; compiled out entirely on non-CUDA builds.
#ifdef BROTENSOR_HAS_CUDA
    const bool prof_enabled = (std::getenv("UNET_PROF") != nullptr);
#else
    const bool prof_enabled = false;
#endif
    struct ProfBlock {
#ifdef BROTENSOR_HAS_CUDA
        cudaEvent_t s{}, e{};
#endif
        float ms_accum{0.0f};
    };
    ProfBlock pb_conv_in, pb_down_res, pb_down_xform, pb_down_samp;
    ProfBlock pb_mid, pb_up_res, pb_up_xform, pb_up_samp, pb_conv_out;
    ProfBlock* pb_all[] = {&pb_conv_in, &pb_down_res, &pb_down_xform, &pb_down_samp,
                           &pb_mid, &pb_up_res, &pb_up_xform, &pb_up_samp, &pb_conv_out};
    (void)pb_all;
#ifdef BROTENSOR_HAS_CUDA
    if (prof_enabled) {
        for (auto* p : pb_all) {
            cudaEventCreate(&p->s);
            cudaEventCreate(&p->e);
        }
    }
#endif
    auto prof_begin = [&]([[maybe_unused]] ProfBlock& p) {
#ifdef BROTENSOR_HAS_CUDA
        if (prof_enabled) cudaEventRecord(p.s);
#endif
    };
    auto prof_end = [&]([[maybe_unused]] ProfBlock& p) {
#ifdef BROTENSOR_HAS_CUDA
        if (prof_enabled) {
            cudaEventRecord(p.e);
            cudaEventSynchronize(p.e);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, p.s, p.e);
            p.ms_accum += ms;
        }
#endif
    };

    const int nb      = static_cast<int>(cfg_.block_out_channels.size());
    const int first_C = cfg_.block_out_channels.front();

    // ── 1. Build the time embedding ────────────────────────────────────────
    std::vector<float> sin_vals;
    compute_sinusoidal_emb(timestep, freq_dim_, sin_vals);
    freq_emb_ = detail::upload_host(sin_vals.data(), 1, freq_dim_);

    // LCM cond_proj: add projected guidance-scale embedding to the freq_emb
    // *before* linear_1. Matches diffusers' TimestepEmbedding.forward:
    //   if condition is not None: sample = sample + cond_proj(condition)
    //   sample = linear_1(sample) -> SiLU -> linear_2
    // cond_proj projects (cond_proj_dim) -> (freq_dim), not -> (time_embed_dim).
    if (gs_emb != nullptr) {
        if (cfg_.time_cond_proj_dim <= 0) fail("forward: internal: gs_emb without cond_proj");
        // diffusers' get_guidance_scale_embedding scales w by 1000 before the
        // sinusoidal embedding.
        std::vector<float> w_vals;
        compute_guidance_scale_emb((*gs_emb) * 1000.0f, cfg_.time_cond_proj_dim, w_vals);
        w_emb_ = detail::upload_host(w_vals.data(), 1, cfg_.time_cond_proj_dim);
        // cond_proj is bias-free: pass nullptr.
        detail::linear_batched(te_cond_W_, /*bias=*/nullptr, w_emb_, temb_cond_);
        bt::add_inplace(freq_emb_, temb_cond_);
    }

    detail::linear_batched(te_l1_W_, &te_l1_b_, freq_emb_, temb_a_);
    bt::silu_forward(temb_a_, temb_a_);
    detail::linear_batched(te_l2_W_, &te_l2_b_, temb_a_, temb_b_);
    // Master temb in temb_b_. Pre-compute SiLU once for reuse across resblocks.
    bt::silu_forward(temb_b_, temb_silu_);

    // Skip stack: each entry is a deep copy of the residual stream at push time.
    // Down-path xattn caches advance xattn_idx (kept in scope for mid + up).
    std::vector<bt::Tensor> skips;
    skips.reserve(static_cast<std::size_t>(nb) *
                  static_cast<std::size_t>(cfg_.layers_per_block + 1));
    int Hc = H, Wc = W;
    int xattn_idx = 0;
    auto cache_at = [&](int i) -> const CrossAttnKVCacheEntry* {
        return xattn_cache ? &(*xattn_cache)[static_cast<std::size_t>(i)] : nullptr;
    };
    auto trace_at = [&](int i) -> bt::Tensor* {
        return trace_out ? &(*trace_out)[static_cast<std::size_t>(i)] : nullptr;
    };
    auto bias_at = [&](int i) -> const bt::Tensor* {
        return attn_logit_biases ? (*attn_logit_biases)[static_cast<std::size_t>(i)]
                                 : nullptr;
    };

    // ── 2. conv_in: in_channels -> first_C ─────────────────────────────────
    x_ = sample.clone();
    prof_begin(pb_conv_in);
    apply_conv3x3_(conv_in_W_, conv_in_b_, cfg_.in_channels, first_C, H, W,
                   /*stride=*/1, /*pad=*/1, x_, y_);
    std::swap(x_, y_);
    prof_end(pb_conv_in);

    skips.push_back(x_.clone());

    // ── 3. down_blocks ─────────────────────────────────────────────────────
    for (int i = 0; i < nb; ++i) {
        DownBlock& d = down_blocks_[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg_.layers_per_block; ++j) {
            prof_begin(pb_down_res);
            apply_resnet_(d.resnets[static_cast<std::size_t>(j)], Hc, Wc, x_, y_);
            prof_end(pb_down_res);
            if (d.has_attention) {
                prof_begin(pb_down_xform);
                const int idx = xattn_idx++;
                apply_transformer_(d.transformers[static_cast<std::size_t>(j)],
                                   encoder_hidden_states,
                                   cache_at(idx),
                                   Hc, Wc, x_,
                                   trace_at(idx), bias_at(idx));
                prof_end(pb_down_xform);
            }
            skips.push_back(x_.clone());
        }
        if (d.has_downsampler) {
            // 3x3 stride-2 conv same-channels.
            prof_begin(pb_down_samp);
            if (d.downsampler.W_q.active()) {
                apply_conv3x3_q_(d.downsampler.W_q, d.downsampler.b,
                                 d.C_out, d.C_out, Hc, Wc,
                                 /*stride=*/2, /*pad=*/1, x_, y_);
            } else {
                apply_conv3x3_(d.downsampler.W, d.downsampler.b,
                               d.C_out, d.C_out, Hc, Wc,
                               /*stride=*/2, /*pad=*/1, x_, y_);
            }
            std::swap(x_, y_);
            prof_end(pb_down_samp);
            Hc /= 2;
            Wc /= 2;
            skips.push_back(x_.clone());
        }
    }

    // ── 4. mid_block: resnet -> transformer -> resnet ──────────────────────
    prof_begin(pb_mid);
    apply_resnet_(mid_.r0, Hc, Wc, x_, y_);
    {
        const int idx = xattn_idx++;
        apply_transformer_(mid_.t, encoder_hidden_states,
                           cache_at(idx),
                           Hc, Wc, x_,
                           trace_at(idx), bias_at(idx));
    }
    apply_resnet_(mid_.r1, Hc, Wc, x_, y_);
    prof_end(pb_mid);

    // ── 5. up_blocks ───────────────────────────────────────────────────────
    for (int i = 0; i < nb; ++i) {
        UpBlock& u = up_blocks_[static_cast<std::size_t>(i)];
        const int layers = cfg_.layers_per_block + 1;
        for (int j = 0; j < layers; ++j) {
            // Channel-axis concat with popped skip.
            bt::Tensor skip = std::move(skips.back());
            skips.pop_back();
            const int C_x_now    = x_.cols / (Hc * Wc);
            const int C_skip_now = skip.cols / (Hc * Wc);
            const std::vector<int> C_parts    = {C_x_now, C_skip_now};
            const std::vector<const bt::Tensor*> parts = {&x_, &skip};
            bt::concat_nchw_channels(parts, /*N=*/1, Hc, Wc, C_parts, cat_buf_);
            std::swap(x_, cat_buf_);

            prof_begin(pb_up_res);
            apply_resnet_(u.resnets[static_cast<std::size_t>(j)], Hc, Wc, x_, y_);
            prof_end(pb_up_res);
            if (u.has_attention) {
                prof_begin(pb_up_xform);
                const int idx = xattn_idx++;
                apply_transformer_(u.transformers[static_cast<std::size_t>(j)],
                                   encoder_hidden_states,
                                   cache_at(idx),
                                   Hc, Wc, x_,
                                   trace_at(idx), bias_at(idx));
                prof_end(pb_up_xform);
            }
        }
        if (u.has_upsampler) {
            prof_begin(pb_up_samp);
            bt::upsample_nearest_2x(x_, 1, u.C_out, Hc, Wc, y_);
            if (u.upsampler.W_q.active()) {
                apply_conv3x3_q_(u.upsampler.W_q, u.upsampler.b,
                                 u.C_out, u.C_out, 2 * Hc, 2 * Wc,
                                 /*stride=*/1, /*pad=*/1, y_, x_);
            } else {
                apply_conv3x3_(u.upsampler.W, u.upsampler.b,
                               u.C_out, u.C_out, 2 * Hc, 2 * Wc,
                               /*stride=*/1, /*pad=*/1, y_, x_);
            }
            prof_end(pb_up_samp);
            Hc *= 2;
            Wc *= 2;
        }
    }

    // ── 6. conv_norm_out -> SiLU -> conv_out ───────────────────────────────
    prof_begin(pb_conv_out);
    bt::group_norm_forward(x_, norm_out_g_, norm_out_b_,
                               1, first_C, Hc, Wc, cfg_.norm_num_groups, cfg_.eps,
                               y_);
    bt::silu_forward(y_, y_);
    apply_conv3x3_(conv_out_W_, conv_out_b_,
                   first_C, cfg_.out_channels, Hc, Wc,
                   /*stride=*/1, /*pad=*/1, y_, out);
    prof_end(pb_conv_out);

    if (prof_enabled) {
        std::fprintf(stderr,
            "[UNET_PROF] conv_in=%.3f down_res=%.3f down_xform=%.3f down_samp=%.3f "
            "mid=%.3f up_res=%.3f up_xform=%.3f up_samp=%.3f conv_out=%.3f "
            "(ms, sum=%.3f)\n",
            pb_conv_in.ms_accum, pb_down_res.ms_accum, pb_down_xform.ms_accum,
            pb_down_samp.ms_accum, pb_mid.ms_accum, pb_up_res.ms_accum,
            pb_up_xform.ms_accum, pb_up_samp.ms_accum, pb_conv_out.ms_accum,
            pb_conv_in.ms_accum + pb_down_res.ms_accum + pb_down_xform.ms_accum +
            pb_down_samp.ms_accum + pb_mid.ms_accum + pb_up_res.ms_accum +
            pb_up_xform.ms_accum + pb_up_samp.ms_accum + pb_conv_out.ms_accum);
#ifdef BROTENSOR_HAS_CUDA
        for (auto* p : pb_all) {
            cudaEventDestroy(p->s);
            cudaEventDestroy(p->e);
        }
#endif
    }
}

}  // namespace brodiffusion::unet
