#include "brodiffusion/detail/unet_blocks.h"

#include "brodiffusion/fused_resblock.h"
#include "brodiffusion/fused_transformer.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <stdexcept>
#include <utility>

namespace brodiffusion::unet::detail {

namespace bt = ::brotensor;

void apply_conv3x3(const bt::Tensor& W, const bt::Tensor& b,
                   int C_in, int C_out, int H, int W_,
                   int stride, int pad,
                   const bt::Tensor& in, bt::Tensor& out) {
    bt::conv2d_forward(in, W, &b,
                           /*N=*/1, C_in, H, W_,
                           C_out, /*kH=*/3, /*kW=*/3,
                           stride, stride, pad, pad, /*dil=*/1, 1,
                           out);
}

void apply_conv3x3_q(const QWeight& Wq, const bt::Tensor& b,
                     int C_in, int C_out, int H, int W_,
                     int stride, int pad,
                     const bt::Tensor& in, bt::Tensor& out) {
    bt::conv2d_int8w_fp16_forward(in, Wq.W_int8, Wq.scales, &b,
                                      /*N=*/1, C_in, H, W_,
                                      C_out, /*kH=*/3, /*kW=*/3,
                                      stride, stride, pad, pad, /*dil=*/1, 1,
                                      /*groups=*/1, out);
}

void apply_resnet(const Resnet& r, int H, int W,
                  bt::Tensor& x, bt::Tensor& tmp,
                  BlockScratch& s,
                  int norm_num_groups, float groupnorm_eps) {
    // Per-resblock time-emb projection: silu(temb) -> Linear -> (1, C_out).
    brodiffusion::detail::linear_batched(r.temb_W, &r.temb_b, s.temb_silu, s.temb_proj);

    if (r.W1_q.active()) {
        const QWeight* skip_q = r.has_shortcut ? &r.Ws_q : nullptr;
        const bt::Tensor* skip_b = r.has_shortcut ? &r.bs : nullptr;
        const bt::Tensor* skip_W_int8   = skip_q ? &skip_q->W_int8 : nullptr;
        const bt::Tensor* skip_W_scales = skip_q ? &skip_q->scales : nullptr;
        brodiffusion::fused_resblock_forward(x,
                                             r.n1g, r.n1b,
                                             r.W1_q.W_int8, r.W1_q.scales, r.b1,
                                             s.temb_proj,
                                             r.n2g, r.n2b,
                                             r.W2_q.W_int8, r.W2_q.scales, r.b2,
                                             skip_W_int8, skip_W_scales, skip_b,
                                             r.C_in, r.C_out, H, W,
                                             norm_num_groups, groupnorm_eps,
                                             tmp);
    } else {
        const bt::Tensor* skip_W = r.has_shortcut ? &r.Ws : nullptr;
        const bt::Tensor* skip_b = r.has_shortcut ? &r.bs : nullptr;
        brodiffusion::fused_resblock_forward(x,
                                             r.n1g, r.n1b,
                                             r.W1, r.b1,
                                             s.temb_proj,
                                             r.n2g, r.n2b,
                                             r.W2, r.b2,
                                             skip_W, skip_b,
                                             r.C_in, r.C_out, H, W,
                                             norm_num_groups, groupnorm_eps,
                                             tmp);
    }
    std::swap(x, tmp);
}

void apply_transformer(const Transformer2D& t,
                       const bt::Tensor& ctx,
                       const CrossAttnKVCacheEntry* cache_entry,
                       int H, int W, bt::Tensor& x,
                       BlockScratch& s,
                       int norm_num_groups, float layernorm_eps,
                       bt::Tensor* trace_out_entry,
                       const bt::Tensor* attn_logit_bias) {
    // Trace mode is incompatible with the cached K/V fast path because
    // brotensor's cross_attention_forward_with_attn reprojects K/V from
    // ctx every call. forward_trace() already ensures no cache is passed.
    if (trace_out_entry != nullptr && cache_entry != nullptr) {
        throw std::runtime_error("unet::detail::apply_transformer: "
            "trace_out_entry and cache_entry are mutually exclusive");
    }
    const int C  = t.C;
    const int H_heads = t.num_heads;

    // 1. residual is `x` itself; we add the post-transformer result back in.
    // 2. GroupNorm in NCHW. Diffusers' Transformer2DModel uses eps=1e-6 on
    //    this outer GroupNorm (distinct from the 1e-5 used on the ResNet
    //    GroupNorms); hard-code it here rather than relying on cfg_.eps.
    bt::group_norm_forward(x, t.gn_g, t.gn_b,
                               1, C, H, W, norm_num_groups, 1e-6f, s.gn);

    // 3. NCHW -> (L, C) sequence.
    bt::nchw_to_sequence(s.gn, 1, C, H, W, s.seq);

    // 4. proj_in: 1x1 conv ≡ Linear over C.
    if (t.pi_q.active()) {
        bt::linear_forward_batched_int8w_fp16(
            t.pi_q.W_int8, t.pi_q.scales, &t.pi_b, s.seq, s.proj_in_seq);
    } else {
        brodiffusion::detail::linear_batched(t.pi_W, &t.pi_b, s.seq, s.proj_in_seq);
    }

    // 5. transformer blocks (always 1 for SD1.5).
    s.tseq = s.proj_in_seq.clone();
    for (const AttnFFN& blk : t.blocks) {
        // ── self-attention (Q/K/V bias-less, Wo biased) ───────────────────
        brodiffusion::detail::layernorm_batched(s.tseq, blk.n1g, blk.n1b, s.ln, layernorm_eps);
        if (blk.Wq1_q.active()) {
            bt::flash_attention_qkvo_int8w_fp16(
                s.ln, /*Ctx=*/nullptr,
                blk.Wq1_q.W_int8, blk.Wq1_q.scales, /*bq=*/nullptr,
                blk.Wk1_q.W_int8, blk.Wk1_q.scales, /*bk=*/nullptr,
                blk.Wv1_q.W_int8, blk.Wv1_q.scales, /*bv=*/nullptr,
                blk.Wo1_q.W_int8, blk.Wo1_q.scales, &blk.bo1,
                /*d_mask=*/nullptr, H_heads, /*causal=*/false,
                s.attn_proj);
        } else {
            bt::flash_attention_qkvo_forward(
                s.ln, /*Ctx=*/nullptr,
                blk.Wq1, /*bq=*/nullptr,
                blk.Wk1, /*bk=*/nullptr,
                blk.Wv1, /*bv=*/nullptr,
                blk.Wo1, &blk.bo1,
                /*d_mask=*/nullptr, H_heads, /*causal=*/false,
                s.attn_proj);
        }
        brodiffusion::add_inplace_vec(s.tseq, s.attn_proj);

        // ── cross-attention (K, V from `ctx` — possibly cached) ───────────
        brodiffusion::detail::layernorm_batched(s.tseq, blk.n2g, blk.n2b, s.ln, layernorm_eps);
        if (trace_out_entry) {
            // Trace path: brotensor's cross_attention_forward_with_attn
            // writes the head-averaged softmax map to AttnAvg. No Wo
            // bias is supported by that op, so we manually add bo2 after.
            // (forward_trace already guards against INT8.)
            bt::cross_attention_forward_with_attn(
                s.ln, ctx,
                blk.Wq2, blk.Wk2, blk.Wv2, blk.Wo2,
                /*d_mask=*/nullptr,
                attn_logit_bias,
                H_heads,
                s.attn_proj, *trace_out_entry);
            // Add output bias bo2 (per-column broadcast across rows of attn_proj).
            brodiffusion::add_inplace_row_bias(s.attn_proj, blk.bo2);
        } else if (cache_entry) {
            // K/V already projected from `ctx` upstream — skip the two
            // per-step matmuls and feed the cached buffers straight in.
            if (blk.Wq2_q.active()) {
                bt::flash_attention_q_with_kv_cached_int8w_fp16(
                    s.ln, cache_entry->K, cache_entry->V,
                    blk.Wq2_q.W_int8, blk.Wq2_q.scales, /*bq=*/nullptr,
                    blk.Wo2_q.W_int8, blk.Wo2_q.scales, &blk.bo2,
                    /*d_mask=*/nullptr, H_heads, /*causal=*/false,
                    s.attn_proj);
            } else {
                bt::flash_attention_q_with_kv_cached_forward(
                    s.ln, cache_entry->K, cache_entry->V,
                    blk.Wq2, /*bq=*/nullptr,
                    blk.Wo2, &blk.bo2,
                    /*d_mask=*/nullptr, H_heads, /*causal=*/false,
                    s.attn_proj);
            }
        } else {
            if (blk.Wq2_q.active()) {
                bt::flash_attention_qkvo_int8w_fp16(
                    s.ln, &ctx,
                    blk.Wq2_q.W_int8, blk.Wq2_q.scales, /*bq=*/nullptr,
                    blk.Wk2_q.W_int8, blk.Wk2_q.scales, /*bk=*/nullptr,
                    blk.Wv2_q.W_int8, blk.Wv2_q.scales, /*bv=*/nullptr,
                    blk.Wo2_q.W_int8, blk.Wo2_q.scales, &blk.bo2,
                    /*d_mask=*/nullptr, H_heads, /*causal=*/false,
                    s.attn_proj);
            } else {
                bt::flash_attention_qkvo_forward(
                    s.ln, &ctx,
                    blk.Wq2, /*bq=*/nullptr,
                    blk.Wk2, /*bk=*/nullptr,
                    blk.Wv2, /*bv=*/nullptr,
                    blk.Wo2, &blk.bo2,
                    /*d_mask=*/nullptr, H_heads, /*causal=*/false,
                    s.attn_proj);
            }
        }
        brodiffusion::add_inplace_vec(s.tseq, s.attn_proj);

        // ── feed-forward (GEGLU) ──────────────────────────────────────────
        brodiffusion::detail::layernorm_batched(s.tseq, blk.n3g, blk.n3b, s.ln, layernorm_eps);
        // Fused FF1 + exact-GEGLU: skips the (B, 2*D) intermediate of FF1.
        // SD1.5's BasicTransformerBlock uses F.gelu(approximate=False).
        if (blk.ff1_q.active()) {
            brodiffusion::fused_linear_geglu(
                s.ln, blk.ff1_q.W_int8, blk.ff1_q.scales, blk.ff1_b, s.ff_act);
        } else {
            brodiffusion::fused_linear_geglu(s.ln, blk.ff1_W, blk.ff1_b, s.ff_act);
        }
        if (blk.ff2_q.active()) {
            bt::linear_forward_batched_int8w_fp16(
                blk.ff2_q.W_int8, blk.ff2_q.scales, &blk.ff2_b, s.ff_act, s.ff_out);
        } else {
            brodiffusion::detail::linear_batched(blk.ff2_W, &blk.ff2_b, s.ff_act, s.ff_out);
        }
        brodiffusion::add_inplace_vec(s.tseq, s.ff_out);
    }

    // 6. proj_out: 1x1 conv ≡ Linear.
    if (t.po_q.active()) {
        bt::linear_forward_batched_int8w_fp16(
            t.po_q.W_int8, t.po_q.scales, &t.po_b, s.tseq, s.proj_out_seq);
    } else {
        brodiffusion::detail::linear_batched(t.po_W, &t.po_b, s.tseq, s.proj_out_seq);
    }

    // 7. seq -> NCHW.
    bt::sequence_to_nchw(s.proj_out_seq, 1, C, H, W, s.proj_out_nchw);

    // 8. residual add.
    bt::add_inplace(x, s.proj_out_nchw);
}

}  // namespace brodiffusion::unet::detail
