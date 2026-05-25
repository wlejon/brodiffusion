#pragma once

// UNet block-level definitions shared between UNet and ControlNet.
//
// The block structs (QWeight, Resnet, AttnFFN, Transformer2D, SampleConv,
// DownBlock, MidBlock, UpBlock) and the per-block forward helpers
// (apply_resnet, apply_transformer, apply_conv3x3, apply_conv3x3_q) used to
// live inside UNet's private section. Phase D1 lifted them into this header
// so a future ControlNet module can reuse the exact same code paths without
// duplication.
//
// Behavioral contract: all helpers below are character-for-character the same
// brotensor op sequence the UNet used before the extraction; UNet behavior
// is bit-identical post-refactor. ControlNet is expected to reuse these in
// the same way: hold a BlockScratch instance, pre-compute `temb_silu` once
// per forward (`bt::silu_forward(temb, scratch.temb_silu)`), then call the
// helpers per layer.

#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::unet::detail {

// Paired INT8 weight + per-output-row FP32 scales. .W_int8.size() == 0 means
// the layer is still using its FP16 weight (set on finalize_weights when
// quantize_weights is enabled).
struct QWeight {
    brotensor::Tensor W_int8;   // INT8 (out, in)
    brotensor::Tensor scales;   // FP32 (out, 1)
    bool active() const { return W_int8.size() > 0; }
};

struct Resnet {
    brotensor::Tensor n1g, n1b, W1, b1;
    brotensor::Tensor temb_W, temb_b;
    brotensor::Tensor n2g, n2b, W2, b2;
    brotensor::Tensor Ws, bs;
    // INT8 counterparts (populated by finalize_weights when enabled).
    QWeight W1_q, W2_q, Ws_q;
    bool has_shortcut = false;
    int  C_in = 0, C_out = 0;
};

struct AttnFFN {
    brotensor::Tensor n1g, n1b;
    brotensor::Tensor Wq1, Wk1, Wv1, Wo1, bo1;
    brotensor::Tensor n2g, n2b;
    brotensor::Tensor Wq2, Wk2, Wv2, Wo2, bo2;
    brotensor::Tensor n3g, n3b;
    brotensor::Tensor ff1_W, ff1_b;
    brotensor::Tensor ff2_W, ff2_b;
    // INT8 counterparts.
    QWeight Wq1_q, Wk1_q, Wv1_q, Wo1_q;
    QWeight Wq2_q, Wk2_q, Wv2_q, Wo2_q;
    QWeight ff1_q, ff2_q;
};

struct Transformer2D {
    brotensor::Tensor gn_g, gn_b;
    brotensor::Tensor pi_W, pi_b;
    brotensor::Tensor po_W, po_b;
    QWeight pi_q, po_q;
    std::vector<AttnFFN> blocks;
    int  C = 0;
    int  num_heads = 0;
};

struct SampleConv {
    brotensor::Tensor W, b;
    QWeight W_q;
};

struct DownBlock {
    std::vector<Resnet>        resnets;
    std::vector<Transformer2D> transformers;
    SampleConv                 downsampler;
    bool has_attention   = false;
    bool has_downsampler = false;
    int  C_out = 0;
};

struct MidBlock {
    Resnet         r0, r1;
    Transformer2D  t;
};

struct UpBlock {
    std::vector<Resnet>        resnets;
    std::vector<Transformer2D> transformers;
    SampleConv                 upsampler;
    bool has_attention = false;
    bool has_upsampler = false;
    int  C_out = 0;
};

// Cached cross-attention K/V projection for a single context tensor — one
// (K, V) pair per Transformer2D block at the compute dtype. See UNet for
// the priming routine and traversal order.
struct CrossAttnKVCacheEntry {
    brotensor::Tensor K;  // (Lk, C)
    brotensor::Tensor V;  // (Lk, C)
};

// Scratch tensors used by apply_resnet / apply_transformer. Bundled into one
// owner so ControlNet can construct its own instance and call the helpers
// without UNet involvement.
//
// Lifetime: each independent forward path (UNet, ControlNet) needs its own
// BlockScratch instance. A single forward call reuses (and resizes) the
// scratch tensors across all layers, so a long-running pipeline allocates
// device memory once and amortizes thereafter.
//
// temb_silu is the SiLU(time_emb) shared across every resblock in one
// forward. The driver (UNet::forward_impl_) writes it once before the down
// pass and the apply_resnet helpers read it from here.
struct BlockScratch {
    // Per-resblock time-emb projection scratch (silu(temb) shared, per-block
    // projected output).
    brotensor::Tensor temb_silu, temb_proj;
    // Transformer2D scratch — GroupNorm, NCHW<->seq, LayerNorm output,
    // proj_in/out, transformer-block intermediate sequence, attention output,
    // GEGLU activations.
    brotensor::Tensor gn, seq, proj_in_seq, tseq, ln;
    brotensor::Tensor attn_proj;
    brotensor::Tensor ff_act, ff_out;
    brotensor::Tensor proj_out_seq, proj_out_nchw;
};

// 3x3 conv2d with bias. Stride/pad symmetric. N=1 batch hard-coded.
void apply_conv3x3(const brotensor::Tensor& W,
                   const brotensor::Tensor& b,
                   int C_in, int C_out, int H, int W_,
                   int stride, int pad,
                   const brotensor::Tensor& in,
                   brotensor::Tensor& out);

// INT8 weight-only variant of apply_conv3x3.
void apply_conv3x3_q(const QWeight& Wq,
                     const brotensor::Tensor& b,
                     int C_in, int C_out, int H, int W_,
                     int stride, int pad,
                     const brotensor::Tensor& in,
                     brotensor::Tensor& out);

// ResnetBlock2D forward. Uses scratch.temb_silu (must be pre-computed by the
// caller for this forward) and scratch.temb_proj. `tmp` is the caller's
// auxiliary buffer used by the fused resblock op; after the call, x and tmp
// are swapped so the new result is in `x`.
void apply_resnet(const Resnet& r, int H, int W,
                  brotensor::Tensor& x, brotensor::Tensor& tmp,
                  BlockScratch& s,
                  int norm_num_groups, float groupnorm_eps);

// Transformer2D forward. If `cache_entry` is non-null, its (K, V) replace
// the cross-attn K/V projections (must have been primed against the same
// `ctx`). Trace plumbing: when `trace_out_entry` is non-null, attn2 is
// routed through cross_attention_forward_with_attn and the head-averaged
// (Lq, Lk) softmax map is written there; `attn_logit_bias` is an optional
// FP32 (Lq, Lk) pre-softmax bias passed straight through.
void apply_transformer(const Transformer2D& t,
                       const brotensor::Tensor& ctx,
                       const CrossAttnKVCacheEntry* cache_entry,
                       int H, int W,
                       brotensor::Tensor& x,
                       BlockScratch& s,
                       int norm_num_groups, float layernorm_eps,
                       brotensor::Tensor* trace_out_entry = nullptr,
                       const brotensor::Tensor* attn_logit_bias = nullptr);

// ── Weight loaders ─────────────────────────────────────────────────────────
//
// Lifted from UNet's private section so ControlNet can reuse them verbatim.
// Behavior is bit-identical to UNet's pre-Phase-D2 private helpers: each
// reads tensors from `f` under the given `prefix` (which must end in '.'),
// uploads them at the brodiffusion compute dtype, and fills in the matching
// struct fields. Throws std::runtime_error on missing tensors or shape
// mismatches (tagged "unet::detail::load_resnet" / "...load_transformer").
//
// `time_embed_dim` is the master temb dim (= cfg.block_out_channels[0] *
// time_embed_dim_mult on the standard UNet); the resnet's time_emb_proj
// weight is shaped (C_out, time_embed_dim).
//
// `cross_attention_dim` is the K/V context width for cross-attention
// (CLIP-L = 768 for SD1.5).
void load_resnet(const brotensor::safetensors::File& f,
                 const std::string& prefix,
                 int C_in, int C_out, int time_embed_dim,
                 Resnet& r);

void load_transformer(const brotensor::safetensors::File& f,
                      const std::string& prefix,
                      int C, int num_heads, int cross_attention_dim,
                      Transformer2D& t);

}  // namespace brodiffusion::unet::detail
