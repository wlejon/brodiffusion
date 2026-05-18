#pragma once

// UNet2DConditionModel for SD1.5 (the noise-prediction backbone).
//
// Inference-only, FP16, batch size N = 1 (the forward path hard-codes N=1
// throughout; the underlying brotensor ops support N > 1, so generalizing
// is mostly a matter of plumbing the batch dim through). Architecture mirrors Hugging
// Face's diffusers `UNet2DConditionModel` defaults for SD1.5:
//
//   block_out_channels = [320, 640, 1280, 1280]
//   layers_per_block   = 2
//   norm_num_groups    = 32
//   cross_attention_dim = 768            (CLIP ViT-L/14 hidden dim)
//   attention_head_dim  = 8              (actually means num_heads — diffusers
//                                          backwards-compat quirk; head_dim is
//                                          block_out_channels[i] / 8)
//   time_embed_dim      = 4 * block_out_channels[0] = 1280
//
// Top-level structure:
//   conv_in (in_channels=4 -> 320, 3x3)
//   time embedding: sinusoidal(t, 320) -> Linear(320,1280) -> SiLU -> Linear(1280,1280)
//   4 down_blocks:
//     [0..nb-2]: CrossAttnDownBlock — layers_per_block × (ResBlock + Transformer2D),
//                then a stride-2 3x3 conv downsampler.
//     [nb-1]:    DownBlock          — layers_per_block × ResBlock, no downsampler.
//   mid_block: ResBlock -> Transformer2D -> ResBlock
//   4 up_blocks (mirrors down, reversed channel order):
//     [0]:       UpBlock            — (layers_per_block+1) × ResBlock, upsample
//     [1..nb-1]: CrossAttnUpBlock   — (layers_per_block+1) × (ResBlock + Transformer2D),
//                                     upsample, EXCEPT the last block has no upsampler.
//     Each up-block layer first concats the latent with a popped skip activation
//     along the channel axis, then runs ResBlock(+Transformer).
//   conv_norm_out (GroupNorm) -> SiLU -> conv_out (320 -> out_channels=4, 3x3)
//
// Transformer2D (BasicTransformerBlock × 1, SD1.5 uses_linear_projection=False):
//   GroupNorm -> 1x1 proj_in -> seq layout
//     LayerNorm -> self-attn (no Q/K/V bias, biased Wo, no causal mask)
//     LayerNorm -> cross-attn (no Q/K/V bias, biased Wo, ctx = text encoder out)
//     LayerNorm -> Linear(D, 8D) -> GEGLU -> Linear(4D, D)   (FF, all biased)
//   1x1 proj_out -> NCHW layout -> residual add
//
// ResBlock (ResnetBlock2D):
//   GroupNorm(C_in) -> SiLU -> Conv3x3 -> + time_emb_proj(SiLU(temb))
//   GroupNorm(C_out) -> SiLU -> Conv3x3
//   + (1x1 conv shortcut if C_in != C_out)
//
// Caller is responsible for cuda_sync() before reading the output. All
// weights and activations are FP16 — convert host-side if your checkpoint
// ships FP32 weights.

#include "brotensor/device_buffer.h"
#include "brotensor/tensor.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace brodiffusion::safetensors { class File; struct TensorView; }

namespace brodiffusion::unet {

struct UNetConfig {
    int in_channels   = 4;
    int out_channels  = 4;
    std::vector<int> block_out_channels = {320, 640, 1280, 1280};
    int layers_per_block = 2;
    int norm_num_groups  = 32;
    float eps            = 1e-5f;
    int cross_attention_dim = 768;
    // Per-level num_heads is computed as block_out_channels[i] / attention_head_dim.
    int attention_head_dim  = 8;
    // time_embed_dim = block_out_channels[0] * time_embed_dim_mult.
    int time_embed_dim_mult = 4;

    // Optional guidance-scale-embedding projection dimension. When > 0 the
    // UNet expects the checkpoint to contain `time_embedding.cond_proj.weight`
    // of shape (time_embed_dim, time_cond_proj_dim), no bias, and the caller
    // must use the forward() overload that takes a `guidance_scale_embedding`
    // float. When 0 (default — vanilla SD1.5) no cond_proj weight is loaded
    // and the existing forward() overloads are used.
    //
    // LCM-distilled checkpoints (e.g. SimianLuo/LCM_Dreamshaper_v7) ship with
    // time_cond_proj_dim = 256; the projected guidance-scale embedding is
    // added to the time embedding *after* linear_1 and *before* the SiLU,
    // matching diffusers' TimestepEmbedding.forward.
    int time_cond_proj_dim = 0;

    // When true, finalize_weights() converts the big UNet weights (ResBlock
    // 3x3/1x1 conv weights, attention Q/K/V/O projections, transformer
    // proj_in/proj_out, FF1/FF2 linears, down/upsampler 3x3 convs) from FP16
    // to INT8 weight-only quantisation (W8A16). Small/sensitive layers
    // (conv_in, conv_out, GroupNorm gain/bias, time embedding, cond_proj,
    // per-resblock time_emb_proj, all biases) stay FP16. Per-output-row
    // symmetric scales (matching brotensor::quantize_int8_per_row_host).
    bool quantize_weights = false;
};

class UNet {
public:
    explicit UNet(const UNetConfig& cfg);
    ~UNet();

    UNet(const UNet&) = delete;
    UNet& operator=(const UNet&) = delete;
    UNet(UNet&&) noexcept = default;
    UNet& operator=(UNet&&) noexcept = default;

    // Load all weights from a safetensors file. Names follow Hugging Face's
    // `UNet2DConditionModel` convention. SD1.5 diffusers exports use an empty
    // prefix; SD1.5 full checkpoints typically use "model.diffusion_model.".
    //
    // Every tensor must be FP16. Throws std::runtime_error on missing names,
    // shape mismatches, or dtype mismatch.
    void load_weights(const brodiffusion::safetensors::File& f,
                      const std::string& prefix = "");

    // Forward pass.
    //   sample:                (1, in_channels * H * W) FP16 — noisy latent
    //   H, W:                  spatial dims of `sample`. H and W must each be
    //                          divisible by 2^(num_blocks-1) (typically 8).
    //   timestep:              continuous timestep value (typically in [0, 1000)).
    //   encoder_hidden_states: (L_text, cross_attention_dim) FP16, e.g. CLIP output.
    //   out:                   (1, out_channels * H * W) FP16, resized as needed.
    void forward(const brotensor::GpuTensor& sample,
                 int H, int W,
                 float timestep,
                 const brotensor::GpuTensor& encoder_hidden_states,
                 brotensor::GpuTensor& out);

    // Cached cross-attention K/V for a single context tensor — one (K, V) pair
    // per Transformer2D block, FP16, layout matching
    // brotensor::flash_attention_forward_gpu's K/V args (Lk, C). The text
    // context is fixed across all denoising steps so projecting K/V once per
    // generate() per CFG branch eliminates 16 × steps × 2 redundant matmuls.
    struct CrossAttnKVCacheEntry {
        brotensor::GpuTensor K;  // (Lk, C)
        brotensor::GpuTensor V;  // (Lk, C)
    };
    using CrossAttnKVCache = std::vector<CrossAttnKVCacheEntry>;

    // Head-averaged cross-attention softmax map per Transformer2D block, in the
    // same traversal order the forward pass visits them. Each entry has shape
    // (Lq, Lk) FP16, where Lq is the layer's spatial token count (H*W at that
    // resolution) and Lk is the text-context length (77 for SD1.5/CLIP). The
    // trace-mode forward overload below populates this vector for downstream
    // research consumers (cross-attention tree search, attention scoring).
    using CrossAttnTrace = std::vector<brotensor::GpuTensor>;

    // Populate `cache` with one (K, V) pair per Transformer2D block (in the
    // same traversal order the forward pass visits them: down blocks,
    // mid block, up blocks). `cache` is resized as needed.
    void prime_xattn_cache(const brotensor::GpuTensor& ctx,
                           CrossAttnKVCache& cache);

    // Variant of forward that uses a pre-primed K/V cache built from the
    // same `encoder_hidden_states` (or, more precisely, from a ctx with the
    // exact tokens the cache was primed against). Cross-attention layers
    // skip the K/V projection step; self-attention is unchanged.
    void forward(const brotensor::GpuTensor& sample,
                 int H, int W,
                 float timestep,
                 const brotensor::GpuTensor& encoder_hidden_states,
                 const CrossAttnKVCache& xattn_cache,
                 brotensor::GpuTensor& out);

    // LCM guidance-scale-embedding forward.
    //
    // Only valid when the UNet was built with `time_cond_proj_dim > 0`. The
    // raw guidance scale `w` (e.g. 7.5) is passed in; this overload internally
    // computes the sinusoidal `w_emb = sinusoidal(w * 1000, time_cond_proj_dim)`
    // matching diffusers' `get_guidance_scale_embedding`, then projects it
    // through the loaded `time_embedding.cond_proj.weight` and adds it to the
    // time embedding before the SiLU between linear_1 and linear_2.
    //
    // The non-LCM overloads above will throw if the UNet was built with
    // `time_cond_proj_dim > 0`; this overload will throw if it was 0.
    void forward(const brotensor::GpuTensor& sample,
                 int H, int W,
                 float timestep,
                 float guidance_scale_embedding,
                 const brotensor::GpuTensor& encoder_hidden_states,
                 const CrossAttnKVCache& xattn_cache,
                 brotensor::GpuTensor& out);

    // Trace-mode forward. Routes each cross-attention (`attn2`) call through
    // brotensor::cross_attention_forward_with_attn_gpu so the head-averaged
    // softmax map can be observed and an optional per-layer FP32 pre-softmax
    // logit bias can be injected. Self-attention (`attn1`) still uses the
    // fast flash path.
    //
    // `trace_out`, if non-null, is resized to num_xattn_blocks() and each
    // entry is filled with the head-averaged (Lq, Lk) attention map for
    // that layer.
    //
    // `attn_logit_biases`, if non-null, must have length num_xattn_blocks();
    // entry `i` is either null (no bias) or a (Lq_i, Lk) FP32 GpuTensor added
    // to the scaled QKᵀ scores before softmax at layer i.
    //
    // Trace mode currently does NOT use the K/V cache (the with-attn brotensor
    // op has no cached variant yet) — K/V are reprojected from `ctx` at every
    // layer. Acceptable cost for experiment-mode tree search; if it bottlenecks
    // we'll ask brotensor for `..._q_with_kv_cached_with_attn_gpu`.
    //
    // INT8 (quantize_weights) is not supported in trace mode and will throw.
    void forward_trace(const brotensor::GpuTensor& sample,
                       int H, int W,
                       float timestep,
                       const brotensor::GpuTensor& encoder_hidden_states,
                       const std::vector<const brotensor::GpuTensor*>* attn_logit_biases,
                       CrossAttnTrace* trace_out,
                       brotensor::GpuTensor& out);

    // Number of Transformer2D (cross-attn) blocks in the model — matches the
    // size of any cache returned by prime_xattn_cache.
    int num_xattn_blocks() const;

    // Fold a LoRA delta into the base FP16 weight identified by `target_path`.
    // `target_path` is a diffusers path *within* the UNet (no "unet." prefix),
    // e.g. "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q" or
    // "ff.net.0.proj". Throws if the path is unknown or shapes don't match.
    //
    // The merge is in-place:
    //     W += scale_total * (lora_up @ lora_down)
    // where `scale_total = (alpha / rank) * user_scale` is computed by the
    // caller. `lora_down` is (rank, in_dim) and `lora_up` is (out_dim, rank);
    // F16 or F32 source are both accepted (F32 converted host-side, matching
    // the rest of the UNet loader).
    //
    // Must be called *after* `load_weights()`. May be called more than once
    // (e.g. to stack multiple LoRAs).
    void apply_lora_delta(const std::string& target_path,
                          const brodiffusion::safetensors::TensorView& lora_down,
                          const brodiffusion::safetensors::TensorView& lora_up,
                          float scale_total);

    // Finalize the UNet weights for inference. When the config has
    // `quantize_weights == true` this converts the big linear / conv weights
    // listed in UNetConfig::quantize_weights to INT8 (per-output-row symmetric
    // FP32 scales) and frees the original FP16 storage. When
    // `quantize_weights == false` this is a no-op except for marking the UNet
    // finalized, after which apply_lora_delta() throws.
    //
    // Idempotent: a second call is a no-op. apply_lora_delta() must be called
    // BEFORE finalize_weights() — once finalized, the FP16 storage backing the
    // LoRA-patchable layers is gone.
    void finalize_weights();
    bool is_finalized() const { return finalized_; }

    const UNetConfig& config() const { return cfg_; }

private:
    // Paired INT8 weight + per-output-row FP32 scales. Populated by
    // finalize_weights() when quantize_weights is true; .W_int8.size() == 0
    // means this layer is still using its FP16 weight.
    struct QWeight {
        brotensor::GpuTensor W_int8;   // INT8 (out, in)
        brotensor::GpuTensor scales;   // FP32 (out, 1)
        bool active() const { return W_int8.size() > 0; }
    };
    struct Resnet {
        brotensor::GpuTensor n1g, n1b, W1, b1;
        brotensor::GpuTensor temb_W, temb_b;
        brotensor::GpuTensor n2g, n2b, W2, b2;
        brotensor::GpuTensor Ws, bs;
        // INT8 counterparts (populated by finalize_weights when enabled).
        QWeight W1_q, W2_q, Ws_q;
        bool has_shortcut = false;
        int  C_in = 0, C_out = 0;
    };
    struct AttnFFN {
        brotensor::GpuTensor n1g, n1b;
        brotensor::GpuTensor Wq1, Wk1, Wv1, Wo1, bo1;
        brotensor::GpuTensor n2g, n2b;
        brotensor::GpuTensor Wq2, Wk2, Wv2, Wo2, bo2;
        brotensor::GpuTensor n3g, n3b;
        brotensor::GpuTensor ff1_W, ff1_b;
        brotensor::GpuTensor ff2_W, ff2_b;
        // INT8 counterparts.
        QWeight Wq1_q, Wk1_q, Wv1_q, Wo1_q;
        QWeight Wq2_q, Wk2_q, Wv2_q, Wo2_q;
        QWeight ff1_q, ff2_q;
    };
    struct Transformer2D {
        brotensor::GpuTensor gn_g, gn_b;
        brotensor::GpuTensor pi_W, pi_b;
        brotensor::GpuTensor po_W, po_b;
        QWeight pi_q, po_q;
        std::vector<AttnFFN> blocks;
        int  C = 0;
        int  num_heads = 0;
    };
    struct SampleConv {
        brotensor::GpuTensor W, b;
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

    void load_resnet_(const brodiffusion::safetensors::File& f,
                      const std::string& prefix,
                      int C_in, int C_out, Resnet& r);
    void load_transformer_(const brodiffusion::safetensors::File& f,
                           const std::string& prefix,
                           int C, int num_heads, Transformer2D& t);

    void apply_resnet_(const Resnet& r, int H, int W,
                       brotensor::GpuTensor& x, brotensor::GpuTensor& tmp);
    // If `cache_entry` is non-null, its (K, V) replace the cross-attn K/V
    // projections (must have been primed against the same `ctx`).
    void apply_transformer_(const Transformer2D& t,
                            const brotensor::GpuTensor& ctx,
                            const CrossAttnKVCacheEntry* cache_entry,
                            int H, int W,
                            brotensor::GpuTensor& x,
                            // Trace plumbing. Both null = fast path (existing
                            // behaviour). When trace_out_entry is non-null,
                            // attn2 is routed through cross_attention_forward_
                            // with_attn_gpu and AttnAvg is written to
                            // *trace_out_entry. attn_logit_bias is an optional
                            // FP32 (Lq, Lk) pre-softmax bias passed straight
                            // through to the brotensor op.
                            brotensor::GpuTensor* trace_out_entry = nullptr,
                            const brotensor::GpuTensor* attn_logit_bias = nullptr);
    // Shared forward worker; xattn_cache may be null (legacy path) or point
    // at a cache with exactly num_xattn_blocks() entries. If `gs_emb` is
    // non-null the LCM cond_proj path is used (requires time_cond_proj_dim>0);
    // otherwise the vanilla SD1.5 time-embedding path is used. `trace_out`
    // and `attn_logit_biases` (both optional) wire the trace-mode plumbing
    // for cross-attention research; see UNet::forward_trace.
    void forward_impl_(const brotensor::GpuTensor& sample,
                       int H, int W,
                       float timestep,
                       const float* gs_emb,
                       const brotensor::GpuTensor& encoder_hidden_states,
                       const CrossAttnKVCache* xattn_cache,
                       const std::vector<const brotensor::GpuTensor*>* attn_logit_biases,
                       CrossAttnTrace* trace_out,
                       brotensor::GpuTensor& out);
    // Returns a pointer to the FP16 base weight identified by `target_path`
    // (a diffusers tail within the UNet, e.g. "down_blocks.0.attentions.0.
    // transformer_blocks.0.attn1.to_q"). Returns nullptr if the path doesn't
    // match a recognized LoRA-patchable layer.
    brotensor::GpuTensor* lora_target_(const std::string& target_path);
    // Sub-dispatchers used by lora_target_. `sub` is the diffusers path tail
    // remaining after the block-level prefix has been consumed.
    static brotensor::GpuTensor* resolve_transformer_target_(Transformer2D& tr,
                                                              const std::string& sub);
    static brotensor::GpuTensor* resolve_resnet_target_(Resnet& r,
                                                        const std::string& tail);

    void apply_conv3x3_(const brotensor::GpuTensor& W,
                        const brotensor::GpuTensor& b,
                        int C_in, int C_out, int H, int W_,
                        int stride, int pad,
                        const brotensor::GpuTensor& in,
                        brotensor::GpuTensor& out);
    // INT8 variant; called when the supplied QWeight is active.
    void apply_conv3x3_q_(const QWeight& Wq,
                          const brotensor::GpuTensor& b,
                          int C_in, int C_out, int H, int W_,
                          int stride, int pad,
                          const brotensor::GpuTensor& in,
                          brotensor::GpuTensor& out);
    // Helper used by finalize_weights() to quantise a single FP16 weight in
    // place: downloads the FP16 weight, runs quantize_int8_per_row_host,
    // uploads INT8 + scales, then frees the FP16 GpuTensor.
    void quantize_weight_inplace_(brotensor::GpuTensor& W_fp16, QWeight& q);

    UNetConfig cfg_;
    bool finalized_ = false;
    int time_embed_dim_ = 0;
    int freq_dim_       = 0;

    brotensor::GpuTensor conv_in_W_,  conv_in_b_;
    brotensor::GpuTensor te_l1_W_, te_l1_b_, te_l2_W_, te_l2_b_;
    // cond_proj weight (LCM only; empty when time_cond_proj_dim == 0).
    brotensor::GpuTensor te_cond_W_;
    std::vector<DownBlock> down_blocks_;
    MidBlock               mid_;
    std::vector<UpBlock>   up_blocks_;
    brotensor::GpuTensor norm_out_g_, norm_out_b_;
    brotensor::GpuTensor conv_out_W_, conv_out_b_;

    brotensor::GpuTensor x_, y_;
    brotensor::GpuTensor freq_emb_, temb_a_, temb_b_, temb_silu_, temb_proj_;
    // LCM scratch: w_emb_ holds the sinusoidal guidance-scale embedding,
    // temb_cond_ holds cond_proj(w_emb_); both unused when time_cond_proj_dim==0.
    brotensor::GpuTensor w_emb_, temb_cond_;
    brotensor::GpuTensor cat_buf_;
    brotensor::GpuTensor gn_, seq_, proj_in_seq_, tseq_, ln_;
    brotensor::GpuTensor attn_proj_;
    brotensor::GpuTensor ff_mid_, ff_act_, ff_out_;
    brotensor::GpuTensor proj_out_seq_, proj_out_nchw_;
};

}  // namespace brodiffusion::unet
