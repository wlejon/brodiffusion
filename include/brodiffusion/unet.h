#pragma once

// UNet2DConditionModel for SD1.5 (the noise-prediction backbone).
//
// Inference-only, batch size N = 1 (the forward path hard-codes N=1
// throughout; the underlying brotensor ops support N > 1, so generalizing
// is mostly a matter of plumbing the batch dim through). Runs on whichever
// backend brotensor resolves at runtime — CPU by default, CUDA when
// available — at that backend's compute dtype (FP32 on CPU, FP16 on a GPU).
// Architecture mirrors Hugging Face's diffusers `UNet2DConditionModel`
// defaults for SD1.5:
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
// Caller is responsible for sync_all() before reading the output. Weights
// and activations carry the compute dtype — FP32 on CPU, FP16 on a GPU
// backend. The safetensors loader accepts F16 or F32 source weights and
// converts as needed.

#include "brodiffusion/denoiser.h"

#include "brotensor/tensor.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; struct TensorView; }

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

    // Guidance-scale-embedding projection dimension for LCM-distilled
    // checkpoints. This is a property of the checkpoint, not a caller choice:
    // load_weights() auto-detects it. A distilled checkpoint ships
    // `time_embedding.cond_proj.weight` of shape (freq_dim, cond_proj_dim),
    // no bias — load_weights() loads it and sets time_cond_proj_dim from the
    // weight's second dim. A vanilla SD1.5 checkpoint has no such tensor, so
    // load_weights() sets time_cond_proj_dim = 0. Whatever value is set here
    // before load_weights() is only a hint and is overwritten to match the
    // file, so a pipeline built either way loads any checkpoint without a
    // spurious "missing tensor" failure.
    //
    // When time_cond_proj_dim > 0 after load, the caller must drive the UNet
    // through the forward()/forward_trace() overload that takes a
    // `guidance_scale_embedding`. The projected guidance-scale embedding is
    // added to the sinusoidal time embedding *before* linear_1, matching
    // diffusers' TimestepEmbedding.forward. LCM-distilled checkpoints
    // (e.g. SimianLuo/LCM_Dreamshaper_v7) ship time_cond_proj_dim = 256.
    int time_cond_proj_dim = 0;

    // When true, finalize_weights() converts the big UNet weights (ResBlock
    // 3x3/1x1 conv weights, attention Q/K/V/O projections, transformer
    // proj_in/proj_out, FF1/FF2 linears, down/upsampler 3x3 convs) to INT8
    // weight-only quantisation (W8A16). Small/sensitive layers (conv_in,
    // conv_out, GroupNorm gain/bias, time embedding, cond_proj, per-resblock
    // time_emb_proj, all biases) keep their FP16 storage. Per-output-row
    // symmetric scales (matching brotensor::quantize_int8_per_row_host).
    //
    // INT8 (W8A16) quantization is GPU-only. On the CPU backend
    // finalize_weights() prints a warning and ignores this flag, so the
    // pipeline still runs end-to-end in FP32.
    bool quantize_weights = false;
};

class UNet final : public Denoiser {
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
    // Source tensors may be F16 or F32; they load at the compute dtype.
    // Throws std::runtime_error on missing names, shape mismatches, or a
    // source dtype that is neither F16 nor F32.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "") override;

    // ── Denoiser interface ────────────────────────────────────────────────
    //
    // Model-agnostic entry points the Pipeline reaches through a
    // unique_ptr<Denoiser>. The legacy UNet-shaped forward overloads below
    // remain for direct callers (and trace mode).
    PreparedConditioning prepare(const Conditioning& cond) override;
    void forward(const brotensor::Tensor& latent, int H_lat, int W_lat,
                 float timestep, const PreparedConditioning& prepared,
                 Branch branch, brotensor::Tensor& out) override;
    int latent_channels() const override { return cfg_.in_channels; }
    PredictionType prediction_type() const override {
        return PredictionType::Epsilon;
    }
    bool uses_cfg() const override { return true; }
    brotensor::Dtype compute_dtype() const override;
    unet::UNet* as_unet() override { return this; }
    const unet::UNet* as_unet() const override { return this; }

    // Forward pass. Activation tensors carry the compute dtype (FP32 on CPU,
    // FP16 on a GPU backend).
    //   sample:                (1, in_channels * H * W) — noisy latent
    //   H, W:                  spatial dims of `sample`. H and W must each be
    //                          divisible by 2^(num_blocks-1) (typically 8).
    //   timestep:              continuous timestep value (typically in [0, 1000)).
    //   encoder_hidden_states: (L_text, cross_attention_dim), e.g. CLIP output.
    //   out:                   (1, out_channels * H * W), resized as needed.
    void forward(const brotensor::Tensor& sample,
                 int H, int W,
                 float timestep,
                 const brotensor::Tensor& encoder_hidden_states,
                 brotensor::Tensor& out);

    // Cached cross-attention K/V for a single context tensor — one (K, V) pair
    // per Transformer2D block, at the compute dtype, layout matching
    // brotensor::flash_attention_forward's K/V args (Lk, C). The text
    // context is fixed across all denoising steps so projecting K/V once per
    // generate() per CFG branch eliminates 16 × steps × 2 redundant matmuls.
    struct CrossAttnKVCacheEntry {
        brotensor::Tensor K;  // (Lk, C)
        brotensor::Tensor V;  // (Lk, C)
    };
    using CrossAttnKVCache = std::vector<CrossAttnKVCacheEntry>;

    // Head-averaged cross-attention softmax map per Transformer2D block, in the
    // same traversal order the forward pass visits them. Each entry has shape
    // (Lq, Lk) at the compute dtype, where Lq is the layer's spatial token count (H*W at that
    // resolution) and Lk is the text-context length (77 for SD1.5/CLIP). The
    // trace-mode forward overload below populates this vector for downstream
    // research consumers (cross-attention tree search, attention scoring).
    // Alias of the model-agnostic brodiffusion::AttentionTrace.
    using CrossAttnTrace = AttentionTrace;

    // Populate `cache` with one (K, V) pair per Transformer2D block (in the
    // same traversal order the forward pass visits them: down blocks,
    // mid block, up blocks). `cache` is resized as needed.
    void prime_xattn_cache(const brotensor::Tensor& ctx,
                           CrossAttnKVCache& cache);

    // Variant of forward that uses a pre-primed K/V cache built from the
    // same `encoder_hidden_states` (or, more precisely, from a ctx with the
    // exact tokens the cache was primed against). Cross-attention layers
    // skip the K/V projection step; self-attention is unchanged.
    void forward(const brotensor::Tensor& sample,
                 int H, int W,
                 float timestep,
                 const brotensor::Tensor& encoder_hidden_states,
                 const CrossAttnKVCache& xattn_cache,
                 brotensor::Tensor& out);

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
    void forward(const brotensor::Tensor& sample,
                 int H, int W,
                 float timestep,
                 float guidance_scale_embedding,
                 const brotensor::Tensor& encoder_hidden_states,
                 const CrossAttnKVCache& xattn_cache,
                 brotensor::Tensor& out);

    // Trace-mode forward. Routes each cross-attention (`attn2`) call through
    // brotensor::cross_attention_forward_with_attn so the head-averaged
    // softmax map can be observed and an optional per-layer FP32 pre-softmax
    // logit bias can be injected. Self-attention (`attn1`) still uses the
    // fast flash path.
    //
    // `trace_out`, if non-null, is resized to num_xattn_blocks() and each
    // entry is filled with the head-averaged (Lq, Lk) attention map for
    // that layer.
    //
    // `attn_logit_biases`, if non-null, must have length num_xattn_blocks();
    // entry `i` is either null (no bias) or a (Lq_i, Lk) FP32 Tensor added
    // to the scaled QKᵀ scores before softmax at layer i.
    //
    // `guidance_scale_embedding` mirrors the LCM `forward` overload: pass a
    // pointer to the raw guidance scale `w` for an LCM-distilled U-Net
    // (time_cond_proj_dim > 0), or null for vanilla SD1.5. The cond_proj path
    // and the trace plumbing are independent inside forward_impl_, so trace
    // mode supports both families — but the pointer must be non-null iff the
    // U-Net was built with time_cond_proj_dim > 0, exactly as the LCM and
    // vanilla forward() overloads enforce.
    //
    // Trace mode currently does NOT use the K/V cache (the with-attn brotensor
    // op has no cached variant yet) — K/V are reprojected from `ctx` at every
    // layer. Acceptable cost for experiment-mode tree search; if it bottlenecks
    // we'll ask brotensor for `..._q_with_kv_cached_with_attn`.
    //
    // INT8 (quantize_weights) is not supported in trace mode and will throw.
    void forward_trace(const brotensor::Tensor& sample,
                       int H, int W,
                       float timestep,
                       const float* guidance_scale_embedding,
                       const brotensor::Tensor& encoder_hidden_states,
                       const std::vector<const brotensor::Tensor*>* attn_logit_biases,
                       CrossAttnTrace* trace_out,
                       brotensor::Tensor& out);

    // ── Denoiser trace seam ───────────────────────────────────────────────
    // Model-agnostic trace entry point. Adapts the generic PreparedConditioning
    // contract onto forward_trace: the raw text context and the LCM guidance
    // scale are pulled from the UNet's prepared payload (forward_trace bypasses
    // the K/V cache, so it needs the context, not the cache).
    void forward_traced(
            const brotensor::Tensor& latent, int H_lat, int W_lat,
            float timestep, const PreparedConditioning& prepared,
            Branch branch,
            const std::vector<const brotensor::Tensor*>* attn_logit_biases,
            AttentionTrace* trace_out, brotensor::Tensor& out) override;

    // Number of Transformer2D (cross-attn) blocks in the model — matches the
    // size of any cache returned by prime_xattn_cache.
    int num_xattn_blocks() const override;

    // Per-Transformer2D-block spatial stride (relative to H_lat, W_lat), in
    // traversal order. SD1.5 schedule is hard-coded: down stages 0/1/2 each
    // emit 2 transformers, mid emits 1, up stages 1/2/3 each emit 3.
    std::vector<int> layer_strides() const;

    // Text-encoder context length (Lk) for cross-attention. SD1.5 / CLIP: 77.
    int context_length() const { return 77; }

    // Fold a LoRA delta into the base weight identified by `target_path`.
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
                          const brotensor::safetensors::TensorView& lora_down,
                          const brotensor::safetensors::TensorView& lora_up,
                          float scale_total);

    // Finalize the UNet weights for inference. When the config has
    // `quantize_weights == true` and the weights are GPU-resident, this
    // converts the big linear / conv weights listed in
    // UNetConfig::quantize_weights to INT8 (per-output-row symmetric FP32
    // scales) and frees the original weight storage. INT8 quantization is
    // GPU-only: on the CPU backend it is skipped (with a warning) so the
    // pipeline still runs in FP32. When `quantize_weights == false` this is a
    // no-op except for marking the UNet finalized, after which
    // apply_lora_delta() throws.
    //
    // Idempotent: a second call is a no-op. apply_lora_delta() must be called
    // BEFORE finalize_weights() — once finalized, the weight storage backing
    // the LoRA-patchable layers may be gone.
    void finalize_weights() override;
    bool is_finalized() const { return finalized_; }

    const UNetConfig& config() const { return cfg_; }

private:
    // Paired INT8 weight + per-output-row FP32 scales. Populated by
    // finalize_weights() when quantize_weights is true; .W_int8.size() == 0
    // means this layer is still using its FP16 weight.
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

    void load_resnet_(const brotensor::safetensors::File& f,
                      const std::string& prefix,
                      int C_in, int C_out, Resnet& r);
    void load_transformer_(const brotensor::safetensors::File& f,
                           const std::string& prefix,
                           int C, int num_heads, Transformer2D& t);

    void apply_resnet_(const Resnet& r, int H, int W,
                       brotensor::Tensor& x, brotensor::Tensor& tmp);
    // If `cache_entry` is non-null, its (K, V) replace the cross-attn K/V
    // projections (must have been primed against the same `ctx`).
    void apply_transformer_(const Transformer2D& t,
                            const brotensor::Tensor& ctx,
                            const CrossAttnKVCacheEntry* cache_entry,
                            int H, int W,
                            brotensor::Tensor& x,
                            // Trace plumbing. Both null = fast path (existing
                            // behaviour). When trace_out_entry is non-null,
                            // attn2 is routed through
                            // cross_attention_forward_with_attn and AttnAvg
                            // is written to
                            // *trace_out_entry. attn_logit_bias is an optional
                            // FP32 (Lq, Lk) pre-softmax bias passed straight
                            // through to the brotensor op.
                            brotensor::Tensor* trace_out_entry = nullptr,
                            const brotensor::Tensor* attn_logit_bias = nullptr);
    // Shared forward worker; xattn_cache may be null (legacy path) or point
    // at a cache with exactly num_xattn_blocks() entries. If `gs_emb` is
    // non-null the LCM cond_proj path is used (requires time_cond_proj_dim>0);
    // otherwise the vanilla SD1.5 time-embedding path is used. `trace_out`
    // and `attn_logit_biases` (both optional) wire the trace-mode plumbing
    // for cross-attention research; see UNet::forward_trace.
    void forward_impl_(const brotensor::Tensor& sample,
                       int H, int W,
                       float timestep,
                       const float* gs_emb,
                       const brotensor::Tensor& encoder_hidden_states,
                       const CrossAttnKVCache* xattn_cache,
                       const std::vector<const brotensor::Tensor*>* attn_logit_biases,
                       CrossAttnTrace* trace_out,
                       brotensor::Tensor& out);
    // Returns a pointer to the base weight identified by `target_path`
    // (a diffusers tail within the UNet, e.g. "down_blocks.0.attentions.0.
    // transformer_blocks.0.attn1.to_q"). Returns nullptr if the path doesn't
    // match a recognized LoRA-patchable layer.
    brotensor::Tensor* lora_target_(const std::string& target_path);
    // Sub-dispatchers used by lora_target_. `sub` is the diffusers path tail
    // remaining after the block-level prefix has been consumed.
    static brotensor::Tensor* resolve_transformer_target_(Transformer2D& tr,
                                                              const std::string& sub);
    static brotensor::Tensor* resolve_resnet_target_(Resnet& r,
                                                        const std::string& tail);

    void apply_conv3x3_(const brotensor::Tensor& W,
                        const brotensor::Tensor& b,
                        int C_in, int C_out, int H, int W_,
                        int stride, int pad,
                        const brotensor::Tensor& in,
                        brotensor::Tensor& out);
    // INT8 variant; called when the supplied QWeight is active.
    void apply_conv3x3_q_(const QWeight& Wq,
                          const brotensor::Tensor& b,
                          int C_in, int C_out, int H, int W_,
                          int stride, int pad,
                          const brotensor::Tensor& in,
                          brotensor::Tensor& out);
    // Helper used by finalize_weights() to quantise a single FP16 weight in
    // place: downloads the FP16 weight, runs quantize_int8_per_row_host,
    // uploads INT8 + scales, then frees the FP16 Tensor.
    void quantize_weight_inplace_(brotensor::Tensor& W_fp16, QWeight& q);

    UNetConfig cfg_;
    bool finalized_ = false;
    int time_embed_dim_ = 0;
    int freq_dim_       = 0;

    brotensor::Tensor conv_in_W_,  conv_in_b_;
    brotensor::Tensor te_l1_W_, te_l1_b_, te_l2_W_, te_l2_b_;
    // cond_proj weight (LCM only; empty when time_cond_proj_dim == 0).
    brotensor::Tensor te_cond_W_;
    std::vector<DownBlock> down_blocks_;
    MidBlock               mid_;
    std::vector<UpBlock>   up_blocks_;
    brotensor::Tensor norm_out_g_, norm_out_b_;
    brotensor::Tensor conv_out_W_, conv_out_b_;

    brotensor::Tensor x_, y_;
    brotensor::Tensor freq_emb_, temb_a_, temb_b_, temb_silu_, temb_proj_;
    // LCM scratch: w_emb_ holds the sinusoidal guidance-scale embedding,
    // temb_cond_ holds cond_proj(w_emb_); both unused when time_cond_proj_dim==0.
    brotensor::Tensor w_emb_, temb_cond_;
    brotensor::Tensor cat_buf_;
    brotensor::Tensor gn_, seq_, proj_in_seq_, tseq_, ln_;
    brotensor::Tensor attn_proj_;
    brotensor::Tensor ff_mid_, ff_act_, ff_out_;
    brotensor::Tensor proj_out_seq_, proj_out_nchw_;
};

}  // namespace brodiffusion::unet
