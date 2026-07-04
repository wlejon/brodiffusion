#pragma once

// Krea2Transformer2DModel — Krea 2's single-stream flow-matching image DiT.
//
// A forward-only port of diffusers' Krea2Transformer2DModel (Krea AI's 12.9B
// text-to-image backbone). Runs on whichever backend brotensor resolves —
// FP32 on CPU, and BF16 on CUDA (dit::flux_compute_dtype; the huge residual
// stream needs BF16's exponent range, exactly as Flux does).
//
// Unlike Flux's double/single-stream split, Krea 2 processes ONE joint
// sequence `[text ; image]` through `num_layers` identical blocks. The pieces:
//
//   - Text conditioning enters as a stack of `num_text_layers` hidden states
//     tapped from a Qwen3-VL text encoder (see krea2_text.h), one stack per
//     token: shape (text_seq, num_text_layers, text_hidden_dim), flattened here
//     to (text_seq*num_text_layers, text_hidden_dim), token-major/layer-minor.
//   - `text_fusion` collapses that stack into one refined text sequence:
//       * `num_layerwise_text_blocks` pre-norm blocks attend across the
//         num_text_layers axis independently PER TOKEN (batched length-N_L
//         attention, no mask, no RoPE),
//       * a learned `projector` (Linear num_text_layers->1) collapses the axis,
//       * `num_refiner_text_blocks` pre-norm blocks attend across the token
//         sequence (masked by the text validity mask, no RoPE).
//     Every fusion attention is plain MHA (text_num_attention_heads ==
//     text_num_key_value_heads) with q/k RMSNorm and a sigmoid output gate.
//   - `txt_in` projects the fused text (text_hidden_dim -> hidden_size);
//     `img_in` projects the packed latent (in_channels -> hidden_size).
//   - The timestep drives every block through ONE shared modulation vector
//     `temb_mod` (time_mod_proj of gelu(time_embed(t))) plus a per-block learned
//     `scale_shift_table` (6, hidden) added to it: prescale/preshift/pregate for
//     the attention sublayer, postscale/postshift/postgate for the FF sublayer.
//     Each block: `x += pregate * attn((1+prescale)*rms(x)+preshift)` then
//     `x += postgate * ff((1+postscale)*rms(x)+postshift)`.
//   - Block attention is grouped-query (num_attention_heads query heads,
//     num_key_value_heads KV heads), q/k RMSNorm, 3-axis (t,h,w) RoPE
//     (rope_theta, NOT the default 10000), a sigmoid output gate, and the
//     combined [text-mask ; all-ones-image] key-padding mask.
//   - `final_layer` applies a 2-entry adaptive RMSNorm driven by the RAW
//     time embedding `temb` (NOT temb_mod) and projects hidden_size ->
//     in_channels, over the image rows only.
//
// RMSNorm here uses a zero-centered gain: the effective scale is `1 + weight`.
// The gains are pre-added to 1 and kept in FP32; the norm runs in FP32
// (upcasting the activations) to preserve the small trained weights, matching
// the reference's `_keep_in_fp32_modules`.
//
// Pairs with the FlowMatch (rectified-flow Euler) scheduler; the forward
// predicts the velocity for the image tokens (prediction_type Velocity).

#include "brodiffusion/denoiser.h"
#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::dit {

struct Krea2Config {
    int in_channels             = 64;    // packed latent dim (= latent_channels*4)
    int num_layers              = 28;
    int attention_head_dim      = 128;
    int num_attention_heads     = 48;    // query heads; hidden = heads*head_dim
    int num_key_value_heads     = 12;    // KV heads (GQA)
    int intermediate_size       = 16384;
    int timestep_embed_dim      = 256;
    int text_hidden_dim         = 2560;
    int num_text_layers         = 12;
    int text_num_attention_heads   = 20;
    int text_num_key_value_heads   = 20;
    int text_intermediate_size  = 6912;
    int num_layerwise_text_blocks = 2;
    int num_refiner_text_blocks   = 2;
    std::vector<int> axes_dims_rope = {32, 48, 48};  // sums to attention_head_dim
    float rope_theta            = 1000.0f;
    float norm_eps              = 1e-5f;

    // When true (and the default device is CUDA), load_weights() quantizes
    // every per-block linear — all Attention (to_q/to_k/to_v/to_gate/to_out)
    // and SwiGLU (gate/up/down) weights in the 28 main TransformerBlocks and
    // the 4 text-fusion FusionBlocks — to INT8 weight-only (W8A16, per-output-
    // row symmetric FP32 scales) as the weights stream in, so the FP16/BF16
    // copy never materialises on the device. That matters: the BF16 Krea 2
    // transformer is ~24.6 GB — beyond a 24 GB card — while INT8 is ~12 GB.
    // The small in/out layers (img_in, time/txt_in projections, final_layer,
    // the projector) keep the compute dtype, as do all RMSNorm gains and the
    // scale/shift tables. On a non-CUDA backend the flag is ignored with a
    // warning (the fused dequant matmuls are GPU-only).
    bool quantize_weights = false;

    int hidden_size() const { return num_attention_heads * attention_head_dim; }
    int latent_channels() const { return in_channels / 4; }
};

class Krea2Transformer2DModel {
public:
    explicit Krea2Transformer2DModel(const Krea2Config& cfg);
    ~Krea2Transformer2DModel();

    Krea2Transformer2DModel(const Krea2Transformer2DModel&) = delete;
    Krea2Transformer2DModel& operator=(const Krea2Transformer2DModel&) = delete;

    // Load from a single file or a sharded set (first match wins across shards;
    // a name missing everywhere throws). Krea 2 ships the transformer sharded.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");
    void load_weights(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix = "");

    // Run the timestep-independent half of the model once per prompt: compact
    // the padded text block to its valid tokens (every text row shares the
    // identity RoPE position and masked rows are excluded from all attention,
    // so dropping them is exact), then run the text_fusion stack (layerwise
    // blocks, projector, refiner blocks) and txt_in.
    //   prompt_embeds: (text_seq*num_text_layers, text_hidden_dim) tapped text
    //       hidden states, token-major/layer-minor (krea2::encode_prompt layout).
    //   prompt_embeds_mask: (text_seq, 1) FP32 validity mask (1 valid / 0 pad).
    //   txt_out: (n_valid, hidden_size) text rows for forward_with_text.
    // Nothing in here sees the timestep, so re-running it every denoise step
    // (the reference pipeline layout) is pure waste: Krea2Denoiser::prepare
    // calls this once per CFG branch per generation.
    void encode_text(const brotensor::Tensor& prompt_embeds,
                     const brotensor::Tensor& prompt_embeds_mask,
                     brotensor::Tensor& txt_out);

    // Predict the flow-matching velocity for the image tokens, from text rows
    // precomputed by encode_text. The joint sequence is (txt.rows + hp*wp)
    // fully-valid rows — no attention mask.
    //   packed_latent: (hp*wp, in_channels) packed noisy latent (FP32 or the
    //       compute dtype; cast internally).
    //   hp, wp: packed latent grid (image_seq_len = hp*wp).
    //   timestep: flow-matching time in [0,1].
    //   out: (hp*wp, in_channels) velocity, at the compute dtype.
    void forward_with_text(const brotensor::Tensor& packed_latent, int hp,
                           int wp, const brotensor::Tensor& txt,
                           float timestep, brotensor::Tensor& out);

    // Single-shot convenience: encode_text + forward_with_text. Kept for the
    // krea2-fwd debug CLI and the DiT parity harness, which drive one forward
    // from raw (embeds, mask) conditioning.
    void forward(const brotensor::Tensor& packed_latent, int hp, int wp,
                 const brotensor::Tensor& prompt_embeds,
                 const brotensor::Tensor& prompt_embeds_mask,
                 float timestep, brotensor::Tensor& out);

    const Krea2Config& config() const { return cfg_; }
    brotensor::Dtype compute_dtype() const;

private:
    // A biased-or-bias-free linear (weight (out,in), bias (out,1) or empty).
    // When the layer was quantized at load (cfg.quantize_weights), W is empty
    // and W_int8 (out,in) + scales (out,1 FP32 per-row) carry the weight
    // instead; the bias, if any, always stays at the compute dtype.
    struct Linear {
        brotensor::Tensor W;   // (out, in) at compute dtype; empty if quantized
        brotensor::Tensor b;   // (out, 1) at compute dtype; empty if bias-free
        brotensor::Tensor W_int8;  // (out, in) INT8 when quantized
        brotensor::Tensor scales;  // (out, 1) FP32 per-row scales when quantized
        bool has_bias() const { return b.size() > 0; }
        bool quantized() const { return W_int8.size() > 0; }
    };

    // Self-attention with GQA, q/k RMSNorm, sigmoid output gate. RoPE is applied
    // only when the caller supplies cos/sin (block attention); the text-fusion
    // attentions pass null.
    struct Attention {
        Linear to_q, to_k, to_v, to_gate, to_out;
        brotensor::Tensor norm_q, norm_k;   // (head_dim,1) FP32 (1+weight)
    };
    struct SwiGLU { Linear gate, up, down; };

    // Pre-norm text-fusion block (no modulation, no RoPE).
    struct FusionBlock {
        brotensor::Tensor norm1, norm2;     // (dim,1) FP32 (1+weight)
        Attention attn;
        SwiGLU ff;
    };
    // Main transformer block (temb modulation + per-block table, RoPE).
    struct TransformerBlock {
        brotensor::Tensor scale_shift_table;  // (6, hidden) at compute dtype
        brotensor::Tensor norm1, norm2;       // (hidden,1) FP32 (1+weight)
        Attention attn;
        SwiGLU ff;
    };

    void load_impl_(const std::vector<const brotensor::safetensors::File*>& shards,
                    const std::string& prefix);

    // One linear at the compute dtype, dispatching dense vs INT8 (W8A16).
    brotensor::Tensor linb_(const Linear& l, const brotensor::Tensor& X);

    Krea2Config cfg_;
    bool loaded_ = false;

    // Top-level weights.
    Linear img_in_;
    Linear time_l1_, time_l2_;   // time_embed.linear_{1,2}
    Linear time_mod_proj_;
    // text_fusion
    std::vector<FusionBlock> layerwise_blocks_;
    brotensor::Tensor projector_;   // (1, num_text_layers) at compute dtype
    std::vector<FusionBlock> refiner_blocks_;
    // txt_in (Krea2TextProjection)
    brotensor::Tensor txt_in_norm_;   // (text_hidden_dim,1) FP32 (1+weight)
    Linear txt_in_l1_, txt_in_l2_;
    // body
    std::vector<TransformerBlock> blocks_;
    // final_layer
    brotensor::Tensor final_scale_shift_table_;   // (2, hidden)
    brotensor::Tensor final_norm_;                // (hidden,1) FP32 (1+weight)
    Linear final_linear_;
};

// Krea2Denoiser — Krea2Transformer2DModel behind brodiffusion's model-agnostic
// Denoiser interface (the sibling of FluxDenoiser / SanaDenoiser the Pipeline
// drives). Wraps one Krea2Transformer2DModel; the Pipeline supplies the packed
// latent as the flat NCHW (1, latent_channels*H_lat*W_lat) the Denoiser contract
// uses, and this class does the 2x2 pack/unpack around the transformer's
// packed-token interface, exactly as FluxDenoiser does.
//
// CFG convention. Krea 2's Raw checkpoint runs REAL classifier-free guidance
// (uses_cfg() == true): the Pipeline evaluates a cond and an uncond branch and
// combines them with the standard formula uncond + scale*(cond-uncond). Krea 2's
// own model card quotes a guidance number `g` under the algebraically identical
// convention cond + g*(cond-uncond), i.e. standard_scale = 1 + g. So the card's
// recommended Raw setting g=4.5 maps to brodiffusion's --guidance-scale 5.5, and
// Turbo's g=0.0 maps to --guidance-scale 1.0 (the default, which disables CFG —
// matching Turbo's intended no-CFG behaviour). GenerateOptions::guidance_scale
// keeps its usual meaning everywhere; only the doc translates.
class Krea2Denoiser final : public Denoiser {
public:
    // patch_size is model_index.json's patch_size (2 for Krea 2); the 2x2
    // pack/unpack helpers assume it, so a value other than 2 is rejected.
    explicit Krea2Denoiser(const Krea2Config& cfg, int patch_size = 2);
    ~Krea2Denoiser();

    Krea2Denoiser(const Krea2Denoiser&) = delete;
    Krea2Denoiser& operator=(const Krea2Denoiser&) = delete;
    Krea2Denoiser(Krea2Denoiser&&) noexcept = default;
    Krea2Denoiser& operator=(Krea2Denoiser&&) noexcept = default;

    // ── Denoiser interface ────────────────────────────────────────────────
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "") override;
    // Sharded overload — the Krea 2 transformer ships in 3 shards.
    void load_weights(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix = "");
    void finalize_weights() override {}   // no quantisation for Krea 2
    PreparedConditioning prepare(const Conditioning& cond) override;
    void forward(const brotensor::Tensor& latent, int H_lat, int W_lat,
                 float timestep, const PreparedConditioning& prepared,
                 Branch branch, brotensor::Tensor& out) override;
    int latent_channels() const override { return model_.config().latent_channels(); }
    PredictionType prediction_type() const override {
        return PredictionType::Velocity;
    }
    bool uses_cfg() const override { return true; }
    brotensor::Dtype compute_dtype() const override {
        return model_.compute_dtype();
    }

    const Krea2Config& config() const { return model_.config(); }

private:
    Krea2Transformer2DModel model_;
    int patch_size_;
    brotensor::Tensor tf_out_;   // transformer output scratch (packed velocity)
};

}  // namespace brodiffusion::dit
