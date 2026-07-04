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

    // Predict the flow-matching velocity for the image tokens.
    //   packed_latent: (hp*wp, in_channels) packed noisy latent (FP32 or the
    //       compute dtype; cast internally).
    //   hp, wp: packed latent grid (image_seq_len = hp*wp).
    //   prompt_embeds: (text_seq*num_text_layers, text_hidden_dim) tapped text
    //       hidden states, token-major/layer-minor (krea2::encode_prompt layout).
    //   prompt_embeds_mask: (text_seq, 1) FP32 validity mask (1 valid / 0 pad).
    //   timestep: flow-matching time in [0,1].
    //   out: (hp*wp, in_channels) velocity, at the compute dtype.
    void forward(const brotensor::Tensor& packed_latent, int hp, int wp,
                 const brotensor::Tensor& prompt_embeds,
                 const brotensor::Tensor& prompt_embeds_mask,
                 float timestep, brotensor::Tensor& out);

    const Krea2Config& config() const { return cfg_; }
    brotensor::Dtype compute_dtype() const;

private:
    struct Linear {
        brotensor::Tensor W;   // (out, in) at compute dtype
        brotensor::Tensor b;   // (out, 1) at compute dtype; empty if bias-free
        bool has_bias() const { return b.size() > 0; }
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

}  // namespace brodiffusion::dit
