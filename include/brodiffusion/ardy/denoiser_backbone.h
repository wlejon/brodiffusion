#pragma once
//
// ARDY denoiser backbone — the shared prefix-conditioned Transformer encoder.
//
// The two-stage denoiser (ardy/model/auto_latent_twostage_denoiser.py) runs two
// of these blocks: a root stage (output 20 = motion_root_dim 5 x fpt 4) and a
// body stage (output 128 = latent_embedding_dim). Both are the same
// `TransformerEncoderBlock` (ardy/model/backbone.py), differing only in
// output_dim. This class is that block.
//
// Per forward: the pre-projected motion tokens x (already at latent_dim; the
// block's input_linear is Identity for the auto-latent denoiser) are prepended
// with a PREFIX of conditioning tokens — [text, timestep, first_heading_angle] —
// and run through a stack of post-norm BIDIRECTIONAL Transformer encoder layers
// (unlike the causal FSQ decoder). The prefix is stripped off the output and the
// remaining motion rows are projected to output_dim.
//
// g152 config: latent_dim 1024, 8 heads (head_dim 128), ff 2048, 8 layers, gelu,
// post-norm; llm_shape [1, 4096] (one 4096-dim text token); use_text_mask false;
// input_first_heading_angle true; positional_encoding_mode
// "learned_prefix_zero_at_first_generation".
//
// Positional encoding:
//   * prefix rows get a LEARNED positional embedding (a 3x1024 table).
//   * motion rows get a non-learned sinusoidal PE indexed by token_index, which
//     may be negative (indices are centered at the first generation token). For
//     any integer index i, PE(i)[2k] = sin(i*div_k), PE(i)[2k+1] = cos(i*div_k),
//     div_k = 10000^(-2k/latent_dim) — so no table is needed, we evaluate it.
//   * timestep is embedded as time_embed(sinusoidalPE[timestep]) via a
//     Linear->SiLU->Linear MLP.

#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::ardy {

class ArdyDenoiserBackbone {
public:
    struct Config {
        int latent_dim       = 1024;
        int num_heads        = 8;      // head_dim = latent_dim / num_heads = 128
        int ff_size          = 2048;
        int num_layers       = 8;
        int llm_dim          = 4096;   // llm_shape[-1]
        int num_text_tokens  = 1;      // llm_shape[0]
        int output_dim       = 20;     // root stage: 20; body stage: 128
        bool input_first_heading_angle = true;
    };

    // NB: a delegating default ctor rather than a `= Config{}` default arg —
    // GCC 12 rejects value-initializing a nested type in a default argument
    // (the enclosing class is still incomplete there). The member-init below
    // runs in complete-class context, where Config's NSDMIs are available.
    ArdyDenoiserBackbone() : ArdyDenoiserBackbone(Config{}) {}
    explicit ArdyDenoiserBackbone(const Config& cfg);
    ~ArdyDenoiserBackbone();

    ArdyDenoiserBackbone(const ArdyDenoiserBackbone&) = delete;
    ArdyDenoiserBackbone& operator=(const ArdyDenoiserBackbone&) = delete;

    const Config& config() const { return cfg_; }

    // Load one stage's weights from the ARDY denoiser.safetensors. `prefix` is
    // the stage prefix, e.g. "denoiser.backbone.root_model." or
    // "denoiser.backbone.body_model.". Throws on any missing tensor.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix);

    // Run the block.
    //   x:            (T, latent_dim) pre-projected motion tokens (device tensor).
    //   text_feat:    (num_text_tokens, llm_dim) host text embedding.
    //   timestep:     diffusion step index (into the sinusoidal PE table).
    //   first_heading_angle: frame-0 heading (radians).
    //   token_index:  (T,) host integer positions for the motion PE (may be < 0).
    //   T:            number of motion tokens.
    //   key_mask:     optional (num_prefix + T,) host FP32 key-padding mask
    //                 (1 attendable / 0 ignored); nullptr = all attendable (the
    //                 first-generation-window case). num_prefix = 2 + (heading?1:0).
    //   out:          (T, output_dim) at compute dtype. Caller syncs before read.
    void forward(const brotensor::Tensor& x,
                 const float* text_feat,
                 int timestep,
                 float first_heading_angle,
                 const int* token_index,
                 int T,
                 const float* key_mask,
                 brotensor::Tensor& out);

    // Number of prefix conditioning tokens (text + timestep [+ heading]).
    int num_prefix() const { return cfg_.num_text_tokens + 1 + (cfg_.input_first_heading_angle ? 1 : 0); }

private:
    struct Linear { brotensor::Tensor W, b; };
    struct Layer {
        Linear q, k, v, out_proj;   // self-attention (packed in_proj split 3x)
        Linear linear1, linear2;    // FFN
        brotensor::Tensor n1g, n1b, n2g, n2b;  // norm1/norm2 affine
    };

    Config cfg_;
    Linear embed_text_;          // llm_dim -> latent_dim
    Linear time0_, time2_;       // timestep MLP: latent_dim -> latent_dim (x2)
    Linear first_heading_;       // 2 -> latent_dim
    Linear output_linear_;       // latent_dim -> output_dim
    brotensor::Tensor learned_prefix_;  // (num_prefix, latent_dim) PE table
    std::vector<Layer> layers_;
};

}  // namespace brodiffusion::ardy
