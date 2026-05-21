#pragma once

// CLIP ViT-L/14 text encoder for SD1.5.
//
// Forward-only. Runs on whichever backend brotensor resolves at runtime —
// CPU by default, CUDA when available — at that backend's compute dtype (FP32
// on CPU, FP16 on a GPU). Architecture (one layer):
//   x = LN1(x)
//   q = x @ Wq + bq;  k = x @ Wk + bk;  v = x @ Wv + bv     (heads stacked)
//   a = causal_self_attention(q, k, v)
//   x = x + a @ Wo + bo
//   x = LN2(x)
//   m = QuickGELU(x @ Wfc1 + bfc1)
//   x = x + m @ Wfc2 + bfc2
// Repeated num_layers times, then a final LayerNorm. Output is the last
// hidden state (L, hidden_dim) — SD1.5 takes this directly as cross-attention
// context; no projection or pooling.
//
// Weights load at the pipeline compute dtype. The safetensors loader accepts
// either F16 or F32 source tensors and converts as needed for the active
// backend.

#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; struct TensorView; }

namespace brodiffusion::clip {

struct TextEncoderConfig {
    int   vocab_size       = 49408;
    int   max_position     = 77;
    int   hidden_dim       = 768;
    int   num_heads        = 12;       // head_dim = hidden_dim / num_heads = 64
    int   num_layers       = 12;
    int   intermediate_dim = 3072;     // FFN inner width
    float layer_norm_eps   = 1e-5f;
    // Token id whose first occurrence marks end-of-text. CLIP ViT-L/14: 49407.
    // The pooled output (when requested) is the final hidden state at this
    // token's first position in the sequence.
    int eos_token_id = 49407;
};

class TextEncoder {
public:
    explicit TextEncoder(const TextEncoderConfig& cfg);
    ~TextEncoder();

    // Non-copyable; movable.
    TextEncoder(const TextEncoder&) = delete;
    TextEncoder& operator=(const TextEncoder&) = delete;
    TextEncoder(TextEncoder&&) noexcept = default;
    TextEncoder& operator=(TextEncoder&&) noexcept = default;

    // Load all weights from a safetensors file under the given prefix. Names
    // follow Hugging Face's convention; the prefix defaults to "text_model."
    // matching `transformers` exports. SD1.5 full checkpoints typically use
    // "cond_stage_model.transformer.text_model." — pass that explicitly.
    //
    // Required tensors (per layer i in [0, num_layers)):
    //   {prefix}embeddings.token_embedding.weight        (V, D)
    //   {prefix}embeddings.position_embedding.weight     (P, D)
    //   {prefix}encoder.layers.{i}.layer_norm1.{weight,bias}   (D,)
    //   {prefix}encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}
    //          weight (D, D), bias (D,)
    //   {prefix}encoder.layers.{i}.layer_norm2.{weight,bias}   (D,)
    //   {prefix}encoder.layers.{i}.mlp.fc1.{weight (FFN, D), bias (FFN,)}
    //   {prefix}encoder.layers.{i}.mlp.fc2.{weight (D, FFN), bias (D,)}
    //   {prefix}final_layer_norm.{weight,bias}                 (D,)
    //
    // Source tensors may be F16 or F32; they load at the compute dtype.
    // Throws std::runtime_error if a name is missing, shape mismatches the
    // config, or the source dtype is neither F16 nor F32.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "text_model.");

    // Forward pass on a length-L sequence of int32 token IDs. L must equal
    // cfg.max_position (CLIP is fixed-length).
    //   ids: host pointer to L int32 token IDs in [0, vocab_size).
    //   out: (L, hidden_dim) Tensor at the compute dtype, resized as needed.
    // brotensor::init() must have been called once before any forward.
    // The caller is responsible for sync_all() before reading `out` to host.
    //
    // If `pooled` is non-null it is filled with the (1, hidden_dim) pooled
    // vector: the final hidden state at the first eos_token_id position in
    // `ids`. When the checkpoint contains a `text_projection.weight`, the
    // pooled vector is additionally projected through it (CLIPTextModelWith-
    // Projection behavior); otherwise it is the raw EOS hidden state
    // (CLIPTextModel behavior, which is what Flux uses).
    void forward(const int32_t* ids, brotensor::Tensor& out,
                 brotensor::Tensor* pooled = nullptr);

    const TextEncoderConfig& config() const { return cfg_; }

    // Fold a LoRA delta into the base weight identified by `target_path`,
    // a diffusers path within the CLIP module (e.g.
    // "text_model.encoder.layers.0.self_attn.q_proj"). Same semantics as
    // brodiffusion::unet::UNet::apply_lora_delta: in-place
    //     W += scale_total * (lora_up @ lora_down)
    // with `scale_total = (alpha / rank) * user_scale` baked in by the caller.
    void apply_lora_delta(const std::string& target_path,
                          const brotensor::safetensors::TensorView& lora_down,
                          const brotensor::safetensors::TensorView& lora_up,
                          float scale_total);

private:
    struct Layer {
        brotensor::Tensor ln1_gamma, ln1_beta;
        brotensor::Tensor Wq, bq, Wk, bk, Wv, bv, Wo, bo;
        brotensor::Tensor ln2_gamma, ln2_beta;
        brotensor::Tensor fc1_W, fc1_b, fc2_W, fc2_b;
    };

    TextEncoderConfig cfg_;

    // Weights.
    brotensor::Tensor token_embed_;     // (V, D)
    brotensor::Tensor position_embed_;  // (P, D)
    std::vector<Layer>   layers_;
    brotensor::Tensor final_gamma_, final_beta_;
    brotensor::Tensor text_projection_;   // (D, D); empty when absent
    bool                 has_text_projection_ = false;

    // Per-call scratch (kept alive across calls to avoid realloc).
    // Device-resident INT32 token-id / position-id buffers.
    brotensor::Tensor ids_dev_;
    brotensor::Tensor positions_dev_;   // [0..P-1] uploaded once
    brotensor::Tensor tok_emb_, pos_emb_;
    brotensor::Tensor x_;                            // residual stream
    brotensor::Tensor ln_out_;
    brotensor::Tensor proj_out_;
    brotensor::Tensor ffn_mid_, ffn_act_, ffn_out_;
    brotensor::Tensor pooled_eos_;   // (1, D) EOS-row scratch for projection
};

}  // namespace brodiffusion::clip
