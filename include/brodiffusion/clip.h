#pragma once

// CLIP ViT-L/14 text encoder for SD1.5.
//
// Forward-only, FP16 throughout. Architecture (one layer):
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
// Weights are FP16. SD1.5 ships its text encoder in FP16 already; if you have
// FP32 weights, convert host-side before upload (brotensor provides the IEEE
// 754 binary16 helpers).

#include "brotensor/device_buffer.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brodiffusion::safetensors { class File; struct TensorView; }

namespace brodiffusion::clip {

struct TextEncoderConfig {
    int   vocab_size       = 49408;
    int   max_position     = 77;
    int   hidden_dim       = 768;
    int   num_heads        = 12;       // head_dim = hidden_dim / num_heads = 64
    int   num_layers       = 12;
    int   intermediate_dim = 3072;     // FFN inner width
    float layer_norm_eps   = 1e-5f;
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
    // Every tensor must be FP16. Throws std::runtime_error if a name is
    // missing, shape mismatches the config, or dtype is wrong.
    void load_weights(const brodiffusion::safetensors::File& f,
                      const std::string& prefix = "text_model.");

    // Forward pass on a length-L sequence of int32 token IDs. L must equal
    // cfg.max_position (CLIP is fixed-length).
    //   ids: host pointer to L int32 token IDs in [0, vocab_size).
    //   out: (L, hidden_dim) FP16 GpuTensor, resized as needed.
    // brotensor::cuda_init() must have been called once before any forward.
    // The caller is responsible for cuda_sync() before reading `out` to host.
    void forward(const int32_t* ids, brotensor::GpuTensor& out);

    const TextEncoderConfig& config() const { return cfg_; }

    // Fold a LoRA delta into the base FP16 weight identified by `target_path`,
    // a diffusers path within the CLIP module (e.g.
    // "text_model.encoder.layers.0.self_attn.q_proj"). Same semantics as
    // brodiffusion::unet::UNet::apply_lora_delta: in-place
    //     W += scale_total * (lora_up @ lora_down)
    // with `scale_total = (alpha / rank) * user_scale` baked in by the caller.
    void apply_lora_delta(const std::string& target_path,
                          const brodiffusion::safetensors::TensorView& lora_down,
                          const brodiffusion::safetensors::TensorView& lora_up,
                          float scale_total);

private:
    struct Layer {
        brotensor::GpuTensor ln1_gamma, ln1_beta;
        brotensor::GpuTensor Wq, bq, Wk, bk, Wv, bv, Wo, bo;
        brotensor::GpuTensor ln2_gamma, ln2_beta;
        brotensor::GpuTensor fc1_W, fc1_b, fc2_W, fc2_b;
    };

    TextEncoderConfig cfg_;

    // Weights.
    brotensor::GpuTensor token_embed_;     // (V, D)
    brotensor::GpuTensor position_embed_;  // (P, D)
    std::vector<Layer>   layers_;
    brotensor::GpuTensor final_gamma_, final_beta_;

    // Per-call scratch (kept alive across calls to avoid realloc).
    brotensor::DeviceBuffer<int32_t> ids_dev_;
    brotensor::DeviceBuffer<int32_t> positions_dev_;   // [0..P-1] uploaded once
    brotensor::GpuTensor tok_emb_, pos_emb_;
    brotensor::GpuTensor x_;                            // residual stream
    brotensor::GpuTensor ln_out_;
    brotensor::GpuTensor proj_out_;
    brotensor::GpuTensor ffn_mid_, ffn_act_, ffn_out_;
};

}  // namespace brodiffusion::clip
