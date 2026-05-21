#pragma once

// T5-XXL text encoder (encoder-only) — the second text encoder Flux uses.
//
// Forward-only. Runs on whichever backend brotensor resolves at runtime —
// CPU by default, CUDA when available — at that backend's compute dtype (FP32
// on CPU, FP16 on a GPU). Architecture (google/t5-v1_1-xxl):
//
//   x = embedding_lookup(shared, ids)          (no position embedding)
//   per block:
//     n = rms_norm(x, ln0)
//     a = self_attention_bias(n, Wq,Wk,Wv,Wo, pos_bias, scale=1.0)
//     x = x + a
//     n = rms_norm(x, ln1)
//     g = gelu(linear(wi_0, n))                (tanh-approx GELU)
//     l = linear(wi_1, n)
//     h = g * l                                (gated-gelu FFN)
//     x = x + linear(wo, h)
//   x = rms_norm(x, final_ln)
//
// RMSNorm has no mean-subtraction and no bias. No linear layer has a bias.
// Relative-position bias replaces position embeddings; the bias table lives
// on block 0 and is shared across all layers. Attention does NOT scale QK^T
// (scale=1.0) — the position bias is defined against unscaled scores.
//
// Weights load at the pipeline compute dtype. The safetensors loader accepts
// F16, F32, or BF16 source tensors and converts as needed for the active
// backend.

#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; struct TensorView; }

namespace brodiffusion::t5 {

struct T5Config {
    int   vocab_size  = 32128;
    int   d_model     = 4096;
    int   d_ff        = 10240;
    int   d_kv        = 64;
    int   num_heads   = 64;     // num_heads * d_kv == d_model
    int   num_layers  = 24;
    int   relative_attention_num_buckets  = 32;
    int   relative_attention_max_distance = 128;
    float layer_norm_eps = 1e-6f;
};

class TextEncoder {  // T5 encoder
public:
    explicit TextEncoder(const T5Config& cfg);
    ~TextEncoder();

    // Non-copyable; movable.
    TextEncoder(const TextEncoder&) = delete;
    TextEncoder& operator=(const TextEncoder&) = delete;
    TextEncoder(TextEncoder&&) noexcept = default;
    TextEncoder& operator=(TextEncoder&&) noexcept = default;

    // Load from a safetensors file. Default prefix "" (standalone
    // T5EncoderModel export). Token embedding: try "{prefix}shared.weight",
    // fall back to "{prefix}encoder.embed_tokens.weight".
    //
    // Required tensors (per block i in [0, num_layers)):
    //   {prefix}encoder.block.{i}.layer.0.layer_norm.weight            (D,)
    //   {prefix}encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight (D,D)
    //   {prefix}encoder.block.{i}.layer.1.layer_norm.weight            (D,)
    //   {prefix}encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight   (d_ff,D)
    //   {prefix}encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight   (d_ff,D)
    //   {prefix}encoder.block.{i}.layer.1.DenseReluDense.wo.weight     (D,d_ff)
    // Plus, only on block 0:
    //   {prefix}encoder.block.0.layer.0.SelfAttention.
    //       relative_attention_bias.weight                  (num_buckets, num_heads)
    // And:
    //   {prefix}encoder.final_layer_norm.weight                        (D,)
    //
    // Throws std::runtime_error on a missing name, shape mismatch, or an
    // unsupported source dtype.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");

    // Forward over a length-L int32 token-id sequence (host pointer).
    //   ids: host pointer to L int32 token IDs in [0, vocab_size).
    //   out: (L, d_model) Tensor at the compute dtype, resized as needed.
    // brotensor::init() must have been called once before any forward.
    // The caller is responsible for sync_all() before reading `out` to host.
    void forward(const int32_t* ids, int L, brotensor::Tensor& out);

    const T5Config& config() const { return cfg_; }

private:
    struct Block {
        brotensor::Tensor ln0;            // (d_model, 1)
        brotensor::Tensor Wq, Wk, Wv, Wo; // each (d_model, d_model)
        brotensor::Tensor ln1;            // (d_model, 1)
        brotensor::Tensor wi_0, wi_1;     // each (d_ff, d_model)
        brotensor::Tensor wo;             // (d_model, d_ff)
    };

    // Rebuild the (num_heads*L, L) FP32 relative-position bias for length L.
    // Cached: only rebuilt when L changes.
    void rebuild_position_bias_(int L);

    T5Config cfg_;

    // Weights.
    brotensor::Tensor token_embed_;          // (vocab_size, d_model)
    std::vector<Block>   blocks_;
    brotensor::Tensor final_ln_;             // (d_model, 1)

    // relative_attention_bias.weight as host FP32 values, row-major
    // (num_buckets, num_heads). Populated at load time.
    std::vector<float> rel_attn_bias_;

    // Cached (num_heads*L, L) FP32 device tensor; rebuilt when L changes.
    brotensor::Tensor pos_bias_;
    int               pos_bias_L_ = -1;

    // Per-call scratch (kept alive across calls to avoid realloc).
    brotensor::Tensor ids_dev_;
    brotensor::Tensor x_;        // residual stream
    brotensor::Tensor n_;        // rms-norm output
    brotensor::Tensor attn_;     // attention sub-layer output
    brotensor::Tensor g_, l_;    // FFN gate / linear branches
    brotensor::Tensor ffn_out_;  // FFN sub-layer output
};

}  // namespace brodiffusion::t5
