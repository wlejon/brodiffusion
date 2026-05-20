#pragma once

// CLIP ViT-L/14 image encoder.
//
// Pairs with brodiffusion/clip.h's text encoder (which is the same ViT-L/14
// architecture's text branch). Used by clip_score.h to score VAE-decoded
// SD outputs against a prompt.
//
// Architecture (default config matches openai/clip-vit-large-patch14):
//   patch_embed:  conv2d 3 -> 1024, k=14 s=14 (no bias)             -> 256 tokens
//   class_embed:  learnable (1024,) prepended                       -> 257 tokens
//   position_embed: (257, 1024) added
//   pre_layernorm
//   24 layers:
//     ln1 -> self-attn (biased q/k/v/out, non-causal) -> +
//     ln2 -> fc1 (4096) -> QuickGELU -> fc2 (1024)    -> +
//   post_layernorm
//
// The final (1, 1024) embedding is the post-LN CLS token (row 0). Caller
// applies the cross-modal projection (clip_score does this).
//
// Inference-only, batch size N = 1. Runs at the pipeline compute dtype —
// FP32 on CPU, FP16 on a GPU backend. Input is a preprocessed activation:
// (1, 3 * 224 * 224) NCHW, already mean/std normalised in CLIP space.
// clip_score.h owns the host-side resize + normalisation that gets you from
// a [-1, 1] SD output to that input.

#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::clip_image {

struct ImageEncoderConfig {
    int image_size      = 224;
    int patch_size      = 14;
    int in_channels     = 3;
    int hidden_dim      = 1024;
    int num_heads       = 16;       // head_dim = 64
    int num_layers      = 24;
    int intermediate_dim = 4096;    // FFN inner width
    float layer_norm_eps = 1.0e-5f;
};

// Number of tokens fed to the transformer = 1 (CLS) + (image/patch)^2.
inline int num_tokens(const ImageEncoderConfig& cfg) {
    const int g = cfg.image_size / cfg.patch_size;
    return 1 + g * g;
}

class ImageEncoder {
public:
    explicit ImageEncoder(const ImageEncoderConfig& cfg);
    ~ImageEncoder();

    ImageEncoder(const ImageEncoder&) = delete;
    ImageEncoder& operator=(const ImageEncoder&) = delete;
    ImageEncoder(ImageEncoder&&) noexcept = default;
    ImageEncoder& operator=(ImageEncoder&&) noexcept = default;

    // Load weights from a Hugging Face openai/clip-vit-large-patch14
    // safetensors export. Default prefix matches the HF `CLIPModel` layout:
    //   "vision_model.embeddings.patch_embedding.weight"     (1024, 3, 14, 14)
    //   "vision_model.embeddings.class_embedding"            (1024,)
    //   "vision_model.embeddings.position_embedding.weight"  (257, 1024)
    //   "vision_model.pre_layrnorm.{weight,bias}"            (1024,)   [sic — HF typo, preserved]
    //   "vision_model.encoder.layers.{i}.layer_norm{1,2}.{weight,bias}"
    //   "vision_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}"
    //   "vision_model.encoder.layers.{i}.mlp.fc{1,2}.{weight,bias}"
    //   "vision_model.post_layernorm.{weight,bias}"          (1024,)
    //
    // All loaded at the compute dtype; F16 or F32 source weights are
    // converted host-side on the fly. If `pre_layrnorm_alt` is true, the
    // loader accepts "pre_layernorm.*" as a fallback for the typo-fixed
    // variant some forks ship.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "vision_model.",
                      bool pre_layrnorm_alt = true);

    // Forward pass.
    //   pixels: (1, 3 * 224 * 224) Tensor at the compute dtype, mean/std-
    //           normalised in CLIP space (see clip_score::preprocess for the
    //           host-side pipeline that produces this).
    //   cls_out: (1, hidden_dim), *post_layernorm(CLS token)*.
    //            Resized as needed. This is the input to the visual
    //            projection that produces the final cross-modal embedding.
    void forward(const brotensor::Tensor& pixels,
                 brotensor::Tensor& cls_out);

    const ImageEncoderConfig& config() const { return cfg_; }

private:
    struct Layer {
        brotensor::Tensor ln1_g, ln1_b;
        brotensor::Tensor Wq, bq, Wk, bk, Wv, bv, Wo, bo;
        brotensor::Tensor ln2_g, ln2_b;
        brotensor::Tensor fc1_W, fc1_b, fc2_W, fc2_b;
    };

    ImageEncoderConfig cfg_;

    brotensor::Tensor patch_W_;       // (D, 3 * P * P)  — conv2d filter
    brotensor::Tensor class_embed_;   // (1, D)          — broadcast into row 0
    brotensor::Tensor position_embed_;// (T, D)
    brotensor::Tensor pre_g_, pre_b_;
    std::vector<Layer>   layers_;
    brotensor::Tensor post_g_, post_b_;

    // Scratch buffers reused across calls.
    brotensor::Tensor patch_nchw_;    // (1, D * G * G)
    brotensor::Tensor seq_;           // (T, D) residual stream
    brotensor::Tensor ln_out_;
    brotensor::Tensor proj_out_;
    brotensor::Tensor ffn_mid_, ffn_act_, ffn_out_;
    brotensor::Tensor post_;          // (T, D)
};

}  // namespace brodiffusion::clip_image
