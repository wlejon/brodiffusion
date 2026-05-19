#pragma once

// CLIP score: cosine similarity between a CLIP-projected image and a
// CLIP-projected prompt. The natural off-the-shelf reward for sd_mcts.
//
// Wraps:
//   - the existing clip::Tokenizer + clip::TextEncoder (text branch)
//   - clip_image::ImageEncoder              (image branch)
//   - text_projection (768, 768)            (text  -> shared space)
//   - visual_projection (768, 1024)         (image -> shared space)
//
// Usage in an sd_mcts run:
//   CLIPScorer scorer(tokenizer, text_encoder, image_encoder);
//   scorer.load_projections(clip_full_safetensors_file);
//   scorer.set_prompt("a photo of an astronaut riding a horse");
//   sampler.set_scorer([&](const std::vector<float>& img, int H, int W) {
//       return scorer.score(img, H, W);
//   });
//
// `score()` is callable as many times as you like after set_prompt(); the
// text-side projection is cached on the active prompt.
//
// Numerics: image preprocessing (resize + normalize) runs on the host —
// 512x512 -> 224x224 is ~150K floats per call, trivial. The transformer
// forward is on the GPU, FP16. Cosine similarity is the final dot
// product on 768 host floats.

#include "brodiffusion/clip.h"
#include "brodiffusion/clip_image.h"
#include "brodiffusion/tokenizer.h"

#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brodiffusion::safetensors { class File; }

namespace brodiffusion::clip_score {

struct Config {
    // Shared cross-modal embedding dim. Both projections land here.
    int projection_dim = 768;

    // CLIP image preprocessing (openai/clip-vit-large-patch14 defaults).
    float mean[3]   = {0.48145466f, 0.4578275f, 0.40821073f};
    float std_[3]   = {0.26862954f, 0.26130258f, 0.27577711f};
};

class CLIPScorer {
public:
    CLIPScorer(const clip::Tokenizer&        tokenizer,
               clip::TextEncoder&            text_encoder,
               clip_image::ImageEncoder&     image_encoder,
               Config                        cfg = {});

    // Load the two cross-modal projections from a Hugging Face
    // openai/clip-vit-large-patch14 safetensors export. Default keys
    // (top-level, no prefix):
    //   "visual_projection.weight"  (projection_dim, vision_hidden_dim)
    //   "text_projection.weight"    (projection_dim, text_hidden_dim)
    //
    // Some forks (e.g. an `open_clip` re-export) ship them under a model.
    // prefix — pass that as `prefix` if needed.
    void load_projections(const safetensors::File& f,
                          const std::string& prefix = "");

    // Tokenize, encode, pool at EOS, project. Cache the projected text
    // feature on the scorer; subsequent score() calls dot against it.
    // The EOS index is argmax(token_ids) — CLIP fills the tail with EOS
    // padding, and the BOS token id (49406) is less than EOS (49407), so
    // argmax lands on the *first* EOS-padded slot (i.e. the post-prompt
    // position the official CLIP pooling rule uses).
    void set_prompt(std::string_view prompt);

    // Score a VAE-decoded image against the cached prompt. Returns the
    // cosine similarity in [-1, 1] (higher = better alignment).
    //   image: (3 * H * W) FP32, NCHW planar, values in [-1, 1] (the
    //          pipeline::Pipeline::generate output format).
    //   H, W:  pixel dimensions of the image.
    float score(const std::vector<float>& image, int H, int W);

    // Accessor exposing the active text feature (post-projection,
    // L2-normalised, length projection_dim). Empty if set_prompt was
    // never called. Useful for sanity-checking.
    const std::vector<float>& text_feature() const { return text_feat_; }

    const Config& config() const { return cfg_; }

private:
    // Host-side: resize input image to 224x224 (bilinear, per-channel) and
    // CLIP-normalise. Output is interleaved planar NCHW FP16 bits, length
    // 3 * 224 * 224, ready for upload via brotensor::upload_fp16.
    std::vector<std::uint16_t> preprocess_(const std::vector<float>& image,
                                           int H, int W) const;

    const clip::Tokenizer&     tok_;
    clip::TextEncoder&         text_enc_;
    clip_image::ImageEncoder&  image_enc_;
    Config                     cfg_;

    // Projection weights. Stored on GPU as FP16; same layout as a linear
    // layer's W (rows=out, cols=in).
    brotensor::GpuTensor visual_proj_W_;   // (P, vision_D)
    brotensor::GpuTensor text_proj_W_;     // (P, text_D)

    // Scratch.
    brotensor::GpuTensor pixels_dev_;      // (1, 3*224*224) FP16
    brotensor::GpuTensor img_cls_;         // (1, vision_D)
    brotensor::GpuTensor img_proj_;        // (1, P)
    brotensor::GpuTensor text_hidden_;     // (L, text_D) — TextEncoder output
    brotensor::GpuTensor text_pooled_;     // (1, text_D)
    brotensor::GpuTensor text_proj_;       // (1, P)

    // Cached, L2-normalised, length projection_dim.
    std::vector<float> text_feat_;
};

}  // namespace brodiffusion::clip_score
