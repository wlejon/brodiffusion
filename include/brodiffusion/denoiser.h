#pragma once

// Denoiser — the model-agnostic noise / velocity-prediction backbone.
//
// brodiffusion's Pipeline owns one Denoiser via unique_ptr and is otherwise
// model-agnostic. The SD1.5 UNet (unet::UNet) implements it today; a Flux DiT
// denoiser will implement it later. The trace / cross-attention tree-search
// API is irreducibly UNet-shaped and stays on Pipeline, reached through
// as_unet() (null for non-UNet denoisers).

#include "brotensor/tensor.h"

#include <memory>
#include <string>

namespace brotensor::safetensors { class File; }

namespace brodiffusion {

namespace unet { class UNet; }

// What the denoiser's forward output represents.
//   Epsilon  — predicted noise epsilon (SD1.5 UNet; DDIM / LCM schedulers).
//   Velocity — rectified-flow velocity v (Flux DiT; flow-match scheduler).
enum class PredictionType { Epsilon, Velocity };

// Which classifier-free-guidance branch a forward() call evaluates.
enum class Branch { Cond, Uncond };

// Model-agnostic conditioning, assembled by the Pipeline from whichever
// encoders a given model needs.
//   SD1.5: text_embeddings  = CLIP token sequence (L, cross_attention_dim).
//          uncond_embeddings = negative-prompt CLIP sequence when has_uncond.
//          guidance          = LCM guidance-scale input w (used only when the
//                              UNet was built with time_cond_proj_dim > 0).
//   Flux:  text_embeddings  = T5 token sequence (L, 4096).
//          pooled            = CLIP pooled vector (1, 768).
//          guidance          = distilled guidance scalar (flux-dev; 0 schnell).
struct Conditioning {
    brotensor::Tensor text_embeddings;
    brotensor::Tensor uncond_embeddings;
    brotensor::Tensor pooled;
    float guidance   = 0.0f;
    bool  has_uncond = false;
};

// Opaque, move-only per-model prepared conditioning. The concrete payload
// (UNet K/V caches; Flux pre-projected T5 context) lives behind Impl so this
// header stays free of model-specific headers. A denoiser's prepare() builds
// one; its forward() consumes it.
class PreparedConditioning {
public:
    // Base for the per-model payload. Concrete subclasses are defined in the
    // owning denoiser's .cpp and never escape it.
    struct Impl {
        virtual ~Impl() = default;
    };

    PreparedConditioning() = default;
    explicit PreparedConditioning(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}

    PreparedConditioning(PreparedConditioning&&) noexcept = default;
    PreparedConditioning& operator=(PreparedConditioning&&) noexcept = default;
    PreparedConditioning(const PreparedConditioning&) = delete;
    PreparedConditioning& operator=(const PreparedConditioning&) = delete;

    Impl* get() const { return impl_.get(); }
    explicit operator bool() const { return impl_ != nullptr; }

private:
    std::unique_ptr<Impl> impl_;
};

// Abstract noise / velocity-prediction backbone.
class Denoiser {
public:
    virtual ~Denoiser() = default;

    // Load weights from a safetensors file; `prefix` is the diffusers key
    // prefix for this sub-module.
    virtual void load_weights(const brotensor::safetensors::File& f,
                              const std::string& prefix) = 0;

    // Finalize weights for inference (e.g. INT8 quantisation). Idempotent.
    virtual void finalize_weights() = 0;

    // Pre-process conditioning once per generation (e.g. project text-context
    // K/V). The result is consumed by forward().
    virtual PreparedConditioning prepare(const Conditioning& cond) = 0;

    // One denoising forward pass for the given CFG branch.
    //   latent:   (1, latent_channels()*H_lat*W_lat) at compute_dtype().
    //   timestep: continuous timestep value.
    //   prepared: result of prepare() for this generation.
    //   out:      (1, latent_channels()*H_lat*W_lat), resized as needed.
    virtual void forward(const brotensor::Tensor& latent,
                         int H_lat, int W_lat,
                         float timestep,
                         const PreparedConditioning& prepared,
                         Branch branch,
                         brotensor::Tensor& out) = 0;

    virtual int latent_channels() const = 0;
    virtual PredictionType prediction_type() const = 0;
    // Whether the pipeline should run a separate uncond branch + CFG combine
    // when the user requests guidance.
    virtual bool uses_cfg() const = 0;
    virtual brotensor::Dtype compute_dtype() const = 0;

    // Non-null only for the SD1.5 UNet. The trace / tree-search API is
    // UNet-shaped and stays on Pipeline, gated through this hook.
    virtual unet::UNet* as_unet() { return nullptr; }
    virtual const unet::UNet* as_unet() const { return nullptr; }
};

}  // namespace brodiffusion
