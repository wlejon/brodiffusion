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
#include <stdexcept>
#include <string>
#include <vector>

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
//   Krea 2: text_embeddings  = Qwen3-VL tapped hidden states, flattened
//                              (max_seq*num_text_layers, text_hidden_dim); the
//                              partner text_embeddings_mask carries the
//                              per-token validity the DiT's text-fusion masks on.
//                              uncond_embeddings / uncond_embeddings_mask are
//                              the negative-prompt pair when has_uncond (real CFG).
struct Conditioning {
    brotensor::Tensor text_embeddings;
    brotensor::Tensor uncond_embeddings;
    brotensor::Tensor pooled;
    // Per-token validity mask travelling alongside text_embeddings /
    // uncond_embeddings. Empty (default) for every model that needs no mask
    // (SD1.5 / Flux / Sana / PixArt); Krea 2 fills it — an (max_seq, 1) FP32
    // tensor, 1.0 for a valid token row and 0.0 for a pad/filler row.
    brotensor::Tensor text_embeddings_mask;
    brotensor::Tensor uncond_embeddings_mask;
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

// Per-attention-block head-averaged image-query → text-key attention map,
// in the denoiser's forward traversal order. Each entry is an (Lq, Lk)
// tensor at the compute dtype: Lq image/spatial query tokens, Lk text-context
// tokens. A denoiser that exposes attention for inspection or steering fills
// one of these per forward_traced() call — the SD1.5 UNet's Transformer2D
// cross-attention blocks, a DiT's joint-attention blocks, etc.
using AttentionTrace = std::vector<brotensor::Tensor>;

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

    // ── CUDA-graph step-capture seam ──────────────────────────────────────
    //
    // A denoiser whose per-step forward is a fixed, allocation-stable op
    // sequence can split it for CUDA-graph capture:
    //
    //   prepare_step(timestep, prepared) — the host-dependent per-step work
    //     (e.g. the time-embedding chain) written into persistent device
    //     buffers. Runs once per step, OUTSIDE any graph capture.
    //   forward_body(latent, ..., branch, out) — the remaining pure
    //     device-side sequence. After prepare_step, calling forward_body for
    //     a branch must produce exactly what forward() produces for that
    //     branch and timestep.
    //
    // forward_body's contract, when supports_step_capture() is true: after
    // one warm-up call at a fixed (H_lat, W_lat) it performs no Tensor
    // (re)allocation of named buffers, no host reads/writes, and launches
    // every kernel on brotensor's current stream — so a Pipeline may record
    // it with brotensor::CudaGraphCapture and replay it each step after
    // refreshing the latent contents and calling prepare_step. Op-internal
    // scratch that is allocated AND freed within the body is fine (the
    // stream-ordered allocator turns those into paired graph memory nodes).
    //
    // The defaults mark the seam unsupported; the Pipeline then runs the
    // plain forward() every step.
    virtual bool supports_step_capture() const { return false; }
    virtual void prepare_step(float timestep,
                              const PreparedConditioning& prepared) {
        (void)timestep; (void)prepared;
        throw std::runtime_error(
            "Denoiser::prepare_step: step capture not supported");
    }
    virtual void forward_body(const brotensor::Tensor& latent,
                              int H_lat, int W_lat,
                              const PreparedConditioning& prepared,
                              Branch branch,
                              brotensor::Tensor& out) {
        (void)latent; (void)H_lat; (void)W_lat; (void)prepared;
        (void)branch; (void)out;
        throw std::runtime_error(
            "Denoiser::forward_body: step capture not supported");
    }

    // ── Attention trace / steering ────────────────────────────────────────
    //
    // Model-agnostic seam for cross-modal attention inspection and steering.
    // The SD1.5 UNet implements it over its Transformer2D cross-attention
    // blocks; a DiT implements it over its joint-attention blocks. A denoiser
    // with no traceable text↔image attention leaves the defaults.

    // Number of attention blocks that expose a head-averaged image→text map
    // (and accept a per-block pre-softmax logit bias). 0 — the default —
    // means this denoiser has no traceable cross-modal attention.
    virtual int num_xattn_blocks() const { return 0; }

    // Trace-mode forward: identical contract to forward(), but additionally
    // captures a per-block AttentionTrace and/or injects per-block pre-softmax
    // logit biases.
    //   attn_logit_biases — null, or exactly num_xattn_blocks() entries;
    //                       entry i is null (no bias) or an (Lq_i, Lk) FP32
    //                       tensor added to block i's scores before softmax.
    //   trace_out         — null, or filled with num_xattn_blocks() maps.
    // The default throws — override when num_xattn_blocks() > 0.
    virtual void forward_traced(
            const brotensor::Tensor& latent, int H_lat, int W_lat,
            float timestep, const PreparedConditioning& prepared,
            Branch branch,
            const std::vector<const brotensor::Tensor*>* attn_logit_biases,
            AttentionTrace* trace_out, brotensor::Tensor& out) {
        (void)latent; (void)H_lat; (void)W_lat; (void)timestep;
        (void)prepared; (void)branch; (void)attn_logit_biases;
        (void)trace_out; (void)out;
        throw std::runtime_error(
            "Denoiser::forward_traced: this denoiser has no traceable "
            "attention");
    }

    // Non-null only for the SD1.5 UNet — the hook for genuinely UNet-specific
    // operations (LoRA target resolution). The attention trace/steer API is
    // NOT routed through this: it is the model-agnostic pair above.
    virtual unet::UNet* as_unet() { return nullptr; }
    virtual const unet::UNet* as_unet() const { return nullptr; }
};

}  // namespace brodiffusion
