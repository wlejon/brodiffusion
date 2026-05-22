#pragma once

// Stable Diffusion 1.5 text-to-image pipeline.
//
// Owns the tokenizer, CLIP text encoder, UNet, VAE decoder, and DDIM
// scheduler — wires them into a single `generate(prompt, ...)` call that
// produces a raw FP32 RGB image in [-1, 1] (the VAE output range). Callers
// clamp + rescale to uint8 / encode to PNG outside this library.
//
// Inference-only, batch size N = 1. Runs on whichever backend brotensor
// resolves at runtime — CPU by default, CUDA when a GPU is available — at
// that backend's compute dtype (FP32 on CPU, FP16 on a GPU). Classifier-free
// guidance runs the U-Net twice per step (conditional + unconditional)
// rather than batching, since the rest of the inference stack is N=1.

#include "brodiffusion/clip.h"
#include "brodiffusion/denoiser.h"
#include "brodiffusion/dit/flux.h"
#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/lcm_scheduler.h"
#include "brodiffusion/model_config.h"
#include "brodiffusion/scheduler.h"
#include "brodiffusion/t5.h"
#include "brodiffusion/tokenizer.h"
#include "brodiffusion/tokenizer_t5.h"
#include "brodiffusion/unet.h"
#include "brodiffusion/vae.h"

#include "brotensor/tensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::pipeline {

struct PipelineConfig {
    // Which model family this pipeline runs. StableDiffusion uses the UNet
    // denoiser + CLIP; Flux uses the DiT denoiser + CLIP (pooled) + T5.
    ModelClass               model_class = ModelClass::StableDiffusion;
    unet::UNetConfig         unet;          // StableDiffusion
    dit::FluxConfig          flux;          // Flux
    vae::DecoderConfig       vae;
    clip::TextEncoderConfig  text_encoder;  // CLIP — both classes
    t5::T5Config             t5;            // Flux
    int                      t5_max_length = 512;
    // DDIM (default, vanilla SD1.5) or LCM (latent-consistency, distilled
    // checkpoints with unet.time_cond_proj_dim > 0) or FlowMatch (Flux). The
    // pipeline branches on the active alternative; existing call sites that
    // don't set this keep working unchanged.
    std::variant<scheduler::DDIMConfig, scheduler::LCMConfig,
                 scheduler::FlowMatchConfig> scheduler;
};

// Snapshot of mid-generation state. Cheap to fork — only the latent
// carries device memory; the rest is host-side scalars / RNG state.
//
// The cross-attention K/V cache (which holds the projected text context) is
// NOT part of this snapshot — it lives on Pipeline, is rebuilt at every
// prime(), and stays constant across all branched states within a generation.
// Forking a state therefore costs one latent clone, not a full UNet replay.
//
// Used by cross-attention tree search: at any step, .clone() the current
// state, advance the clone differently from the original (different RNG, a
// different attn_logit_bias, etc.), score, and pick a winner.
struct PipelineState {
    brotensor::Tensor latent;  // (1, C_lat * H_lat * W_lat) at compute dtype
    std::mt19937_64 rng;          // initial-noise + per-step LCM noise stream
    int step_index = 0;           // 0-based: how many step_once() calls have run
    int n_steps    = 0;           // total scheduled steps for this generation
    int H_lat      = 0;
    int W_lat      = 0;

    // Deep clone: copies the latent on the active device. RNG and ints are
    // trivial copies.
    PipelineState clone() const;
};

// Source of the initial latent noise.
//   Internal — brodiffusion's own std::mt19937_64 + std::normal_distribution
//              stream (the historical default).
//   Torch    — bit-compatible with torch.randn() filled by a CPU Generator
//              seeded with `seed`. Lets `--seed N` reproduce a PyTorch
//              reference run's starting latent exactly, so the two pipelines
//              can be diffed with the RNG removed as a variable.
enum class NoiseSource { Internal, Torch };

struct GenerateOptions {
    // Image dimensions in pixels. Latent dims are H/8, W/8. Both must be
    // multiples of 8 and at least 8 (so latent dims are >= 1).
    int height = 512;
    int width  = 512;

    int   num_inference_steps = 30;
    float guidance_scale      = 7.5f;   // CFG; 1.0 disables (uncond is skipped).
    std::string negative_prompt;

    // RNG seed for the initial latent noise.
    std::uint64_t seed = 0;

    // Which RNG produces the initial latent noise from `seed`. Ignored when
    // init_noise is supplied. See NoiseSource above.
    NoiseSource noise_source = NoiseSource::Internal;

    // Optional explicit initial latent noise, in raw N(0,1) units (i.e. the
    // output of randn(), before init_noise_sigma is applied), NCHW flat,
    // length = C_lat * (height/8) * (width/8). When non-empty, prime() uses
    // these values instead of seeding the RNG — this lets a caller feed a
    // noise tensor generated elsewhere (e.g. a PyTorch reference run) so the
    // two implementations start from a bit-identical latent and any output
    // difference is attributable to the model, not the RNG. The scheduler's
    // init_noise_sigma is still applied on top.
    std::vector<float> init_noise;
};

class Pipeline {
public:
    // StableDiffusion constructor: builds a UNet denoiser. Valid only when
    // cfg.model_class == ModelClass::StableDiffusion.
    Pipeline(const PipelineConfig& cfg, clip::Tokenizer tokenizer);

    // Flux constructor: builds a FluxDenoiser plus the second (T5) text
    // encoder + tokenizer. Valid only when cfg.model_class == ModelClass::Flux.
    Pipeline(const PipelineConfig& cfg, clip::Tokenizer clip_tok,
             t5::Tokenizer t5_tok);

    // Build a fully-loaded Pipeline from a diffusers model directory: reads the
    // JSON configs, constructs the right sub-modules, loads all component
    // weights and tokenizers. Supports StableDiffusion and Flux model
    // directories. Pipeline is move-only; this returns by value.
    static Pipeline from_model_dir(const std::string& model_dir);

    // Move-only (owns move-only sub-modules; copies make no sense).
    Pipeline(Pipeline&&) noexcept = default;
    Pipeline& operator=(Pipeline&&) noexcept = default;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Load every sub-module's weights from a single safetensors file using
    // the prefixes used by an SD1.5 full checkpoint:
    //   "cond_stage_model.transformer.text_model." for CLIP
    //   "model.diffusion_model."                   for the U-Net
    //   "first_stage_model.decoder."               for the VAE decoder
    void load_weights(const brotensor::safetensors::File& f);

    // Load with explicit prefixes (e.g. when sub-modules ship in separate
    // diffusers exports).
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& text_prefix,
                      const std::string& unet_prefix,
                      const std::string& vae_prefix);

    // Load each module from its own diffusers-format safetensors file. Uses
    // the diffusers default prefixes ("text_model.", "", "decoder.").
    void load_weights(const brotensor::safetensors::File& text_file,
                      const brotensor::safetensors::File& unet_file,
                      const brotensor::safetensors::File& vae_file);

    // Merge a LoRA file's deltas into the base UNet and CLIP weights.
    //
    // Must be called *after* load_weights() and *before* generate(). The
    // cross-attention K/V cache is re-primed inside every generate() call,
    // so calling apply_lora() between generates is safe — the next generate
    // will rebuild the cache against the updated K/V projections.
    //
    // `scale` is a user multiplier applied on top of the per-LoRA alpha/rank
    // factor (default 1.0 = use as-shipped). Negative values are allowed
    // (subtract / undo). May be called more than once to stack multiple
    // LoRAs.
    //
    // Accepts both kohya-ss/A1111 and diffusers/PEFT key conventions; the
    // format is auto-detected from the key prefixes. Throws if the file
    // contains LoRA tensors that don't map to a known SD1.5 target.
    void apply_lora(const brotensor::safetensors::File& f, float scale = 1.0f);

    // Generate an image. Returns a freshly-allocated host buffer of
    // 3 * height * width FP32 values in NCHW (C=3, [-1, 1]).
    //
    // Internally: prime() → loop(step_once) → decode(). Bit-equivalent to
    // calling those three primitives directly.
    std::vector<float> generate(std::string_view prompt,
                                const GenerateOptions& opts);

    // ── Step-wise API (for cross-attention tree search and similar) ───────
    //
    // prime():        encode prompt(s), build xattn caches, allocate initial
    //                 latent noise. Returns a state with step_index=0.
    // step_once():    advance one denoising step (mutates state). If
    //                 trace_out is non-null, the UNet forward runs in trace
    //                 mode (no K/V cache reuse, no INT8 quantization) and
    //                 fills the attention-map trace. Throws if state is
    //                 already at n_steps.
    // decode():       VAE-decode a state's latent to an FP32 host buffer
    //                 (same shape and units as generate()).
    //
    // The xattn cache lives on `this` and is shared across all branched
    // states from the same prime() call. Calling apply_lora() between
    // prime() and step_once() leaves the cache stale — re-prime() to refresh.
    PipelineState prime(std::string_view prompt, const GenerateOptions& opts);
    // If `attn_logit_biases` is non-null, trace mode is used and the bias
    // vector is forwarded to the active denoiser's forward_traced (length must
    // equal num_xattn_blocks(); per-entry nulls allowed for no-bias blocks).
    // When biases are supplied, trace_out may still be null — the trace is
    // computed but discarded. Trace mode requires a denoiser whose
    // num_xattn_blocks() > 0.
    void step_once(PipelineState& state, const GenerateOptions& opts,
                   AttentionTrace* trace_out = nullptr,
                   const std::vector<const brotensor::Tensor*>*
                       attn_logit_biases = nullptr);
    std::vector<float> decode(const PipelineState& state);

    // Accessors for research callers (e.g. tree search needs xattn block count
    // to size the bias vector). Throws if the active denoiser is not a UNet.
    const unet::UNet& unet() const;

    // Denoiser-generic count of traceable / steerable cross-attention blocks
    // for the active denoiser (16 for the SD1.5 UNet, 57 for the Flux DiT,
    // 0 for a denoiser with no trace support). Lets a caller size an
    // attn_logit_biases vector or a trace array without knowing the denoiser
    // type. Always valid — never throws.
    int num_xattn_blocks() const { return denoiser_->num_xattn_blocks(); }

    const PipelineConfig& config() const { return cfg_; }

private:
    void encode_prompt_(std::string_view prompt, brotensor::Tensor& out);

    PipelineConfig            cfg_;
    ModelClass                model_class_;
    clip::Tokenizer           tokenizer_;       // CLIP — both classes
    clip::TextEncoder         text_encoder_;    // CLIP — both classes
    std::optional<t5::Tokenizer>   t5_tokenizer_;   // Flux only
    std::optional<t5::TextEncoder> t5_encoder_;     // Flux only
    std::unique_ptr<Denoiser> denoiser_;
    vae::Decoder              vae_;
    std::variant<scheduler::DDIM, scheduler::LCM, scheduler::FlowMatch>
        scheduler_;

    // Model-agnostic conditioning, rebuilt each prime(). `conditioning_` keeps
    // the raw text context around for trace-mode access; `prepared_` holds the
    // per-denoiser prepared payload (K/V caches), shared across all branched
    // states from the same prime().
    Conditioning          conditioning_;
    PreparedConditioning  prepared_;

    // Working buffers reused across step_once() calls. The current latent
    // lives on PipelineState, not here.
    brotensor::Tensor noise_pred_cond_, noise_pred_uncond_;
    brotensor::Tensor decoded_, scratch_;
    // Per-step Gaussian noise used by LCM (resampled at every non-final step
    // from the same RNG stream as the initial latent noise). Unused in DDIM.
    brotensor::Tensor noise_step_;
};

}  // namespace brodiffusion::pipeline
