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

#include "brolm/clip.h"
#include "brodiffusion/controlnet.h"
#include "brodiffusion/denoiser.h"
#include "brodiffusion/dit/flux.h"
#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/lcm_scheduler.h"
#include "brodiffusion/model_config.h"
#include "brodiffusion/scheduler.h"
#include "brolm/t5.h"
#include "brolm/tokenizer.h"
#include "brolm/tokenizer_t5.h"
#include "brodiffusion/unet.h"
#include "brodiffusion/vae.h"

#include "brotensor/tensor.h"

#include <cstdint>
#include <memory>
#include <optional>
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
    brolm::clip::TextEncoderConfig  text_encoder;  // CLIP — both classes
    brolm::t5::T5Config             t5;            // Flux
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
    // Philox key for initial-noise + per-step LCM noise. Initialised from
    // opts.seed in prime(); branched states should mutate this on the clone
    // (e.g. XOR a branch id) to diverge their noise streams.
    std::uint64_t rng_key = 0;
    int step_index = 0;           // 0-based: how many step_once() calls have run
    int n_steps    = 0;           // total scheduled steps for this generation
    int H_lat      = 0;
    int W_lat      = 0;

    // Deep clone: copies the latent on the active device. RNG and ints are
    // trivial copies.
    PipelineState clone() const;
};

// Source of the initial latent noise.
//   Internal — brotensor's device-side Philox stream keyed by `seed`. Same
//              stream is reused for per-step LCM noise.
//   Torch    — bit-compatible with torch.randn() filled by a CPU Generator
//              seeded with `seed`. Lets `--seed N` reproduce a PyTorch
//              reference run's starting latent exactly, so the two pipelines
//              can be diffed with the RNG removed as a variable.
enum class NoiseSource { Internal, Torch };

// One conditioning image for a single registered ControlNet (SD1.5 only).
// `scale` is the per-net `conditioning_scale` (diffusers parity); 0.0
// disables that net's contribution for the run, 1.0 is the HF default.
// `start_step` / `end_step` are the schedule fractions (in [0, 1]) over
// which this net contributes — outside the window, the net is skipped and
// its residual stays zero. Defaults cover the full schedule. The window is
// half-open on the right: a step at fractional position `f` runs the net
// iff `start_step <= f < end_step`.
struct ControlNetInput {
    std::string image_path;
    float       scale      = 1.0f;
    float       start_step = 0.0f;
    float       end_step   = 1.0f;
};

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

    // ── img2img / inpaint priming ─────────────────────────────────────────
    // If non-empty, prime() encodes this image with the VAE encoder and
    // noises it to the appropriate point in the schedule (governed by
    // `strength`) instead of starting from pure Gaussian noise. The image
    // is decoded by broimage and resized to (width, height). For img2img
    // the model only denoises the remaining (1 - strength) fraction of the
    // schedule, so the generated image stays close to the init.
    //
    // SD1.5 only for now; throws on Flux. Empty path = txt2img (existing
    // behavior). Throws if both init_image_path and init_noise are set.
    // noise_source / init_noise are ignored when init_image_path is set.
    std::string init_image_path;

    // 0..1. Higher = more noise added = more freedom from the init image.
    // 1.0 reduces to (almost) full-schedule txt2img against the encoded init.
    // Ignored when init_image_path is empty.
    float strength = 0.8f;

    // VAE encode mode: deterministic (use the mean, default) or sampled
    // (mean + exp(0.5*logvar) * eps, where eps is drawn from the same Philox
    // stream as the schedule noise at a non-overlapping counter offset).
    // Ignored when init_image_path is empty.
    bool vae_encode_sample = false;

    // ── ControlNet (SD1.5 only) ───────────────────────────────────────────
    // One ControlNetInput per registered ControlNet (in the order they were
    // added via add_controlnet). When non-empty, prime() loads each control
    // image and step_once() runs every registered ControlNet, summing their
    // residuals position-wise (each weighted by its own ControlNetInput::
    // scale) before feeding them into UNet's skip connections. Size must
    // equal the number of registered ControlNets. SD1.5 only; throws on
    // Flux. Empty = no ControlNet for this run.
    std::vector<ControlNetInput> controls;

    // Inpaint mask path. When non-empty, the pipeline runs in inpaint mode:
    // init_image_path must ALSO be set, and at each scheduler step (except
    // the final one) the unmasked region of the latent is replaced by a
    // re-noised version of the encoded init latent at the next timestep.
    // The mask is decoded as 1-channel and resized to latent dims with
    // nearest-neighbor (see load_mask_as_latent). White (>= 128) = inpaint,
    // black = keep. SD1.5 only; throws on Flux.
    std::string mask_image_path;
};

class Pipeline {
public:
    // StableDiffusion constructor: builds a UNet denoiser. Valid only when
    // cfg.model_class == ModelClass::StableDiffusion.
    Pipeline(const PipelineConfig& cfg, brolm::clip::Tokenizer tokenizer);

    // Flux constructor: builds a FluxDenoiser plus the second (T5) text
    // encoder + tokenizer. Valid only when cfg.model_class == ModelClass::Flux.
    Pipeline(const PipelineConfig& cfg, brolm::clip::Tokenizer clip_tok,
             brolm::t5::Tokenizer t5_tok);

    // Build a fully-loaded Pipeline from a diffusers model directory: reads the
    // JSON configs, constructs the right sub-modules, loads all component
    // weights and tokenizers. Supports StableDiffusion and Flux model
    // directories. Pipeline is move-only; this returns by value.
    struct ModelDirOptions {
        // Quantize the denoiser (SD1.5 U-Net / Flux transformer) — and, for
        // Flux, the T5-XXL encoder — to INT8 weight-only (W8A16) while
        // loading. GPU-only; ignored with a warning on the CPU backend.
        // Flux.1 FP16 (~24 GB) does not fit a 24 GB card next to T5: INT8
        // (~12 GB) is how Flux runs there at all.
        bool quantize = false;
    };
    static Pipeline from_model_dir(const std::string& model_dir,
                                   const ModelDirOptions& opts = {});

    // Move-only (owns move-only sub-modules; copies make no sense). The
    // move members and destructor are defined in pipeline.cpp where
    // StepGraphSession is complete.
    ~Pipeline();
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;
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

    // Register a ControlNet safetensors file. The returned index is the
    // addressing key used elsewhere (remove_controlnet, plus the position
    // into GenerateOptions::controls). SD1.5 only; throws on Flux. The
    // default config matches HF's lllyasviel/sd-controlnet-* zoo; pass an
    // explicit ControlNetConfig if a non-default checkpoint shape is needed.
    int add_controlnet(const brotensor::safetensors::File& f);
    int add_controlnet(const brotensor::safetensors::File& f,
                       const controlnet::ControlNetConfig& cfg);

    // Drop one registered ControlNet by index. Subsequent indices shift down.
    void remove_controlnet(int index);

    // Drop all registered ControlNets.
    void clear_controlnets();

    // Number of currently registered ControlNets.
    int num_controlnets() const { return static_cast<int>(controlnets_.size()); }

    // True iff at least one ControlNet has been loaded.
    bool has_controlnet() const { return !controlnets_.empty(); }

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
    brolm::clip::Tokenizer    tokenizer_;       // CLIP — both classes
    brolm::clip::TextEncoder  text_encoder_;    // CLIP — both classes
    std::optional<brolm::t5::Tokenizer>   t5_tokenizer_;   // Flux only
    std::optional<brolm::t5::TextEncoder> t5_encoder_;     // Flux only
    std::unique_ptr<Denoiser> denoiser_;
    vae::Decoder              vae_;
    // VAE encoder mirrors vae_; used by img2img / inpaint priming. Always
    // constructed (config derived from cfg.vae); weights are loaded by every
    // load_weights() overload. SD1.5 only — the Flux img2img path is TODO.
    // Robustness: if encoder weights are missing in the safetensors file
    // (some decoder-only derivative checkpoints), the load throws clearly.
    vae::Encoder              vae_encoder_;
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

    // Inpaint blending state. Built in prime() when opts.mask_image_path is
    // non-empty; consumed by step_once() to blend the unmasked region of the
    // generated latent with a re-noised version of x0 at each step. `mask_b`
    // is the per-channel-broadcast mask (shape matches the latent); raw
    // `mask` of shape (1, H_lat*W_lat) is kept around for tests / debug.
    // Empty tensors + inpaint_active_=false mean inpaint mode is off for the
    // current generation. Lives on Pipeline (not PipelineState) — shared
    // across branched states from the same prime() call, same as the K/V
    // cache.
    brotensor::Tensor inpaint_x0_;          // (1, C_lat*H_lat*W_lat)
    brotensor::Tensor inpaint_mask_;        // (1, H_lat*W_lat), {0,1}
    brotensor::Tensor inpaint_mask_b_;      // (1, C_lat*H_lat*W_lat) broadcast
    brotensor::Tensor inpaint_one_minus_b_; // (1, C_lat*H_lat*W_lat) = 1 - mask_b
    brotensor::Tensor inpaint_renoise_buf_; // (1, C_lat*H_lat*W_lat) scratch
    brotensor::Tensor inpaint_noise_step_;  // (1, C_lat*H_lat*W_lat) Philox draw
    bool              inpaint_active_ = false;

    // ControlNet plumbing. `controlnets_` is the registered stack (lifetime
    // tied to Pipeline; appended via add_controlnet, dropped via
    // remove_controlnet / clear_controlnets). `control_inputs_` is the
    // per-run input list copied from GenerateOptions::controls in prime();
    // size == controlnets_.size() when active. `control_images_` holds one
    // image tensor per registered ControlNet, built in prime() at the
    // active device + compute dtype and reused for every step in the
    // current generation. `controlnet_active_` mirrors "GenerateOptions
    // .controls is non-empty AND the registered count matches" — checked
    // by step_once to branch into the residual-aware UNet path.
    //
    // Step buffers: `cn_down_residuals_` / `cn_mid_residual_` are the
    // summed-across-nets residuals fed into the UNet skips; the first net
    // writes directly into them and subsequent nets accumulate via
    // bt::add_inplace through the *_scratch_ buffers. With one registered
    // ControlNet the scratch buffers are unused.
    std::vector<std::unique_ptr<controlnet::ControlNet>> controlnets_;
    std::vector<ControlNetInput>                         control_inputs_;
    std::vector<brotensor::Tensor>                       control_images_;
    bool                                                 controlnet_active_ = false;
    std::vector<brotensor::Tensor>                       cn_down_residuals_;
    brotensor::Tensor                                    cn_mid_residual_;
    std::vector<brotensor::Tensor>                       cn_down_residuals_scratch_;
    brotensor::Tensor                                    cn_mid_residual_scratch_;

    // CUDA-graph denoising-step session (defined in pipeline.cpp; CUDA-only
    // payload). When the denoiser supports step capture and the latent is
    // CUDA-resident, step_once runs the first two steps of a generation
    // eagerly through the capture seam (settling every scratch buffer at its
    // high-water capacity), records the cond(+uncond) forward bodies as one
    // CUDA graph at the end of the second step, and replays that graph for
    // every later step — one launch instead of hundreds. The session is keyed
    // on (latent pointer, prepared payload, H, W, CFG); any mismatch (new
    // prime(), forked state, resolution change) falls back to eager stepping
    // and re-captures. Trace-mode and ControlNet steps always run eager.
    struct StepGraphSession;
    std::unique_ptr<StepGraphSession> step_graph_;
    // Runs cond(+uncond) denoiser passes for one step via the capture seam,
    // managing warm-up / capture / replay. Outputs land in noise_pred_cond_
    // (+ noise_pred_uncond_), exactly like the eager path.
    void step_denoise_captured_(PipelineState& state, float t, bool do_cfg);
};

}  // namespace brodiffusion::pipeline
