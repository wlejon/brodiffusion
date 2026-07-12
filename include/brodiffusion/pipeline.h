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
#include "brodiffusion/cond_control.h"
#include "brodiffusion/controlnet.h"
#include "brodiffusion/denoiser.h"
#include "brodiffusion/dit/flux.h"
#include "brodiffusion/dit/sana.h"
#include "brodiffusion/dit/pixart.h"
#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/scm_scheduler.h"
#include "brodiffusion/lcm_scheduler.h"
#include "brodiffusion/dpm_solver.h"
#include "brodiffusion/model_config.h"
#include "brodiffusion/scheduler.h"
#include "brolm/gemma2.h"
#include "brolm/gemma_tokenizer.h"
#include "brolm/t5.h"
#include "brolm/tokenizer.h"
#include "brolm/tokenizer_t5.h"
#include "brodiffusion/unet.h"
#include "brodiffusion/vae.h"
#include "brodiffusion/vae_dcae.h"
#include "brodiffusion/vae_qwenimage.h"
#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"
#include "brolm/qwen3vl_vision.h"
#include "brolm/qwen3vl_prompt.h"
#include "brodiffusion/krea2_text.h"

#include "brotensor/tensor.h"

#include <cstdint>
#include <functional>
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
    dit::SanaConfig          sana;          // Sana (linear DiT transformer)
    dit::PixArtConfig        pixart;        // PixArt-Sigma (DiT transformer)
    vae::DecoderConfig       vae;
    dcae::DecoderConfig      dcae;          // Sana (DC-AE f32c32 decoder)
    brolm::clip::TextEncoderConfig  text_encoder;  // CLIP — SD / Flux
    brolm::t5::T5Config             t5;            // Flux
    int                      t5_max_length = 512;
    brolm::gemma::Gemma2Config gemma;       // Sana (Gemma-2 text encoder)
    int                      sana_max_seq_len = 300;  // Gemma caption length
    Krea2ModelConfig         krea2;         // Krea 2 (transformer + VAE + Qwen3-VL)
    // DDIM (default, vanilla SD1.5) or LCM (latent-consistency, distilled
    // checkpoints with unet.time_cond_proj_dim > 0) or FlowMatch (Flux). The
    // pipeline branches on the active alternative; existing call sites that
    // don't set this keep working unchanged.
    std::variant<scheduler::DDIMConfig, scheduler::LCMConfig,
                 scheduler::FlowMatchConfig, scheduler::SCMConfig,
                 scheduler::DPMSolverConfig> scheduler;
};

// Snapshot of mid-generation state. Cheap to fork — only the latent
// carries device memory; the rest is host-side scalars / RNG state plus a
// shared handle to the prepared conditioning.
//
// The prepared conditioning (cross-attention K/V cache / projected text
// context) rides the state as a shared_ptr: prime() builds one per call and
// every clone() of that state shares it, so forking still costs one latent
// clone, not a re-encode. Because each prime() owns its conditioning, states
// from DIFFERENT prime() calls can be stepped interleaved — the spatial-paint
// dual-state loop primes once plain and once axis-steered, then steps both in
// lockstep, and each denoises under its own conditioning.
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

    // The conditioning this state denoises under. Set by prime(); shared
    // (not copied) by clone(). step_once()/decode() read the state's own
    // conditioning, never a pipeline-global one.
    std::shared_ptr<PreparedConditioning> prepared;

    // Deep clone: copies the latent on the active device; shares `prepared`.
    // RNG and ints are trivial copies.
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

    // Optional cooperative cancellation, honored by generate() only (the
    // step-wise prime/step_once API leaves pacing to the caller, who can stop
    // looping whenever it likes). Checked once per denoising step and once
    // before the VAE decode; if it returns true, generate() throws
    // GenerateCancelled. Empty by default (no cancellation). Same convention
    // as triposplat::SamplerOptions::should_cancel.
    std::function<bool()> should_cancel;
};

// Thrown by generate() when should_cancel() returns true. Distinct type so a
// caller can tell a cancel apart from a real failure.
struct GenerateCancelled : std::exception {
    const char* what() const noexcept override { return "pipeline: generate cancelled"; }
};

// LoadCancelled (thrown by from_model_dir when ModelDirOptions::should_cancel
// fires) lives in denoiser.h — the sharded DiT loaders that throw it sit below
// this header and only include denoiser.h.

class Pipeline {
public:
    // StableDiffusion constructor: builds a UNet denoiser. Valid only when
    // cfg.model_class == ModelClass::StableDiffusion.
    Pipeline(const PipelineConfig& cfg, brolm::clip::Tokenizer tokenizer);

    // Flux constructor: builds a FluxDenoiser plus the second (T5) text
    // encoder + tokenizer. Valid only when cfg.model_class == ModelClass::Flux.
    Pipeline(const PipelineConfig& cfg, brolm::clip::Tokenizer clip_tok,
             brolm::t5::Tokenizer t5_tok);

    // Sana constructor: builds a SanaDenoiser (linear DiT) + the DC-AE f32c32
    // decoder + the Gemma-2 text encoder, and owns the Gemma tokenizer. Valid
    // only when cfg.model_class == ModelClass::Sana. The CLIP / KL-VAE members
    // are default-constructed and unused.
    Pipeline(const PipelineConfig& cfg, brolm::gemma::Tokenizer gemma_tok);

    // PixArt constructor: builds a PixArtDenoiser (DiT) + the KL-VAE decoder +
    // the T5-XXL text encoder, and owns the T5 tokenizer. Valid only when
    // cfg.model_class == ModelClass::PixArt. There is no CLIP frontend (the
    // CLIP tokenizer/encoder members stay default-constructed and unused).
    Pipeline(const PipelineConfig& cfg, brolm::t5::Tokenizer t5_tok);

    // Krea 2 constructor: builds a Krea2Denoiser (single-stream flow DiT) + the
    // Qwen-Image VAE decoder + the Qwen3-VL-4B text encoder, and owns the
    // Qwen3-VL tokenizer. Valid only when cfg.model_class == ModelClass::Krea2.
    // No CLIP frontend / KL-VAE (those members stay default-constructed).
    Pipeline(const PipelineConfig& cfg, brolm::qwen3vl::Tokenizer qwen3vl_tok);

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

        // Optional cooperative cancellation for the load itself. from_model_dir
        // polls it between components — and, for the big sharded DiTs (Krea 2),
        // once per transformer block — so a caller (e.g. an app shutting down
        // mid-load) can abort a multi-GB read promptly instead of blocking a
        // join until all ~26 GB finish. When it returns true, from_model_dir
        // throws LoadCancelled. Empty by default (no cancellation). Same
        // convention as GenerateOptions::should_cancel.
        std::function<bool()> should_cancel;
    };
    static Pipeline from_model_dir(const std::string& model_dir,
                                   const ModelDirOptions& opts);
    // Overload instead of a defaulted `opts = {}` argument: GCC rejects the
    // brace-init default because ModelDirOptions has an in-class member
    // initializer (default member init needed outside a member function).
    static Pipeline from_model_dir(const std::string& model_dir) {
        return from_model_dir(model_dir, ModelDirOptions{});
    }

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

    // Apply a LoRA file.
    //
    // SD1.5: merges the deltas into the base UNet and CLIP weights.
    // Krea 2: attaches the file as ONE runtime-adapter group on the DiT (the
    // base linears may be INT8-quantized, so nothing is merged; each adapted
    // linear adds scale * (x @ down^T) @ up^T per forward). Runtime groups
    // are indexed in apply order — see set_lora_scale() / clear_loras().
    //
    // Must be called *after* load_weights() and *before* generate(). The
    // cross-attention K/V cache is re-primed inside every generate() call,
    // so calling apply_lora() between generates is safe — the next generate
    // will rebuild the cache against the updated K/V projections. For Krea 2
    // the fused text conditioning is rebuilt at every prime(), so a LoRA on
    // the text-fusion blocks likewise lands on the next generation.
    //
    // `scale` is a user multiplier applied on top of the per-LoRA alpha/rank
    // factor (default 1.0 = use as-shipped). Negative values are allowed
    // (subtract / undo). May be called more than once to stack multiple
    // LoRAs.
    //
    // Accepts kohya-ss/A1111 and diffusers/PEFT key conventions for SD1.5,
    // plus the Krea2-style DiT conventions (transformer. / diffusion_model. /
    // kohya-mangled transformer_blocks) — see lora.h. The format is
    // auto-detected from the key prefixes. Throws if the file contains LoRA
    // tensors that don't map to a target of the ACTIVE model family.
    //
    // Returns the runtime-adapter group index (Krea 2; pass it to
    // set_lora_scale), or -1 when the LoRA was merged in place (SD1.5).
    int apply_lora(const brotensor::safetensors::File& f, float scale = 1.0f);

    // Runtime-adapter LoRA controls — Krea 2 only (SD1.5 LoRAs are merged
    // irreversibly; these throw for other model classes). `index` is the
    // apply_lora() call order, 0-based.
    void set_lora_scale(int index, float scale);
    void clear_loras();
    int  num_loras() const;

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
    // The xattn cache rides the returned state (state.prepared) and is shared
    // across all branched states cloned from it; states from separate prime()
    // calls each keep their own, so they can be stepped interleaved (the
    // spatial-paint dual-state loop). Calling apply_lora() between prime()
    // and step_once() leaves the cache stale — re-prime() to refresh.
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

    // The sigma schedule of the most recent prime()/generate() when the
    // active scheduler is FlowMatch: length num_inference_steps + 1 with a
    // trailing 0.0 (see FlowMatch::sigmas()). Empty for non-flow-match
    // schedulers or before any schedule has been set. Lets a step-wise
    // caller compute the x0 preview x_i - sigmas[i]*v from consecutive
    // latents without re-running the model.
    std::vector<float> schedule_sigmas() const;

    // Accessors for research callers (e.g. tree search needs xattn block count
    // to size the bias vector). Throws if the active denoiser is not a UNet.
    const unet::UNet& unet() const;

    // Denoiser-generic count of traceable / steerable cross-attention blocks
    // for the active denoiser (16 for the SD1.5 UNet, 57 for the Flux DiT,
    // 0 for a denoiser with no trace support). Lets a caller size an
    // attn_logit_biases vector or a trace array without knowing the denoiser
    // type. Always valid — never throws.
    int num_xattn_blocks() const { return denoiser_->num_xattn_blocks(); }

    // Latent→pixel upscale factor of the active autoencoder: 8 for the SD /
    // Flux KL-VAE, 32 for Sana's DC-AE f32c32. A decoded image is therefore
    // (state.H_lat * vae_scale_factor()) × (state.W_lat * vae_scale_factor())
    // pixels — the dimensions of decode()'s buffer. Always valid; never throws.
    int vae_scale_factor() const {
        return model_class_ == ModelClass::Sana ? 32 : 8;
    }

    const PipelineConfig& config() const { return cfg_; }

    // Conditioning-space control axes (the sana-research seam). Load a
    // dictionary, set per-axis weights, and every subsequent prime() injects
    // the weighted directions into the positive text conditioning before the
    // denoiser. Empty / all-zero by default (no effect). See cond_control.h.
    CondControl&       cond_control()       { return cond_control_; }
    const CondControl& cond_control() const { return cond_control_; }

    // Encode `prompt` into the model's text-conditioning sequence — the exact
    // (L, hidden) embeddings the denoiser cross-attends to (row 0 = BOS). This
    // is the same encode prime() runs, exposed so callers can build control
    // directions in the encoder's own space (e.g. a diff-of-means axis from two
    // phrase sets). Routes by model class: Gemma-2 for Sana, CLIP otherwise.
    // Requires weights to be loaded.
    brotensor::Tensor encode_conditioning(std::string_view prompt);

    // ── Reference-attention identity anchor (the frame/sana-research seam) ──
    //
    // Sana's self-attention is linear, so a reference image's appearance can be
    // injected training-free: capture a neutral anchor's per-(branch,step,block)
    // attention summaries during one denoise, then add them (scaled by a weight)
    // into every later generation. The anchor carries visual IDENTITY; each
    // generation's own prompt still sets pose / expression. This lets a steered
    // attribute (e.g. an emotion cond_control axis pushed to the extreme) move
    // freely while the subject stays the same person — text-carries-structure,
    // image-carries-identity, no training and no clamp. Sana-only (throws on
    // other model classes); see dit::SanaDenoiser's ref_* methods.
    //
    // capture_identity_anchor() runs one full generation of `prompt` with the
    // denoiser recording, returns that anchor image, and arms the seam. The
    // captured step count should match later generations' num_inference_steps
    // for tight t-alignment (a shorter run reuses the anchor's final step).
    std::vector<float> capture_identity_anchor(std::string_view prompt,
                                               const GenerateOptions& opts);
    // Injection strength. 0 (default) disables injection even with an anchor
    // armed; ~1 reproduces the reference faithfully; higher over-anchors. Takes
    // effect on the next generate() / prime().
    void  set_identity_weight(float weight);
    float identity_weight() const;
    // True once an anchor has been captured (until clear_identity_anchor()).
    bool  has_identity_anchor() const;
    // Drop the cached anchor and zero the weight (frees the summary cache).
    void  clear_identity_anchor();

    // ── Krea 2 research hooks (the krea2_capi / krea-research seam) ─────────
    //
    // Direct forwarding to dit::Krea2Transformer2DModel's research hooks (see
    // dit/krea2.h for full semantics of each). Krea2-only; every method below
    // throws if model_class_ != Krea2. These mirror the Sana identity-anchor
    // methods above in spirit (denoiser-specific research plumbing exposed at
    // the Pipeline level) but are independent of that seam.

    // AdaLN mod-delta: add `delta` (1, 6*krea_hidden_size()) to the shared
    // per-step modulation for body blocks [block_lo, block_hi) on every
    // subsequent step_once(). Empty tensor clears.
    void krea_set_mod_delta(const brotensor::Tensor& delta,
                            int block_lo, int block_hi);

    // Timestep-embedding readout at `timestep` (the SAME 0..1000-scale value
    // krea_step_timestep() returns / step_once() consumes internally — this
    // method does the flow-time (/1000) conversion itself). No forward pass.
    // temb_out: (1, krea_hidden_size()); mod_out: (1, 6*krea_hidden_size()).
    void krea_time_mod(float timestep, brotensor::Tensor& temb_out,
                       brotensor::Tensor& mod_out);

    // The active scheduler's timestep for `state.step_index` — the same
    // 0..1000-scale value step_once() feeds the denoiser internally. Lets a
    // caller build a krea_time_mod() query / mod-delta for the step about to
    // run. Krea2 pairs exclusively with the FlowMatch scheduler.
    float krea_step_timestep(const PipelineState& state) const;

    // Attention-gate scale: scale the sigmoid output gate of body-block
    // attention, text rows and image rows separately, for blocks
    // [block_lo, block_hi). (1, 1) clears.
    void krea_set_gate_scale(float txt_scale, float img_scale,
                             int block_lo, int block_hi);

    // Per-token gate mask over blocks [block_lo, block_hi); mask holds
    // text_seq + img_len values in forward order. Empty tensor clears.
    void krea_set_gate_mask(const brotensor::Tensor& mask,
                            int block_lo, int block_hi);

    // Gate activity capture. When enabled, every subsequent step_once()
    // overwrites the internal sink; krea_gates() reads it back, row-major
    // (krea_num_layers(), text_seq + img_len).
    void krea_capture_gates(bool enable);
    std::vector<float> krea_gates() const;

    // Sizing accessors so a caller can allocate buffers without hardcoding
    // Krea 2's constants (6144 / 28).
    int krea_hidden_size() const;
    int krea_num_layers() const;

    // ── Krea 2 raw-taps entry points (band dial / image-as-prompt seam) ────
    //
    // Krea2's conditioning pipeline exposes two independently-callable
    // stages: raw per-layer Qwen3-VL taps (krea_encode_prompt_taps) and the
    // fusion stack that collapses them into the (n_valid, krea_hidden_size())
    // conditioning the DiT cross-attends to (krea_encode_text — the same
    // space cond_control axes are applied in, see cond_control()). Between
    // the two, a caller can edit specific tap rows (e.g. scale layers 7-10
    // for the "deep-band" literal<->stylized dial) or substitute an entirely
    // different tap source (e.g. image tokens through the same Qwen3-VL
    // backbone — see krea_encode_image_prompt). krea_prime_from_taps() then
    // primes a step-wise generation from the (possibly edited) raw taps,
    // exactly as prime() does internally for a plain text prompt.
    krea2::TextConditioning krea_encode_prompt_taps(std::string_view prompt);

    brotensor::Tensor krea_encode_text(
        const brotensor::Tensor& prompt_embeds,
        const brotensor::Tensor& prompt_embeds_mask);

    // Prime a step-wise generation from caller-supplied raw taps (as
    // returned by krea_encode_prompt_taps(), optionally edited) instead of
    // encoding `prompt` internally. uncond_embeds/uncond_mask may be null —
    // when guidance_scale != 1.0 (do_cfg) the uncond branch then falls back
    // to encoding opts.negative_prompt normally.
    PipelineState krea_prime_from_taps(const brotensor::Tensor& embeds,
                                       const brotensor::Tensor& mask,
                                       const brotensor::Tensor* uncond_embeds,
                                       const brotensor::Tensor* uncond_mask,
                                       const GenerateOptions& opts);

    // Krea 2 image-as-prompt: encode `pixels` (FP32 CHW, [0,1] range, shape
    // (3, H, W)) through Krea 2's own Qwen3-VL-4B vision tower into the SAME
    // raw-taps shape krea_encode_prompt_taps() produces for text — feed the
    // result straight into krea_encode_text()/krea_prime_from_taps(). No
    // separate model — the checkpoint's text encoder ships its vision tower
    // too (unused for plain text prompting).
    krea2::TextConditioning krea_encode_image_prompt(const float* pixels,
                                                     int H, int W);

private:
    // Encode a prompt to the CLIP (77, hidden) conditioning. If content_end is
    // non-null, it receives the EOS index (first eos_id) = end of the content
    // rows, for the conditioning-control seam's content-rows policy.
    void encode_prompt_(std::string_view prompt, brotensor::Tensor& out,
                        int* content_end = nullptr);
    // Sana txt2img priming: Gemma-encode prompt(s), prepare conditioning, and
    // allocate the FP32 initial latent (32x downsample, 32 channels). Returns a
    // step_index=0 state. Called from prime() when model_class_ == Sana.
    PipelineState prime_sana_(std::string_view prompt,
                              const GenerateOptions& opts);
    // Sana-Sprint denoising step (SCMScheduler / TrigFlow). The few-step,
    // guidance-distilled (no-CFG) path: maps the latent into the sCM input
    // parameterisation, runs one DiT forward with the embedded guidance scale,
    // reconstructs the model output, and applies one TrigFlow scheduler step.
    // Called from step_once() when the active scheduler is scheduler::SCM.
    void step_once_scm_(PipelineState& state, const GenerateOptions& opts);

    PipelineConfig            cfg_;
    ModelClass                model_class_;
    // CLIP tokenizer for SD / Flux. Empty for Sana (which has no CLIP
    // tokenizer; clip::Tokenizer is not default-constructible, hence optional).
    std::optional<brolm::clip::Tokenizer> tokenizer_;
    brolm::clip::TextEncoder  text_encoder_;    // CLIP — SD / Flux (unused Sana)
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
    std::variant<scheduler::DDIM, scheduler::LCM, scheduler::FlowMatch,
                 scheduler::SCM, scheduler::DPMSolverMultistep>
        scheduler_;

    // ── Sana-only sub-modules ─────────────────────────────────────────────
    // The SanaDenoiser lives in denoiser_; these are the pieces that have no
    // SD / Flux analogue. decode() routes to dcae_ for Sana (vae_ is unused);
    // prime_sana_ Gemma-encodes the prompt via gemma_model_ / gemma_tokenizer_.
    // All empty / default for non-Sana pipelines.
    std::optional<dcae::Decoder>             dcae_;
    std::optional<brolm::gemma::Gemma2Model> gemma_model_;
    std::optional<brolm::gemma::Tokenizer>   gemma_tokenizer_;

    // ── Krea 2-only sub-modules ───────────────────────────────────────────
    // The Krea2Denoiser lives in denoiser_; these have no SD / Flux analogue.
    // decode() routes to vae_qwen_ for Krea 2; prime() Qwen3-VL-encodes the
    // prompt (tapped hidden states + validity mask) via qwen3vl_model_ /
    // qwen3vl_tokenizer_. All empty for non-Krea2 pipelines.
    std::optional<vae_qwenimage::Decoder>    vae_qwen_;
    std::optional<brolm::qwen3vl::TextModel> qwen3vl_model_;
    std::optional<brolm::qwen3vl::Tokenizer> qwen3vl_tokenizer_;
    // Krea 2's vision tower + image preprocessor config — unused by plain
    // text prompting, loaded (from the SAME text_encoder shard(s)
    // qwen3vl_model_ loads) only so krea_encode_image_prompt() can work.
    // Empty for non-Krea2 pipelines and left unconstructed until
    // from_model_dir() loads its weights.
    std::optional<brolm::qwen3vl::VisionTower> qwen3vl_vision_;
    brolm::qwen3vl::PreprocessConfig           qwen3vl_pp_;

    // Model-agnostic raw conditioning, rebuilt each prime(). Kept around for
    // trace-mode / ControlNet access to the raw text context (those paths
    // reflect the LATEST prime()). The per-denoiser prepared payload (K/V
    // caches) is NOT held here — prime() moves it onto the returned
    // PipelineState (state.prepared), so states from different prime() calls
    // each step under their own conditioning.
    Conditioning          conditioning_;

    // Conditioning-space control axes, applied to the positive text embeddings
    // in every prime() (no-op until a dictionary is loaded + a weight set).
    CondControl           cond_control_;

    // Reference-attention identity anchor state (Sana only). identity_anchor_ is
    // set once capture_identity_anchor() succeeds; identity_weight_ scales the
    // injection (0 = off). capturing_anchor_ is true only while
    // capture_identity_anchor() drives the recording generation, so prime_sana_
    // leaves the denoiser in Capture mode instead of switching it to Inject.
    bool  identity_anchor_   = false;
    bool  capturing_anchor_  = false;
    float identity_weight_   = 0.0f;

    // Backing store for krea_capture_gates()/krea_gates() — owned here so its
    // lifetime outlives the Krea2Transformer2DModel::capture_gates() pointer
    // registration across step_once() calls.
    std::vector<float> krea_gate_sink_;

    // Set by krea_prime_from_taps() immediately before delegating to prime();
    // prime()'s Krea2 branch consumes (moves out of) these instead of calling
    // krea2::encode_prompt() when present, then clears them. Lets the two
    // prime paths share every non-text-encode step (latent alloc, RNG,
    // scheduler setup, CUDA graph keying) with zero duplication.
    std::optional<krea2::TextConditioning> krea_taps_override_;
    std::optional<krea2::TextConditioning> krea_uncond_taps_override_;

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
