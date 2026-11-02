#pragma once

// Configuration, state, and option types for brodiffusion::pipeline::Pipeline.
// Included automatically by <brodiffusion/pipeline.h>.

#include "brolm/clip.h"
#include "brodiffusion/cond_control.h"
#include "brodiffusion/control_schedule.h"
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
#include "brodiffusion/vae_qwenimage21.h"
#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"
#include "brolm/qwen3vl_vision.h"
#include "brolm/qwen3vl_prompt.h"
#include "brodiffusion/krea2_text.h"
#include "brodiffusion/qwenimage21_text.h"
#include "brodiffusion/dit/qwenimage21.h"

#include "brotensor/tensor.h"

#include <array>
#include <cstddef>
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
    QwenImage21ModelConfig   qwenimage21;   // Qwen-Image 2.1 (DiT + VAE + Qwen3-VL)
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
// Hands out a fresh identity for every PipelineState ever constructed. Process
// -wide and monotonic: unlike a heap or device address, a value from here is
// never handed out twice, so two different states can never compare equal.
std::uint64_t next_state_id();

// Used by cross-attention tree search: at any step, .clone() the current
// state, advance the clone differently from the original (different RNG, a
// different attn_logit_bias, etc.), score, and pick a winner.
struct PipelineState {
    brotensor::Tensor latent;  // (1, C_lat * H_lat * W_lat) at compute dtype

    // Unique, never-reused identity for this state's buffer set. The CUDA-graph
    // step session keys on this.
    //
    // It used to key on the raw addresses of `latent.data` and `prepared`, and
    // that is unsound: both are freed and re-allocated at the same size on every
    // generate(), so a pooled allocator hands the next generation the SAME
    // addresses, the key compares equal, and a graph captured against the
    // PREVIOUS generation's buffers gets replayed against this one's. It survives
    // on the accident that the rest of the buffers usually land at their old
    // addresses too; perturb the allocator (encode an image with another model
    // between generations, say) and one of them moves, and the replay reads a
    // freed pointer — an illegal access if the block is unmapped, and, worse, a
    // silently mis-conditioned image if it is not. An address that can be recycled
    // is not an identity. A counter cannot be recycled.
    std::uint64_t id = next_state_id();

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

// One condition image for Qwen-Image 2.1's image-conditioned generation.
// Either a file path (decoded by broimage) or pixels already in memory —
// exactly one of the two. In-memory pixels are PLANAR CHW in [0, 1], 3
// channels (RGB) or 4 (RGBA, alpha preserved for the autoencoder and
// composited over white for the vision tower).
//
// The image is resized before use — see GenerateOptions::output_resolution —
// so it does not have to arrive at any particular size.
struct ConditionImage {
    std::string path;
    std::vector<float> pixels;
    int channels = 3;
    int H = 0;
    int W = 0;
};

struct GenerateOptions {
    // Image dimensions in pixels. Latent dims are H/8, W/8. Both must be
    // multiples of 8 and at least 8 (so latent dims are >= 1).
    //
    // Qwen-Image 2.1 with condition images accepts 0 for both, meaning
    // "derive the canvas from the LAST condition image's aspect ratio at
    // output_resolution" — which is what the reference pipeline does when
    // height/width are omitted. Every other path still demands a positive
    // multiple of the autoencoder stride.
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
    // SD1.5 and Flux. Flux follows diffusers' FluxImg2ImgPipeline: the VAE
    // encode applies (z - shift_factor) * scaling_factor, and the flow-match
    // add_noise is (1 - sigma) * x0 + sigma * noise at the t_start sigma.
    // Inpaint (mask_image_path) is still SD1.5 only. Empty path = txt2img (existing
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

    // ── Qwen-Image 2.1 image conditioning (edit / reference image) ────────
    //
    // Condition images, in the order the prompt should see them. Each one is
    // read twice: the Qwen3-VL vision tower turns it into rows of the prompt
    // stream, and the autoencoder turns it into latent tokens that sit in the
    // joint sequence ahead of the noise. The target latents are the only
    // thing the sampler touches — a condition image is context, never
    // something being denoised, which is what separates this from img2img.
    //
    // Empty = plain text-to-image. QwenImage21 only; other model classes
    // throw. The vision tower must be resident (it is, unless the checkpoint
    // shipped without one).
    std::vector<ConditionImage> condition_images;

    // Target side length used both to size the generated canvas (when
    // height/width are 0) and to resize every condition image, via the
    // reference's calculate_dimensions(output_resolution², aspect). 1024 is
    // the reference default; drop it to fit a smaller card. Ignored when
    // condition_images is empty.
    int output_resolution = 1024;

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

// Options for Pipeline::from_model_dir (aliased as Pipeline::ModelDirOptions).
struct PipelineModelDirOptions {
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

    // Krea 2 only: override the text-encoder (Qwen3-VL-4B) backbone. When
    // non-empty, the text backbone's language-model weights are loaded from
    // this path instead of `<model_dir>/text_encoder`. Accepts:
    //   • a llama.cpp `.gguf` checkpoint (e.g. a Q8_0 quant) — text-only, so
    //     the vision tower still loads from `<model_dir>/text_encoder`;
    //   • a diffusers `.safetensors` file or a directory of them — its
    //     `language_model.*` (or `model.language_model.*`) subtree is used,
    //     and its `visual.*` tower too when present (else the dir's).
    // The checkpoint must be the same Qwen3-VL-4B architecture the model
    // dir's config describes (per-tensor shape checks enforce it). Empty
    // (default) keeps the bundled text encoder. Ignored for non-Krea2.
    std::string text_encoder_path;
};

// One saved Qwen-Image 2.1 prefix cache slot (cond + optional uncond branch).
struct Qi21PrefixSlot {
    dit::QwenImage21PrefixCache cond;
    dit::QwenImage21PrefixCache uncond;
    bool uncond_valid = false;
};

// One entry in the Qwen-Image 2.1 LRU prompt memo.
struct Qi21PromptMemo {
    std::string prompt;
    qwenimage21::TextConditioning cond;
};

// Expand an init-noise tensor (NCHW, c×h×w, raw N(0,1)) by an integer spatial
// factor k, preserving its low-frequency structure: the result is exactly
// i.i.d. N(0,1), but each k×k block's mean is tied to the source value
// (H = E − mean_k(E) + src/k, with E a fresh Philox stream keyed by `seed`).
// Because a generation's identity/composition is decided by the coarse
// structure of the init noise, this transports an identity found at one
// resolution to a k× larger render — a seed cannot do that (the same seed at
// a different latent shape is an unrelated noise field). Deterministic for a
// given (src, seed). Output length = c * h*k * w*k.
std::vector<float> expand_init_noise(const float* src, int c, int h, int w,
                                     int k, std::uint64_t seed);

}  // namespace brodiffusion::pipeline
