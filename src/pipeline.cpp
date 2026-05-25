#include "brodiffusion/pipeline.h"

#include "brolm/clip.h"
#include "brodiffusion/denoiser.h"
#include "brodiffusion/dit/flux.h"
#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/lcm_scheduler.h"
#include "brodiffusion/lora.h"
#include "brodiffusion/model_config.h"
#include "brotensor/safetensors.h"
#include "brodiffusion/scheduler.h"
#include "brolm/t5.h"
#include "brolm/tokenizer.h"
#include "brolm/tokenizer_t5.h"
#include "brodiffusion/unet.h"
#include "brodiffusion/vae.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/safetensors_dir.h"
#include "brodiffusion/detail/torch_rng.h"
#include "brodiffusion/image_io.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace brodiffusion::pipeline {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("pipeline::Pipeline: " + msg);
}

using SchedulerVariant =
    std::variant<scheduler::DDIM, scheduler::LCM, scheduler::FlowMatch>;

// Construct the scheduler variant from the matching config variant.
SchedulerVariant
make_scheduler(const std::variant<scheduler::DDIMConfig, scheduler::LCMConfig,
                                  scheduler::FlowMatchConfig>& v) {
    if (std::holds_alternative<scheduler::LCMConfig>(v)) {
        return SchedulerVariant{std::in_place_type<scheduler::LCM>,
                                std::get<scheduler::LCMConfig>(v)};
    }
    if (std::holds_alternative<scheduler::FlowMatchConfig>(v)) {
        return SchedulerVariant{std::in_place_type<scheduler::FlowMatch>,
                                std::get<scheduler::FlowMatchConfig>(v)};
    }
    return SchedulerVariant{std::in_place_type<scheduler::DDIM>,
                            std::get<scheduler::DDIMConfig>(v)};
}

// Fetch the timestep at step `i` as a float, uniformly across schedulers
// (DDIM / LCM return int timesteps; FlowMatch returns continuous floats).
float timestep_at(const SchedulerVariant& v, int i) {
    return std::visit([i](const auto& s) -> float {
        return static_cast<float>(s.timesteps()[i]);
    }, v);
}

// Fill `out` (allocated to (1, n) at compute_dtype() on the default device)
// with n N(0,1) draws from the Philox stream keyed by (key, counter). When
// compute_dtype() is FP16 the draws are made into an FP32 scratch tensor and
// cast — brotensor::randn is FP32-only by design. `out` is always
// reassigned via Tensor::empty so a default-constructed (= CPU) input lands
// on the active default device.
void randn_compute(std::uint64_t key, std::uint64_t counter, int n,
                   bt::Tensor& out) {
    const bt::Dtype dt = compute_dtype();
    out = bt::Tensor::empty(1, n, dt);
    if (dt == bt::Dtype::FP16) {
        bt::Tensor fp32 = bt::Tensor::empty(1, n, bt::Dtype::FP32);
        bt::randn(key, counter, fp32);
        bt::cast(fp32, out, bt::Dtype::FP16);
    } else {
        bt::randn(key, counter, out);
    }
}

// Run one scheduler step on `state.latent`. LCM resamples per-step Gaussian
// noise from the state's Philox stream (key = state.rng_key, counter offset
// past the initial latent + all earlier steps); FlowMatch / DDIM are
// deterministic.
void scheduler_step(SchedulerVariant& sched, const bt::Tensor& pred,
                    int step_index, PipelineState& state, int n_lat,
                    bt::Tensor& scratch, bt::Tensor& noise_step) {
    if (std::holds_alternative<scheduler::LCM>(sched)) {
        const std::uint64_t counter =
            static_cast<std::uint64_t>(1 + step_index) *
            static_cast<std::uint64_t>(n_lat);
        randn_compute(state.rng_key, counter, n_lat, noise_step);
        std::get<scheduler::LCM>(sched).step(
            pred, step_index, state.latent, noise_step, scratch);
    } else if (std::holds_alternative<scheduler::FlowMatch>(sched)) {
        std::get<scheduler::FlowMatch>(sched).step(
            pred, step_index, state.latent, scratch);
    } else {
        std::get<scheduler::DDIM>(sched).step(
            pred, step_index, state.latent, scratch);
    }
}

// Build an EncoderConfig from a DecoderConfig. The two structs share fields
// by design — both describe the same SD1.5 AutoencoderKL with mirrored
// topology. Used by Pipeline's encoder member so callers don't have to
// duplicate channel counts in PipelineConfig.
vae::EncoderConfig encoder_config_from_decoder(const vae::DecoderConfig& d) {
    vae::EncoderConfig e;
    e.in_channels         = d.in_channels;
    e.out_channels        = d.out_channels;
    e.block_out_channels  = d.block_out_channels;
    e.layers_per_block    = d.layers_per_block;
    e.norm_num_groups     = d.norm_num_groups;
    e.scaling_factor      = d.scaling_factor;
    e.shift_factor        = d.shift_factor;
    e.eps                 = d.eps;
    e.num_attention_heads = d.num_attention_heads;
    return e;
}

// Derive the encoder-weight prefix from a decoder-weight prefix by stripping
// a trailing "decoder." and appending "encoder.". This keeps the existing
// (text, unet, vae) 3-prefix load_weights signature working for img2img:
// callers don't pass an explicit encoder prefix because in every diffusers /
// CompVis layout we've seen, the encoder is a sibling subtree of the decoder
// under the same VAE parent.
std::string encoder_prefix_from_decoder(const std::string& vae_prefix) {
    const std::string tail = "decoder.";
    if (vae_prefix.size() >= tail.size() &&
        vae_prefix.compare(vae_prefix.size() - tail.size(), tail.size(),
                           tail) == 0) {
        return vae_prefix.substr(0, vae_prefix.size() - tail.size()) +
               "encoder.";
    }
    // Fallback: caller passed something unusual — just append "encoder.".
    return vae_prefix + "encoder.";
}

// img2img t_start: how many steps to skip at the front of the schedule.
//   init_timestep = max(1, floor(n_steps * strength))   (clamped to 1..n_steps)
//   t_start       = n_steps - init_timestep             (clamped to 0..n_steps-1)
// strength=1.0 -> t_start=0 (full schedule); strength near 0 -> t_start near
// n_steps - 1 (almost no denoising). Mirrors diffusers'
// StableDiffusionImg2ImgPipeline.get_timesteps.
int img2img_t_start(int n_steps, float strength) {
    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;
    int init_t = static_cast<int>(static_cast<float>(n_steps) * strength);
    if (init_t < 1) init_t = 1;
    if (init_t > n_steps) init_t = n_steps;
    int t_start = n_steps - init_t;
    if (t_start < 0) t_start = 0;
    if (t_start >= n_steps) t_start = n_steps - 1;
    return t_start;
}

// Construct the denoiser matching the model class.
std::unique_ptr<Denoiser> make_denoiser(const PipelineConfig& cfg) {
    if (cfg.model_class == ModelClass::Flux) {
        return std::make_unique<dit::FluxDenoiser>(cfg.flux);
    }
    return std::make_unique<unet::UNet>(cfg.unet);
}

}  // namespace

Pipeline::Pipeline(const PipelineConfig& cfg, brolm::clip::Tokenizer tokenizer)
    : cfg_(cfg),
      model_class_(cfg.model_class),
      tokenizer_(std::move(tokenizer)),
      text_encoder_(cfg.text_encoder),
      denoiser_(make_denoiser(cfg)),
      vae_(cfg.vae),
      vae_encoder_(encoder_config_from_decoder(cfg.vae)),
      scheduler_(make_scheduler(cfg.scheduler)) {
    if (cfg.model_class == ModelClass::Flux) {
        fail("Pipeline: Flux model_class requires the (cfg, clip_tok, t5_tok) "
             "constructor");
    }
}

Pipeline::Pipeline(const PipelineConfig& cfg, brolm::clip::Tokenizer clip_tok,
                   brolm::t5::Tokenizer t5_tok)
    : cfg_(cfg),
      model_class_(cfg.model_class),
      tokenizer_(std::move(clip_tok)),
      text_encoder_(cfg.text_encoder),
      t5_tokenizer_(std::move(t5_tok)),
      t5_encoder_(std::in_place, cfg.t5),
      denoiser_(make_denoiser(cfg)),
      vae_(cfg.vae),
      vae_encoder_(encoder_config_from_decoder(cfg.vae)),
      scheduler_(make_scheduler(cfg.scheduler)) {
    if (cfg.model_class != ModelClass::Flux) {
        fail("Pipeline: the (cfg, clip_tok, t5_tok) constructor requires "
             "model_class == Flux");
    }
}

const unet::UNet& Pipeline::unet() const {
    const unet::UNet* u = denoiser_->as_unet();
    if (u == nullptr) fail("unet(): the active denoiser is not a UNet");
    return *u;
}

Pipeline Pipeline::from_model_dir(const std::string& model_dir) {
    namespace fs = std::filesystem;
    const ModelConfig mc = load_model_config(model_dir);

    PipelineConfig cfg;
    cfg.model_class   = mc.model_class;
    cfg.unet          = mc.unet;
    cfg.flux          = mc.flux;
    cfg.vae           = mc.vae;
    cfg.text_encoder  = mc.text_encoder;
    cfg.t5            = mc.t5;
    cfg.t5_max_length = mc.t5_max_length;
    cfg.scheduler     = mc.scheduler;

    const fs::path root(model_dir);

    // CLIP tokenizer (both classes).
    brolm::clip::Tokenizer clip_tok = brolm::clip::Tokenizer::load(
        (root / "tokenizer" / "vocab.json").string(),
        (root / "tokenizer" / "merges.txt").string());

    if (mc.model_class == ModelClass::Flux) {
        // T5 tokenizer for the second text encoder.
        brolm::t5::Tokenizer t5_tok = brolm::t5::Tokenizer::load(
            (root / "tokenizer_2" / "tokenizer.json").string());

        Pipeline p(cfg, std::move(clip_tok), std::move(t5_tok));

        // Load component weights. CLIP + VAE are single-file; the Flux
        // transformer and the T5-XXL encoder may be sharded — search every
        // shard by name (no .index.json parse needed).
        auto te_files  = detail::open_component_files(
            (root / "text_encoder").string());
        auto vae_files = detail::open_component_files(
            (root / "vae").string());
        auto tf_files  = detail::open_component_files(
            (root / "transformer").string());
        auto t52_files = detail::open_component_files(
            (root / "text_encoder_2").string());

        p.text_encoder_.load_weights(te_files.front(), "text_model.");
        p.vae_.load_weights(vae_files.front(), "decoder.");
        // Load the VAE encoder too — Flux img2img isn't wired yet, but the
        // diffusers Flux VAE ships an encoder and loading it now keeps the
        // model-dir load complete (and matches the SD branch which loads
        // the encoder via the (text, unet, vae) load_weights overload).
        p.vae_encoder_.load_weights(vae_files.front(), "encoder.");

        std::vector<const brotensor::safetensors::File*> tf_ptrs;
        for (const auto& f : tf_files) tf_ptrs.push_back(&f);
        auto* flux = dynamic_cast<dit::FluxDenoiser*>(p.denoiser_.get());
        if (!flux) fail("from_model_dir: Flux denoiser construction failed");
        flux->load_weights(tf_ptrs, "");

        std::vector<const brotensor::safetensors::File*> t52_ptrs;
        for (const auto& f : t52_files) t52_ptrs.push_back(&f);
        p.t5_encoder_->load_weights(t52_ptrs, "");

        return p;
    }

    // StableDiffusion: single-file CLIP / UNet / VAE.
    Pipeline p(cfg, std::move(clip_tok));
    auto te_files   = detail::open_component_files(
        (root / "text_encoder").string());
    auto unet_files = detail::open_component_files(
        (root / "unet").string());
    auto vae_files  = detail::open_component_files(
        (root / "vae").string());
    p.load_weights(te_files.front(), unet_files.front(), vae_files.front());
    return p;
}

void Pipeline::load_weights(const brotensor::safetensors::File& f) {
    load_weights(f,
                 "cond_stage_model.transformer.text_model.",
                 "model.diffusion_model.",
                 "first_stage_model.decoder.");
}

void Pipeline::load_weights(const brotensor::safetensors::File& f,
                            const std::string& text_prefix,
                            const std::string& unet_prefix,
                            const std::string& vae_prefix) {
    text_encoder_.load_weights(f, text_prefix);
    denoiser_->load_weights(f, unet_prefix);
    vae_.load_weights(f, vae_prefix);
    // The encoder prefix is derived from the decoder prefix: strip trailing
    // "decoder." and append "encoder." (so "first_stage_model.decoder." ->
    // "first_stage_model.encoder."). Encoder's load_weights handles the
    // sibling quant_conv lookup off its parent automatically.
    vae_encoder_.load_weights(f, encoder_prefix_from_decoder(vae_prefix));
}

void Pipeline::load_weights(const brotensor::safetensors::File& text_file,
                            const brotensor::safetensors::File& unet_file,
                            const brotensor::safetensors::File& vae_file) {
    text_encoder_.load_weights(text_file, "text_model.");
    denoiser_->load_weights(unet_file, "");
    vae_.load_weights(vae_file, "decoder.");
    // Encoder lives in the same diffusers VAE safetensors as the decoder.
    vae_encoder_.load_weights(vae_file, "encoder.");
}

void Pipeline::apply_lora(const brotensor::safetensors::File& f, float scale) {
    const std::vector<lora::Triple> triples = lora::enumerate(f);
    if (triples.empty()) {
        fail("apply_lora: no LoRA triples found in file");
    }
    for (const lora::Triple& t : triples) {
        const float scale_total = (static_cast<float>(t.alpha) /
                                   static_cast<float>(t.rank)) * scale;
        const brotensor::safetensors::TensorView& down = f.get(t.down_key);
        const brotensor::safetensors::TensorView& up   = f.get(t.up_key);
        if (t.domain == "unet") {
            unet::UNet* u = denoiser_->as_unet();
            if (u == nullptr) {
                fail("apply_lora: UNet LoRA target but the active denoiser "
                     "is not a UNet");
            }
            u->apply_lora_delta(t.target_path, down, up, scale_total);
        } else if (t.domain == "text_encoder") {
            text_encoder_.apply_lora_delta(t.target_path, down, up, scale_total);
        } else {
            fail("apply_lora: unknown domain '" + t.domain + "'");
        }
    }
}

void Pipeline::encode_prompt_(std::string_view prompt, bt::Tensor& out) {
    std::vector<std::int32_t> ids = tokenizer_.encode(prompt);
    if (static_cast<int>(ids.size()) != cfg_.text_encoder.max_position) {
        fail("tokenizer returned " + std::to_string(ids.size()) +
             " ids, expected " + std::to_string(cfg_.text_encoder.max_position));
    }
    text_encoder_.forward(ids.data(), out);
}

PipelineState PipelineState::clone() const {
    PipelineState out;
    out.latent     = latent.clone();
    out.rng_key    = rng_key;
    out.step_index = step_index;
    out.n_steps    = n_steps;
    out.H_lat      = H_lat;
    out.W_lat      = W_lat;
    return out;
}

PipelineState Pipeline::prime(std::string_view prompt,
                              const GenerateOptions& opts) {
    if (opts.height <= 0 || opts.width <= 0 ||
        opts.height % 8 != 0 || opts.width % 8 != 0) {
        fail("height and width must be positive multiples of 8");
    }
    if (opts.num_inference_steps <= 0) fail("num_inference_steps must be positive");
    if (!opts.init_image_path.empty() && !opts.init_noise.empty()) {
        fail("prime: init_image_path and init_noise cannot both be set "
             "(img2img and explicit-noise priming are mutually exclusive)");
    }
    if (!opts.init_image_path.empty() &&
        model_class_ != ModelClass::StableDiffusion) {
        fail("prime: img2img (init_image_path) is currently SD1.5 only; "
             "Flux img2img is not yet supported");
    }

    const int H_lat = opts.height / 8;
    const int W_lat = opts.width  / 8;
    // Latent channel count is denoiser-defined (4 for SD1.5, 16 for Flux).
    const int C_lat = denoiser_->latent_channels();
    const int n_lat = C_lat * H_lat * W_lat;
    const bool is_lcm = std::holds_alternative<scheduler::LCM>(scheduler_);
    const bool do_cfg = denoiser_->uses_cfg() && !is_lcm &&
                        (opts.guidance_scale != 1.0f);

    // 0. Finalize denoiser weights (W8A16 quantisation happens here if enabled).
    denoiser_->finalize_weights();

    // 1. Encode prompt(s) into the model-agnostic Conditioning struct.
    if (model_class_ == ModelClass::Flux) {
        // Flux: T5 token sequence is the cross-attention context; the CLIP
        // pooled vector feeds the AdaLN time-text embedding. No CFG branch.
        if (!t5_tokenizer_ || !t5_encoder_) {
            fail("prime: Flux pipeline missing T5 tokenizer / encoder");
        }
        std::vector<std::int32_t> t5_ids =
            t5_tokenizer_->encode(prompt, cfg_.t5_max_length);
        // encode() pads to t5_max_length with <pad>; pass pad_id so the T5
        // encoder masks those positions out of self-attention (HF parity).
        t5_encoder_->forward(t5_ids.data(),
                             static_cast<int>(t5_ids.size()),
                             conditioning_.text_embeddings,
                             t5_tokenizer_->pad_id());

        // CLIP pooled vector — discard the CLIP sequence output for Flux.
        std::vector<std::int32_t> clip_ids = tokenizer_.encode(prompt);
        if (static_cast<int>(clip_ids.size()) !=
            cfg_.text_encoder.max_position) {
            fail("prime: CLIP tokenizer returned " +
                 std::to_string(clip_ids.size()) + " ids, expected " +
                 std::to_string(cfg_.text_encoder.max_position));
        }
        text_encoder_.forward(clip_ids.data(), scratch_,
                              &conditioning_.pooled);

        conditioning_.has_uncond = false;
        conditioning_.uncond_embeddings = brotensor::Tensor{};
        conditioning_.guidance =
            cfg_.flux.guidance_embeds ? opts.guidance_scale : 0.0f;
    } else {
        encode_prompt_(prompt, conditioning_.text_embeddings);
        if (do_cfg) {
            encode_prompt_(opts.negative_prompt,
                           conditioning_.uncond_embeddings);
            conditioning_.has_uncond = true;
        } else {
            conditioning_.has_uncond = false;
            conditioning_.uncond_embeddings = brotensor::Tensor{};
        }
        conditioning_.guidance = is_lcm ? opts.guidance_scale : 0.0f;
    }

    // 1b. Pre-process conditioning once per generation (cross-attention K/V
    // projection for the UNet; pre-projected T5 context for Flux). Shared
    // across all branched states.
    prepared_ = denoiser_->prepare(conditioning_);

    // 2. Build the initial state shell (latent assigned below).
    PipelineState state;
    state.H_lat = H_lat;
    state.W_lat = W_lat;
    // Stash the seed: LCM resamples per-step noise from this same Philox key,
    // and branched states diverge by mutating rng_key on the clone.
    state.rng_key = opts.seed;
    const float sigma = std::visit(
        [](const auto& s) { return s.init_noise_sigma(); }, scheduler_);

    // 3. Set the timestep schedule (lives on the scheduler — shared across
    // all branches). FlowMatch with dynamic shifting (flux-dev) needs the
    // image token count; DDIM / LCM ignore the second argument. We set the
    // schedule *before* building the initial latent so img2img can pick a
    // t_start off the populated timesteps_ vector.
    const int image_seq_len = (H_lat / 2) * (W_lat / 2);
    std::visit([&](auto& s) {
        using S = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<S, scheduler::FlowMatch>) {
            s.set_timesteps(opts.num_inference_steps, image_seq_len);
        } else {
            s.set_timesteps(opts.num_inference_steps);
        }
    }, scheduler_);
    state.n_steps = std::visit(
        [](const auto& s) { return s.num_inference_steps(); }, scheduler_);

    if (!opts.init_image_path.empty()) {
        // ── img2img: VAE-encode the init image, then noise it to t_start ──
        // Counter layout for the img2img Philox stream:
        //   counter 0..n_lat               : add_noise noise (reuses the
        //                                    txt2img initial-latent slot,
        //                                    which is unused here).
        //   counter n_lat..2*n_lat         : VAE encoder eps when
        //                                    vae_encode_sample is true.
        //   counter (1+step)*n_lat..       : LCM per-step noise (unchanged).
        // The eps slot at offset n_lat does NOT collide with the LCM
        // per-step slots because LCM starts at counter (1+0)*n_lat = n_lat
        // — but step 0's noise is only drawn on the *first* step_once(), by
        // which time we've already consumed eps. Treat them as
        // non-overlapping in time even though the counter ranges abut.
        bt::Tensor image_nchw = load_image_as_latent_input(
            opts.init_image_path, opts.width, opts.height);

        bt::Tensor x0_latent;
        if (opts.vae_encode_sample) {
            bt::Tensor eps;
            randn_compute(opts.seed,
                          static_cast<std::uint64_t>(n_lat),
                          n_lat, eps);
            vae_encoder_.encode(image_nchw, opts.height, opts.width,
                                &eps, x0_latent);
        } else {
            vae_encoder_.encode(image_nchw, opts.height, opts.width,
                                /*eps=*/nullptr, x0_latent);
        }

        // Draw the add-noise noise (raw N(0,1)) from counter 0 — same
        // Philox slot a txt2img run would have used for its initial latent.
        bt::Tensor noise;
        randn_compute(opts.seed, 0, n_lat, noise);

        const int t_start = img2img_t_start(state.n_steps, opts.strength);
        std::visit([&](auto& s) {
            s.add_noise(x0_latent, noise, t_start, state.latent, scratch_);
        }, scheduler_);
        state.step_index = t_start;
        return state;
    }

    // ── txt2img: initial latent is pure Gaussian noise * init_noise_sigma ──
    if (!opts.init_noise.empty()) {
        // Caller-supplied initial noise (raw N(0,1)); used for exact
        // cross-implementation comparison. Still scaled by init_noise_sigma.
        if (static_cast<int>(opts.init_noise.size()) != n_lat) {
            fail("prime: init_noise has " +
                 std::to_string(opts.init_noise.size()) +
                 " values, expected " + std::to_string(n_lat));
        }
        std::vector<float> noise(n_lat);
        for (int i = 0; i < n_lat; ++i) {
            noise[i] = sigma * opts.init_noise[i];
        }
        state.latent = detail::upload_host(noise.data(), 1, n_lat);
    } else if (opts.noise_source == NoiseSource::Torch) {
        // torch.randn-compatible noise: makes `seed` reproduce a PyTorch
        // reference run's starting latent (see detail/torch_rng.h). This
        // path stays host-side by construction — the bit-exact Box-Muller
        // is the whole point.
        std::vector<float> noise =
            detail::torch_randn_f32(opts.seed, static_cast<std::size_t>(n_lat));
        if (sigma != 1.0f) {
            for (int i = 0; i < n_lat; ++i) noise[i] *= sigma;
        }
        state.latent = detail::upload_host(noise.data(), 1, n_lat);
    } else {
        // Internal: device-side Philox stream. Counter 0..n_lat is the
        // initial latent; per-step LCM noise uses (1+step)*n_lat onwards.
        randn_compute(opts.seed, 0, n_lat, state.latent);
        if (sigma != 1.0f) bt::scale_inplace(state.latent, sigma);
    }

    state.step_index = 0;
    return state;
}

void Pipeline::step_once(PipelineState& state, const GenerateOptions& opts,
                         AttentionTrace* trace_out,
                         const std::vector<const bt::Tensor*>*
                             attn_logit_biases) {
    if (state.step_index >= state.n_steps) {
        fail("step_once: step_index (" + std::to_string(state.step_index) +
             ") >= n_steps (" + std::to_string(state.n_steps) + ")");
    }
    const bool is_lcm = std::holds_alternative<scheduler::LCM>(scheduler_);
    const bool do_cfg = denoiser_->uses_cfg() && !is_lcm &&
                        (opts.guidance_scale != 1.0f);
    const int i = state.step_index;
    const float t = timestep_at(scheduler_, i);
    const int n_lat = denoiser_->latent_channels() *
                      state.H_lat * state.W_lat;

    // Trace mode is forced if either the caller wants a trace OR is injecting
    // attention biases (forward_trace is the only path that accepts biases).
    const bool trace_mode = (trace_out != nullptr) || (attn_logit_biases != nullptr);

    // ── Denoiser forward ──────────────────────────────────────────────────
    if (trace_mode) {
        // Trace mode bypasses the K/V cache + INT8 inside the denoiser and
        // captures the conditional pass only; a CFG uncond branch (if any)
        // still uses the fast prepared path. A scratch trace absorbs the maps
        // when the caller wants biases but not the trace itself. The denoiser
        // pulls the raw context (and any LCM guidance) from prepared_.
        if (denoiser_->num_xattn_blocks() == 0) {
            fail("step_once: trace mode requires a denoiser with traceable "
                 "attention blocks");
        }
        AttentionTrace scratch_trace;
        AttentionTrace* trace_dst = trace_out ? trace_out : &scratch_trace;
        denoiser_->forward_traced(state.latent, state.H_lat, state.W_lat, t,
                                  prepared_, Branch::Cond, attn_logit_biases,
                                  trace_dst, noise_pred_cond_);
        if (do_cfg) {
            denoiser_->forward(state.latent, state.H_lat, state.W_lat, t,
                               prepared_, Branch::Uncond, noise_pred_uncond_);
        }
    } else {
        denoiser_->forward(state.latent, state.H_lat, state.W_lat, t,
                           prepared_, Branch::Cond, noise_pred_cond_);
        if (do_cfg) {
            denoiser_->forward(state.latent, state.H_lat, state.W_lat, t,
                               prepared_, Branch::Uncond, noise_pred_uncond_);
        }
    }
    // CFG combine (DDIM only; LCM has no uncond branch).
    if (do_cfg) {
        bt::scale_inplace(noise_pred_cond_, opts.guidance_scale);
        bt::scale_inplace(noise_pred_uncond_, 1.0f - opts.guidance_scale);
        bt::add_inplace(noise_pred_cond_, noise_pred_uncond_);
    }

    // ── Scheduler step ────────────────────────────────────────────────────
    scheduler_step(scheduler_, noise_pred_cond_, i, state, n_lat,
                   scratch_, noise_step_);
    ++state.step_index;
}

std::vector<float> Pipeline::decode(const PipelineState& state) {
    vae_.decode(state.latent, state.H_lat, state.W_lat, decoded_);
    bt::sync_all();
    const int n_img = cfg_.vae.out_channels *
                       (state.H_lat * 8) * (state.W_lat * 8);
    // The decoded tensor carries the compute dtype — FP16 on a GPU backend,
    // FP32 on CPU. Convert FP16 bits to float as needed.
    if (decoded_.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> dec_bits(static_cast<std::size_t>(n_img));
        decoded_.copy_to_host_fp16(dec_bits.data());
        std::vector<float> out(static_cast<std::size_t>(n_img));
        for (int i = 0; i < n_img; ++i) {
            out[static_cast<std::size_t>(i)] =
                bt::fp16_bits_to_fp32(dec_bits[static_cast<std::size_t>(i)]);
        }
        return out;
    }
    return decoded_.to_host_vector();
}

std::vector<float> Pipeline::generate(std::string_view prompt,
                                       const GenerateOptions& opts) {
    PipelineState state = prime(prompt, opts);
    // img2img priming sets state.step_index to a non-zero start (t_start);
    // txt2img leaves it at 0. Loop until the schedule is exhausted —
    // step_once increments step_index itself.
    while (state.step_index < state.n_steps) {
        step_once(state, opts, /*trace_out=*/nullptr);
    }
    return decode(state);
}

}  // namespace brodiffusion::pipeline
