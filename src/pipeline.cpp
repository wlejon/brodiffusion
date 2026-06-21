#include "brodiffusion/pipeline.h"

#include "brolm/clip.h"
#include "brodiffusion/controlnet.h"
#include "brodiffusion/denoiser.h"
#include "brodiffusion/dit/flux.h"
#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/scm_scheduler.h"
#include "brodiffusion/lcm_scheduler.h"
#include "brodiffusion/detail/device.h"
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
#include "brodiffusion/dit/sana.h"
#include "brodiffusion/sana_text.h"
#include "brodiffusion/vae_dcae.h"
#include "brodiffusion/image_io.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#ifdef BROTENSOR_HAS_CUDA
#include "brotensor/cuda_graph.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    std::variant<scheduler::DDIM, scheduler::LCM, scheduler::FlowMatch,
                 scheduler::SCM>;

// Construct the scheduler variant from the matching config variant.
SchedulerVariant
make_scheduler(const std::variant<scheduler::DDIMConfig, scheduler::LCMConfig,
                                  scheduler::FlowMatchConfig,
                                  scheduler::SCMConfig>& v) {
    if (std::holds_alternative<scheduler::LCMConfig>(v)) {
        return SchedulerVariant{std::in_place_type<scheduler::LCM>,
                                std::get<scheduler::LCMConfig>(v)};
    }
    if (std::holds_alternative<scheduler::FlowMatchConfig>(v)) {
        return SchedulerVariant{std::in_place_type<scheduler::FlowMatch>,
                                std::get<scheduler::FlowMatchConfig>(v)};
    }
    if (std::holds_alternative<scheduler::SCMConfig>(v)) {
        return SchedulerVariant{std::in_place_type<scheduler::SCM>,
                                std::get<scheduler::SCMConfig>(v)};
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
    e.force_upcast        = d.force_upcast;
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

// Escape hatch: set BRODIFFUSION_DISABLE_STEP_GRAPH=1 to force eager
// denoiser stepping even when the CUDA-graph session would be eligible.
// Read per step so tests can flip it between generations in-process.
bool step_graph_disabled() {
    const char* e = std::getenv("BRODIFFUSION_DISABLE_STEP_GRAPH");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
}

// Construct the denoiser matching the model class.
std::unique_ptr<Denoiser> make_denoiser(const PipelineConfig& cfg) {
    if (cfg.model_class == ModelClass::Flux) {
        return std::make_unique<dit::FluxDenoiser>(cfg.flux);
    }
    if (cfg.model_class == ModelClass::Sana) {
        return std::make_unique<dit::SanaDenoiser>(cfg.sana);
    }
    return std::make_unique<unet::UNet>(cfg.unet);
}

}  // namespace

// CUDA-graph denoising-step session. One per (latent buffer, prepared
// payload, H, W, CFG mode) identity — see the member doc in pipeline.h.
// `eager_steps` counts the warm-up steps run through the capture seam for
// this key; the second one settles every U-Net scratch buffer at its
// high-water capacity (the x_/y_ ping-pong roles permute per body call, so
// one step is not enough), after which the body's pointer set is stable and
// the capture at the end of that step records a replayable graph.
struct Pipeline::StepGraphSession {
#ifdef BROTENSOR_HAS_CUDA
    bt::CudaGraph graph;
#endif
    const void* latent_ptr  = nullptr;  // state.latent.data
    const void* prepared_id = nullptr;  // prepared_.get()
    int  H = 0, W = 0;
    bool do_cfg = false;
    int  eager_steps = 0;
};

Pipeline::~Pipeline() = default;
Pipeline::Pipeline(Pipeline&&) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;

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

Pipeline::Pipeline(const PipelineConfig& cfg, brolm::gemma::Tokenizer gemma_tok)
    : cfg_(cfg),
      model_class_(cfg.model_class),
      tokenizer_(std::nullopt),           // CLIP unused for Sana
      text_encoder_(cfg.text_encoder),    // CLIP unused for Sana
      denoiser_(make_denoiser(cfg)),
      vae_(cfg.vae),                       // KL-VAE unused for Sana
      vae_encoder_(encoder_config_from_decoder(cfg.vae)),
      scheduler_(make_scheduler(cfg.scheduler)),
      dcae_(std::in_place, cfg.dcae),
      gemma_model_(std::in_place, cfg.gemma),
      gemma_tokenizer_(std::move(gemma_tok)) {
    if (cfg.model_class != ModelClass::Sana) {
        fail("Pipeline: the (cfg, gemma_tok) constructor requires "
             "model_class == Sana");
    }
}

const unet::UNet& Pipeline::unet() const {
    const unet::UNet* u = denoiser_->as_unet();
    if (u == nullptr) fail("unet(): the active denoiser is not a UNet");
    return *u;
}

Pipeline Pipeline::from_model_dir(const std::string& model_dir,
                                  const ModelDirOptions& dir_opts) {
    namespace fs = std::filesystem;
    const ModelConfig mc = load_model_config(model_dir);

    PipelineConfig cfg;
    cfg.model_class   = mc.model_class;
    cfg.unet          = mc.unet;
    cfg.flux          = mc.flux;
    cfg.sana          = mc.sana;
    cfg.vae           = mc.vae;
    cfg.dcae          = mc.dcae;
    cfg.text_encoder  = mc.text_encoder;
    cfg.t5            = mc.t5;
    cfg.t5_max_length = mc.t5_max_length;
    cfg.gemma         = mc.gemma;
    cfg.sana_max_seq_len = mc.sana_max_seq_len;
    cfg.scheduler     = mc.scheduler;
    if (dir_opts.quantize) {
        cfg.unet.quantize_weights = true;
        cfg.flux.quantize_weights = true;
        cfg.t5.quantize_weights   = true;
    }

    const fs::path root(model_dir);

    if (mc.model_class == ModelClass::Sana) {
        // Sana's text frontend is Gemma-2, not CLIP/T5 — load that tokenizer
        // (the CLIP vocab.json / merges.txt below don't exist in a Sana dir).
        brolm::gemma::Tokenizer gemma_tok = brolm::gemma::Tokenizer::load(
            (root / "tokenizer" / "tokenizer.json").string());

        Pipeline p(cfg, std::move(gemma_tok));

        // transformer (single-file) → SanaDenoiser; DC-AE decoder under the
        // "decoder." subtree of the VAE file; Gemma-2 text encoder may be
        // sharded (2 fp16 shards) — search every shard by name.
        const bool time_load = std::getenv("BRODIFFUSION_TIME") != nullptr;
        auto stamp = [&](const char* what, auto t0) {
            if (time_load) {
                std::fprintf(stderr, "[time]   %s: %.2f s\n", what,
                             std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - t0).count());
            }
            return std::chrono::steady_clock::now();
        };
        auto t = std::chrono::steady_clock::now();
        auto tf_files  = detail::open_component_files(
            (root / "transformer").string());
        auto vae_files = detail::open_component_files(
            (root / "vae").string());
        auto te_files  = detail::open_component_files(
            (root / "text_encoder").string());
        t = stamp("open files (mmap)", t);

        p.denoiser_->load_weights(tf_files.front(), "");
        t = stamp("DiT weights", t);
        p.dcae_->load_weights(vae_files.front(), "decoder.");
        t = stamp("DC-AE weights", t);

        std::vector<const brotensor::safetensors::File*> te_ptrs;
        for (const auto& f : te_files) te_ptrs.push_back(&f);
        p.gemma_model_->load_weights(te_ptrs, "");
        t = stamp("Gemma weights", t);

        return p;
    }

    // CLIP tokenizer (SD / Flux).
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

int Pipeline::add_controlnet(const brotensor::safetensors::File& f) {
    return add_controlnet(f, controlnet::ControlNetConfig{});
}

int Pipeline::add_controlnet(const brotensor::safetensors::File& f,
                             const controlnet::ControlNetConfig& cfg) {
    if (model_class_ != ModelClass::StableDiffusion) {
        fail("add_controlnet: Flux not supported (SD1.5 only)");
    }
    auto net = std::make_unique<controlnet::ControlNet>(cfg);
    net->load_weights(f, "");
    controlnets_.push_back(std::move(net));
    return static_cast<int>(controlnets_.size()) - 1;
}

void Pipeline::remove_controlnet(int index) {
    if (index < 0 || index >= static_cast<int>(controlnets_.size())) {
        fail("remove_controlnet: index " + std::to_string(index) +
             " out of range (have " + std::to_string(controlnets_.size()) +
             ")");
    }
    controlnets_.erase(controlnets_.begin() + index);
    if (index < static_cast<int>(control_images_.size())) {
        control_images_.erase(control_images_.begin() + index);
    }
    if (index < static_cast<int>(control_inputs_.size())) {
        control_inputs_.erase(control_inputs_.begin() + index);
    }
    if (controlnets_.empty()) controlnet_active_ = false;
}

void Pipeline::clear_controlnets() {
    controlnets_.clear();
    control_images_.clear();
    control_inputs_.clear();
    controlnet_active_ = false;
}

void Pipeline::encode_prompt_(std::string_view prompt, bt::Tensor& out) {
    if (!tokenizer_) fail("encode_prompt_: CLIP tokenizer not present");
    std::vector<std::int32_t> ids = tokenizer_->encode(prompt);
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

PipelineState Pipeline::prime_sana_(std::string_view prompt,
                                    const GenerateOptions& opts) {
    // Sana downsamples 32x (DC-AE f32c32), not the 8x of SD / Flux.
    if (opts.height <= 0 || opts.width <= 0 ||
        opts.height % 32 != 0 || opts.width % 32 != 0) {
        fail("Sana: height and width must be positive multiples of 32");
    }
    if (opts.num_inference_steps <= 0) {
        fail("num_inference_steps must be positive");
    }
    // img2img / inpaint / ControlNet are not wired for Sana yet (init_noise
    // and the txt2img RNG paths below are).
    if (!opts.init_image_path.empty() || !opts.mask_image_path.empty() ||
        !opts.controls.empty()) {
        fail("Sana: img2img / inpaint / ControlNet are not supported");
    }
    if (!gemma_model_ || !gemma_tokenizer_ || !dcae_) {
        fail("Sana: pipeline missing Gemma encoder / DC-AE decoder");
    }

    // Reset the SD-only run state so a previous non-Sana generation can't leak.
    inpaint_active_    = false;
    controlnet_active_ = false;
    control_inputs_.clear();
    control_images_.clear();

    const int H_lat = opts.height / 32;
    const int W_lat = opts.width  / 32;
    const int C_lat = denoiser_->latent_channels();      // 32
    const int n_lat = C_lat * H_lat * W_lat;
    // Base Sana is not guidance-distilled: a separate uncond branch + CFG
    // combine runs whenever guidance != 1.0 (uses_cfg() is true). Sana-Sprint
    // (SCMScheduler) is guidance-distilled — uses_cfg() is false, so do_cfg
    // stays false and the guidance scale is fed in as an embedding instead.
    const bool is_scm = std::holds_alternative<scheduler::SCM>(scheduler_);
    const bool do_cfg = denoiser_->uses_cfg() && (opts.guidance_scale != 1.0f);

    denoiser_->finalize_weights();

    // 1. Gemma-encode the positive prompt (and, under CFG, the negative). The
    //    Sana DiT cross-attends over exactly the returned valid token rows.
    conditioning_.text_embeddings = brodiffusion::sana::encode_prompt(
        *gemma_model_, *gemma_tokenizer_, std::string(prompt),
        cfg_.sana_max_seq_len);
    // Conditioning-space control seam: add the weighted control axes to the
    // positive caption embeddings (rows [1, L), BOS untouched) before the DiT
    // projects them. No-op unless a dictionary is loaded with a nonzero weight.
    // The negative branch is left clean (steer the prompt, not the baseline).
    cond_control_.apply(conditioning_.text_embeddings);
    if (do_cfg) {
        conditioning_.uncond_embeddings = brodiffusion::sana::encode_prompt(
            *gemma_model_, *gemma_tokenizer_,
            std::string(opts.negative_prompt), cfg_.sana_max_seq_len);
        conditioning_.has_uncond = true;
    } else {
        conditioning_.has_uncond = false;
        conditioning_.uncond_embeddings = bt::Tensor{};
    }
    // Sana-Sprint embeds the raw guidance scale (the denoiser applies
    // guidance_embeds_scale); base Sana uses CFG instead, so leaves it at 0.
    conditioning_.guidance = is_scm ? opts.guidance_scale : 0.0f;

    // 1b. Project the caption context once per generation (caption_projection
    //     + caption_norm), shared across all branched states.
    prepared_ = denoiser_->prepare(conditioning_);

    // 2. State shell + timestep schedule (rectified-flow FlowMatch, shift=3).
    PipelineState state;
    state.H_lat   = H_lat;
    state.W_lat   = W_lat;
    state.rng_key = opts.seed;
    std::visit([&](auto& s) { s.set_timesteps(opts.num_inference_steps); },
               scheduler_);
    state.n_steps = std::visit(
        [](const auto& s) { return s.num_inference_steps(); }, scheduler_);
    const float sigma = std::visit(
        [](const auto& s) { return s.init_noise_sigma(); }, scheduler_);

    // 3. Initial latent. The Sana DiT runs FP32 (FP16 overflows), so the latent
    //    — which the rectified-flow velocity is added to in place — must be FP32
    //    too, regardless of the GPU backend's FP16 compute dtype. brotensor's
    //    randn is FP32-native, so the device path needs no cast.
    if (!opts.init_noise.empty()) {
        if (static_cast<int>(opts.init_noise.size()) != n_lat) {
            fail("prime: init_noise has " +
                 std::to_string(opts.init_noise.size()) +
                 " values, expected " + std::to_string(n_lat));
        }
        std::vector<float> noise(static_cast<std::size_t>(n_lat));
        for (int i = 0; i < n_lat; ++i) noise[i] = sigma * opts.init_noise[i];
        state.latent =
            bt::Tensor::from_host(noise.data(), 1, n_lat).to(bt::default_device());
    } else if (opts.noise_source == NoiseSource::Torch) {
        std::vector<float> noise =
            detail::torch_randn_f32(opts.seed, static_cast<std::size_t>(n_lat));
        if (sigma != 1.0f) for (int i = 0; i < n_lat; ++i) noise[i] *= sigma;
        state.latent =
            bt::Tensor::from_host(noise.data(), 1, n_lat).to(bt::default_device());
    } else {
        state.latent = bt::Tensor::empty(1, n_lat, bt::Dtype::FP32);
        bt::randn(opts.seed, 0, state.latent);
        if (sigma != 1.0f) bt::scale_inplace(state.latent, sigma);
    }

    // Sana-Sprint works in sCM coordinates: the initial latent is scaled by
    // sigma_data (diffusers: `latents = latents * sigma_data`). The matching
    // 1/sigma_data is applied to the denoised latent after the final step.
    if (is_scm) {
        bt::scale_inplace(state.latent,
                          std::get<scheduler::SCM>(scheduler_).sigma_data());
    }

    state.step_index = 0;
    return state;
}

PipelineState Pipeline::prime(std::string_view prompt,
                              const GenerateOptions& opts) {
    // Sana has its own (32x downsample, FP32 latent, Gemma-encoded) priming
    // path; the SD / Flux machinery below (8x latent, CLIP / T5, img2img /
    // inpaint / ControlNet) does not apply.
    if (model_class_ == ModelClass::Sana) {
        return prime_sana_(prompt, opts);
    }
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
    if (!opts.mask_image_path.empty() && opts.init_image_path.empty()) {
        fail("inpaint: --mask requires --init (mask_image_path is set but "
             "init_image_path is empty)");
    }
    if (!opts.mask_image_path.empty() &&
        model_class_ != ModelClass::StableDiffusion) {
        fail("inpaint: Flux is not yet supported (SD1.5 only)");
    }

    // Reset inpaint state so a previous inpaint generation doesn't leak into
    // the next call. Re-armed below in the img2img branch when mask_image_path
    // is set.
    inpaint_active_ = false;

    // Reset ControlNet activation; armed below when controls are supplied.
    controlnet_active_ = false;
    control_inputs_.clear();
    control_images_.clear();

    if (!opts.controls.empty()) {
        if (model_class_ != ModelClass::StableDiffusion) {
            fail("prime: ControlNet is currently SD1.5 only; "
                 "Flux ControlNet is not supported");
        }
        if (controlnets_.empty()) {
            fail("prime: GenerateOptions.controls is non-empty but no "
                 "ControlNet has been registered — call add_controlnet() "
                 "first");
        }
        if (opts.controls.size() != controlnets_.size()) {
            fail("prime: GenerateOptions has " +
                 std::to_string(opts.controls.size()) +
                 " ControlNet input(s) but " +
                 std::to_string(controlnets_.size()) +
                 " ControlNet(s) are registered — counts must match");
        }
        // Control images live at FULL image resolution; each ControlNet's
        // conditioning_embedding does the 8x downsample to latent space.
        // Pixel range is [0, 1] (UnsignedUnit) to match diffusers — the HF
        // ControlNetPipeline's preprocessor does pixel/255 with no recentering.
        control_images_.reserve(opts.controls.size());
        for (const auto& ci : opts.controls) {
            if (ci.image_path.empty()) {
                fail("prime: ControlNetInput.image_path must be non-empty");
            }
            if (!(ci.start_step >= 0.0f && ci.end_step <= 1.0f &&
                  ci.start_step <= ci.end_step)) {
                fail("prime: ControlNetInput.start_step/end_step must "
                     "satisfy 0 <= start <= end <= 1");
            }
            control_images_.push_back(brodiffusion::load_image_as_latent_input(
                ci.image_path, opts.width, opts.height,
                brodiffusion::PixelRange::UnsignedUnit));
        }
        control_inputs_   = opts.controls;
        controlnet_active_ = true;
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
        std::vector<std::int32_t> clip_ids = tokenizer_->encode(prompt);
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

        // Inpaint: stash a clone of x0 BEFORE add_noise consumes it via
        // state.latent. Cheap (one extra latent-sized tensor per generation).
        if (!opts.mask_image_path.empty()) {
            inpaint_x0_   = x0_latent.clone();
            inpaint_mask_ = brodiffusion::load_mask_as_latent(
                                opts.mask_image_path, H_lat, W_lat);

            // Broadcast the (1, H_lat*W_lat) mask across C_lat channels on the
            // host once per generation — the device-side per-step blend then
            // reduces to plain elementwise mul + add against the cached
            // broadcast tensors. Faster than rebuilding the broadcast each
            // step, and avoids needing a broadcasting op in brotensor.
            const int mask_n = H_lat * W_lat;
            std::vector<float> mask_host(static_cast<std::size_t>(mask_n));
            if (inpaint_mask_.dtype == bt::Dtype::FP16) {
                std::vector<std::uint16_t> bits(
                    static_cast<std::size_t>(mask_n));
                inpaint_mask_.copy_to_host_fp16(bits.data());
                bt::sync_all();
                for (int i = 0; i < mask_n; ++i) {
                    mask_host[static_cast<std::size_t>(i)] =
                        bt::fp16_bits_to_fp32(bits[static_cast<std::size_t>(i)]);
                }
            } else {
                mask_host = inpaint_mask_.to_host_vector();
            }
            std::vector<float> mask_b_host(static_cast<std::size_t>(n_lat));
            std::vector<float> one_minus_b_host(
                static_cast<std::size_t>(n_lat));
            for (int c = 0; c < C_lat; ++c) {
                for (int i = 0; i < mask_n; ++i) {
                    const std::size_t k =
                        static_cast<std::size_t>(c) *
                        static_cast<std::size_t>(mask_n) +
                        static_cast<std::size_t>(i);
                    const float m =
                        mask_host[static_cast<std::size_t>(i)];
                    mask_b_host[k]      = m;
                    one_minus_b_host[k] = 1.0f - m;
                }
            }
            inpaint_mask_b_      = detail::upload_host(mask_b_host.data(),
                                                       1, n_lat);
            inpaint_one_minus_b_ = detail::upload_host(one_minus_b_host.data(),
                                                       1, n_lat);
            inpaint_active_ = true;
        }

        std::visit([&](auto& s) {
            using S = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<S, scheduler::DDIM> ||
                          std::is_same_v<S, scheduler::LCM> ||
                          std::is_same_v<S, scheduler::FlowMatch>) {
                s.add_noise(x0_latent, noise, t_start, state.latent, scratch_);
            } else {
                fail("prime: img2img add_noise is not supported for the "
                     "active scheduler");
            }
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

void Pipeline::step_denoise_captured_(PipelineState& state, float t,
                                      bool do_cfg) {
#ifdef BROTENSOR_HAS_CUDA
    StepGraphSession* s = step_graph_.get();
    const bool key_match =
        s != nullptr &&
        s->latent_ptr  == state.latent.data &&
        s->prepared_id == prepared_.get() &&
        s->H == state.H_lat && s->W == state.W_lat &&
        s->do_cfg == do_cfg;
    if (!key_match) {
        step_graph_ = std::make_unique<StepGraphSession>();
        s = step_graph_.get();
        s->latent_ptr  = state.latent.data;
        s->prepared_id = prepared_.get();
        s->H      = state.H_lat;
        s->W      = state.W_lat;
        s->do_cfg = do_cfg;
    }

    // Host-dependent per-step inputs (time-embedding chain) — always eager,
    // writes the persistent temb buffers the captured body reads.
    denoiser_->prepare_step(t, prepared_);

    if (s->graph.valid()) {
        s->graph.launch();
        return;
    }

    // Eager warm-up step through the capture seam: computes this step's real
    // outputs and settles every body buffer at its high-water capacity.
    denoiser_->forward_body(state.latent, state.H_lat, state.W_lat,
                            prepared_, Branch::Cond, noise_pred_cond_);
    if (do_cfg) {
        denoiser_->forward_body(state.latent, state.H_lat, state.W_lat,
                                prepared_, Branch::Uncond, noise_pred_uncond_);
    }
    ++s->eager_steps;

    // Capture only once every buffer-role assignment the captured calls can
    // start from has already been warmed. The UNet body ping-pongs a pool of
    // three buffers (x_/y_/cat_buf_) via std::swap, so the per-call role
    // permutation has period at most 3 — after 3 eager body calls, any
    // subsequent call repeats an already-warmed assignment and performs no
    // reallocation (an alloc/free of non-graph memory mid-capture is illegal
    // and poisons the graph).
    const int body_calls = s->eager_steps * (do_cfg ? 2 : 1);
    if (body_calls < 3) return;

    // Capture at the end of the second eager step: re-issue the identical
    // bodies on the capture stream. Capture records the op sequence without
    // executing it (the eager outputs above stand for this step); every
    // later step replays the whole sequence with one cudaGraphLaunch.
    bt::sync_all();
    {
        bt::CudaGraphCapture cap;
        denoiser_->forward_body(state.latent, state.H_lat, state.W_lat,
                                prepared_, Branch::Cond, noise_pred_cond_);
        if (do_cfg) {
            denoiser_->forward_body(state.latent, state.H_lat, state.W_lat,
                                    prepared_, Branch::Uncond,
                                    noise_pred_uncond_);
        }
        s->graph = cap.finish();
    }
#else
    // No CUDA backend in this build: plain eager forwards.
    denoiser_->forward(state.latent, state.H_lat, state.W_lat, t,
                       prepared_, Branch::Cond, noise_pred_cond_);
    if (do_cfg) {
        denoiser_->forward(state.latent, state.H_lat, state.W_lat, t,
                           prepared_, Branch::Uncond, noise_pred_uncond_);
    }
#endif
}

void Pipeline::step_once_scm_(PipelineState& state,
                              const GenerateOptions& opts) {
    (void)opts;  // guidance is baked into prepared_; steps/seed live elsewhere
    auto& sched = std::get<scheduler::SCM>(scheduler_);
    const int i      = state.step_index;
    const int n_lat  = denoiser_->latent_channels() *
                       state.H_lat * state.W_lat;
    const float sigma_data = sched.sigma_data();

    // sCM time transform. The schedule angle s maps to the network's input
    // timestep scm_t = sin(s)/(cos(s)+sin(s)); the latent is rescaled into the
    // network's input parameterisation by `scale`. (diffusers
    // SanaSprintPipeline.__call__ denoising loop.)
    const float s     = sched.timesteps()[i];
    const float scm_t = std::sin(s) / (std::cos(s) + std::sin(s));
    const float scale =
        std::sqrt(scm_t * scm_t + (1.0f - scm_t) * (1.0f - scm_t));

    // latent_model_input = (latent / sigma_data) * scale, into scratch_. This
    // tensor is both the DiT input and the `lmi` term in the output
    // reconstruction below, so it must survive the forward (which reads, never
    // writes, its latent argument).
    detail::resize_like(scratch_, 1, n_lat, state.latent.dtype,
                        state.latent.device);
    bt::copy_d2d(state.latent, 0, scratch_, 0, n_lat);
    bt::scale_inplace(scratch_, scale / sigma_data);

    // One DiT forward at the sCM input timestep. Guidance (already embedded via
    // the prepared conditioning) makes this a single, CFG-free pass.
    denoiser_->forward(scratch_, state.H_lat, state.W_lat, scm_t, prepared_,
                       Branch::Cond, noise_pred_cond_);

    // Reconstruct the scheduler's model_output from the network output:
    //   np = ((1 - 2t)·lmi + (1 - 2t + 2t²)·np) / scale · sigma_data
    const float a = 1.0f - 2.0f * scm_t;
    const float b = 1.0f - 2.0f * scm_t + 2.0f * scm_t * scm_t;
    bt::axpby_inplace(noise_pred_cond_, scratch_, /*a=*/b, /*b=*/a);
    bt::scale_inplace(noise_pred_cond_, sigma_data / scale);

    // Fresh unit-variance noise for the inter-step injection (the scheduler
    // applies sigma_data). FP32 to match Sana's FP32 latent. Counter offset
    // past the initial-latent draw (counters 0..n_lat), mirroring LCM.
    const std::uint64_t counter =
        static_cast<std::uint64_t>(1 + i) * static_cast<std::uint64_t>(n_lat);
    noise_step_ = bt::Tensor::empty(1, n_lat, bt::Dtype::FP32);
    bt::randn(state.rng_key, counter, noise_step_);

    sched.step(noise_pred_cond_, i, state.latent, noise_step_);
    ++state.step_index;

    // The final step returns the denoised latent in sCM coordinates; diffusers
    // decodes `denoised / sigma_data`. Undo the sigma_data scaling applied in
    // prime_sana_ so decode() sees a plain DC-AE latent.
    if (state.step_index >= state.n_steps) {
        bt::scale_inplace(state.latent, 1.0f / sigma_data);
    }
}

void Pipeline::step_once(PipelineState& state, const GenerateOptions& opts,
                         AttentionTrace* trace_out,
                         const std::vector<const bt::Tensor*>*
                             attn_logit_biases) {
    if (state.step_index >= state.n_steps) {
        fail("step_once: step_index (" + std::to_string(state.step_index) +
             ") >= n_steps (" + std::to_string(state.n_steps) + ")");
    }
    // Sana-Sprint (SCMScheduler / TrigFlow): a dedicated few-step, no-CFG,
    // no-trace path with model-specific input/output parameterisation.
    if (std::holds_alternative<scheduler::SCM>(scheduler_)) {
        if (trace_out != nullptr || attn_logit_biases != nullptr) {
            fail("step_once: attention trace / steering is not supported for "
                 "Sana-Sprint (SCMScheduler)");
        }
        step_once_scm_(state, opts);
        return;
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

    // ── ControlNet-augmented forward ──────────────────────────────────────
    // When a ControlNet is active, run it once per step against the cond
    // branch's RAW text context (HF default guess_mode=false reuses the same
    // residuals for both CFG branches). The residual-aware UNet overload is
    // reached through Denoiser::as_unet() — bypassing the Denoiser virtual
    // dispatch, which (intentionally) has no residual-carrying forward.
    if (controlnet_active_) {
        unet::UNet* u = denoiser_->as_unet();
        if (u == nullptr) {
            fail("step_once: ControlNet requires a UNet denoiser");
        }
        const bt::Tensor& ctx_cond = conditioning_.text_embeddings;

        // Active-window membership: a net contributes at step i iff
        // start_step <= i/n_steps < end_step. Avoids running the forward
        // for inactive nets; the residual contribution is simply skipped.
        const float frac = (state.n_steps > 0)
            ? (static_cast<float>(i) / static_cast<float>(state.n_steps))
            : 0.0f;
        auto in_window = [&](const ControlNetInput& ci) {
            return frac >= ci.start_step && frac < ci.end_step;
        };

        // Sum residuals across all registered ControlNets. First active net
        // writes directly into the summed buffers; subsequent active nets
        // run into per-net scratch and accumulate via bt::add_inplace.
        // When no nets are in-window this step the summed buffers retain
        // the previous step's contents — we zero them by writing the first
        // net's contribution. If literally no net is active we feed
        // nullptrs to the UNet (== plain non-CN forward).
        bool any_active = false;
        for (std::size_t k = 0; k < controlnets_.size(); ++k) {
            const auto& ci = control_inputs_[k];
            if (!in_window(ci)) continue;
            if (!any_active) {
                controlnets_[k]->forward(
                    state.latent, state.H_lat, state.W_lat, t,
                    ctx_cond, control_images_[k], ci.scale,
                    cn_down_residuals_, cn_mid_residual_);
                any_active = true;
            } else {
                controlnets_[k]->forward(
                    state.latent, state.H_lat, state.W_lat, t,
                    ctx_cond, control_images_[k], ci.scale,
                    cn_down_residuals_scratch_, cn_mid_residual_scratch_);
                for (std::size_t r = 0; r < cn_down_residuals_.size(); ++r) {
                    bt::add_inplace(cn_down_residuals_[r],
                                    cn_down_residuals_scratch_[r]);
                }
                bt::add_inplace(cn_mid_residual_, cn_mid_residual_scratch_);
            }
        }

        std::vector<const bt::Tensor*> down_ptrs;
        const bt::Tensor* mid_ptr = nullptr;
        if (any_active) {
            down_ptrs.reserve(cn_down_residuals_.size());
            for (const auto& rt : cn_down_residuals_) down_ptrs.push_back(&rt);
            mid_ptr = &cn_mid_residual_;
        }

        if (trace_mode) {
            // Trace mode bypasses the K/V cache (same as the non-CN trace
            // path) and captures the cond pass only. For LCM the guidance
            // scale is passed via the cond_proj path; for vanilla SD1.5 the
            // pointer is null. A CFG uncond branch (DDIM only) still uses
            // the fast cached + residual-aware path.
            if (u->num_xattn_blocks() == 0) {
                fail("step_once: trace mode requires a denoiser with "
                     "traceable attention blocks");
            }
            AttentionTrace scratch_trace;
            AttentionTrace* trace_dst =
                trace_out ? trace_out : &scratch_trace;
            const float gs   = conditioning_.guidance;
            const float* gsp = is_lcm ? &gs : nullptr;
            u->forward_trace(state.latent, state.H_lat, state.W_lat, t,
                             gsp, ctx_cond, down_ptrs, mid_ptr,
                             attn_logit_biases, trace_dst,
                             noise_pred_cond_);
            if (do_cfg) {
                const auto& cache_uncond =
                    u->kv_cache_for(prepared_, Branch::Uncond);
                const bt::Tensor& ctx_uncond =
                    conditioning_.uncond_embeddings;
                u->forward(state.latent, state.H_lat, state.W_lat, t,
                           ctx_uncond, cache_uncond, down_ptrs, mid_ptr,
                           noise_pred_uncond_);
            }
        } else {
            const auto& cache_cond =
                u->kv_cache_for(prepared_, Branch::Cond);
            if (is_lcm) {
                // LCM-distilled UNet: cond_proj path adds the guidance
                // embedding to the time embedding. No CFG branch under LCM.
                u->forward(state.latent, state.H_lat, state.W_lat, t,
                           conditioning_.guidance, ctx_cond, cache_cond,
                           down_ptrs, mid_ptr, noise_pred_cond_);
            } else {
                u->forward(state.latent, state.H_lat, state.W_lat, t,
                           ctx_cond, cache_cond, down_ptrs, mid_ptr,
                           noise_pred_cond_);
                if (do_cfg) {
                    const auto& cache_uncond =
                        u->kv_cache_for(prepared_, Branch::Uncond);
                    const bt::Tensor& ctx_uncond =
                        conditioning_.uncond_embeddings;
                    u->forward(state.latent, state.H_lat, state.W_lat, t,
                               ctx_uncond, cache_uncond, down_ptrs, mid_ptr,
                               noise_pred_uncond_);
                }
            }
        }
    } else if (trace_mode) {
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
        // Plain fast path. When the denoiser exposes the step-capture seam
        // and the latent is CUDA-resident, run it through the CUDA-graph
        // session (warm-up → capture → single-launch replay); otherwise the
        // classic eager forwards.
        const bool graph_eligible =
            denoiser_->supports_step_capture() &&
            state.latent.device == bt::Device::CUDA &&
            !step_graph_disabled();
        if (graph_eligible) {
            step_denoise_captured_(state, t, do_cfg);
        } else {
            denoiser_->forward(state.latent, state.H_lat, state.W_lat, t,
                               prepared_, Branch::Cond, noise_pred_cond_);
            if (do_cfg) {
                denoiser_->forward(state.latent, state.H_lat, state.W_lat, t,
                                   prepared_, Branch::Uncond,
                                   noise_pred_uncond_);
            }
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

    // ── Inpaint latent blend ──────────────────────────────────────────────
    // After the scheduler step, the latent represents the state at the next
    // timestep (i+1). Re-noise x0 to that same timestep and replace the
    // unmasked region (mask=0) with it — keeping the inpaint region (mask=1)
    // on the model's own trajectory. Skip on the final step: the latent is at
    // t=0 (clean) and re-noising would just re-add noise to a clean image.
    if (inpaint_active_ && state.step_index < state.n_steps) {
        // Per-step renoise noise: separate Philox slot from the LCM per-step
        // stream. We XOR the state's rng_key with the golden-ratio constant
        // to decorrelate from both the initial-noise (counter 0..n_lat) and
        // LCM per-step (counter (1+step)*n_lat..) draws on the same key.
        const std::uint64_t inpaint_key =
            state.rng_key ^ 0x9E3779B97F4A7C15ULL;
        const std::uint64_t counter =
            static_cast<std::uint64_t>(state.step_index) *
            static_cast<std::uint64_t>(n_lat);
        randn_compute(inpaint_key, counter, n_lat, inpaint_noise_step_);

        // x0_renoised = add_noise(x0, noise_step, state.step_index).
        // Only DDIM / LCM have add_noise on the SD1.5 path; FlowMatch is
        // unreachable here (the Flux guard in prime() rejected it).
        std::visit([&](auto& s) {
            using S = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<S, scheduler::DDIM> ||
                          std::is_same_v<S, scheduler::LCM>) {
                s.add_noise(inpaint_x0_, inpaint_noise_step_,
                            state.step_index, inpaint_renoise_buf_, scratch_);
            } else {
                fail("step_once: inpaint blend reached non-SD1.5 scheduler "
                     "(this should be impossible — Flux guard in prime() "
                     "should have rejected it)");
            }
        }, scheduler_);

        // Blend: latent = mask_b * latent + (1 - mask_b) * x0_renoised.
        bt::mul_inplace(state.latent, inpaint_mask_b_);
        bt::mul_inplace(inpaint_renoise_buf_, inpaint_one_minus_b_);
        bt::add_inplace(state.latent, inpaint_renoise_buf_);
    }
}

std::vector<float> Pipeline::decode(const PipelineState& state) {
    int n_img;
    if (model_class_ == ModelClass::Sana) {
        // DC-AE f32c32 decoder: 32x upsample, 3-channel image. The decoder
        // applies latent/scaling_factor internally and casts the FP32 latent to
        // its own compute dtype at the boundary.
        if (!dcae_) fail("decode: Sana DC-AE decoder not loaded");
        dcae_->decode(state.latent, state.H_lat, state.W_lat, decoded_);
        n_img = cfg_.dcae.image_channels *
                (state.H_lat * 32) * (state.W_lat * 32);
    } else {
        vae_.decode(state.latent, state.H_lat, state.W_lat, decoded_);
        n_img = cfg_.vae.out_channels *
                (state.H_lat * 8) * (state.W_lat * 8);
    }
    bt::sync_all();
    // The decoded tensor carries the VAE's arithmetic dtype — FP16 on a GPU
    // backend, BF16 for a force_upcast VAE (Flux), FP32 on CPU. Convert the
    // 16-bit cases to float as needed.
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
    if (decoded_.dtype == bt::Dtype::BF16) {
        std::vector<std::uint16_t> dec_bits(static_cast<std::size_t>(n_img));
        decoded_.copy_to_host_bf16(dec_bits.data());
        std::vector<float> out(static_cast<std::size_t>(n_img));
        for (int i = 0; i < n_img; ++i) {
            out[static_cast<std::size_t>(i)] =
                bt::bf16_bits_to_fp32(dec_bits[static_cast<std::size_t>(i)]);
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
        if (opts.should_cancel && opts.should_cancel()) throw GenerateCancelled();
        step_once(state, opts, /*trace_out=*/nullptr);
    }
    if (opts.should_cancel && opts.should_cancel()) throw GenerateCancelled();
    return decode(state);
}

}  // namespace brodiffusion::pipeline
