#include "brodiffusion/pipeline.h"

#include "brodiffusion/clip.h"
#include "brodiffusion/denoiser.h"
#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/lcm_scheduler.h"
#include "brodiffusion/lora.h"
#include "brotensor/safetensors.h"
#include "brodiffusion/scheduler.h"
#include "brodiffusion/tokenizer.h"
#include "brodiffusion/unet.h"
#include "brodiffusion/vae.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
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

// Run one scheduler step on `state.latent`. LCM resamples per-step Gaussian
// noise from the state's RNG stream; FlowMatch / DDIM are deterministic.
void scheduler_step(SchedulerVariant& sched, const bt::Tensor& pred,
                    int step_index, PipelineState& state, int n_lat,
                    bt::Tensor& scratch, bt::Tensor& noise_step) {
    if (std::holds_alternative<scheduler::LCM>(sched)) {
        std::normal_distribution<float> nrm(0.0f, 1.0f);
        std::vector<float> noise_vals(static_cast<std::size_t>(n_lat));
        for (int k = 0; k < n_lat; ++k) {
            noise_vals[static_cast<std::size_t>(k)] = nrm(state.rng);
        }
        noise_step = detail::upload_host(noise_vals.data(), 1, n_lat);
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

}  // namespace

Pipeline::Pipeline(const PipelineConfig& cfg, clip::Tokenizer tokenizer)
    : cfg_(cfg),
      tokenizer_(std::move(tokenizer)),
      text_encoder_(cfg.text_encoder),
      denoiser_(std::make_unique<unet::UNet>(cfg.unet)),
      vae_(cfg.vae),
      scheduler_(make_scheduler(cfg.scheduler)) {}

const unet::UNet& Pipeline::unet() const {
    const unet::UNet* u = denoiser_->as_unet();
    if (u == nullptr) fail("unet(): the active denoiser is not a UNet");
    return *u;
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
}

void Pipeline::load_weights(const brotensor::safetensors::File& text_file,
                            const brotensor::safetensors::File& unet_file,
                            const brotensor::safetensors::File& vae_file) {
    text_encoder_.load_weights(text_file, "text_model.");
    denoiser_->load_weights(unet_file, "");
    vae_.load_weights(vae_file, "decoder.");
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
    out.rng        = rng;          // mt19937_64 is trivially copyable
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

    const int H_lat = opts.height / 8;
    const int W_lat = opts.width  / 8;
    const int C_lat = cfg_.unet.in_channels;
    const int n_lat = C_lat * H_lat * W_lat;
    const bool is_lcm = std::holds_alternative<scheduler::LCM>(scheduler_);
    const bool do_cfg = denoiser_->uses_cfg() && !is_lcm &&
                        (opts.guidance_scale != 1.0f);

    // 0. Finalize denoiser weights (W8A16 quantisation happens here if enabled).
    denoiser_->finalize_weights();

    // 1. Encode prompt(s) into the model-agnostic Conditioning struct.
    encode_prompt_(prompt, conditioning_.text_embeddings);
    if (do_cfg) {
        encode_prompt_(opts.negative_prompt, conditioning_.uncond_embeddings);
        conditioning_.has_uncond = true;
    } else {
        conditioning_.has_uncond = false;
        conditioning_.uncond_embeddings = brotensor::Tensor{};
    }
    conditioning_.guidance = is_lcm ? opts.guidance_scale : 0.0f;

    // 1b. Pre-process conditioning once per generation (cross-attention K/V
    // projection for the UNet). Shared across all branched states.
    prepared_ = denoiser_->prepare(conditioning_);

    // 2. Build the initial state.
    PipelineState state;
    state.H_lat = H_lat;
    state.W_lat = W_lat;
    state.rng.seed(opts.seed);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    const float sigma = std::visit(
        [](const auto& s) { return s.init_noise_sigma(); }, scheduler_);
    std::vector<float> noise(n_lat);
    for (int i = 0; i < n_lat; ++i) {
        noise[i] = sigma * nrm(state.rng);
    }
    state.latent = detail::upload_host(noise.data(), 1, n_lat);

    // 3. Set the timestep schedule (lives on the scheduler — shared across
    // all branches).
    std::visit([&](auto& s) { s.set_timesteps(opts.num_inference_steps); },
               scheduler_);
    state.n_steps = std::visit(
        [](const auto& s) { return s.num_inference_steps(); }, scheduler_);
    state.step_index = 0;
    return state;
}

void Pipeline::step_once(PipelineState& state, const GenerateOptions& opts,
                         unet::UNet::CrossAttnTrace* trace_out,
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
        // forward_trace bypasses K/V cache + INT8 + LCM cond_proj. We capture
        // the conditional pass only; CFG uncond (if any) still uses the fast
        // prepared path. Use a scratch trace when the caller asked for biases
        // but not the trace itself. Trace mode is UNet-only.
        unet::UNet* u = denoiser_->as_unet();
        if (u == nullptr) fail("trace mode requires a UNet denoiser");
        unet::UNet::CrossAttnTrace scratch_trace;
        unet::UNet::CrossAttnTrace* trace_dst =
            trace_out ? trace_out : &scratch_trace;
        u->forward_trace(state.latent, state.H_lat, state.W_lat, t,
                         conditioning_.text_embeddings, attn_logit_biases,
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
    for (int i = 0; i < state.n_steps; ++i) {
        step_once(state, opts, /*trace_out=*/nullptr);
    }
    return decode(state);
}

}  // namespace brodiffusion::pipeline
