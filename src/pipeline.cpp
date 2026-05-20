#include "brodiffusion/pipeline.h"

#include "brodiffusion/clip.h"
#include "brodiffusion/lcm_scheduler.h"
#include "brodiffusion/lora.h"
#include "brodiffusion/safetensors.h"
#include "brodiffusion/scheduler.h"
#include "brodiffusion/tokenizer.h"
#include "brodiffusion/unet.h"
#include "brodiffusion/vae.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
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

// Construct the scheduler variant from the matching config variant.
std::variant<scheduler::DDIM, scheduler::LCM>
make_scheduler(const std::variant<scheduler::DDIMConfig, scheduler::LCMConfig>& v) {
    if (std::holds_alternative<scheduler::LCMConfig>(v)) {
        return std::variant<scheduler::DDIM, scheduler::LCM>{
            std::in_place_type<scheduler::LCM>,
            std::get<scheduler::LCMConfig>(v)};
    }
    return std::variant<scheduler::DDIM, scheduler::LCM>{
        std::in_place_type<scheduler::DDIM>,
        std::get<scheduler::DDIMConfig>(v)};
}

}  // namespace

Pipeline::Pipeline(const PipelineConfig& cfg, clip::Tokenizer tokenizer)
    : cfg_(cfg),
      tokenizer_(std::move(tokenizer)),
      text_encoder_(cfg.text_encoder),
      unet_(cfg.unet),
      vae_(cfg.vae),
      scheduler_(make_scheduler(cfg.scheduler)) {}

void Pipeline::load_weights(const safetensors::File& f) {
    load_weights(f,
                 "cond_stage_model.transformer.text_model.",
                 "model.diffusion_model.",
                 "first_stage_model.decoder.");
}

void Pipeline::load_weights(const safetensors::File& f,
                            const std::string& text_prefix,
                            const std::string& unet_prefix,
                            const std::string& vae_prefix) {
    text_encoder_.load_weights(f, text_prefix);
    unet_.load_weights(f, unet_prefix);
    vae_.load_weights(f, vae_prefix);
}

void Pipeline::load_weights(const safetensors::File& text_file,
                            const safetensors::File& unet_file,
                            const safetensors::File& vae_file) {
    text_encoder_.load_weights(text_file, "text_model.");
    unet_.load_weights(unet_file, "");
    vae_.load_weights(vae_file, "decoder.");
}

void Pipeline::apply_lora(const safetensors::File& f, float scale) {
    const std::vector<lora::Triple> triples = lora::enumerate(f);
    if (triples.empty()) {
        fail("apply_lora: no LoRA triples found in file");
    }
    for (const lora::Triple& t : triples) {
        const float scale_total = (static_cast<float>(t.alpha) /
                                   static_cast<float>(t.rank)) * scale;
        const safetensors::TensorView& down = f.get(t.down_key);
        const safetensors::TensorView& up   = f.get(t.up_key);
        if (t.domain == "unet") {
            unet_.apply_lora_delta(t.target_path, down, up, scale_total);
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
    const bool do_cfg = !is_lcm && (opts.guidance_scale != 1.0f);

    // 0. Finalize UNet weights (W8A16 quantisation happens here if enabled).
    unet_.finalize_weights();

    // 1. Encode prompt(s).
    encode_prompt_(prompt, ctx_cond_);
    if (do_cfg) encode_prompt_(opts.negative_prompt, ctx_uncond_);

    // 1b. Pre-project cross-attention K/V once per generation (shared across
    // all branched states).
    unet_.prime_xattn_cache(ctx_cond_, xattn_cache_cond_);
    if (do_cfg) unet_.prime_xattn_cache(ctx_uncond_, xattn_cache_uncond_);

    // 2. Build the initial state.
    PipelineState state;
    state.H_lat = H_lat;
    state.W_lat = W_lat;
    state.rng.seed(opts.seed);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    const float sigma = std::visit(
        [](const auto& s) { return s.init_noise_sigma(); }, scheduler_);
    std::vector<std::uint16_t> noise(n_lat);
    for (int i = 0; i < n_lat; ++i) {
        noise[i] = bt::fp32_to_fp16_bits(sigma * nrm(state.rng));
    }
    state.latent = brotensor::Tensor::from_host_fp16(noise.data(), 1, n_lat);

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
    const bool do_cfg = !is_lcm && (opts.guidance_scale != 1.0f);
    const int i = state.step_index;
    const int t_int = std::visit(
        [i](const auto& s) { return s.timesteps()[i]; }, scheduler_);
    const float t = static_cast<float>(t_int);
    const int n_lat = cfg_.unet.in_channels * state.H_lat * state.W_lat;

    // Trace mode is forced if either the caller wants a trace OR is injecting
    // attention biases (forward_trace is the only path that accepts biases).
    const bool trace_mode = (trace_out != nullptr) || (attn_logit_biases != nullptr);

    // ── UNet forward ──────────────────────────────────────────────────────
    if (trace_mode) {
        // forward_trace bypasses K/V cache + INT8 + LCM cond_proj. We capture
        // the conditional pass only; CFG uncond (if any) still uses the fast
        // cached path. Use a scratch trace when the caller asked for biases
        // but not the trace itself.
        unet::UNet::CrossAttnTrace scratch_trace;
        unet::UNet::CrossAttnTrace* trace_dst =
            trace_out ? trace_out : &scratch_trace;
        unet_.forward_trace(state.latent, state.H_lat, state.W_lat, t,
                            ctx_cond_, attn_logit_biases,
                            trace_dst, noise_pred_cond_);
        if (do_cfg) {
            unet_.forward(state.latent, state.H_lat, state.W_lat, t,
                          ctx_uncond_, xattn_cache_uncond_, noise_pred_uncond_);
        }
    } else if (is_lcm) {
        if (cfg_.unet.time_cond_proj_dim > 0) {
            unet_.forward(state.latent, state.H_lat, state.W_lat, t,
                          opts.guidance_scale,
                          ctx_cond_, xattn_cache_cond_, noise_pred_cond_);
        } else {
            unet_.forward(state.latent, state.H_lat, state.W_lat, t,
                          ctx_cond_, xattn_cache_cond_, noise_pred_cond_);
        }
    } else {
        unet_.forward(state.latent, state.H_lat, state.W_lat, t,
                      ctx_cond_, xattn_cache_cond_, noise_pred_cond_);
        if (do_cfg) {
            unet_.forward(state.latent, state.H_lat, state.W_lat, t,
                          ctx_uncond_, xattn_cache_uncond_, noise_pred_uncond_);
        }
    }
    // CFG combine (DDIM only; LCM has no uncond branch).
    if (do_cfg) {
        bt::scale_inplace(noise_pred_cond_, opts.guidance_scale);
        bt::scale_inplace(noise_pred_uncond_, 1.0f - opts.guidance_scale);
        bt::add_inplace(noise_pred_cond_, noise_pred_uncond_);
    }

    // ── Scheduler step ────────────────────────────────────────────────────
    if (is_lcm) {
        std::normal_distribution<float> nrm(0.0f, 1.0f);
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(n_lat));
        for (int k = 0; k < n_lat; ++k) {
            bits[k] = bt::fp32_to_fp16_bits(nrm(state.rng));
        }
        noise_step_ = brotensor::Tensor::from_host_fp16(bits.data(), 1, n_lat);
        std::get<scheduler::LCM>(scheduler_).step(
            noise_pred_cond_, i, state.latent, noise_step_, scratch_);
    } else {
        std::get<scheduler::DDIM>(scheduler_).step(
            noise_pred_cond_, i, state.latent, scratch_);
    }
    ++state.step_index;
}

std::vector<float> Pipeline::decode(const PipelineState& state) {
    vae_.decode(state.latent, state.H_lat, state.W_lat, decoded_);
    bt::sync_all();
    const int n_img = cfg_.vae.out_channels *
                       (state.H_lat * 8) * (state.W_lat * 8);
    std::vector<std::uint16_t> dec_bits(static_cast<std::size_t>(n_img));
    decoded_.copy_to_host_fp16(dec_bits.data());
    std::vector<float> out(static_cast<std::size_t>(n_img));
    for (int i = 0; i < n_img; ++i) {
        out[static_cast<std::size_t>(i)] =
            bt::fp16_bits_to_fp32(dec_bits[static_cast<std::size_t>(i)]);
    }
    return out;
}

std::vector<float> Pipeline::generate(std::string_view prompt,
                                       const GenerateOptions& opts) {
    const bool prof = std::getenv("BRODIFF_PROF") != nullptr;
    using clk = std::chrono::high_resolution_clock;
    double unet_ms = 0.0, vae_ms = 0.0;

    PipelineState state = prime(prompt, opts);
    for (int i = 0; i < state.n_steps; ++i) {
        auto t0 = clk::now();
        step_once(state, opts, /*trace_out=*/nullptr);
        if (prof) bt::sync_all();
        auto t1 = clk::now();
        unet_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    auto tv0 = clk::now();
    auto out = decode(state);
    auto tv1 = clk::now();
    vae_ms = std::chrono::duration<double, std::milli>(tv1 - tv0).count();
    if (prof) {
        std::fprintf(stderr, "    [prof] unet_total=%.1f ms  vae=%.1f ms\n", unet_ms, vae_ms);
        std::fflush(stderr);
    }
    return out;
}

}  // namespace brodiffusion::pipeline
