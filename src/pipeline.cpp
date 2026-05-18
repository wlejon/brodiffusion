#include "brodiffusion/pipeline.h"

#include "brodiffusion/clip.h"
#include "brodiffusion/lcm_scheduler.h"
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

void Pipeline::encode_prompt_(std::string_view prompt, bt::GpuTensor& out) {
    std::vector<std::int32_t> ids = tokenizer_.encode(prompt);
    if (static_cast<int>(ids.size()) != cfg_.text_encoder.max_position) {
        fail("tokenizer returned " + std::to_string(ids.size()) +
             " ids, expected " + std::to_string(cfg_.text_encoder.max_position));
    }
    text_encoder_.forward(ids.data(), out);
}

std::vector<float> Pipeline::generate(std::string_view prompt,
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
    // LCM skips the uncond pass entirely: w is baked into the UNet via the
    // cond_proj input, so there is no second CFG branch.
    const bool do_cfg = !is_lcm && (opts.guidance_scale != 1.0f);

    // 1. Encode prompt(s).
    encode_prompt_(prompt, ctx_cond_);
    if (do_cfg) encode_prompt_(opts.negative_prompt, ctx_uncond_);

    // 1b. Pre-project cross-attention K/V for the (cond, uncond) text contexts.
    // Text context is fixed across all denoising steps, so projecting K/V
    // once eliminates 16 × steps × (1 + do_cfg) redundant matmuls per layer.
    unet_.prime_xattn_cache(ctx_cond_, xattn_cache_cond_);
    if (do_cfg) unet_.prime_xattn_cache(ctx_uncond_, xattn_cache_uncond_);

    // 2. Initial noise. randn(1, C_lat*H_lat*W_lat) FP16 * init_noise_sigma.
    // The RNG stream is shared with the per-step LCM noise resampling below
    // (continuation, no re-seed) so a given seed deterministically reproduces
    // both the initial latent AND every step's resampled noise.
    std::mt19937_64 rng(opts.seed);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    const float sigma = std::visit(
        [](const auto& s) { return s.init_noise_sigma(); }, scheduler_);
    std::vector<std::uint16_t> noise(n_lat);
    for (int i = 0; i < n_lat; ++i) {
        noise[i] = bt::fp32_to_fp16_bits(sigma * nrm(rng));
    }
    bt::upload_fp16(noise.data(), 1, n_lat, latent_);

    // 3. Schedule + denoising loop.
    const bool prof = std::getenv("BRODIFF_PROF") != nullptr;
    using clk = std::chrono::high_resolution_clock;
    double unet_ms = 0.0, vae_ms = 0.0;

    std::visit([&](auto& s) { s.set_timesteps(opts.num_inference_steps); },
               scheduler_);
    const int n_steps = std::visit(
        [](const auto& s) { return s.num_inference_steps(); }, scheduler_);

    std::vector<std::uint16_t> step_noise_bits;
    if (is_lcm) step_noise_bits.resize(n_lat);

    for (int i = 0; i < n_steps; ++i) {
        const int t_int = std::visit(
            [i](const auto& s) { return s.timesteps()[i]; }, scheduler_);
        const float t = static_cast<float>(t_int);

        auto t0 = clk::now();
        if (is_lcm) {
            unet_.forward(latent_, H_lat, W_lat, t,
                          opts.guidance_scale,
                          ctx_cond_, xattn_cache_cond_, noise_pred_cond_);
        } else {
            unet_.forward(latent_, H_lat, W_lat, t, ctx_cond_,
                          xattn_cache_cond_, noise_pred_cond_);
            if (do_cfg) {
                unet_.forward(latent_, H_lat, W_lat, t, ctx_uncond_,
                              xattn_cache_uncond_, noise_pred_uncond_);
                bt::scale_inplace_gpu(noise_pred_cond_, opts.guidance_scale);
                bt::scale_inplace_gpu(noise_pred_uncond_, 1.0f - opts.guidance_scale);
                bt::add_inplace_gpu(noise_pred_cond_, noise_pred_uncond_);
            }
        }
        if (prof) bt::cuda_sync();
        auto t1 = clk::now();
        unet_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (is_lcm) {
            // LCM resamples fresh Gaussian noise from the same RNG stream
            // every step; the scheduler ignores it on the final step.
            for (int k = 0; k < n_lat; ++k) {
                step_noise_bits[k] = bt::fp32_to_fp16_bits(nrm(rng));
            }
            bt::upload_fp16(step_noise_bits.data(), 1, n_lat, noise_step_);
            std::get<scheduler::LCM>(scheduler_).step(
                noise_pred_cond_, i, latent_, noise_step_, scratch_);
        } else {
            std::get<scheduler::DDIM>(scheduler_).step(
                noise_pred_cond_, i, latent_, scratch_);
        }
    }

    // 4. Decode through the VAE (scaling factor applied internally).
    auto tv0 = clk::now();
    vae_.decode(latent_, H_lat, W_lat, decoded_);
    bt::cuda_sync();
    auto tv1 = clk::now();
    vae_ms = std::chrono::duration<double, std::milli>(tv1 - tv0).count();
    if (prof) {
        std::fprintf(stderr, "    [prof] unet_total=%.1f ms  vae=%.1f ms\n", unet_ms, vae_ms);
        std::fflush(stderr);
    }

    // 5. Download FP16 → FP32 host buffer.
    const int n_img = cfg_.vae.out_channels * opts.height * opts.width;
    std::vector<std::uint16_t> dec_bits(n_img);
    bt::download_fp16(decoded_, dec_bits.data());
    std::vector<float> out(n_img);
    for (int i = 0; i < n_img; ++i) {
        out[i] = bt::fp16_bits_to_fp32(dec_bits[i]);
    }
    return out;
}

}  // namespace brodiffusion::pipeline
