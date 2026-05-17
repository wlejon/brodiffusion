#include "brodiffusion/pipeline.h"

#include "brodiffusion/clip.h"
#include "brodiffusion/safetensors.h"
#include "brodiffusion/scheduler.h"
#include "brodiffusion/tokenizer.h"
#include "brodiffusion/unet.h"
#include "brodiffusion/vae.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace brodiffusion::pipeline {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("pipeline::Pipeline: " + msg);
}

}  // namespace

Pipeline::Pipeline(const PipelineConfig& cfg, clip::Tokenizer tokenizer)
    : cfg_(cfg),
      tokenizer_(std::move(tokenizer)),
      text_encoder_(cfg.text_encoder),
      unet_(cfg.unet),
      vae_(cfg.vae),
      scheduler_(cfg.scheduler) {}

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
    const bool do_cfg = opts.guidance_scale != 1.0f;

    // 1. Encode prompt(s).
    encode_prompt_(prompt, ctx_cond_);
    if (do_cfg) encode_prompt_(opts.negative_prompt, ctx_uncond_);

    // 2. Initial noise. randn(1, C_lat*H_lat*W_lat) FP16 * init_noise_sigma.
    std::mt19937_64 rng(opts.seed);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    const float sigma = scheduler_.init_noise_sigma();
    std::vector<std::uint16_t> noise(n_lat);
    for (int i = 0; i < n_lat; ++i) {
        noise[i] = bt::fp32_to_fp16_bits(sigma * nrm(rng));
    }
    bt::upload_fp16(noise.data(), 1, n_lat, latent_);

    // 3. Schedule + denoising loop.
    scheduler_.set_timesteps(opts.num_inference_steps);
    for (int i = 0; i < scheduler_.num_inference_steps(); ++i) {
        const float t = static_cast<float>(scheduler_.timesteps()[i]);

        unet_.forward(latent_, H_lat, W_lat, t, ctx_cond_, noise_pred_cond_);

        if (do_cfg) {
            unet_.forward(latent_, H_lat, W_lat, t, ctx_uncond_, noise_pred_uncond_);
            // noise_pred = uncond + scale * (cond - uncond)
            //            = scale * cond + (1 - scale) * uncond
            bt::scale_inplace_gpu(noise_pred_cond_, opts.guidance_scale);
            bt::scale_inplace_gpu(noise_pred_uncond_, 1.0f - opts.guidance_scale);
            bt::add_inplace_gpu(noise_pred_cond_, noise_pred_uncond_);
        }

        scheduler_.step(noise_pred_cond_, i, latent_, scratch_);
    }

    // 4. Decode through the VAE (scaling factor applied internally).
    vae_.decode(latent_, H_lat, W_lat, decoded_);
    bt::cuda_sync();

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
