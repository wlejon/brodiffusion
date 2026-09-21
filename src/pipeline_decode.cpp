// The output end of Pipeline: the sigma schedule readback, VAE routing for
// every model family, the generate() convenience loop, and the
// coarse-to-fine noise expander. Moved verbatim out of pipeline.cpp — see
// pipeline_detail.h for the file map.

#include "brodiffusion/pipeline.h"

#include "pipeline_detail.h"

#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/vae_dcae.h"
#include "brodiffusion/vae_qwenimage.h"
#include "brodiffusion/vae_qwenimage21.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <variant>
#include <vector>

namespace brodiffusion::pipeline {

namespace bt = ::brotensor;

using namespace detail_pipe;

std::vector<float> Pipeline::schedule_sigmas() const {
    if (const auto* fm = std::get_if<scheduler::FlowMatch>(&scheduler_)) {
        return fm->sigmas();
    }
    return {};
}

std::vector<float> Pipeline::decode(const PipelineState& state) {
    const bool vae_time = std::getenv("BRODIFFUSION_TIME") != nullptr;
    if (vae_time) brotensor::sync_all();
    const auto vae_t0 = std::chrono::steady_clock::now();
    int n_img;
    if (model_class_ == ModelClass::Sana) {
        // DC-AE f32c32 decoder: 32x upsample, 3-channel image. The decoder
        // applies latent/scaling_factor internally and casts the FP32 latent to
        // its own compute dtype at the boundary.
        if (!dcae_) fail("decode: Sana DC-AE decoder not loaded");
        dcae_->decode(state.latent, state.H_lat, state.W_lat, decoded_);
        n_img = cfg_.dcae.image_channels *
                (state.H_lat * 32) * (state.W_lat * 32);
    } else if (model_class_ == ModelClass::Krea2) {
        // Qwen-Image VAE: 8x upsample, 3-channel image. decode() applies the
        // per-channel latents_mean/std denormalization internally (see
        // vae_qwenimage.h), so the unpacked NCHW latent goes straight in.
        if (!vae_qwen_) fail("decode: Krea2 Qwen-Image decoder not loaded");
        vae_qwen_->decode(state.latent, state.H_lat, state.W_lat, decoded_);
        n_img = cfg_.krea2.vae.input_channels *
                (state.H_lat * 8) * (state.W_lat * 8);
    } else if (model_class_ == ModelClass::QwenImage21) {
        // Qwen-Image 2.1's VAE is 16x and RGBA on both ends. decode() applies
        // the per-channel latents_mean/std denormalisation itself (exactly the
        // `latents * latents_std + latents_mean` the reference pipeline does
        // just before vae.decode), so the raw pipeline-scale latent goes
        // straight in — pre-scaling it here would apply the affine twice. The
        // fourth (alpha) plane is dropped below.
        if (!vae_qi21_) fail("decode: Qwen-Image 2.1 decoder not loaded");
        vae_qi21_->decode(state.latent, state.H_lat, state.W_lat, decoded_);
        n_img = cfg_.qwenimage21.vae.out_channels *
                (state.H_lat * 16) * (state.W_lat * 16);
    } else {
        vae_.decode(state.latent, state.H_lat, state.W_lat, decoded_);
        n_img = cfg_.vae.out_channels *
                (state.H_lat * 8) * (state.W_lat * 8);
    }
    bt::sync_all();
    if (vae_time) {
        std::fprintf(stderr, "[time] VAE decode: %.3f s\n",
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - vae_t0).count());
    }
    // The decoded tensor carries the VAE's arithmetic dtype — FP16 on a GPU
    // backend, BF16 for a force_upcast VAE (Flux, Qwen-Image 2.1), FP32 on
    // CPU. Convert the 16-bit cases to float as needed.
    std::vector<float> out;
    if (decoded_.dtype == bt::Dtype::FP16 || decoded_.dtype == bt::Dtype::BF16) {
        const bool is_fp16 = decoded_.dtype == bt::Dtype::FP16;
        std::vector<std::uint16_t> dec_bits(static_cast<std::size_t>(n_img));
        if (is_fp16) decoded_.copy_to_host_fp16(dec_bits.data());
        else         decoded_.copy_to_host_bf16(dec_bits.data());
        out.resize(static_cast<std::size_t>(n_img));
        for (int i = 0; i < n_img; ++i) {
            const std::uint16_t b = dec_bits[static_cast<std::size_t>(i)];
            out[static_cast<std::size_t>(i)] =
                is_fp16 ? bt::fp16_bits_to_fp32(b) : bt::bf16_bits_to_fp32(b);
        }
    } else {
        out = decoded_.to_host_vector();
    }

    // Qwen-Image 2.1's autoencoder is RGBA: it reconstructs four planes and
    // the pipeline's contract is three. The layout is planar NCHW, so the RGB
    // the caller wants is exactly the leading 3/4 of the buffer and dropping
    // alpha is a truncation.
    if (model_class_ == ModelClass::QwenImage21 &&
        cfg_.qwenimage21.vae.out_channels == 4) {
        const std::size_t rgb =
            static_cast<std::size_t>(3) * (state.H_lat * 16) * (state.W_lat * 16);
        if (out.size() >= rgb) out.resize(rgb);
    }
    return out;
}

std::vector<float> Pipeline::generate(std::string_view prompt,
                                       const GenerateOptions& opts) {
    PipelineState state = prime(prompt, opts);
    // img2img priming sets state.step_index to a non-zero start (t_start);
    // txt2img leaves it at 0. Loop until the schedule is exhausted —
    // step_once increments step_index itself.
    const bool loop_time = std::getenv("BRODIFFUSION_TIME") != nullptr;
    const int steps_run = state.n_steps - state.step_index;
    const auto loop_t0 = std::chrono::steady_clock::now();
    while (state.step_index < state.n_steps) {
        if (opts.should_cancel && opts.should_cancel()) throw GenerateCancelled();
        step_once(state, opts, /*trace_out=*/nullptr);
    }
    if (loop_time && steps_run > 0) {
        bt::sync_all();
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - loop_t0).count();
        std::fprintf(stderr, "[time] denoise: %.3f s (%d steps, %.3f s/step)\n",
                     secs, steps_run, secs / steps_run);
    }
    if (opts.should_cancel && opts.should_cancel()) throw GenerateCancelled();
    return decode(state);
}

std::vector<float> expand_init_noise(const float* src, int c, int h, int w,
                                     int k, std::uint64_t seed) {
    if (!src || c < 1 || h < 1 || w < 1 || k < 1) {
        throw std::invalid_argument("expand_init_noise: bad dimensions");
    }
    const int H = h * k, W = w * k;
    const std::size_t n = static_cast<std::size_t>(c) * H * W;

    // Fine-level complement E: brotensor's Philox stream, same generator as
    // the pipeline's Internal noise source.
    std::vector<float> E(n);
    {
        bt::init();
        bt::Tensor t = bt::Tensor::empty(1, static_cast<int>(n), bt::Dtype::FP32);
        bt::randn(seed, 0, t);
        bt::sync_all();
        bt::Tensor host = t.to(bt::Device::CPU);
        std::memcpy(E.data(), host.data, sizeof(float) * n);
    }

    // Conditional Gaussian: within each k×k block, subtract E's block mean
    // (leaving a zero-mean residual) and add src/k. Block means of the result
    // are then src/k — exactly the distribution of a fresh field's block
    // means when src ~ N(0,1) — and every element stays i.i.d. N(0,1).
    std::vector<float> out(n);
    const float inv_kk = 1.0f / static_cast<float>(k * k);
    const float inv_k = 1.0f / static_cast<float>(k);
    for (int ch = 0; ch < c; ++ch) {
        const std::size_t srcBase = static_cast<std::size_t>(ch) * h * w;
        const std::size_t dstBase = static_cast<std::size_t>(ch) * H * W;
        for (int by = 0; by < h; ++by) {
            for (int bx = 0; bx < w; ++bx) {
                float mean = 0.0f;
                for (int dy = 0; dy < k; ++dy) {
                    const std::size_t row = dstBase +
                        static_cast<std::size_t>(by * k + dy) * W + bx * k;
                    for (int dx = 0; dx < k; ++dx) mean += E[row + dx];
                }
                mean *= inv_kk;
                const float lv =
                    src[srcBase + static_cast<std::size_t>(by) * w + bx] * inv_k;
                for (int dy = 0; dy < k; ++dy) {
                    const std::size_t row = dstBase +
                        static_cast<std::size_t>(by * k + dy) * W + bx * k;
                    for (int dx = 0; dx < k; ++dx) {
                        out[row + dx] = E[row + dx] - mean + lv;
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace brodiffusion::pipeline
