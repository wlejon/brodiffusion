#include "brodiffusion/cifar_pipeline.h"
#include "brodiffusion/safetensors.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::cifar_pipeline {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("cifar_pipeline::Pipeline: " + msg);
}

}  // namespace

PipelineState PipelineState::clone() const {
    PipelineState other;
    other.sample     = sample.clone();
    other.rng        = rng;
    other.step_index = step_index;
    other.n_steps    = n_steps;
    other.H          = H;
    other.W          = W;
    return other;
}

Pipeline::Pipeline(const PipelineConfig& cfg)
    : cfg_(cfg),
      unet_(cfg.unet),
      scheduler_(cfg.scheduler) {}

void Pipeline::load_weights(const safetensors::File& unet_file) {
    unet_.load_weights(unet_file, /*prefix=*/"");
}

PipelineState Pipeline::prime(const GenerateOptions& opts) {
    if (opts.num_inference_steps <= 0) fail("num_inference_steps must be positive");
    if (opts.height <= 0 || opts.width <= 0) fail("height/width must be positive");

    const int H = opts.height;
    const int W = opts.width;
    const int C = cfg_.unet.in_channels;
    const int n = C * H * W;

    PipelineState state;
    state.H = H;
    state.W = W;
    state.rng.seed(opts.seed);

    // Initial sample ~ N(0, init_noise_sigma^2). DDPM uses sigma=1.
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    const float sigma = scheduler_.init_noise_sigma();
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        bits[static_cast<std::size_t>(i)] =
            bt::fp32_to_fp16_bits(sigma * nrm(state.rng));
    }
    bt::upload_fp16(bits.data(), 1, n, state.sample);

    scheduler_.set_timesteps(opts.num_inference_steps);
    state.n_steps    = scheduler_.num_inference_steps();
    state.step_index = 0;
    return state;
}

void Pipeline::step_once(PipelineState& state) {
    if (state.step_index >= state.n_steps) {
        fail("step_once: step_index (" + std::to_string(state.step_index) +
             ") >= n_steps (" + std::to_string(state.n_steps) + ")");
    }
    const int i = state.step_index;
    const int t_int = scheduler_.timesteps()[static_cast<std::size_t>(i)];
    const float t = static_cast<float>(t_int);
    const int C = cfg_.unet.in_channels;
    const int n = C * state.H * state.W;

    // 1) eps_pred = UNet(sample, t).
    unet_.forward(state.sample, state.H, state.W, t, eps_pred_);

    // 2) Sample fresh ancestral noise for this step. The scheduler ignores
    //    it on the final step but still wants a tensor of the right shape;
    //    allocating + filling unconditionally is simpler than branching here.
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        bits[static_cast<std::size_t>(k)] = bt::fp32_to_fp16_bits(nrm(state.rng));
    }
    bt::upload_fp16(bits.data(), 1, n, noise_step_);

    // 3) Scheduler step: x_t → x_{t-1}.
    scheduler_.step(eps_pred_, i, state.sample, noise_step_, scratch_);
    ++state.step_index;
}

std::vector<float> Pipeline::decode(const PipelineState& state) {
    const int C = cfg_.unet.out_channels;
    const int n = C * state.H * state.W;

    // Defensive clamp — with `clip_sample=true` the scheduler already enforces
    // [-1, 1] on the predicted x0, but the final-step posterior_mean is a
    // convex blend of x0_pred and x_t, and x_t may have small overshoot.
    decoded_.resize(1, n, bt::Dtype::FP16);
    bt::copy_d2d_gpu(state.sample, 0, decoded_, 0, state.sample.size());
    bt::clamp_gpu(decoded_, -1.0f, 1.0f);

    bt::cuda_sync();
    std::vector<std::uint16_t> dec_bits(static_cast<std::size_t>(n));
    bt::download_fp16(decoded_, dec_bits.data());
    std::vector<float> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] =
            bt::fp16_bits_to_fp32(dec_bits[static_cast<std::size_t>(i)]);
    }
    return out;
}

std::vector<float> Pipeline::generate(const GenerateOptions& opts) {
    PipelineState state = prime(opts);
    for (int i = 0; i < state.n_steps; ++i) {
        step_once(state);
    }
    return decode(state);
}

}  // namespace brodiffusion::cifar_pipeline
