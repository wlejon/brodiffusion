#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/detail/device.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace brodiffusion::scheduler {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("scheduler::FlowMatch: " + msg);
}

}  // namespace

FlowMatch::FlowMatch(const FlowMatchConfig& cfg) : cfg_(cfg) {
    if (cfg_.num_train_timesteps <= 0) fail("num_train_timesteps must be positive");
    if (cfg_.shift <= 0.0f) fail("shift must be positive");
}

void FlowMatch::set_timesteps(int num_inference_steps, int image_seq_len) {
    if (num_inference_steps <= 0) fail("num_inference_steps must be positive");

    const int N = num_inference_steps;

    // 1. Raw sigmas: linspace(1.0, 1.0/N, N).
    std::vector<float> sigma_raw(static_cast<std::size_t>(N));
    if (N == 1) {
        sigma_raw[0] = 1.0f;
    } else {
        const float lo   = 1.0f / static_cast<float>(N);
        const float span = (1.0f - lo) / static_cast<float>(N - 1);
        for (int i = 0; i < N; ++i) {
            sigma_raw[static_cast<std::size_t>(i)] =
                1.0f - static_cast<float>(i) * span;
        }
    }

    // 2. Apply the resolution shift.
    std::vector<float> sigma(static_cast<std::size_t>(N));
    if (cfg_.use_dynamic_shifting) {
        // flux-dev: resolution-dependent shift mu derived from the image
        // latent token count.
        const float m = (cfg_.max_shift - cfg_.base_shift) /
                        static_cast<float>(cfg_.max_image_seq_len -
                                           cfg_.base_image_seq_len);
        const float b = cfg_.base_shift -
                        m * static_cast<float>(cfg_.base_image_seq_len);
        const float mu = static_cast<float>(image_seq_len) * m + b;
        const float exp_mu = std::exp(mu);
        for (int i = 0; i < N; ++i) {
            const float sr = sigma_raw[static_cast<std::size_t>(i)];
            sigma[static_cast<std::size_t>(i)] =
                exp_mu / (exp_mu + (1.0f / sr - 1.0f));
        }
    } else {
        // Static shift (flux-schnell / SD3).
        for (int i = 0; i < N; ++i) {
            const float sr = sigma_raw[static_cast<std::size_t>(i)];
            sigma[static_cast<std::size_t>(i)] =
                cfg_.shift * sr / (1.0f + (cfg_.shift - 1.0f) * sr);
        }
    }

    // 3. Continuous timesteps = sigma * num_train_timesteps.
    timesteps_.resize(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        timesteps_[static_cast<std::size_t>(i)] =
            sigma[static_cast<std::size_t>(i)] *
            static_cast<float>(cfg_.num_train_timesteps);
    }

    // 4. sigmas_ of length N+1 with a trailing 0.0 (consumed by step()).
    sigmas_.resize(static_cast<std::size_t>(N) + 1);
    for (int i = 0; i < N; ++i) {
        sigmas_[static_cast<std::size_t>(i)] = sigma[static_cast<std::size_t>(i)];
    }
    sigmas_[static_cast<std::size_t>(N)] = 0.0f;
}

void FlowMatch::step(const bt::Tensor& v,
                     int step_index,
                     bt::Tensor& sample,
                     bt::Tensor& scratch) const {
    if (timesteps_.empty()) fail("set_timesteps() not called");
    if (step_index < 0 || step_index >= static_cast<int>(timesteps_.size())) {
        fail("step_index out of range");
    }
    if (v.dtype != sample.dtype) {
        fail("v and sample must share a dtype");
    }
    if (v.rows != sample.rows || v.cols != sample.cols) {
        fail("v and sample shape mismatch");
    }

    // Rectified-flow Euler update:  sample += (sigma_next - sigma_t) * v.
    const float sigma_t    = sigmas_[static_cast<std::size_t>(step_index)];
    const float sigma_next = sigmas_[static_cast<std::size_t>(step_index) + 1];
    const float d_sigma    = sigma_next - sigma_t;

    detail::resize_like(scratch, v.rows, v.cols,
                        compute_dtype(), v.device);

    bt::copy_d2d(v, 0, scratch, 0, v.size());
    bt::scale_inplace(scratch, d_sigma);
    bt::add_inplace(sample, scratch);
}

void FlowMatch::add_noise(const bt::Tensor& x0,
                          const bt::Tensor& noise,
                          int step_index,
                          bt::Tensor& sample,
                          bt::Tensor& scratch) const {
    if (timesteps_.empty()) fail("add_noise: set_timesteps() not called");
    if (step_index < 0 || step_index >= static_cast<int>(timesteps_.size())) {
        fail("add_noise: step_index out of range");
    }
    if (x0.dtype != noise.dtype) fail("add_noise: x0 and noise must share a dtype");
    if (x0.rows != noise.rows || x0.cols != noise.cols) {
        fail("add_noise: x0 and noise shape mismatch");
    }

    const float sigma_t = sigmas_[static_cast<std::size_t>(step_index)];
    const float one_minus = 1.0f - sigma_t;

    detail::resize_like(sample,  x0.rows, x0.cols, compute_dtype(), x0.device);
    detail::resize_like(scratch, x0.rows, x0.cols, compute_dtype(), x0.device);

    bt::copy_d2d(x0,    0, sample,  0, x0.size());
    bt::scale_inplace(sample, one_minus);
    bt::copy_d2d(noise, 0, scratch, 0, noise.size());
    bt::scale_inplace(scratch, sigma_t);
    bt::add_inplace(sample, scratch);
}

}  // namespace brodiffusion::scheduler
