#include "brodiffusion/scheduler.h"
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
    throw std::runtime_error("scheduler::DDIM: " + msg);
}

}  // namespace

DDIM::DDIM(const DDIMConfig& cfg) : cfg_(cfg) {
    if (cfg_.num_train_timesteps <= 0) fail("num_train_timesteps must be positive");
    if (cfg_.beta_start <= 0.0f || cfg_.beta_end <= 0.0f ||
        cfg_.beta_end <= cfg_.beta_start) {
        fail("beta_start/beta_end invalid");
    }

    // scaled_linear betas: linspace(sqrt(b0), sqrt(b1), N)^2.
    const int N = cfg_.num_train_timesteps;
    alphas_cumprod_.resize(N);
    const float s0 = std::sqrt(cfg_.beta_start);
    const float s1 = std::sqrt(cfg_.beta_end);
    double cumprod = 1.0;
    for (int t = 0; t < N; ++t) {
        const float u = (N > 1) ? static_cast<float>(t) / static_cast<float>(N - 1) : 0.0f;
        const float s = s0 + (s1 - s0) * u;
        const float beta  = s * s;
        const float alpha = 1.0f - beta;
        cumprod *= static_cast<double>(alpha);
        alphas_cumprod_[t] = static_cast<float>(cumprod);
    }

    final_alpha_cumprod_ = cfg_.set_alpha_to_one ? 1.0f : alphas_cumprod_[0];
}

void DDIM::set_timesteps(int num_inference_steps) {
    if (num_inference_steps <= 0) fail("num_inference_steps must be positive");
    if (num_inference_steps > cfg_.num_train_timesteps) {
        fail("num_inference_steps cannot exceed num_train_timesteps");
    }
    // Leading spacing: arange(0, N_inf) * (N_train // N_inf), +steps_offset, reversed.
    const int step_ratio = cfg_.num_train_timesteps / num_inference_steps;
    timesteps_.resize(num_inference_steps);
    for (int i = 0; i < num_inference_steps; ++i) {
        const int t = i * step_ratio + cfg_.steps_offset;
        timesteps_[num_inference_steps - 1 - i] = t;
    }
}

void DDIM::step(const bt::Tensor& noise_pred,
                int step_index,
                bt::Tensor& sample,
                bt::Tensor& scratch) const {
    if (timesteps_.empty()) fail("set_timesteps() not called");
    if (step_index < 0 || step_index >= static_cast<int>(timesteps_.size())) {
        fail("step_index out of range");
    }
    if (noise_pred.dtype != sample.dtype) {
        fail("noise_pred and sample must share a dtype");
    }
    if (noise_pred.rows != sample.rows || noise_pred.cols != sample.cols) {
        fail("noise_pred and sample shape mismatch");
    }

    const int t = timesteps_[step_index];
    const int t_prev_idx = step_index + 1;
    const int N = cfg_.num_train_timesteps;
    const int t_clamped = (t >= N) ? N - 1 : t;
    const float alpha_t = alphas_cumprod_[t_clamped];
    const float alpha_t_prev =
        (t_prev_idx < static_cast<int>(timesteps_.size()))
            ? alphas_cumprod_[timesteps_[t_prev_idx] < N ? timesteps_[t_prev_idx] : N - 1]
            : final_alpha_cumprod_;

    const float sqrt_at      = std::sqrt(alpha_t);
    const float sqrt_at_prev = std::sqrt(alpha_t_prev);
    const float sqrt_1mat    = std::sqrt(1.0f - alpha_t);
    const float sqrt_1mat_prev = std::sqrt(1.0f - alpha_t_prev);

    // x_{t-1} = A * x_t + B * eps
    const float A = sqrt_at_prev / sqrt_at;
    const float B = sqrt_1mat_prev - sqrt_at_prev * sqrt_1mat / sqrt_at;

    detail::resize_like(scratch, noise_pred.rows, noise_pred.cols,
                        compute_dtype(), noise_pred.device);
    bt::copy_d2d(noise_pred, 0, scratch, 0, noise_pred.size());
    bt::scale_inplace(scratch, B);
    bt::scale_inplace(sample, A);
    bt::add_inplace(sample, scratch);
}

}  // namespace brodiffusion::scheduler
