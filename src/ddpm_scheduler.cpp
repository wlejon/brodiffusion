#include "brodiffusion/ddpm_scheduler.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace brodiffusion::scheduler {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("scheduler::DDPM: " + msg);
}

}  // namespace

DDPM::DDPM(const DDPMConfig& cfg) : cfg_(cfg) {
    if (cfg_.num_train_timesteps <= 0) fail("num_train_timesteps must be positive");
    if (cfg_.beta_start <= 0.0f || cfg_.beta_end <= 0.0f ||
        cfg_.beta_end <= cfg_.beta_start) {
        fail("beta_start/beta_end invalid");
    }

    // Linear betas: linspace(beta_start, beta_end, N).
    const int N = cfg_.num_train_timesteps;
    betas_.resize(static_cast<std::size_t>(N));
    alphas_cumprod_.resize(static_cast<std::size_t>(N));
    double cumprod = 1.0;
    for (int t = 0; t < N; ++t) {
        const float u = (N > 1) ? static_cast<float>(t) / static_cast<float>(N - 1) : 0.0f;
        const float beta  = cfg_.beta_start + (cfg_.beta_end - cfg_.beta_start) * u;
        betas_[static_cast<std::size_t>(t)] = beta;
        cumprod *= static_cast<double>(1.0f - beta);
        alphas_cumprod_[static_cast<std::size_t>(t)] = static_cast<float>(cumprod);
    }
}

void DDPM::set_timesteps(int num_inference_steps) {
    if (num_inference_steps <= 0) fail("num_inference_steps must be positive");
    if (num_inference_steps > cfg_.num_train_timesteps) {
        fail("num_inference_steps cannot exceed num_train_timesteps");
    }

    // Diffusers' DDPMScheduler default ("leading" spacing): pick evenly-spaced
    // timesteps then reverse. step_ratio = N_train // N_inf; timesteps are
    // arange(0, N_inf) * step_ratio (no +1 offset for vanilla DDPM).
    const int step_ratio = cfg_.num_train_timesteps / num_inference_steps;
    timesteps_.resize(static_cast<std::size_t>(num_inference_steps));
    for (int i = 0; i < num_inference_steps; ++i) {
        const int t = i * step_ratio;
        timesteps_[static_cast<std::size_t>(num_inference_steps - 1 - i)] = t;
    }
}

void DDPM::step(const bt::GpuTensor& eps_pred,
                int step_index,
                bt::GpuTensor& sample,
                const bt::GpuTensor& noise,
                bt::GpuTensor& scratch) const {
    if (timesteps_.empty()) fail("set_timesteps() not called");
    if (step_index < 0 || step_index >= static_cast<int>(timesteps_.size())) {
        fail("step_index out of range");
    }
    if (eps_pred.dtype != bt::Dtype::FP16 || sample.dtype != bt::Dtype::FP16) {
        fail("eps_pred and sample must be FP16");
    }
    if (eps_pred.rows != sample.rows || eps_pred.cols != sample.cols) {
        fail("eps_pred / sample shape mismatch");
    }

    const int N = cfg_.num_train_timesteps;
    const int t = timesteps_[static_cast<std::size_t>(step_index)];
    const int t_clamped = (t >= N) ? N - 1 : (t < 0 ? 0 : t);
    const bool is_final = (step_index + 1 >= static_cast<int>(timesteps_.size()));

    const float alpha_bar_t  = alphas_cumprod_[static_cast<std::size_t>(t_clamped)];
    const float alpha_bar_tp =
        is_final ? 1.0f
                 : alphas_cumprod_[static_cast<std::size_t>(
                       std::max(0, std::min(N - 1,
                           timesteps_[static_cast<std::size_t>(step_index + 1)])))];

    // Diffusers DDPMScheduler with variance_type="fixed_large" derives current
    // beta and alpha from the cumprods so that subsampling the timestep grid
    // still produces the right per-step beta. Match that exactly:
    //   current_beta = 1 - alpha_bar_t / alpha_bar_tp
    //   current_alpha = 1 - current_beta
    const float current_beta  = 1.0f - alpha_bar_t / alpha_bar_tp;
    const float current_alpha = 1.0f - current_beta;

    const float sqrt_at      = std::sqrt(alpha_bar_t);
    const float sqrt_1mat    = std::sqrt(1.0f - alpha_bar_t);

    // 1) predicted x0 = (sample - sqrt(1-a_t)*eps_pred) / sqrt(a_t).
    //    Compute in small-magnitude order to avoid FP16 cancellation at high t
    //    (same pattern the LCM scheduler uses).
    if (scratch.rows != eps_pred.rows || scratch.cols != eps_pred.cols ||
        scratch.dtype != bt::Dtype::FP16) {
        scratch.resize(eps_pred.rows, eps_pred.cols, bt::Dtype::FP16);
    }
    bt::copy_d2d_gpu(eps_pred, 0, scratch, 0, eps_pred.size());
    bt::scale_inplace_gpu(scratch, -sqrt_1mat);
    bt::add_inplace_gpu(scratch, sample);
    bt::scale_inplace_gpu(scratch, 1.0f / sqrt_at);
    if (cfg_.clip_sample) {
        bt::clamp_gpu(scratch, -1.0f, 1.0f);
    }

    // 2) posterior_mean = c0 * x0_pred + c1 * sample, where
    //    c0 = sqrt(alpha_bar_tp) * current_beta / (1 - alpha_bar_t)
    //    c1 = sqrt(current_alpha) * (1 - alpha_bar_tp) / (1 - alpha_bar_t)
    const float one_minus_at = 1.0f - alpha_bar_t;
    const float c0 = std::sqrt(alpha_bar_tp) * current_beta / one_minus_at;
    const float c1 = std::sqrt(current_alpha) * (1.0f - alpha_bar_tp) / one_minus_at;

    //    In-place via sample. After this block, sample holds posterior_mean.
    bt::scale_inplace_gpu(scratch, c0);
    bt::scale_inplace_gpu(sample, c1);
    bt::add_inplace_gpu(sample, scratch);

    // 3) sample = posterior_mean + sigma_t * z  (only when not the final step).
    //    variance_type=fixed_large => sigma_t^2 = current_beta.
    if (!is_final) {
        if (noise.dtype != bt::Dtype::FP16) fail("noise must be FP16");
        if (noise.rows != sample.rows || noise.cols != sample.cols) {
            fail("noise shape mismatch");
        }
        const float sigma_t = std::sqrt(current_beta);
        bt::copy_d2d_gpu(noise, 0, scratch, 0, noise.size());
        bt::scale_inplace_gpu(scratch, sigma_t);
        bt::add_inplace_gpu(sample, scratch);
    }
}

}  // namespace brodiffusion::scheduler
