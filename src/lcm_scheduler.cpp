#include "brodiffusion/lcm_scheduler.h"
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
    throw std::runtime_error("scheduler::LCM: " + msg);
}

// sigma_data is fixed at 0.5 in diffusers' LCMScheduler boundary-condition
// parameterization; the c_skip/c_out formulas are sensitive to this value, so
// keep it here (not in LCMConfig) to avoid users accidentally breaking the
// distilled-weight math.
constexpr float kSigmaData = 0.5f;

}  // namespace

LCM::LCM(const LCMConfig& cfg) : cfg_(cfg) {
    if (cfg_.num_train_timesteps <= 0) fail("num_train_timesteps must be positive");
    if (cfg_.beta_start <= 0.0f || cfg_.beta_end <= 0.0f ||
        cfg_.beta_end <= cfg_.beta_start) {
        fail("beta_start/beta_end invalid");
    }
    if (cfg_.original_inference_steps <= 0) {
        fail("original_inference_steps must be positive");
    }
    if (cfg_.original_inference_steps > cfg_.num_train_timesteps) {
        fail("original_inference_steps cannot exceed num_train_timesteps");
    }
    if (cfg_.timestep_scaling <= 0.0f) fail("timestep_scaling must be positive");

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

void LCM::set_timesteps(int num_inference_steps) {
    if (num_inference_steps <= 0) fail("num_inference_steps must be positive");
    if (num_inference_steps > cfg_.original_inference_steps) {
        fail("num_inference_steps cannot exceed original_inference_steps");
    }

    // k = num_train_timesteps // original_inference_steps.
    const int k = cfg_.num_train_timesteps / cfg_.original_inference_steps;
    // Current diffusers' LCMScheduler picks indices into the reversed origin
    // schedule via floor(linspace(0, K, N, endpoint=False)) where
    //   K = original_inference_steps, N = num_inference_steps.
    // For integer K/N that simplifies to j = (i * K) / N (integer floor).
    // Reversed origin timestep at index j is (K - j) * k - 1.
    const int K = cfg_.original_inference_steps;
    timesteps_.resize(num_inference_steps);
    for (int i = 0; i < num_inference_steps; ++i) {
        const int j = (i * K) / num_inference_steps;
        timesteps_[i] = (K - j) * k - 1;
    }
}

void LCM::step(const bt::Tensor& noise_pred,
               int step_index,
               bt::Tensor& sample,
               const bt::Tensor& noise,
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

    const int N = cfg_.num_train_timesteps;
    const int t = timesteps_[step_index];
    const int t_clamped = (t >= N) ? N - 1 : (t < 0 ? 0 : t);
    const int t_prev_idx = step_index + 1;
    const bool is_final = (t_prev_idx >= static_cast<int>(timesteps_.size()));

    const float alpha_t = alphas_cumprod_[t_clamped];
    float alpha_t_prev;
    if (!is_final) {
        const int t_prev = timesteps_[t_prev_idx];
        const int tp_clamped = (t_prev >= N) ? N - 1 : (t_prev < 0 ? 0 : t_prev);
        alpha_t_prev = alphas_cumprod_[tp_clamped];
    } else {
        alpha_t_prev = final_alpha_cumprod_;
    }

    const float sqrt_at        = std::sqrt(alpha_t);
    const float sqrt_1mat      = std::sqrt(1.0f - alpha_t);
    const float sqrt_at_prev   = std::sqrt(alpha_t_prev);
    const float sqrt_1mat_prev = std::sqrt(1.0f - alpha_t_prev);

    // Boundary-condition scalings. `scaled_t = t * timestep_scaling` matches
    // diffusers' LCMScheduler exactly; with timestep_scaling=10 and SD1.5
    // timesteps, c_skip is ~1e-9 and c_out is ~1.0, so the boundary scaling
    // is effectively a no-op for distilled LCM / LCM-LoRA inference. The
    // formula is kept faithful for parity with future schedules.
    const float scaled_t = static_cast<float>(t) * cfg_.timestep_scaling;
    const float sigma2   = kSigmaData * kSigmaData;
    const float denom    = scaled_t * scaled_t + sigma2;
    const float c_skip   = sigma2 / denom;
    const float c_out    = scaled_t / std::sqrt(denom);

    detail::resize_like(scratch, noise_pred.rows, noise_pred.cols,
                        compute_dtype(), noise_pred.device);

    // Compute predicted_x0 = (sample - sqrt(1-a_t)*noise_pred) / sqrt(a_t).
    // CRITICAL: do the subtraction in small-magnitude space first, THEN
    // divide by sqrt(a_t). The expanded form
    //   (1/sqrt_at)*sample + (-sqrt_1mat/sqrt_at)*noise_pred
    // has both terms at magnitude ~14 at t=999 and the difference is what
    // matters — in FP16 we'd lose ~10 bits of precision to catastrophic
    // cancellation, which compounds across LCM's few denoising steps.
    //   scratch = -sqrt_1mat * noise_pred
    bt::copy_d2d(noise_pred, 0, scratch, 0, noise_pred.size());
    bt::scale_inplace(scratch, -sqrt_1mat);
    //   scratch += sample           (small-magnitude subtraction)
    bt::add_inplace(scratch, sample);
    //   scratch *= 1/sqrt_at        (predicted_x0)
    bt::scale_inplace(scratch, 1.0f / sqrt_at);

    // denoised = c_out * predicted_x0 + c_skip * sample. In place: sample
    // becomes denoised via   sample = c_skip*sample + c_out*scratch.
    bt::scale_inplace(sample, c_skip);
    bt::scale_inplace(scratch, c_out);
    bt::add_inplace(sample, scratch);

    // Final step: sample = denoised (no re-noising).
    // Otherwise:  sample = sqrt(a_t_prev)*denoised + sqrt(1 - a_t_prev)*noise.
    if (!is_final) {
        if (noise.dtype != sample.dtype) fail("noise must share sample's dtype");
        if (noise.rows != sample.rows || noise.cols != sample.cols) {
            fail("noise shape mismatch");
        }
        bt::scale_inplace(sample, sqrt_at_prev);
        bt::copy_d2d(noise, 0, scratch, 0, noise.size());
        bt::scale_inplace(scratch, sqrt_1mat_prev);
        bt::add_inplace(sample, scratch);
    }
}

void LCM::add_noise(const bt::Tensor& x0,
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

    const int N = cfg_.num_train_timesteps;
    const int t = timesteps_[step_index];
    const int t_clamped = (t >= N) ? N - 1 : (t < 0 ? 0 : t);
    const float alpha = alphas_cumprod_[t_clamped];
    const float sqrt_a   = std::sqrt(alpha);
    const float sqrt_1ma = std::sqrt(1.0f - alpha);

    detail::resize_like(sample,  x0.rows, x0.cols, compute_dtype(), x0.device);
    detail::resize_like(scratch, x0.rows, x0.cols, compute_dtype(), x0.device);

    bt::copy_d2d(x0,    0, sample,  0, x0.size());
    bt::scale_inplace(sample, sqrt_a);
    bt::copy_d2d(noise, 0, scratch, 0, noise.size());
    bt::scale_inplace(scratch, sqrt_1ma);
    bt::add_inplace(sample, scratch);
}

}  // namespace brodiffusion::scheduler
