#include "brodiffusion/lcm_scheduler.h"

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
        const int t = (K - j) * k - 1 + cfg_.steps_offset;
        timesteps_[i] = t;
    }
}

void LCM::step(const bt::GpuTensor& noise_pred,
               int step_index,
               bt::GpuTensor& sample,
               const bt::GpuTensor& noise,
               bt::GpuTensor& scratch) const {
    if (timesteps_.empty()) fail("set_timesteps() not called");
    if (step_index < 0 || step_index >= static_cast<int>(timesteps_.size())) {
        fail("step_index out of range");
    }
    if (noise_pred.dtype != bt::Dtype::FP16 || sample.dtype != bt::Dtype::FP16) {
        fail("noise_pred and sample must be FP16");
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

    // Consistency-function boundary conditions. `scaled_t = t / timestep_scaling`
    // is what diffusers calls `c_t` after the LCMScheduler's internal scaling.
    const float scaled_t = static_cast<float>(t) / cfg_.timestep_scaling;
    const float sigma2   = kSigmaData * kSigmaData;
    const float denom    = scaled_t * scaled_t + sigma2;
    const float c_skip   = sigma2 / denom;
    const float c_out    = scaled_t / std::sqrt(denom);

    // predicted_original = (sample - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
    //                    = (1/sqrt_at) * sample + (-sqrt_1mat/sqrt_at) * noise_pred
    // denoised           = c_out * predicted_original + c_skip * sample
    //
    // Combining:
    //   denoised = (c_out / sqrt_at) * sample
    //            + (-c_out * sqrt_1mat / sqrt_at) * noise_pred
    //            + c_skip * sample
    //            = A0 * sample + B0 * noise_pred
    // where:
    const float A0 = c_out / sqrt_at + c_skip;
    const float B0 = -c_out * sqrt_1mat / sqrt_at;

    // Final step: sample = denoised.
    // Otherwise:  sample = sqrt(alpha_t_prev) * denoised + sqrt(1 - alpha_t_prev) * noise
    //                    = (sqrt_at_prev * A0) * sample
    //                    + (sqrt_at_prev * B0) * noise_pred
    //                    + sqrt_1mat_prev      * noise
    float A, B, C;
    if (is_final) {
        A = A0; B = B0; C = 0.0f;
    } else {
        A = sqrt_at_prev * A0;
        B = sqrt_at_prev * B0;
        C = sqrt_1mat_prev;
    }

    if (scratch.rows != noise_pred.rows || scratch.cols != noise_pred.cols ||
        scratch.dtype != bt::Dtype::FP16) {
        scratch.resize(noise_pred.rows, noise_pred.cols, bt::Dtype::FP16);
    }

    // scratch = B * noise_pred
    bt::copy_d2d_gpu(noise_pred, 0, scratch, 0, noise_pred.size());
    bt::scale_inplace_gpu(scratch, B);
    // sample = A * sample
    bt::scale_inplace_gpu(sample, A);
    // sample += scratch  (= A*sample + B*noise_pred)
    bt::add_inplace_gpu(sample, scratch);
    // sample += C * noise  (only non-final steps)
    if (!is_final && C != 0.0f) {
        if (noise.dtype != bt::Dtype::FP16) fail("noise must be FP16");
        if (noise.rows != sample.rows || noise.cols != sample.cols) {
            fail("noise shape mismatch");
        }
        bt::copy_d2d_gpu(noise, 0, scratch, 0, noise.size());
        bt::scale_inplace_gpu(scratch, C);
        bt::add_inplace_gpu(sample, scratch);
    }
}

}  // namespace brodiffusion::scheduler
