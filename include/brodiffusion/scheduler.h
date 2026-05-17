#pragma once

// DDIM scheduler for SD1.5 (eta = 0, deterministic).
//
// Matches Hugging Face diffusers' DDIMScheduler with SD1.5's default config:
//
//   num_train_timesteps   = 1000
//   beta_schedule         = "scaled_linear"   (linspace(sqrt(b0), sqrt(b1), N)^2)
//   beta_start            = 0.00085
//   beta_end              = 0.012
//   set_alpha_to_one      = false             (use alphas_cumprod[0] as final α)
//   steps_offset          = 1                 (add 1 to all inference timesteps)
//   prediction_type       = "epsilon"
//   timestep_spacing      = "leading"
//
// Math is host-side FP32; per-step GPU work is a copy + scale + add on the
// FP16 sample/noise tensors. Single-batch (N=1) like the rest of the
// inference path.

#include "brotensor/tensor.h"

#include <cstdint>
#include <vector>

namespace brodiffusion::scheduler {

struct DDIMConfig {
    int   num_train_timesteps = 1000;
    float beta_start          = 0.00085f;
    float beta_end            = 0.012f;
    int   steps_offset        = 1;
    bool  set_alpha_to_one    = false;
};

class DDIM {
public:
    explicit DDIM(const DDIMConfig& cfg = {});

    // Configure the inference schedule. Builds `timesteps_` of length
    // num_inference_steps, ordered from highest to lowest. Must be called
    // before step()/init_noise_sigma().
    void set_timesteps(int num_inference_steps);

    // The factor latent noise is multiplied by before the first denoising
    // step. DDIM uses sigma=1, so this is 1.0. Provided for API parity with
    // schedulers like Euler/Karras that scale the initial noise.
    float init_noise_sigma() const { return 1.0f; }

    // Inference timesteps, ordered first-to-run (high noise → low noise).
    const std::vector<int>& timesteps() const { return timesteps_; }
    int num_inference_steps() const { return static_cast<int>(timesteps_.size()); }

    // Run one DDIM step.
    //   noise_pred: (1, C*H*W) FP16 — UNet output ε̂.
    //   step_index: index into timesteps() (0 = first step).
    //   sample:    (1, C*H*W) FP16 — current x_t, overwritten in place with x_{t-1}.
    //   scratch:   FP16 GpuTensor reused across steps; resized as needed.
    // Caller is responsible for cuda_sync() before reading sample to host.
    void step(const brotensor::GpuTensor& noise_pred,
              int step_index,
              brotensor::GpuTensor& sample,
              brotensor::GpuTensor& scratch) const;

    const DDIMConfig& config() const { return cfg_; }

private:
    DDIMConfig         cfg_;
    std::vector<float> alphas_cumprod_;   // length num_train_timesteps
    std::vector<int>   timesteps_;        // inference order, high → low
    float              final_alpha_cumprod_ = 1.0f;
};

}  // namespace brodiffusion::scheduler
