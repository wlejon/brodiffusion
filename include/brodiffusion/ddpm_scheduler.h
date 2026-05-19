#pragma once

// DDPM ancestral sampler (Ho et al. 2020) — pixel-space, unconditional.
//
// Matches Hugging Face diffusers' DDPMScheduler defaults used by
// google/ddpm-cifar10-32:
//
//   num_train_timesteps   = 1000
//   beta_schedule         = "linear"           (linspace(b0, b1, N))
//   beta_start            = 0.0001
//   beta_end              = 0.02
//   variance_type         = "fixed_large"      (sigma_t^2 = beta_t)
//   clip_sample           = true               (clip x0_pred to [-1, 1])
//   prediction_type       = "epsilon"
//
// Inference runs `num_train_timesteps` denoising steps by default (i.e. the
// full DDPM chain) — picking fewer inference steps subsamples the schedule
// uniformly, the same way DDIM does.

#include "brotensor/tensor.h"

#include <vector>

namespace brodiffusion::scheduler {

struct DDPMConfig {
    int   num_train_timesteps = 1000;
    float beta_start          = 1.0e-4f;
    float beta_end            = 2.0e-2f;
    bool  clip_sample         = true;
};

class DDPM {
public:
    explicit DDPM(const DDPMConfig& cfg = {});

    // Build the inference schedule. timesteps_ is filled high → low.
    void set_timesteps(int num_inference_steps);

    // The factor latent noise is multiplied by before the first step. DDPM
    // starts from N(0, 1), so init_noise_sigma() == 1.
    float init_noise_sigma() const { return 1.0f; }

    const std::vector<int>& timesteps() const { return timesteps_; }
    int num_inference_steps() const { return static_cast<int>(timesteps_.size()); }

    // One DDPM step.
    //   eps_pred:   (1, C*H*W) FP16, predicted noise from the UNet.
    //   step_index: index into timesteps() (0 = first step, highest noise).
    //   sample:     (1, C*H*W) FP16, x_t in/out, replaced with x_{t-1}.
    //   noise:      (1, C*H*W) FP16, fresh Gaussian noise. Unused on the
    //               final step (no re-noising), but must still have the
    //               right shape — pass any FP16 tensor of that size.
    //   scratch:    FP16 GpuTensor reused across steps; resized as needed.
    //
    // Caller is responsible for cuda_sync() before reading sample to host.
    void step(const brotensor::GpuTensor& eps_pred,
              int step_index,
              brotensor::GpuTensor& sample,
              const brotensor::GpuTensor& noise,
              brotensor::GpuTensor& scratch) const;

    const DDPMConfig& config() const { return cfg_; }

private:
    DDPMConfig         cfg_;
    std::vector<float> betas_;            // length num_train_timesteps
    std::vector<float> alphas_cumprod_;   // length num_train_timesteps
    std::vector<int>   timesteps_;        // inference order, high → low
};

}  // namespace brodiffusion::scheduler
