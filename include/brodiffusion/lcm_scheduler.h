#pragma once

// LCM (Latent Consistency Model) scheduler for SD1.5.
//
// Matches Hugging Face diffusers' LCMScheduler with SD1.5's default config:
//
//   num_train_timesteps      = 1000
//   beta_schedule            = "scaled_linear" (linspace(sqrt(b0), sqrt(b1), N)^2)
//   beta_start               = 0.00085
//   beta_end                 = 0.012
//   original_inference_steps = 50            (the LCM "skipping" base schedule)
//   timestep_scaling         = 10.0          (boundary-condition scaling factor
//                                              applied to t when computing
//                                              c_skip / c_out)
//   prediction_type          = "epsilon"     (only mode supported)
//   set_alpha_to_one         = false
//   steps_offset             = 0             (differs from DDIM)
//
// `clip_sample` and `thresholding` from diffusers are NOT supported here;
// LCM-Dreamshaper does not use them.
//
// Unlike DDIM, the LCM step takes fresh Gaussian noise per (non-final) step.
// The caller must produce and upload that noise (same shape as the sample);
// see `step()` below.
//
// Math is host-side FP32; per-step GPU work is a small chain of copy/scale/add
// on the FP16 sample/noise tensors. Single-batch (N=1) like the rest of the
// inference path.

#include "brotensor/tensor.h"

#include <cstdint>
#include <vector>

namespace brodiffusion::scheduler {

struct LCMConfig {
    int   num_train_timesteps      = 1000;
    float beta_start               = 0.00085f;
    float beta_end                 = 0.012f;
    int   original_inference_steps = 50;
    float timestep_scaling         = 10.0f;
    bool  set_alpha_to_one         = false;
    int   steps_offset             = 0;
    // sigma_data is fixed at 0.5 for the LCM consistency parameterization in
    // diffusers; not exposed here.
};

class LCM {
public:
    explicit LCM(const LCMConfig& cfg = {});

    // Configure the inference schedule. Picks `num_inference_steps` entries
    // out of the original_inference_steps LCM origin timesteps:
    //
    //   k             = num_train_timesteps / original_inference_steps   (= 20)
    //   lcm_origin_ts = [1..original_inference_steps] * k - 1            (= 19,39,...,999)
    //   indices       = floor(linspace(0, K, N, endpoint=False))
    //                 = floor(i * K / N) for i in [0, N)
    //   timesteps_[i] = lcm_origin_ts_reversed[indices[i]]
    //                 = (K - floor(i*K/N)) * k - 1
    //
    // For num_inference_steps=4 this yields [999, 759, 499, 259] (diffusers ref).
    // Must be called before step()/init_noise_sigma().
    void set_timesteps(int num_inference_steps);

    // The factor latent noise is multiplied by before the first denoising
    // step. LCM uses sigma=1.
    float init_noise_sigma() const { return 1.0f; }

    // Inference timesteps, ordered first-to-run (high noise -> low noise).
    const std::vector<int>& timesteps() const { return timesteps_; }
    int num_inference_steps() const { return static_cast<int>(timesteps_.size()); }

    // Run one LCM step.
    //   noise_pred:  (1, C*H*W) FP16 — UNet output ε̂.
    //   step_index:  index into timesteps() (0 = first step).
    //   sample:      (1, C*H*W) FP16 — current x_t, overwritten in place with
    //                x_{t-1} (or the final denoised image on the last step).
    //   noise:       (1, C*H*W) FP16 — fresh standard-Gaussian noise produced
    //                by the caller for this step. Ignored on the final step
    //                (the final step is deterministic). Must outlive the call.
    //   scratch:     FP16 GpuTensor reused across steps; resized as needed.
    //
    // Caller is responsible for cuda_sync() before reading sample to host.
    void step(const brotensor::GpuTensor& noise_pred,
              int step_index,
              brotensor::GpuTensor& sample,
              const brotensor::GpuTensor& noise,
              brotensor::GpuTensor& scratch) const;

    const LCMConfig& config() const { return cfg_; }

private:
    LCMConfig          cfg_;
    std::vector<float> alphas_cumprod_;     // length num_train_timesteps
    std::vector<int>   timesteps_;          // inference order, high -> low
    float              final_alpha_cumprod_ = 1.0f;
};

}  // namespace brodiffusion::scheduler
