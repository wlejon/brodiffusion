#pragma once

// FlowMatch — rectified-flow Euler scheduler for Flux / SD3.
//
// Matches Hugging Face diffusers' FlowMatchEulerDiscreteScheduler. Unlike
// DDIM / LCM (which work in alpha-bar / epsilon-prediction space), this
// scheduler operates on a continuous sigma schedule in [1, 0] and consumes a
// velocity prediction v: the rectified-flow ODE update is the plain Euler
// step  x += (sigma_next - sigma_t) * v.
//
//   num_train_timesteps  = 1000
//   shift                = 3.0            (static resolution-independent shift,
//                                            flux-schnell / SD3)
//   use_dynamic_shifting = false          (flux-dev uses a resolution-dependent
//                                            shift derived from the image token
//                                            count; see set_timesteps)
//   prediction_type      = "velocity"     (rectified flow)
//
// The schedule math is host-side FP32; the per-step tensor work is a
// copy + scale + add on the sample / velocity tensors, run on whichever
// backend brotensor resolves at runtime (CPU by default, CUDA when
// available) at that backend's compute dtype — FP32 on CPU, FP16 on a GPU.
// Single-batch (N=1) like the rest of the inference path.

#include "brotensor/tensor.h"

#include <cstdint>
#include <vector>

namespace brodiffusion::scheduler {

struct FlowMatchConfig {
    int   num_train_timesteps = 1000;
    float shift               = 3.0f;    // static shift (flux-schnell)
    bool  use_dynamic_shifting = false;  // flux-dev resolution-dependent shift
    float base_shift          = 0.5f;
    float max_shift           = 1.15f;
    int   base_image_seq_len  = 256;
    int   max_image_seq_len   = 4096;
};

class FlowMatch {
public:
    explicit FlowMatch(const FlowMatchConfig& cfg = {});

    // Configure the inference schedule. Builds `timesteps_` of length
    // num_inference_steps, ordered from highest to lowest (continuous), and
    // the matching `sigmas_` of length num_inference_steps + 1 (with a
    // trailing 0.0 sigma used by step()). Must be called before step().
    //
    // `image_seq_len` is the number of image latent tokens; it is only used
    // when use_dynamic_shifting is true (flux-dev). It has a default so a
    // plain `set_timesteps(N)` call compiles uniformly across the DDIM / LCM /
    // FlowMatch scheduler variant.
    void set_timesteps(int num_inference_steps, int image_seq_len = 0);

    // The factor latent noise is multiplied by before the first denoising
    // step. Rectified flow starts at sigma=1, so this is 1.0.
    float init_noise_sigma() const { return 1.0f; }

    // Inference timesteps, ordered first-to-run (high noise -> low noise).
    // NOTE: continuous (vector<float>), unlike DDIM / LCM which return
    // vector<int>.
    const std::vector<float>& timesteps() const { return timesteps_; }
    int num_inference_steps() const { return static_cast<int>(timesteps_.size()); }

    // Run one rectified-flow Euler step. All tensors carry the pipeline
    // compute dtype (FP32 on CPU, FP16 on a GPU backend) and must share it.
    //   v:          (1, C*H*W) — model velocity prediction.
    //   step_index: index into timesteps() (0 = first step).
    //   sample:     (1, C*H*W) — current x_t, overwritten in place with
    //               x_{t-1}  (sample += (sigma_next - sigma_t) * v).
    //   scratch:    Tensor reused across steps; resized as needed.
    //
    // Caller is responsible for sync_all() before reading sample to host.
    void step(const brotensor::Tensor& v,
              int step_index,
              brotensor::Tensor& sample,
              brotensor::Tensor& scratch) const;

    // Rectified-flow forward noising for img2img priming:
    //   x_t = (1 - sigma_t) * x_0 + sigma_t * noise,
    // where sigma_t = sigmas_[step_index]. Must be called after set_timesteps().
    void add_noise(const brotensor::Tensor& x0,
                   const brotensor::Tensor& noise,
                   int step_index,
                   brotensor::Tensor& sample,
                   brotensor::Tensor& scratch) const;

    const FlowMatchConfig& config() const { return cfg_; }

private:
    FlowMatchConfig    cfg_;
    std::vector<float> timesteps_;   // inference order, high -> low (continuous)
    std::vector<float> sigmas_;      // length num_inference_steps + 1, trailing 0
};

}  // namespace brodiffusion::scheduler
