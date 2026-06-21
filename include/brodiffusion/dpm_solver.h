#pragma once

// DPMSolverMultistepScheduler — DPM-Solver++ (2M) for PixArt-Sigma.
//
// Matches Hugging Face diffusers' DPMSolverMultistepScheduler with the config
// PixArt-Sigma ships:
//
//   num_train_timesteps   = 1000
//   beta_schedule         = "linear"          (linspace(b0, b1, N))
//   beta_start            = 0.0001
//   beta_end              = 0.02
//   prediction_type       = "epsilon"
//   algorithm_type        = "dpmsolver++"
//   solver_order          = 2                  (multistep, "2M")
//   solver_type           = "midpoint"
//   timestep_spacing      = "linspace"
//   lower_order_final     = true
//   thresholding          = false  /  use_karras_sigmas = false
//
// The data-prediction (dpmsolver++) formulation in alpha/sigma space, where
// alpha_t = sqrt(alphas_cumprod_t), sigma_t = sqrt(1 - alphas_cumprod_t), and
// lambda_t = log(alpha_t) - log(sigma_t):
//
//   convert (epsilon -> x0):  x0 = (x_t - sigma_t * eps) / alpha_t
//   first order:   x_{t} = (sigma_t/sigma_s) x_s - alpha_t (e^{-h} - 1) x0_s
//   second order (midpoint, multistep): with h = lambda_t - lambda_s0,
//                  h0 = lambda_s0 - lambda_s1, r0 = h0/h, D0 = x0_s0,
//                  D1 = (x0_s0 - x0_s1)/r0,
//                  x_t = (sigma_t/sigma_s0) x_s0 - alpha_t (e^{-h}-1) D0
//                        - 0.5 alpha_t (e^{-h}-1) D1
//
// Unlike DDIM/FlowMatch (stateless per step), DPM-Solver multistep keeps the
// previous step's x0 prediction, so step() is NOT const and the scheduler
// holds per-generation state. set_timesteps() resets that state. (A forked /
// branched PipelineState would not carry this history — DPM is the linear
// generate() path, not the tree-search path.)
//
// Host-side FP32 schedule math; the per-step tensor work is copy + scale + add
// on the sample / model-output tensors at the pipeline compute dtype (FP32 on
// CPU, FP16 on a GPU backend). Single-batch (N=1) like the rest of the stack.

#include "brotensor/tensor.h"

#include <vector>

namespace brodiffusion::scheduler {

struct DPMSolverConfig {
    int   num_train_timesteps = 1000;
    float beta_start          = 0.0001f;
    float beta_end            = 0.02f;
    int   solver_order        = 2;
    bool  lower_order_final    = true;
    int   steps_offset        = 0;
};

class DPMSolverMultistep {
public:
    explicit DPMSolverMultistep(const DPMSolverConfig& cfg = {});

    // Configure the inference schedule (linspace spacing). Builds `timesteps_`
    // of length num_inference_steps, highest noise first, and resets the
    // multistep history. Must be called before step().
    void set_timesteps(int num_inference_steps);

    // DPM-Solver++ uses sigma = 1 at the start (latent = randn * 1).
    float init_noise_sigma() const { return 1.0f; }

    const std::vector<int>& timesteps() const { return timesteps_; }
    int num_inference_steps() const { return static_cast<int>(timesteps_.size()); }

    // Run one DPM-Solver++ step. All tensors carry the pipeline compute dtype
    // and must share it.
    //   model_output: (1, C*H*W) — the denoiser's epsilon prediction.
    //   step_index:   index into timesteps() (0 = first step).
    //   sample:       (1, C*H*W) — current x_t, overwritten with x_{t-1}.
    //   scratch:      reused across steps; resized as needed.
    // Mutates the internal x0 history. Caller syncs before reading sample.
    void step(const brotensor::Tensor& model_output,
              int step_index,
              brotensor::Tensor& sample,
              brotensor::Tensor& scratch);

    const DPMSolverConfig& config() const { return cfg_; }

private:
    // alpha_t, sigma_t at training timestep t (clamped into range).
    void alpha_sigma_at(int t, float& alpha, float& sigma) const;

    DPMSolverConfig    cfg_;
    std::vector<float> alphas_cumprod_;   // length num_train_timesteps
    std::vector<int>   timesteps_;        // inference order, high -> low

    // Multistep history: x0 prediction from the previous step (m_prev_) and a
    // scratch for the current step's x0 (m_cur_). lower_order_nums_ tracks the
    // warm-up so step 0 runs first-order even with solver_order = 2.
    brotensor::Tensor  m_cur_;
    brotensor::Tensor  m_prev_;
    // FP32 working buffers (the step runs in FP32 regardless of latent dtype).
    brotensor::Tensor  work_;   // FP32 sample
    brotensor::Tensor  mo32_;   // FP32 model output
    brotensor::Tensor  sc32_;   // FP32 scratch
    int                lower_order_nums_ = 0;
    bool               have_prev_ = false;
};

}  // namespace brodiffusion::scheduler
