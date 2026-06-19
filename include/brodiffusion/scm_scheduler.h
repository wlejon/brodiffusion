#pragma once

// SCM — TrigFlow consistency-model scheduler for Sana-Sprint.
//
// Matches Hugging Face diffusers' SCMScheduler (scheduling_scm.py), the
// few-step sampler behind the guidance-distilled Sana-Sprint models. Unlike
// the rectified-flow FlowMatch / DPM schedulers (sigma in [1,0], Euler on a
// velocity), SCM works on a trigonometric "TrigFlow" schedule of angles in
// [pi/2, 0] and consumes the model's denoised reconstruction:
//
//   pred_x0    = cos(s) * x_s - sin(s) * model_output
//   x_{t}      = cos(t) * pred_x0 + sin(t) * (sigma_data * z),   z ~ N(0, I)
//
// where s is the current angle and t the next (smaller) angle. The trailing
// angle is 0, so the final step returns pred_x0 unchanged (sin(0) = 0). The
// per-step Gaussian z is only injected between steps (multi-step inference);
// the last step is deterministic.
//
//   num_train_timesteps  = 1000
//   prediction_type      = "trigflow"
//   sigma_data           = 0.5
//
// The angle schedule for num_inference_steps == 2 is the sCM default
// [max_timesteps, intermediate_timesteps, 0]; otherwise it is
// linspace(max_timesteps, 0, num_inference_steps + 1). The pipeline runs
// num_inference_steps steps (the schedule minus its trailing 0).
//
// The schedule math is host-side FP32; the per-step tensor work is a pair of
// FP32 axpby blends on the sample / model-output / noise tensors. Single-batch
// (N = 1) like the rest of the inference path. Sana runs its latent in FP32
// even on a GPU backend, so these tensors are FP32.

#include "brotensor/tensor.h"

#include <vector>

namespace brodiffusion::scheduler {

struct SCMConfig {
    int   num_train_timesteps = 1000;
    float sigma_data          = 0.5f;
};

class SCM {
public:
    explicit SCM(const SCMConfig& cfg = {});

    // Configure the inference schedule. Builds the angle schedule of length
    // num_inference_steps + 1 (trailing 0) and exposes its first
    // num_inference_steps entries as the run timesteps. Must be called before
    // step(). max_timesteps / intermediate_timesteps mirror diffusers'
    // SCMScheduler.set_timesteps defaults; intermediate_timesteps is only used
    // when num_inference_steps == 2.
    void set_timesteps(int num_inference_steps,
                       float max_timesteps = 1.57080f,
                       float intermediate_timesteps = 1.3f);

    // SCM starts from unit-variance noise (the pipeline applies the sigma_data
    // scaling of the initial latent itself), so this is 1.0.
    float init_noise_sigma() const { return 1.0f; }

    // Run timesteps (angles), ordered first-to-run (high → low). Length
    // num_inference_steps — the full schedule without its trailing 0.
    const std::vector<float>& timesteps() const { return timesteps_; }
    int num_inference_steps() const {
        return static_cast<int>(timesteps_.size());
    }

    float sigma_data() const { return cfg_.sigma_data; }

    // Run one TrigFlow step on `sample` in place.
    //   model_output: (1, C*H*W) — the model's reconstruction term (already
    //                 mapped out of the sCM input parameterisation by the
    //                 pipeline). Same dtype/shape as sample.
    //   step_index:   index into timesteps() (0 = first step).
    //   sample:       (1, C*H*W) — current x_s, overwritten with x_t.
    //   noise:        (1, C*H*W) — fresh N(0,1) draw for the inter-step noise
    //                 injection. Ignored on the final step (sin(0) = 0) but
    //                 must be supplied for uniformity.
    // Caller syncs before reading sample to host.
    void step(const brotensor::Tensor& model_output,
              int step_index,
              brotensor::Tensor& sample,
              const brotensor::Tensor& noise) const;

    const SCMConfig& config() const { return cfg_; }

private:
    SCMConfig          cfg_;
    std::vector<float> schedule_;   // full angle schedule, length N+1 (trailing 0)
    std::vector<float> timesteps_;  // run angles, length N (schedule_ minus tail)
};

}  // namespace brodiffusion::scheduler
