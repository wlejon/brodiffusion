#pragma once
//
// terrain-diffusion samplers — the schedule/solver half of the world pipeline.
//
// Port of xandergos/terrain-diffusion (MIT) —
// terrain_diffusion/scheduler/dpmsolver.py (EDMDPMSolverMultistepScheduler) and
// the denoise loops in terrain_diffusion/inference/world_pipeline.py
// (_coarse_inference / _latent_inference / _decoder_inference).
//
// The pipeline runs three MPUNet stages under TWO different samplers:
//
//   coarse   EDM preconditioning + DPM-Solver++ 2nd-order multistep, 20 steps
//            over a Karras sigma schedule (rho=7, sigma_min=0.002,
//            sigma_max=80, sigma_data=0.5). `final_sigmas_type="zero"` appends
//            a trailing sigma of 0, which both terminates the trajectory at the
//            clean sample and forces the last step back to first order — which
//            is what keeps lambda = -log(sigma) finite.
//
//   base     TrigFlow consistency, 2 NFE: t_init then atan(0.35/0.5).
//   decoder  TrigFlow consistency, 1 NFE: t_init only.
//
// Two traps worth stating up front, because both are easy to "fix" wrongly:
//
//   * The scheduler's own timesteps are 0.25*log(sigma), but the MODEL is never
//     fed them. It is fed atan(sigma/sigma_data) — upstream's
//     trigflow_precondition_noise. The 0.25*log(sigma) array exists only so the
//     scheduler can map a timestep back to a step index. This port drops it and
//     tracks the step index directly, so it never appears here.
//
//   * The TrigFlow update negates the model output before using it
//     (`pred = -model(...)`). That is upstream's actual sign convention, not a
//     transcription slip.
//
// Everything below is batch-1 and host-side FP32: the sampler owns the schedule
// arithmetic in double/float on the CPU and only crosses to the device to call
// MPUNet::forward. The tensors involved are tiny (16x16x6 coarse, 64x64x5 base)
// so the transfers are noise next to the UNet itself, and keeping the solver in
// FP32 keeps the CUDA build's FP16 error confined to the network.

#include "brodiffusion/terrain/mp_unet.h"

#include <cstddef>
#include <vector>

namespace brodiffusion::terrain {

// EDM constants. These are the checkpoint's training constants, not tunables —
// every stage of the shipped terrain-diffusion model was trained at these.
inline constexpr float kSigmaData = 0.5f;
inline constexpr float kSigmaMin  = 0.002f;
inline constexpr float kSigmaMax  = 80.0f;
inline constexpr float kRho       = 7.0f;

// Karras et al. (EDM) sigma schedule, with the trailing zero appended
// (final_sigmas_type == "zero"). Returns num_steps + 1 entries, descending from
// ~sigma_max to exactly 0:
//   ramp[i]   = i / (num_steps - 1)
//   sigmas[i] = (max^(1/rho) + ramp[i] * (min^(1/rho) - max^(1/rho)))^rho
std::vector<float> karras_sigmas(int num_steps,
                                 float sigma_min = kSigmaMin,
                                 float sigma_max = kSigmaMax,
                                 float rho       = kRho);

// The scheduler's nominal timesteps, 0.25*log(sigma), one per step (num_steps
// entries — the trailing zero sigma has no timestep). Provided for completeness
// and diagnostics; the denoise loop does not need it, and it is emphatically
// NOT what the model is conditioned on. See the header comment.
std::vector<float> karras_timesteps(const std::vector<float>& sigmas);

// What the UNet is actually conditioned on at a given sigma.
float trigflow_precondition_noise(float sigma, float sigma_data = kSigmaData);

// DPM-Solver++ multistep solver over a fixed Karras schedule.
//
// Because the sampler pre-scales its inputs, the diffusion is variance-
// exploding with alpha_t == 1 throughout, so lambda_t = -log(sigma_t) and the
// exponential-integrator coefficients collapse to the forms in step().
//
// Usage is one instance per denoise: construct, then call step() once per
// schedule index in order. The instance carries the 2-slot x0 history and the
// step counter, so it must not be shared across trajectories (call reset()).
class DPMSolverMultistep {
public:
    // solver_order 1 or 2. Upstream's default for the coarse stage is 2.
    explicit DPMSolverMultistep(int num_steps, int solver_order = 2,
                                float sigma_data = kSigmaData,
                                float sigma_min  = kSigmaMin,
                                float sigma_max  = kSigmaMax,
                                float rho        = kRho);

    const std::vector<float>& sigmas() const { return sigmas_; }
    int num_steps() const { return num_steps_; }
    int step_index() const { return step_index_; }

    // Convert a raw model output at sigmas[step_index()] into the denoised x0
    // prediction (prediction_type "epsilon"):
    //   c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
    //   c_out  = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
    //   x0     = c_skip * sample + c_out * model_output
    // Exposed separately so callers can inspect the x0 trajectory; step() calls
    // it internally.
    void precondition_outputs(const float* model_output, const float* sample,
                              std::size_t n, float sigma,
                              std::vector<float>& x0) const;

    // One solver step: consumes the model output at the current step index and
    // returns the sample for the next index. Advances the step counter and
    // rotates the x0 history. `out` may alias neither `model_output` nor
    // `sample`; it is resized to n.
    //
    // Order selection replicates diffusers exactly: the first step is forced to
    // first order (the history is not yet full) and, because
    // final_sigmas_type == "zero", so is the last. Everything between runs the
    // 2nd-order midpoint update.
    void step(const float* model_output, const float* sample, std::size_t n,
              std::vector<float>& out);

    void reset();

private:
    void first_order_(const float* sample, std::size_t n, float sigma_t,
                      float sigma_s0, std::vector<float>& out) const;
    void second_order_(const float* sample, std::size_t n, float sigma_t,
                       float sigma_s0, float sigma_s1,
                       std::vector<float>& out) const;

    int   num_steps_;
    int   solver_order_;
    float sigma_data_;
    std::vector<float> sigmas_;      // num_steps_ + 1, sigmas_.back() == 0

    // model_outputs_[1] is the newest x0 prediction, [0] the previous one —
    // the same slot convention as the reference's `self.model_outputs` list.
    std::vector<float> model_outputs_[2];
    int step_index_      = 0;
    int lower_order_nums_ = 0;
};

// t_init = atan(sigma_max / sigma_data), the TrigFlow arc angle corresponding
// to pure noise.
float trigflow_t_init(float sigma_max = kSigmaMax, float sigma_data = kSigmaData);

// The stage's TrigFlow step schedule: {t_init} for the 1-NFE decoder,
// {t_init, atan(0.35/sigma_data)} for the 2-NFE base model.
std::vector<float> trigflow_t_list(bool two_step);

// ---------------------------------------------------------------------------
// Full denoise loops. All buffers are host FP32, NCHW, batch 1.
// ---------------------------------------------------------------------------

// Coarse stage: EDM + DPM-Solver++ over `num_steps`.
//   noise     (1, C_out, S, S) — C_out = net.config().out_channels (6).
//   cond_img  (1, C_cond, S, S) conditioning image concatenated on channels,
//             where C_cond = in_channels - out_channels (5). May be null only
//             if the stage has no conditioning image.
//   cond      one host vector per net.config().conditional_inputs entry.
//   out       (1, C_out, S, S), already divided by sigma_data.
void sample_coarse(MPUNet& net, const float* noise, const float* cond_img,
                   const std::vector<std::vector<float>>& cond, int S,
                   int num_steps, std::vector<float>& out);

// Base / decoder stages: TrigFlow consistency sampling.
//   t_list    one angle per NFE (see trigflow_t_list).
//   noises    t_list.size() noise fields of (1, C_out, S, S), concatenated in
//             order — one per step, matching the reference's zip(t_list, noises).
//   latents   (1, C_extra, S, S) channel-concatenated onto the model input,
//             where C_extra = in_channels - out_channels. Null when the stage
//             takes no latents (base: in_channels == out_channels).
//   out       (1, C_out, S, S), already divided by sigma_data.
void sample_trigflow(MPUNet& net, const std::vector<float>& t_list,
                     const float* noises, const float* latents,
                     const std::vector<std::vector<float>>& cond, int S,
                     std::vector<float>& out);

}  // namespace brodiffusion::terrain
