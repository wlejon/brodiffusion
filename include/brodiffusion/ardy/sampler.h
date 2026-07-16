#pragma once
//
// ARDY diffusion schedule + spaced-DDIM window sampler (x0-prediction).
//
// Mirrors ardy/model/diffusion.py + the generation loop in ardy/model/
// ardy_model.py (denoising_step / _generate_window) for the text-only, no-
// history window: a fresh gen_horizon (52-frame / 13-token) block denoised from
// pure noise with classifier-free guidance.
//
// Schedule (ArdyDiffusion): a cosine beta schedule over num_base_steps, then a
// subsampled denoising schedule via space_timesteps + calc_diffusion_vars. The
// buffers are recomputed (betas from the gathered alphas_cumprod, re-cumprod'd,
// clamped) exactly as the reference does. DDIM is deterministic (eta = 0).
//
// Guidance (ArdyWindowSampler): the reference uses cfg_type "separated"
// (text / constraint / uncond passes). For a text-only window with no
// constraints the constraint pass is bit-identical to the unconditional pass, so
// separated CFG collapses to regular CFG — each step is two denoiser forwards
// (real text, zero text) combined as uncond + w * (text - uncond).

#include "brodiffusion/ardy/denoiser.h"

#include <vector>

namespace brodiffusion::ardy {

// Cosine-schedule diffusion process + spaced DDIM mapping (ardy diffusion.py).
class ArdyDiffusion {
public:
    explicit ArdyDiffusion(int num_base_steps = 10);

    // Per-step diffusion buffers for a subsampled denoising schedule. Following
    // the reference, space_timesteps always yields num_base_steps entries (the
    // subsample only changes their spacing); the sampling loop walks step
    // indices [0, num_denoising_steps). Vectors are indexed by the subsampled
    // step index t; base_timestep[t] is the original timestep the denoiser sees.
    struct Schedule {
        int num_denoising_steps = 0;
        std::vector<int> base_timestep;               // use_timesteps == map_tensor
        std::vector<float> sqrt_recip_alphas_cumprod;
        std::vector<float> sqrt_recipm1_alphas_cumprod;
        std::vector<float> alphas_cumprod_prev;
    };
    Schedule make_schedule(int num_denoising_steps) const;

    // One deterministic DDIM step (eta = 0): produce x_{t-1} from x_t and the
    // clean-x0 prediction at subsampled step index t. out must not alias x_t/x0.
    //   eps = (sqrt_recip_ac[t] * x_t - x0) / sqrt_recipm1_ac[t]
    //   out = x0 * sqrt(ac_prev[t]) + sqrt(1 - ac_prev[t]) * eps
    static void ddim_step(const Schedule& s, int t, const float* x_t,
                          const float* x0, int n, float* out);

    int num_base_steps() const { return num_base_steps_; }

private:
    int num_base_steps_;
    std::vector<float> alphas_cumprod_base_;  // length num_base_steps_
};

// Text-to-motion window sampler: wraps ArdyDenoiser with spaced DDIM + text-only
// classifier-free guidance.
class ArdyWindowSampler {
public:
    explicit ArdyWindowSampler(ArdyDenoiser& denoiser, int num_base_steps = 10);

    // Denoise a fresh generation window from initial noise.
    //   x_init:    (T_tok, 148) initial noise (host, row-major).
    //   text_feat: (1, 4096) host text embedding.
    //   T_tok:     number of tokens (num_frames = T_tok * fpt).
    //   first_heading_angle: frame-0 heading (radians).
    //   num_denoising_steps: subsampled schedule length.
    //   cfg_weight: text guidance weight.
    //   out:       (T_tok * 148) predicted clean hybrid (host, row-major).
    void sample(const float* x_init, const float* text_feat, int T_tok,
                float first_heading_angle, int num_denoising_steps,
                float cfg_weight, std::vector<float>& out);

    const ArdyDiffusion& diffusion() const { return diffusion_; }

private:
    ArdyDiffusion diffusion_;
    ArdyDenoiser& denoiser_;
};

}  // namespace brodiffusion::ardy
