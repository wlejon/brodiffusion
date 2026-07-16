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
#include "brodiffusion/ardy/fsq_decoder.h"

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

    // Denoise one autoregressive window: fixed clean history tokens conditioning
    // a fresh block of generation noise. Only the generation tokens are stepped
    // (history is held), producing the newly generated hybrid tokens.
    //   history:    (num_history_tokens, 148) clean history (host); may be null
    //               when num_history_tokens == 0.
    //   gen_noise:  (num_gen_tokens, 148) initial noise for the generation block.
    //   out_gen:    (num_gen_tokens * 148) the denoised generation tokens (host).
    void sample_ar_window(const float* history, int num_history_tokens,
                          const float* gen_noise, int num_gen_tokens,
                          const float* text_feat, float first_heading_angle,
                          int num_denoising_steps, float cfg_weight,
                          std::vector<float>& out_gen);

    const ArdyDiffusion& diffusion() const { return diffusion_; }

private:
    ArdyDiffusion diffusion_;
    ArdyDenoiser& denoiser_;
};

// Autoregressive text-to-motion generator: chains ArdyWindowSampler windows into
// an arbitrary-length hybrid motion sequence, recentering + requantizing the
// history between windows and tracking the global translation, exactly as
// ardy_model.py Ardy.__call__ does for the text-only (no history, no constraint,
// no crop) path. Produces the full hybrid sequence in the original world frame
// (before FSQ detokenization to explicit motion).
class ArdyMotionGenerator {
public:
    ArdyMotionGenerator(ArdyDenoiser& denoiser, FsqMotionDecoder& fsq,
                        int gen_horizon_len = 52, int num_base_steps = 10);

    // Number of hybrid tokens produced for a requested frame count: one window is
    // gen_horizon_len frames == gen_horizon_len/fpt tokens, and num_frames is
    // rounded up to a whole number of windows.
    int num_windows(int num_frames) const;
    int num_tokens(int num_frames) const;

    // Generate the full hybrid motion sequence for a text embedding.
    //   text_feat:  (4096) host text embedding.
    //   num_frames: requested length (rounded up to a whole window).
    //   first_heading_angle: frame-0 heading (radians).
    //   gen_noise:  (num_windows * (gen_horizon_len/fpt) * 148) per-window noise,
    //               window-major — one fresh generation-block noise per window.
    //   out_hybrid: (num_tokens * 148) hybrid tokens in the world frame.
    //   out_T_tok:  num_tokens(num_frames).
    void generate_hybrid(const float* text_feat, int num_frames,
                         float first_heading_angle, int num_denoising_steps,
                         float cfg_weight, const float* gen_noise,
                         std::vector<float>& out_hybrid, int& out_T_tok);

    // Detokenize a world-frame hybrid sequence into explicit ARDY motion
    // features: (F frames x motion_rep_dim, 414 = [global_root 5, body 409]),
    // F = T_tok * fpt. Runs the FSQ decoder on the body latents conditioned on
    // the local root derived from the hybrid's global root, then concatenates
    // the global root with the decoded body (get_explicit_motion_from_hybrid).
    void detokenize_to_motion(const float* hybrid, int T_tok,
                              std::vector<float>& out_motion);

private:
    ArdyWindowSampler sampler_;
    ArdyDenoiser& denoiser_;
    FsqMotionDecoder& fsq_;
    int gen_horizon_len_;
};

}  // namespace brodiffusion::ardy
