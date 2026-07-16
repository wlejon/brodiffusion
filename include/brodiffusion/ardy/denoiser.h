#pragma once
//
// ARDY two-stage motion denoiser (x0-prediction, generation window).
//
// The full model (ardy/model/auto_latent_twostage_denoiser.py) denoises the
// hybrid motion rep — per token: [global root 20 = motion_root_dim 5 x fpt 4]
// ++ [FSQ body latent 128] = 148 — in two stages:
//   stage 1 (root): project the hybrid token + zeroed constraint slots, run the
//     root backbone -> global root prediction (13 tokens x 20).
//   convert the predicted GLOBAL root to a per-frame LOCAL root (heading-rate +
//     planar velocity + height), which conditions stage 2.
//   stage 2 (body): project [local root ++ body latent] + zeroed constraints,
//     run the body backbone -> body latent prediction (13 x 128).
//   output: [global root pred 20 ++ body latent pred 128] = 148, x0-pred.
//
// This class implements the GENERATION-window case: all tokens are generation
// tokens (history_len 0, no future constraints), num_tokens*fpt == num_frames,
// motion_mask/observed_motion all zero (text-only). That is exactly one CFG pass
// of a fresh text-to-motion window of gen_horizon (52) frames. Under CFG-
// separated with no constraints the constraint and unconditional passes coincide,
// so a text pass (real text) and an unconditional pass (zero text) suffice — both
// are this same forward.
//
// History conditioning, future/pose constraints, and the autoregressive rollout
// build on top of this and land in later stages.

#include "brodiffusion/ardy/denoiser_backbone.h"
#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::ardy {

class ArdyDenoiser {
public:
    struct Config {
        int num_frames_per_token = 4;
        int nframe_root_dim      = 20;    // motion_root_dim(5) * fpt(4)
        int latent_embedding_dim = 128;   // FSQ body latent
        int motion_root_dim      = 5;     // global root (pos 3 + heading 2)
        int local_root_dim       = 4;     // rot-vel 1 + planar vel 2 + height 1
        int motion_rep_dim       = 414;   // full explicit feature dim
        int body_dim             = 409;   // motion_rep_dim - motion_root_dim
        int latent_dim           = 1024;
        float fps                = 25.0f;
    };

    // Delegating default ctor (not a `= Config{}` default arg): GCC 12 rejects
    // value-initializing a nested type in a default argument. See denoiser_backbone.h.
    ArdyDenoiser() : ArdyDenoiser(Config{}) {}
    explicit ArdyDenoiser(const Config& cfg);
    ~ArdyDenoiser();

    ArdyDenoiser(const ArdyDenoiser&) = delete;
    ArdyDenoiser& operator=(const ArdyDenoiser&) = delete;

    const Config& config() const { return cfg_; }

    // hybrid token dim = nframe_root_dim + latent_embedding_dim = 148.
    int hybrid_dim() const { return cfg_.nframe_root_dim + cfg_.latent_embedding_dim; }

    // Load all denoiser weights (both backbones + the two hybrid projections used
    // by the generation path) from the ARDY denoiser.safetensors. Throws on any
    // missing tensor.
    void load_weights(const brotensor::safetensors::File& f);

    // Motion normalization stats. The ardy motion stats file bundles
    // [global_root 5, local_root 4, body 409] = 418 mean/std entries (NOT the
    // 414-dim explicit feature vector). Slices out the global-root [0:5] and
    // local-root [5:9] stats used by the global->local root conversion. eps
    // defaults to the ardy Stats default (1e-5).
    int stats_dim() const {
        return cfg_.motion_root_dim + cfg_.local_root_dim + cfg_.body_dim;  // 418
    }
    void set_motion_stats(const float* mean, const float* std, int n,
                          float eps = 1e-5f);

    // One x0-prediction forward over a window of history + generation tokens.
    //   hybrid:    (T_tok, 148) noisy hybrid tokens (host, row-major). The first
    //              num_history_tokens are clean conditioning history; the rest are
    //              the generation tokens being denoised.
    //   text_feat: (1, 4096) host text embedding (zero it for the uncond pass).
    //   timestep:  diffusion step index.
    //   first_heading_angle: frame-0 heading (radians).
    //   T_tok:     total number of tokens (num_frames = T_tok * fpt).
    //   num_history_tokens: leading history tokens (0 for a fresh generation
    //              window). History tokens use the plain hybrid projections and a
    //              negative token index (origin = history_len // fpt); they are
    //              carried through unchanged (the denoiser is a no-op on them).
    //   out:       (T_tok, 148) predicted clean hybrid at compute dtype. Caller
    //              syncs before reading.
    void forward(const float* hybrid, const float* text_feat, int timestep,
                 float first_heading_angle, int T_tok, brotensor::Tensor& out,
                 int num_history_tokens = 0);

private:
    struct Linear { brotensor::Tensor W, b; };

    // global root (T_frames, 5) normalized -> local root (T_frames, 4) normalized.
    void global_root_to_local_root(const float* groot_norm, int T_frames,
                                   float* lroot_norm) const;

    Config cfg_;
    ArdyDenoiserBackbone root_model_;
    ArdyDenoiserBackbone body_model_;
    Linear global_root_hybrid_proj_;              // 148  -> 1024 (history tokens)
    Linear local_root_hybrid_proj_;               // 144  -> 1024 (history tokens)
    Linear global_root_hybrid_constraints_proj_;  // 3440 -> 1024 (generation tokens)
    Linear local_root_hybrid_constraints_proj_;   // 3436 -> 1024 (generation tokens)
    // Global/local root normalization stats (eps-folded denom), sliced from motion.
    std::vector<float> gr_mean_, gr_stdeps_;  // [motion_root_dim]
    std::vector<float> lr_mean_, lr_stdeps_;  // [local_root_dim]
};

}  // namespace brodiffusion::ardy
