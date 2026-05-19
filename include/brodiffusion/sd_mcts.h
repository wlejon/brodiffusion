#pragma once

// MCTS-guided sampling for the Stable Diffusion 1.5 pipeline.
//
// Differences from cifar_mcts::Sampler (and the reasons each one exists):
//
//   - **Action = attn-bias injection, not RNG perturbation.** SD1.5's
//     default DDIM scheduler is deterministic per step: it consumes RNG
//     only for the initial latent. Mixing the RNG mid-trajectory (as
//     cifar_mcts does for DDPM's ancestral noise) is a no-op for DDIM —
//     all branches collapse to identical futures. Instead, each action
//     selects one of B pre-generated FP32 bias tensors that are added to
//     the pre-softmax cross-attention logits at the UNet's mid block for
//     every step in the next decision interval. This routes through
//     pipeline::Pipeline::step_once's `attn_logit_biases` parameter,
//     which forces the conditional UNet pass into trace mode (bypasses
//     the K/V cache); the unconditional CFG pass still uses the fast
//     cached path. Per-step cond cost rises, but branching now actually
//     produces distinct trajectories.
//
//   - **prime() takes a text prompt.** The cross-attention K/V cache
//     lives on the pipeline (rebuilt per prime()) and is therefore
//     shared across all branched states within a single generate() —
//     exactly the invariant PipelineState::clone() relies on. We never
//     re-prime during MCTS, only fork the latent.
//
//   - **step_once takes opts.** We capture opts once at the top of
//     generate() and forward it through every per-step call inside the
//     env lambdas.
//
// Scoring: the bundled `mean_luminance_score` is a smoke-test only —
// meaningless on 512x512 SD outputs. Override via `set_scorer` with a
// CLIP score / ImageReward / HPSv2 / aesthetic predictor for real
// experiments (clip_score::CLIPScorer is the in-tree option).

#include "brodiffusion/pipeline.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace brodiffusion::sd_mcts {

// Same callback shape as cifar_mcts::ScoreFn: NCHW FP32 in [-1, 1], spatial
// dims H, W (in pixels — VAE-decoded resolution, not latent). Return any
// scalar; PUCT only cares about ordering.
using ScoreFn = std::function<float(const std::vector<float>& image,
                                    int H, int W)>;

// Smoke-test scorer. Mean luminance (ITU-R BT.709) on the [-1, 1] decoded
// image. Use ONLY to validate plumbing — replace before any real run.
float mean_luminance_score(const std::vector<float>& image, int H, int W);

struct Config {
    int branching_factor   = 3;     // children per node, B >= 2.
    int iterations         = 6;     // MCTS iterations per decision.
    int decision_interval  = 6;     // sub-steps advanced per MCTS edge.
                                    // SD1.5 default is 30 inference steps,
                                    // so ~5 decisions is the typical regime.
    float c_puct           = 1.5f;
    // Magnitude of the per-action attn-bias pattern (std of the Gaussian
    // each entry is drawn from). The pattern is added to pre-softmax xattn
    // logits at the UNet's mid block. Too small (<0.1) and branches don't
    // visibly differ; too large (>3.0) and the conditioning collapses to
    // noise. 1.0 is a reasonable starting point — softmax temperature on
    // SD's mid block is implicitly ~1/sqrt(head_dim) so this lands in the
    // same order of magnitude as the logits themselves.
    float bias_magnitude   = 1.0f;
    std::uint64_t seed     = 0;     // seeds the bias patterns + rollout policy.
};

class Sampler {
public:
    Sampler(pipeline::Pipeline& pipe, Config cfg = {});

    void set_scorer(ScoreFn fn);

    // Search-guided generation. Drives the SD pipeline through its full
    // schedule, running a fresh MCTS search at every decision point and
    // committing the most-visited action. Returns the VAE-decoded image
    // (same shape and units as pipeline::Pipeline::generate).
    std::vector<float> generate(const std::string& prompt,
                                const pipeline::GenerateOptions& opts);

    // Per-decision stats from the most recent generate() call, in order.
    struct DecisionStat {
        int step_index_before = 0;
        int best_action       = -1;
        int best_visits       = 0;
        int tree_size         = 0;
        // Normalised root visit distribution (length = branching_factor).
        // Empty if MCTS returned no choice. Useful for diagnosing exploration
        // vs. exploitation regime — uniform ~ pure exploration; peaked
        // ~ Q dominated.
        std::vector<float> root_visits;
    };
    const std::vector<DecisionStat>& last_decisions() const { return decisions_; }

    // Diagnostic: skip MCTS entirely. Run one full generation per action
    // (B trajectories), score each, return them. Use this to verify that
    // (a) the action set actually steers generation, (b) the scorer
    // discriminates between resulting images. If RolloutResult::score is
    // ~constant across actions, MCTS will never have a usable signal —
    // raise bias_magnitude or pick a different action / scorer before
    // touching PUCT hyperparameters.
    struct RolloutResult {
        int action;
        float score;
        std::vector<float> image;   // (3 * H * W) FP32 in [-1, 1]

        // Snapshots of the latent (downloaded host-side, FP32) at the start
        // of each decision interval — i.e. step_index = 0, decision_interval,
        // 2*decision_interval, … Only populated when enumerate_actions is
        // called with `capture_latents = true`. Length D where D is the
        // number of decision boundaries in the schedule (= number of
        // intervals = ceil(n_steps / decision_interval)). Each entry has
        // C_lat * H_lat * W_lat = 4 * (H/8) * (W/8) floats.
        //
        // Intended use: training data for a value head that internalises
        // MCTS — pairs (latent_snapshot[t], bias_pattern[a]) → score lets
        // a small head learn to score candidate biases without rollouts.
        // The bias patterns are deterministic in (cfg_.seed, action), so
        // the training side can reconstruct them from the action index.
        std::vector<std::vector<float>> decision_latents;
        std::vector<int>                decision_step_indices;
    };
    std::vector<RolloutResult> enumerate_actions(
        const std::string& prompt,
        const pipeline::GenerateOptions& opts,
        bool capture_latents = false);

private:
    pipeline::Pipeline&       pipe_;
    Config                    cfg_;
    ScoreFn                   scorer_;
    std::vector<DecisionStat> decisions_;
};

}  // namespace brodiffusion::sd_mcts
