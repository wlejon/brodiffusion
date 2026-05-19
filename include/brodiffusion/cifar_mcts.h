#pragma once

// MCTS-guided sampling for the CIFAR DDPM pipeline.
//
// Wraps `cifar_pipeline::Pipeline` in a `brogameagent::mcts::GenericMcts`
// env. At each "decision" (every `decision_interval` sub-steps) the sampler
// runs a PUCT search over `branching_factor` discrete noise-branch actions,
// rolls out each child to the terminal step, decodes, and scores the
// resulting image with a user-supplied callback. The highest-visit action
// is committed to the outer state, then we recurse until the schedule is
// drained.
//
// Action semantics: action `a` re-seeds the snapshot's RNG with a mixing
// hash of (current rng output, a) at the start of a decision interval.
// Replaying the same action on a deep-clone of the same state therefore
// produces bit-identical futures — required so MCTS's tree statistics
// correspond to reproducible rollouts.
//
// Cost model: each MCTS iteration descends the tree from the root,
// stepping `decision_interval` sub-steps per edge, rolling out random
// children to terminal, decoding, scoring. One iteration costs ≈
// (n_steps − root.step_index) × per-step-ms + one decode. Tune
// `iterations` and `decision_interval` against your time budget.

#include "brodiffusion/cifar_pipeline.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace brodiffusion::cifar_mcts {

// Callback that scores a final, decoded image. Inputs: NCHW FP32 in [-1, 1]
// (the same layout the pipeline's decode() / generate() return), spatial
// dims H, W. Return a scalar reward (any range — MCTS backprop and PUCT
// only care about ordering and the `c_puct` scale).
using ScoreFn = std::function<float(const std::vector<float>& image,
                                    int H, int W)>;

// Default scorer: mean luminance (0.2126 R + 0.7152 G + 0.0722 B), shifted
// so a fully-black image scores -1 and a fully-white image scores +1.
// Trivial — present only to smoke-test the MCTS wiring. Real experiments
// must override this via `set_scorer`.
float mean_luminance_score(const std::vector<float>& image, int H, int W);

struct Config {
    int branching_factor   = 4;     // children per node, B ≥ 2
    int iterations         = 8;     // MCTS iterations per decision
    int decision_interval  = 50;    // sub-steps advanced per MCTS edge
    float c_puct           = 1.5f;  // PUCT exploration constant
    std::uint64_t seed     = 0;     // for MCTS rollout policy + action mixing
};

class Sampler {
public:
    Sampler(cifar_pipeline::Pipeline& pipe, Config cfg = {});

    void set_scorer(ScoreFn fn);

    // Search-guided generation. Drives the pipeline through its full
    // schedule, running a fresh MCTS search at every decision point and
    // committing the most-visited action. Returns the decoded image
    // (same layout as cifar_pipeline::Pipeline::generate).
    std::vector<float> generate(const cifar_pipeline::GenerateOptions& opts);

    // Per-decision stats from the most recent generate() call. One entry
    // per decision point, in order. Useful for plotting "did MCTS actually
    // discriminate at this decision step, or did all children look equal?".
    struct DecisionStat {
        int   step_index_before = 0;   // sample's step index at search() time
        int   best_action       = -1;
        int   best_visits       = 0;
        int   tree_size         = 0;
    };
    const std::vector<DecisionStat>& last_decisions() const { return decisions_; }

private:
    cifar_pipeline::Pipeline& pipe_;
    Config                     cfg_;
    ScoreFn                    scorer_;
    std::vector<DecisionStat>  decisions_;
};

}  // namespace brodiffusion::cifar_mcts
