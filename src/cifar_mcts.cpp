#include "brodiffusion/cifar_mcts.h"

#include "brogameagent/generic_mcts.h"

#include <algorithm>
#include <any>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace brodiffusion::cifar_mcts {

namespace ga = ::brogameagent::mcts;

namespace {

// 64-bit avalanche constant (Fibonacci hash / splitmix family). Used to mix
// the action index into the snapshot's RNG state so different actions
// produce different — but reproducible — noise streams.
constexpr std::uint64_t kMixConst = 0x9E3779B97F4A7C15ULL;

// Deterministically perturb `state.rng` for a given action `a`. Same state
// + same `a` ⇒ same post-perturbation rng; different `a` ⇒ different stream.
void mix_action_into_rng(cifar_pipeline::PipelineState& s, int a) {
    const std::uint64_t r = s.rng();
    s.rng.seed(r ^ (static_cast<std::uint64_t>(a + 1) * kMixConst));
}

// Snapshot type the env passes around inside std::any. A shared_ptr so
// std::any's copy-on-assign doesn't deep-clone the latent every time the
// MCTS internals shuffle nodes.
using Snap = std::shared_ptr<cifar_pipeline::PipelineState>;

}  // namespace

float mean_luminance_score(const std::vector<float>& image, int H, int W) {
    if (image.size() != static_cast<std::size_t>(3 * H * W)) return 0.0f;
    const int plane = H * W;
    double sum = 0.0;
    for (int i = 0; i < plane; ++i) {
        // ITU-R BT.709 luma coefficients on linear-ish [-1, 1] channels.
        sum += 0.2126 * image[static_cast<std::size_t>(0 * plane + i)] +
               0.7152 * image[static_cast<std::size_t>(1 * plane + i)] +
               0.0722 * image[static_cast<std::size_t>(2 * plane + i)];
    }
    return static_cast<float>(sum / plane);
}

Sampler::Sampler(cifar_pipeline::Pipeline& pipe, Config cfg)
    : pipe_(pipe), cfg_(cfg), scorer_(mean_luminance_score) {
    if (cfg_.branching_factor < 2) {
        throw std::runtime_error("cifar_mcts::Sampler: branching_factor must be >= 2");
    }
    if (cfg_.iterations < 1) {
        throw std::runtime_error("cifar_mcts::Sampler: iterations must be >= 1");
    }
    if (cfg_.decision_interval < 1) {
        throw std::runtime_error("cifar_mcts::Sampler: decision_interval must be >= 1");
    }
}

void Sampler::set_scorer(ScoreFn fn) {
    scorer_ = fn ? std::move(fn) : ScoreFn(mean_luminance_score);
}

std::vector<float> Sampler::generate(const cifar_pipeline::GenerateOptions& opts) {
    decisions_.clear();

    cifar_pipeline::PipelineState outer = pipe_.prime(opts);

    // env_state is what GenericMcts mutates. We refill it from a clone of
    // outer before every search; restore_fn replaces it within iterations.
    cifar_pipeline::PipelineState env_state = outer.clone();

    ga::GenericEnv env;
    env.num_actions = cfg_.branching_factor;

    env.snapshot_fn = [&env_state]() -> std::any {
        return std::any(std::make_shared<cifar_pipeline::PipelineState>(env_state.clone()));
    };
    env.restore_fn = [&env_state](const std::any& s) {
        const Snap& sp = std::any_cast<const Snap&>(s);
        env_state = sp->clone();
    };
    env.step_fn = [this, &env_state](int a) -> ga::GenericStepResult {
        mix_action_into_rng(env_state, a);
        const int target = std::min(env_state.step_index + cfg_.decision_interval,
                                    env_state.n_steps);
        while (env_state.step_index < target) {
            pipe_.step_once(env_state);
        }
        if (env_state.step_index >= env_state.n_steps) {
            // Terminal: decode and score. Reward attaches to the final
            // transition only; all earlier edges return 0.
            auto img = pipe_.decode(env_state);
            const float r = scorer_(img, env_state.H, env_state.W);
            return ga::GenericStepResult{r, true};
        }
        return ga::GenericStepResult{0.0f, false};
    };
    env.legal_actions_fn = [this]() -> std::vector<int> {
        std::vector<int> v(static_cast<std::size_t>(cfg_.branching_factor));
        for (int i = 0; i < cfg_.branching_factor; ++i) v[static_cast<std::size_t>(i)] = i;
        return v;
    };
    env.observe_fn = []() -> std::vector<float> { return {}; };

    ga::GenericMcts mcts(std::move(env));
    ga::GenericMctsConfig mcfg;
    mcfg.iterations = cfg_.iterations;
    mcfg.c_puct = cfg_.c_puct;
    // Diffusion has no notion of "discount" — reward is a single terminal
    // signal applied to whichever leaf reached it. gamma=1 makes deeper
    // leaves' rewards weigh equally with shallower ones.
    mcfg.gamma = 1.0f;
    // Rollout depth must cover the worst-case distance to terminal so
    // random rollouts actually decode an image instead of bailing at
    // depth limit (and returning 0 reward).
    const int decisions_remaining_at_root =
        (outer.n_steps + cfg_.decision_interval - 1) / cfg_.decision_interval;
    mcfg.rollout_depth = decisions_remaining_at_root + 1;
    mcfg.seed = cfg_.seed ^ 0xC1FA8ULL;
    mcts.set_config(mcfg);

    while (outer.step_index < outer.n_steps) {
        // Seed env from outer; GenericMcts will snapshot inside search().
        env_state = outer.clone();

        const int before = outer.step_index;
        const int best = mcts.search();
        const auto& s = mcts.last_stats();

        // Apply the chosen action to outer using the same mixing rule
        // step_fn used inside MCTS. Bit-identical evolution.
        if (best < 0) {
            // Shouldn't happen (branching_factor >= 2), but be defensive:
            // just advance without branching.
            const int target = std::min(outer.step_index + cfg_.decision_interval,
                                        outer.n_steps);
            while (outer.step_index < target) pipe_.step_once(outer);
        } else {
            mix_action_into_rng(outer, best);
            const int target = std::min(outer.step_index + cfg_.decision_interval,
                                        outer.n_steps);
            while (outer.step_index < target) pipe_.step_once(outer);
        }

        DecisionStat stat;
        stat.step_index_before = before;
        stat.best_action       = best;
        stat.best_visits       = s.best_visits;
        stat.tree_size         = s.tree_size;
        decisions_.push_back(stat);

        // Fresh tree for the next decision. We could `advance_root(best)`
        // to reuse the subtree, but its children's stored rollouts are
        // tied to noise streams derived from the OLD parent rng — after
        // we advance outer, the env's snapshot is different, and stored
        // Q values no longer correspond to anything reproducible. Reset.
        mcts.reset();
    }

    return pipe_.decode(outer);
}

}  // namespace brodiffusion::cifar_mcts
