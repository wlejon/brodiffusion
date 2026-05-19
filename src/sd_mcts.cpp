#include "brodiffusion/sd_mcts.h"

#include "brogameagent/generic_mcts.h"

#include "brotensor/tensor.h"

#include <algorithm>
#include <any>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace brodiffusion::sd_mcts {

namespace ga = ::brogameagent::mcts;
namespace bt = ::brotensor;

namespace {

using Snap = std::shared_ptr<pipeline::PipelineState>;

// Pick the lowest-resolution (largest-stride) xattn block. SD1.5 has 16
// transformer blocks; the mid block has the largest stride (8 for 512x512
// generation) and therefore the smallest Lq = (H_lat/stride) * (W_lat/stride)
// — cheapest place to inject bias tensors, and also the most semantically
// load-bearing layer (it sits at the UNet bottleneck where layout decisions
// crystallise). Ties broken by lowest index, which puts the bias in the
// down-side mid block when there are multiple equal-stride candidates.
int pick_bias_block(const std::vector<int>& strides) {
    if (strides.empty()) {
        throw std::runtime_error("sd_mcts::Sampler: pipeline reports zero xattn blocks");
    }
    int best_i = 0;
    int best_s = strides[0];
    for (int i = 1; i < static_cast<int>(strides.size()); ++i) {
        if (strides[static_cast<std::size_t>(i)] > best_s) {
            best_s = strides[static_cast<std::size_t>(i)];
            best_i = i;
        }
    }
    return best_i;
}

}  // namespace

float mean_luminance_score(const std::vector<float>& image, int H, int W) {
    if (image.size() != static_cast<std::size_t>(3 * H * W)) return 0.0f;
    const int plane = H * W;
    double sum = 0.0;
    for (int i = 0; i < plane; ++i) {
        sum += 0.2126 * image[static_cast<std::size_t>(0 * plane + i)] +
               0.7152 * image[static_cast<std::size_t>(1 * plane + i)] +
               0.0722 * image[static_cast<std::size_t>(2 * plane + i)];
    }
    return static_cast<float>(sum / plane);
}

Sampler::Sampler(pipeline::Pipeline& pipe, Config cfg)
    : pipe_(pipe), cfg_(cfg), scorer_(mean_luminance_score) {
    if (cfg_.branching_factor < 2) {
        throw std::runtime_error("sd_mcts::Sampler: branching_factor must be >= 2");
    }
    if (cfg_.iterations < 1) {
        throw std::runtime_error("sd_mcts::Sampler: iterations must be >= 1");
    }
    if (cfg_.decision_interval < 1) {
        throw std::runtime_error("sd_mcts::Sampler: decision_interval must be >= 1");
    }
    if (cfg_.bias_magnitude < 0.0f) {
        throw std::runtime_error("sd_mcts::Sampler: bias_magnitude must be >= 0");
    }
}

void Sampler::set_scorer(ScoreFn fn) {
    scorer_ = fn ? std::move(fn) : ScoreFn(mean_luminance_score);
}

std::vector<float> Sampler::generate(const std::string& prompt,
                                     const pipeline::GenerateOptions& opts) {
    decisions_.clear();

    pipeline::PipelineState outer = pipe_.prime(prompt, opts);

    // ── Pre-generate B attn-bias patterns at the mid block ────────────────
    // Action a picks pattern a; the SAME pattern is applied at every
    // step_once inside one decision interval, then a fresh decision picks
    // again. Patterns are deterministic in (cfg_.seed, a) so MCTS rollouts
    // are reproducible.
    const auto strides = pipe_.unet().layer_strides();
    const int num_blocks = pipe_.unet().num_xattn_blocks();
    if (static_cast<int>(strides.size()) != num_blocks) {
        throw std::runtime_error(
            "sd_mcts::Sampler: layer_strides()/num_xattn_blocks() disagree");
    }
    const int bias_idx    = pick_bias_block(strides);
    const int bias_stride = strides[static_cast<std::size_t>(bias_idx)];
    const int Lq_h        = outer.H_lat / bias_stride;
    const int Lq_w        = outer.W_lat / bias_stride;
    const int Lq          = Lq_h * Lq_w;
    const int Lk          = pipe_.unet().context_length();   // 77

    std::vector<bt::GpuTensor> bias_patterns(
        static_cast<std::size_t>(cfg_.branching_factor));
    for (int a = 0; a < cfg_.branching_factor; ++a) {
        // Independent stream per action.
        std::mt19937_64 rng(cfg_.seed ^ (0xA77B1A5ULL * (static_cast<std::uint64_t>(a) + 1)));
        std::normal_distribution<float> nrm(0.0f, cfg_.bias_magnitude);
        std::vector<float> host(static_cast<std::size_t>(Lq) * Lk);
        for (auto& v : host) v = nrm(rng);
        bt::upload(host.data(), Lq, Lk, bias_patterns[static_cast<std::size_t>(a)]);
    }

    // Reusable bias-pointer vector: all nulls except the target block.
    // The vector itself is rebuilt per action only by swapping the one
    // non-null slot's pointer — cheap.
    std::vector<const bt::GpuTensor*> bias_vec(
        static_cast<std::size_t>(num_blocks), nullptr);

    auto set_active_action = [&](int a) {
        bias_vec[static_cast<std::size_t>(bias_idx)] =
            &bias_patterns[static_cast<std::size_t>(a)];
    };

    pipeline::PipelineState env_state = outer.clone();

    ga::GenericEnv env;
    env.num_actions = cfg_.branching_factor;

    env.snapshot_fn = [&env_state]() -> std::any {
        return std::any(std::make_shared<pipeline::PipelineState>(env_state.clone()));
    };
    env.restore_fn = [&env_state](const std::any& s) {
        const Snap& sp = std::any_cast<const Snap&>(s);
        env_state = sp->clone();
    };
    env.step_fn = [&, opts](int a) -> ga::GenericStepResult {
        set_active_action(a);
        const int target = std::min(env_state.step_index + cfg_.decision_interval,
                                    env_state.n_steps);
        while (env_state.step_index < target) {
            pipe_.step_once(env_state, opts, /*trace_out=*/nullptr, &bias_vec);
        }
        if (env_state.step_index >= env_state.n_steps) {
            auto img = pipe_.decode(env_state);
            const float r = scorer_(img, opts.height, opts.width);
            return ga::GenericStepResult{r, true};
        }
        return ga::GenericStepResult{0.0f, false};
    };
    env.legal_actions_fn = [this]() -> std::vector<int> {
        std::vector<int> v(static_cast<std::size_t>(cfg_.branching_factor));
        for (int i = 0; i < cfg_.branching_factor; ++i)
            v[static_cast<std::size_t>(i)] = i;
        return v;
    };
    env.observe_fn = []() -> std::vector<float> { return {}; };

    ga::GenericMcts mcts(std::move(env));
    ga::GenericMctsConfig mcfg;
    mcfg.iterations = cfg_.iterations;
    mcfg.c_puct = cfg_.c_puct;
    mcfg.gamma = 1.0f;
    const int decisions_remaining_at_root =
        (outer.n_steps + cfg_.decision_interval - 1) / cfg_.decision_interval;
    mcfg.rollout_depth = decisions_remaining_at_root + 1;
    mcfg.seed = cfg_.seed ^ 0x5D15ULL;
    mcts.set_config(mcfg);

    while (outer.step_index < outer.n_steps) {
        env_state = outer.clone();

        const int before = outer.step_index;
        const int best = mcts.search();
        const auto& s = mcts.last_stats();

        const int target = std::min(outer.step_index + cfg_.decision_interval,
                                    outer.n_steps);
        if (best >= 0) {
            set_active_action(best);
        } else {
            // Shouldn't happen with branching_factor >= 2, but if MCTS
            // returns no choice fall back to action 0's pattern so the
            // trajectory still progresses deterministically.
            set_active_action(0);
        }
        while (outer.step_index < target) {
            pipe_.step_once(outer, opts, /*trace_out=*/nullptr, &bias_vec);
        }

        DecisionStat stat;
        stat.step_index_before = before;
        stat.best_action       = best;
        stat.best_visits       = s.best_visits;
        stat.tree_size         = s.tree_size;
        stat.root_visits       = mcts.root_visits();
        decisions_.push_back(stat);

        // Fresh tree per decision. Same reasoning as cifar_mcts: the
        // sub-tree's stored Q values are conditioned on the outer state
        // at the time of search, which we've now advanced past.
        mcts.reset();
    }

    return pipe_.decode(outer);
}

std::vector<Sampler::RolloutResult> Sampler::enumerate_actions(
    const std::string& prompt, const pipeline::GenerateOptions& opts) {

    pipeline::PipelineState outer = pipe_.prime(prompt, opts);

    // Same bias-pattern generation as generate(); kept in sync by hand.
    // (No shared helper because the state these touch — strides, bias_idx,
    // Lq — would still need to be re-derived per call; cleaner to inline.)
    const auto strides = pipe_.unet().layer_strides();
    const int num_blocks = pipe_.unet().num_xattn_blocks();
    const int bias_idx = pick_bias_block(strides);
    const int bias_stride = strides[static_cast<std::size_t>(bias_idx)];
    const int Lq = (outer.H_lat / bias_stride) * (outer.W_lat / bias_stride);
    const int Lk = pipe_.unet().context_length();

    std::vector<bt::GpuTensor> bias_patterns(
        static_cast<std::size_t>(cfg_.branching_factor));
    for (int a = 0; a < cfg_.branching_factor; ++a) {
        std::mt19937_64 rng(cfg_.seed ^ (0xA77B1A5ULL * (static_cast<std::uint64_t>(a) + 1)));
        std::normal_distribution<float> nrm(0.0f, cfg_.bias_magnitude);
        std::vector<float> host(static_cast<std::size_t>(Lq) * Lk);
        for (auto& v : host) v = nrm(rng);
        bt::upload(host.data(), Lq, Lk, bias_patterns[static_cast<std::size_t>(a)]);
    }

    std::vector<const bt::GpuTensor*> bias_vec(
        static_cast<std::size_t>(num_blocks), nullptr);

    std::vector<RolloutResult> results;
    results.reserve(static_cast<std::size_t>(cfg_.branching_factor));

    for (int a = 0; a < cfg_.branching_factor; ++a) {
        bias_vec[static_cast<std::size_t>(bias_idx)] =
            &bias_patterns[static_cast<std::size_t>(a)];
        pipeline::PipelineState s = outer.clone();
        while (s.step_index < s.n_steps) {
            pipe_.step_once(s, opts, /*trace_out=*/nullptr, &bias_vec);
        }
        RolloutResult r;
        r.action = a;
        r.image  = pipe_.decode(s);
        r.score  = scorer_(r.image, opts.height, opts.width);
        results.push_back(std::move(r));
    }
    return results;
}

}  // namespace brodiffusion::sd_mcts
