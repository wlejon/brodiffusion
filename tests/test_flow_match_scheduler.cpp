// FlowMatch (rectified-flow Euler) scheduler smoke test.
//
// Verifies Flux / SD3 defaults: the static-shift timestep schedule matches the
// diffusers FlowMatchEulerDiscreteScheduler reference for 4 inference steps,
// init_noise_sigma is 1, the schedule is strictly decreasing, a 4-step step()
// loop with fixed-seed inputs produces a finite result that's deterministic
// across runs, and the dynamic-shifting path yields a sane schedule.

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/flow_match_scheduler.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace sc = brodiffusion::scheduler;
namespace bt = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

int main() {
    // ── 1. Static-shift schedule (default config: shift=3.0, non-dynamic) ──
    //
    // Reference (diffusers FlowMatchEulerDiscreteScheduler, shift=3):
    //   raw sigmas = linspace(1, 0.25, 4) = {1.0, 0.75, 0.5, 0.25}
    //   shifted    = 3s / (1 + 2s)        = {1.0, 0.9, 0.75, 0.5}
    //   timesteps  = sigma * 1000         = {1000, 900, 750, 500}
    sc::FlowMatch fm;
    fm.set_timesteps(4);
    const auto& ts4 = fm.timesteps();
    CHECK(ts4.size() == 4);
    const float ref4[4] = {1000.0f, 900.0f, 750.0f, 500.0f};
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(std::fabs(ts4[i] - ref4[i]) < 0.5f);
    }

    // ── 2. init_noise_sigma ───────────────────────────────────────────────
    CHECK(fm.init_noise_sigma() == 1.0f);

    // ── 3. Strictly decreasing schedule ───────────────────────────────────
    for (std::size_t i = 1; i < ts4.size(); ++i) CHECK(ts4[i] < ts4[i - 1]);

    // ── runtime init for the step() loop ──────────────────────────────────
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }
    // Flux has 16 latent channels.
    const int C = 16, H = 8, W = 8;
    const int n = C * H * W;

    // ── 4. 4-step step() loop: finite + deterministic across runs ──────────
    auto run_loop = [&](std::vector<float>& out_vals) {
        std::mt19937 rng(0xF10A);
        std::normal_distribution<float> nrm(0.0f, 1.0f);

        std::vector<float> sample_vals(n), v_vals(n);
        for (int i = 0; i < n; ++i) {
            sample_vals[i] = nrm(rng);
            v_vals[i]      = 0.1f * nrm(rng);
        }

        bt::Tensor sample, v, scratch;
        sample = bdtest::bd_upload(sample_vals, 1, n);
        v      = bdtest::bd_upload(v_vals,      1, n);

        sc::FlowMatch s;
        s.set_timesteps(4);
        for (int i = 0; i < s.num_inference_steps(); ++i) {
            s.step(v, i, sample, scratch);
        }
        out_vals = bdtest::bd_download(sample);
    };

    std::vector<float> out_a, out_b;
    run_loop(out_a);
    run_loop(out_b);

    // Deterministic across runs (same fixed-seed inputs).
    CHECK(out_a == out_b);

    // No Inf/NaN in the result.
    int n_bad = 0;
    for (float val : out_a) {
        if (!bdtest::bd_finite(val)) ++n_bad;
    }
    CHECK(n_bad == 0);

    // ── 5. Dynamic-shifting schedule (flux-dev) ───────────────────────────
    sc::FlowMatchConfig dyn_cfg;
    dyn_cfg.use_dynamic_shifting = true;
    sc::FlowMatch fm_dyn(dyn_cfg);
    fm_dyn.set_timesteps(8, /*image_seq_len=*/4096);
    const auto& ts_dyn = fm_dyn.timesteps();
    CHECK(ts_dyn.size() == 8);
    // All finite, strictly decreasing.
    for (float val : ts_dyn) CHECK(bdtest::bd_finite(val));
    for (std::size_t i = 1; i < ts_dyn.size(); ++i) {
        CHECK(ts_dyn[i] < ts_dyn[i - 1]);
    }
    // First timestep close to 1000 (raw sigma 1 maps to ~1).
    CHECK(std::fabs(ts_dyn.front() - 1000.0f) < 1.0f);
    // Consistency with the dynamic-shift formula for the first sample:
    //   m  = (max_shift - base_shift) / (max_seq - base_seq)
    //   b  = base_shift - m * base_seq
    //   mu = image_seq_len * m + b
    //   sigma[0] = exp(mu) / (exp(mu) + (1/sigma_raw[0] - 1)), sigma_raw[0]=1
    //            = exp(mu) / (exp(mu) + 0) = 1
    {
        const auto& c = dyn_cfg;
        const float m = (c.max_shift - c.base_shift) /
                        static_cast<float>(c.max_image_seq_len -
                                           c.base_image_seq_len);
        const float b = c.base_shift -
                        m * static_cast<float>(c.base_image_seq_len);
        const float mu = 4096.0f * m + b;
        const float exp_mu = std::exp(mu);
        const float sigma0 = exp_mu / (exp_mu + 0.0f);  // sigma_raw[0] == 1
        const float ref_t0 = sigma0 * 1000.0f;
        CHECK(std::fabs(ts_dyn.front() - ref_t0) < 1.0f);
    }

    return g_failures == 0 ? 0 : 1;
}
