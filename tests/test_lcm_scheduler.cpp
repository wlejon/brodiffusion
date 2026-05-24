// LCM scheduler smoke test.
//
// Verifies SD1.5-LCM defaults: timestep schedule matches the diffusers
// reference for 4 and 8 inference steps, init_noise_sigma is 1, and a 4-step
// step() loop with fixed-seed noise produces a finite FP16 result that's
// deterministic across runs.

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/lcm_scheduler.h"

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
    sc::LCM lcm;

    // Reference values produced by diffusers' LCMScheduler with default config:
    //   k = 1000/50 = 20; lcm_origin = [19, 39, ..., 999]
    //   indices = floor(linspace(0, 50, N, endpoint=False))
    //   set_timesteps(4): indices=[0,12,25,37] -> [999, 759, 499, 259]
    //   set_timesteps(8): indices=[0,6,12,18,25,31,37,43]
    //                        -> [999, 879, 759, 639, 499, 379, 259, 139]
    lcm.set_timesteps(4);
    const auto& ts4 = lcm.timesteps();
    CHECK(ts4.size() == 4);
    CHECK(ts4[0] == 999);
    CHECK(ts4[1] == 759);
    CHECK(ts4[2] == 499);
    CHECK(ts4[3] == 259);

    lcm.set_timesteps(8);
    const auto& ts8 = lcm.timesteps();
    CHECK(ts8.size() == 8);
    const int ref8[8] = {999, 879, 759, 639, 499, 379, 259, 139};
    for (std::size_t i = 0; i < 8; ++i) CHECK(ts8[i] == ref8[i]);
    for (std::size_t i = 1; i < ts8.size(); ++i) CHECK(ts8[i] < ts8[i - 1]);

    CHECK(lcm.init_noise_sigma() == 1.0f);

    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }
    const int C = 4, H = 8, W = 8;
    const int n = C * H * W;

    auto run_loop = [&](std::vector<float>& out_vals) {
        std::mt19937 rng(0xCAFE);
        std::normal_distribution<float> nrm(0.0f, 1.0f);

        std::vector<float> latent_vals(n), pred_vals(n);
        for (int i = 0; i < n; ++i) {
            latent_vals[i] = nrm(rng);
            pred_vals[i]   = 0.1f * nrm(rng);
        }

        bt::Tensor sample, noise_pred, noise, scratch;
        sample     = bdtest::bd_upload(latent_vals, 1, n);
        noise_pred = bdtest::bd_upload(pred_vals,   1, n);

        sc::LCM s;
        s.set_timesteps(4);
        std::vector<float> noise_vals(n);
        for (int i = 0; i < s.num_inference_steps(); ++i) {
            for (int k = 0; k < n; ++k) {
                noise_vals[k] = nrm(rng);
            }
            noise = bdtest::bd_upload(noise_vals, 1, n);
            s.step(noise_pred, i, sample, noise, scratch);
        }
        out_vals = bdtest::bd_download(sample);
    };

    std::vector<float> out_a, out_b;
    run_loop(out_a);
    run_loop(out_b);

    // Deterministic across runs (same RNG seed, same sequence).
    CHECK(out_a == out_b);

    // No Inf/NaN in the result.
    int n_bad = 0;
    for (float v : out_a) {
        if (!bdtest::bd_finite(v)) ++n_bad;
    }
    CHECK(n_bad == 0);

    // ── add_noise checks ──────────────────────────────────────────────────
    //
    // LCM uses the same alpha_cumprod schedule as DDIM for forward noising.
    {
        sc::LCM ln;
        ln.set_timesteps(4);
        const int step_index = 0;
        const int t = ln.timesteps()[step_index];   // = 999
        // Reconstruct alpha_cumprod host-side.
        sc::LCMConfig cfg;
        std::vector<float> alphas(static_cast<std::size_t>(cfg.num_train_timesteps));
        {
            double cumprod = 1.0;
            const float s0 = std::sqrt(cfg.beta_start);
            const float s1 = std::sqrt(cfg.beta_end);
            const int N = cfg.num_train_timesteps;
            for (int i = 0; i < N; ++i) {
                const float u = static_cast<float>(i) / static_cast<float>(N - 1);
                const float s = s0 + (s1 - s0) * u;
                const float beta = s * s;
                cumprod *= static_cast<double>(1.0f - beta);
                alphas[static_cast<std::size_t>(i)] = static_cast<float>(cumprod);
            }
        }
        const int t_clamp = (t >= cfg.num_train_timesteps) ? cfg.num_train_timesteps - 1 : t;
        const float sqrt_a = std::sqrt(alphas[static_cast<std::size_t>(t_clamp)]);

        std::vector<float> x0_vals(n), z_vals(n, 0.0f);
        for (int i = 0; i < n; ++i) x0_vals[i] = static_cast<float>((i % 7) - 3) * 0.1f;
        bt::Tensor x0 = bdtest::bd_upload(x0_vals, 1, n);
        bt::Tensor zn = bdtest::bd_upload(z_vals,  1, n);
        bt::Tensor sample, scratch;
        ln.add_noise(x0, zn, step_index, sample, scratch);
        std::vector<float> got = bdtest::bd_download(sample);
        const float tol = 1e-3f;
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            float expect = sqrt_a * x0_vals[i];
            if (std::fabs(got[static_cast<std::size_t>(i)] - expect) > tol) ++bad;
        }
        CHECK(bad == 0);

        std::mt19937 rng2(0xBEEF);
        std::normal_distribution<float> nrm2(0.0f, 1.0f);
        std::vector<float> r_x0(n), r_n(n);
        for (int i = 0; i < n; ++i) { r_x0[i] = nrm2(rng2); r_n[i] = nrm2(rng2); }
        bt::Tensor rx = bdtest::bd_upload(r_x0, 1, n);
        bt::Tensor rn = bdtest::bd_upload(r_n,  1, n);
        bt::Tensor s2, sc2;
        ln.add_noise(rx, rn, 2, s2, sc2);
        std::vector<float> got2 = bdtest::bd_download(s2);
        CHECK(s2.rows == 1 && s2.cols == n);
        int bad2 = 0;
        for (float v : got2) if (!bdtest::bd_finite(v)) ++bad2;
        CHECK(bad2 == 0);
    }

    return g_failures == 0 ? 0 : 1;
}
