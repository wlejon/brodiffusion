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

    return g_failures == 0 ? 0 : 1;
}
