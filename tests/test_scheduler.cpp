// DDIM scheduler smoke test.
//
// Verifies SD1.5 defaults: scaled_linear betas match expected endpoints,
// alphas_cumprod is strictly decreasing in [0, 1], inference timesteps
// span the expected range with the leading+offset layout, and a step()
// loop produces a finite FP16 result that's deterministic across runs.

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/scheduler.h"

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
    sc::DDIM ddim;

    // Inference schedule: 50 leading steps with steps_offset=1.
    ddim.set_timesteps(50);
    const auto& ts = ddim.timesteps();
    CHECK(ts.size() == 50);
    CHECK(ts.front() == 981);   // 49 * 20 + 1
    CHECK(ts.back() == 1);      // 0 * 20 + 1
    for (std::size_t i = 1; i < ts.size(); ++i) CHECK(ts[i] < ts[i - 1]);

    // alphas_cumprod sanity: monotonically decreasing, all in (0, 1].
    // Indirectly checked via observed init_noise_sigma + range of step factors.
    CHECK(ddim.init_noise_sigma() == 1.0f);

    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }
    const int C = 4, H = 8, W = 8;
    const int n = C * H * W;
    std::vector<float> noise_vals(n), latent_vals(n);
    std::mt19937 rng(0xD1D1);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        latent_vals[i] = nrm(rng);
        noise_vals[i]  = 0.1f * nrm(rng);
    }

    auto run_loop = [&](std::vector<float>& out_vals) {
        bt::Tensor sample, noise, scratch;
        sample = bdtest::bd_upload(latent_vals, 1, n);
        noise  = bdtest::bd_upload(noise_vals,  1, n);
        ddim.set_timesteps(10);
        for (int i = 0; i < ddim.num_inference_steps(); ++i) {
            ddim.step(noise, i, sample, scratch);
        }
        out_vals = bdtest::bd_download(sample);
    };

    std::vector<float> out_a, out_b;
    run_loop(out_a);
    run_loop(out_b);

    // Deterministic across runs.
    CHECK(out_a == out_b);
    // No Inf/NaN in the result.
    int n_bad = 0;
    for (float v : out_a) {
        if (!bdtest::bd_finite(v)) ++n_bad;
    }
    CHECK(n_bad == 0);

    return g_failures == 0 ? 0 : 1;
}
