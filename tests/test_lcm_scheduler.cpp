// LCM scheduler smoke test.
//
// Verifies SD1.5-LCM defaults: timestep schedule matches the diffusers
// reference for 4 and 8 inference steps, init_noise_sigma is 1, and a 4-step
// step() loop with fixed-seed noise produces a finite FP16 result that's
// deterministic across runs.

#include "brodiffusion/lcm_scheduler.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

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
        bt::cuda_init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cuda_init failed: %s\n", e.what());
        return 1;
    }

    const int C = 4, H = 8, W = 8;
    const int n = C * H * W;

    auto run_loop = [&](std::vector<std::uint16_t>& out_bits) {
        std::mt19937 rng(0xCAFE);
        std::normal_distribution<float> nrm(0.0f, 1.0f);

        std::vector<std::uint16_t> latent_bits(n), pred_bits(n);
        for (int i = 0; i < n; ++i) {
            latent_bits[i] = bt::fp32_to_fp16_bits(nrm(rng));
            pred_bits[i]   = bt::fp32_to_fp16_bits(0.1f * nrm(rng));
        }

        bt::GpuTensor sample, noise_pred, noise, scratch;
        bt::upload_fp16(latent_bits.data(), 1, n, sample);
        bt::upload_fp16(pred_bits.data(),   1, n, noise_pred);

        sc::LCM s;
        s.set_timesteps(4);
        std::vector<std::uint16_t> noise_bits(n);
        for (int i = 0; i < s.num_inference_steps(); ++i) {
            for (int k = 0; k < n; ++k) {
                noise_bits[k] = bt::fp32_to_fp16_bits(nrm(rng));
            }
            bt::upload_fp16(noise_bits.data(), 1, n, noise);
            s.step(noise_pred, i, sample, noise, scratch);
        }
        bt::cuda_sync();
        out_bits.resize(n);
        bt::download_fp16(sample, out_bits.data());
    };

    std::vector<std::uint16_t> out_a, out_b;
    run_loop(out_a);
    run_loop(out_b);

    // Deterministic across runs (same RNG seed, same sequence).
    CHECK(std::memcmp(out_a.data(), out_b.data(), n * 2) == 0);

    // No Inf/NaN in the result.
    int n_bad = 0;
    for (std::uint16_t b : out_a) {
        const std::uint16_t exp = (b >> 10) & 0x1F;
        if (exp == 0x1F) ++n_bad;
    }
    CHECK(n_bad == 0);

    return g_failures == 0 ? 0 : 1;
}
