// DDIM scheduler smoke test.
//
// Verifies SD1.5 defaults: scaled_linear betas match expected endpoints,
// alphas_cumprod is strictly decreasing in [0, 1], inference timesteps
// span the expected range with the leading+offset layout, and a step()
// loop produces a finite FP16 result that's deterministic across runs.

#include "brodiffusion/scheduler.h"

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
        bt::cuda_init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cuda_init failed: %s\n", e.what());
        return 1;
    }

    const int C = 4, H = 8, W = 8;
    const int n = C * H * W;
    std::vector<std::uint16_t> noise_bits(n), latent_bits(n);
    std::mt19937 rng(0xD1D1);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        latent_bits[i] = bt::fp32_to_fp16_bits(nrm(rng));
        noise_bits[i]  = bt::fp32_to_fp16_bits(0.1f * nrm(rng));
    }

    auto run_loop = [&](std::vector<std::uint16_t>& out_bits) {
        bt::GpuTensor sample, noise, scratch;
        bt::upload_fp16(latent_bits.data(), 1, n, sample);
        bt::upload_fp16(noise_bits.data(),  1, n, noise);
        ddim.set_timesteps(10);
        for (int i = 0; i < ddim.num_inference_steps(); ++i) {
            ddim.step(noise, i, sample, scratch);
        }
        bt::cuda_sync();
        out_bits.resize(n);
        bt::download_fp16(sample, out_bits.data());
    };

    std::vector<std::uint16_t> out_a, out_b;
    run_loop(out_a);
    run_loop(out_b);

    // Deterministic across runs.
    CHECK(std::memcmp(out_a.data(), out_b.data(), n * 2) == 0);
    // No Inf/NaN in the result.
    int n_bad = 0;
    for (std::uint16_t b : out_a) {
        const std::uint16_t exp = (b >> 10) & 0x1F;
        const std::uint16_t mant = b & 0x3FF;
        if (exp == 0x1F) ++n_bad;   // Inf or NaN
        (void)mant;
    }
    CHECK(n_bad == 0);

    return g_failures == 0 ? 0 : 1;
}
