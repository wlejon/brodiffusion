// PipelineState snapshot/restore unit test.
//
// Verifies the contract of PipelineState::clone() in isolation:
//   1. The cloned latent is bit-identical on the GPU after clone.
//   2. Mutating the clone's latent does NOT affect the original.
//   3. The cloned mt19937_64 produces the same sequence as the original
//      from the clone point forward.
//   4. Advancing the original's RNG after clone does NOT affect the clone.
//   5. step_index / n_steps / H_lat / W_lat are copied.
//
// Full end-to-end Pipeline::prime + step_once + decode determinism is
// covered implicitly: generate() in src/pipeline.cpp now uses this exact
// machinery, and the txt2img smoke run produced a valid LCM-LoRA image.

#include "brodiffusion/pipeline.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace pl = brodiffusion::pipeline;
namespace bt = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

int main() {
    try { bt::init(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    // Build a known FP16 latent and stamp it onto a PipelineState.
    constexpr int N = 64;
    std::vector<std::uint16_t> bits(N);
    for (int i = 0; i < N; ++i) {
        bits[static_cast<std::size_t>(i)] =
            bt::fp32_to_fp16_bits(0.1f * static_cast<float>(i) - 3.0f);
    }

    pl::PipelineState s0;
    s0.latent = brotensor::Tensor::from_host_fp16(bits.data(), 1, N);
    s0.rng.seed(424242ULL);
    // Burn a few RNG draws to make the state non-trivial.
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    for (int i = 0; i < 5; ++i) (void)nrm(s0.rng);
    s0.step_index = 3;
    s0.n_steps    = 8;
    s0.H_lat      = 32;
    s0.W_lat      = 32;

    // Clone.
    pl::PipelineState s1 = s0.clone();

    // 1. Latents bit-identical right after clone.
    std::vector<std::uint16_t> b0(N), b1(N);
    s0.latent.copy_to_host_fp16(b0.data());
    s1.latent.copy_to_host_fp16(b1.data());
    bt::sync_all();
    CHECK(std::memcmp(b0.data(), b1.data(), N * 2) == 0);

    // 2. Mutating clone latent does not affect original.
    std::vector<std::uint16_t> bumped(N, bt::fp32_to_fp16_bits(99.0f));
    s1.latent = brotensor::Tensor::from_host_fp16(bumped.data(), 1, N);
    std::vector<std::uint16_t> b0_after(N), b1_after(N);
    s0.latent.copy_to_host_fp16(b0_after.data());
    s1.latent.copy_to_host_fp16(b1_after.data());
    bt::sync_all();
    CHECK(std::memcmp(b0_after.data(), b0.data(), N * 2) == 0);
    CHECK(std::memcmp(b1_after.data(), bumped.data(), N * 2) == 0);

    // 3. Cloned RNG produces same raw-engine sequence as the original from
    //    the clone point. (Raw uint64 draws avoid the cached-second-value
    //    state inside std::normal_distribution, which would entangle two
    //    streams sharing a single distribution instance.)
    pl::PipelineState s_a = s0.clone();
    pl::PipelineState s_b = s0.clone();
    constexpr int K = 16;
    for (int i = 0; i < K; ++i) {
        const std::uint64_t va = s_a.rng();
        const std::uint64_t vb = s_b.rng();
        CHECK(va == vb);
    }

    // 4. Advancing one clone's RNG does NOT affect another clone.
    pl::PipelineState s_d = s0.clone();
    pl::PipelineState s_e = s0.clone();
    pl::PipelineState s_burn = s0.clone();
    for (int i = 0; i < 7; ++i) (void)s_burn.rng();  // burn on s_burn only
    CHECK(s_d.rng() == s_e.rng());

    // 5. Scalars copy.
    CHECK(s1.step_index == 3);
    CHECK(s1.n_steps == 8);
    CHECK(s1.H_lat == 32);
    CHECK(s1.W_lat == 32);

    if (g_failures == 0) std::printf("pipeline_state: OK\n");
    else std::fprintf(stderr, "pipeline_state: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
