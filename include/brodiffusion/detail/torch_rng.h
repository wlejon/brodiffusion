#pragma once

// PyTorch-compatible CPU random noise.
//
// Reproduces torch.randn(..., dtype=torch.float32) as filled by a CPU
// Generator: at::mt19937 driving the scalar `normal_fill` Box-Muller
// transform from ATen's DistributionTemplates.h. This lets brodiffusion's
// `--seed N` produce the *same* initial latent noise as a PyTorch reference
// run seeded with torch.Generator().manual_seed(N) — so the two pipelines
// can be compared with the RNG eliminated as a source of difference.
//
// Why std::mt19937 works: at::mt19937 is a textbook MT19937 — same seeding
// recurrence (1812433253 multiplier) and same tempering as the C++ standard
// engine — so its 32-bit integer stream is bit-identical to std::mt19937.
// We drive std::mt19937 directly and only reimplement the float transforms.
//
// Bit-exactness vs PyTorch: exact against a PyTorch run that uses the scalar
// normal kernel (ATEN_CPU_CAPABILITY=default). Against an AVX2 build the
// vectorized log/sincos approximations differ in the last 1-2 ulps — far
// below FP16 noise and irrelevant for image comparison.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace brodiffusion::detail {

// torch.randn(n) for a contiguous float32 tensor, n >= 16 (every SD/Flux
// latent is far larger). Row-major fill order matches torch, so the result
// drops straight into an NCHW latent buffer.
inline std::vector<float> torch_randn_f32(std::uint64_t seed, std::size_t n) {
    if (n < 16) {
        throw std::runtime_error(
            "torch_randn_f32: n must be >= 16 (torch's vectorized float path)");
    }
    std::vector<float> data(n);

    // at::mt19937 == std::mt19937: the low 32 bits of the seed drive the
    // engine (torch does `state[0] = seed & 0xffffffff`).
    std::mt19937 eng(static_cast<std::uint32_t>(seed));

    // torch's uniform_real_distribution<float>: 24 random bits / 2^24 -> [0,1).
    auto next_uniform = [&eng]() -> float {
        return static_cast<float>(eng() & 0xFFFFFFu) * (1.0f / 16777216.0f);
    };
    for (std::size_t i = 0; i < n; ++i) data[i] = next_uniform();

    // normal_fill_16: Box-Muller over 8 (u1,u2) pairs, in place. Mirrors
    // ATen's normal_fill_16<float> exactly (mean=0, std=1).
    auto fill16 = [](float* d) {
        constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
        for (int j = 0; j < 8; ++j) {
            const float u1 = 1.0f - d[j];          // (0,1] -> finite log
            const float u2 = d[j + 8];
            const float radius = std::sqrt(-2.0f * std::log(u1));
            // torch forms theta in double (c10::pi<double>), stores it float.
            const float theta =
                static_cast<float>(kTwoPi * static_cast<double>(u2));
            d[j]     = radius * std::cos(theta);
            d[j + 8] = radius * std::sin(theta);
        }
    };
    for (std::size_t i = 0; i + 16 <= n; i += 16) fill16(data.data() + i);

    // Tail: torch recomputes the final 16 elements with fresh draws when the
    // length is not a multiple of 16 (consuming 16 extra RNG outputs).
    if (n % 16 != 0) {
        float* tail = data.data() + n - 16;
        for (int j = 0; j < 16; ++j) tail[j] = next_uniform();
        fill16(tail);
    }
    return data;
}

}  // namespace brodiffusion::detail
