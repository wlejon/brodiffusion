// CPU↔Metal parity tests for brodiffusion's fused ops.
//
// brodiffusion ships SD1.5-tuned fused Metal kernels (src/fused_resblock.mm,
// src/fused_transformer.mm). The public brodiffusion::fused_* entry points
// (src/fused.cpp) dispatch on tensor residency: a Metal-resident input runs
// the fused Metal kernel; a CPU-resident input runs the FP32 fallback
// composed from brotensor's CPU ops. Both paths claim identical math.
//
// This test exercises both paths from a single Metal-enabled binary: every
// op is run once with CPU FP32 tensors and once with Metal FP16 tensors over
// numerically identical inputs (host values are canonicalised through FP16
// first, so the only difference under test is CPU-FP32 vs Metal-FP16 compute
// precision). Outputs are compared within an FP16-appropriate tolerance.
//
// Covers the FP16 ops that have a CPU fallback:
//   * fused_resblock_forward   (no-skip and 1x1-skip)
//   * fused_linear_geglu
//   * add_inplace_vec
//   * add_inplace_row_bias
// The W8A16 (INT8 weight-only) overloads are GPU-only by design — the CPU
// fallback throws — so there is no CPU reference to compare against.
//
// Skips cleanly (exit 0) when no Metal backend is available, so the same
// source builds and passes in the CPU-only configuration.

#include "brodiffusion/fused_resblock.h"
#include "brodiffusion/fused_transformer.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace bt = brotensor;
namespace bd = brodiffusion;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

namespace {

// Deterministic host values in [-scale, scale] + bias, each canonicalised to
// an FP16-representable value so the CPU-FP32 and Metal-FP16 paths consume
// byte-identical numeric inputs — the test then isolates kernel compute
// precision rather than input-quantisation noise.
std::vector<float> gen(std::size_t n, std::uint64_t seed, float scale,
                       float bias = 0.0f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float v = d(rng) * scale + bias;
        out[i] = bt::fp16_bits_to_fp32(bt::fp32_to_fp16_bits(v));
    }
    return out;
}

bt::Tensor cpu_t(const std::vector<float>& v, int r, int c) {
    return bt::Tensor::from_host_on(bt::Device::CPU, v.data(), r, c);
}

bt::Tensor metal_t(const std::vector<float>& v, int r, int c) {
    std::vector<std::uint16_t> bits(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        bits[i] = bt::fp32_to_fp16_bits(v[i]);
    }
    return bt::Tensor::from_host_fp16_on(bt::Device::Metal, bits.data(), r, c);
}

// Max absolute error and reference magnitude across two host-side results.
struct Cmp {
    float max_abs_err = 0.0f;
    float max_abs_ref = 0.0f;
    int   nonfinite   = 0;
};

Cmp compare(const std::vector<float>& cpu, const std::vector<float>& gpu) {
    Cmp c;
    const std::size_t n = std::min(cpu.size(), gpu.size());
    if (cpu.size() != gpu.size()) { c.nonfinite = 1; return c; }
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(cpu[i]) || !std::isfinite(gpu[i])) {
            ++c.nonfinite;
            continue;
        }
        c.max_abs_err = std::max(c.max_abs_err, std::fabs(cpu[i] - gpu[i]));
        c.max_abs_ref = std::max(c.max_abs_ref, std::fabs(cpu[i]));
    }
    return c;
}

// Assert a parity result: finite, non-empty, and within tol_rel * ref + tol_abs.
void expect_parity(const char* label, const Cmp& c, std::size_t n,
                   float tol_rel, float tol_abs) {
    const float bound = tol_rel * c.max_abs_ref + tol_abs;
    std::printf("  %-26s err=%.5f ref=%.4f bound=%.5f%s\n",
                label, c.max_abs_err, c.max_abs_ref, bound,
                c.max_abs_err <= bound ? "" : "  <-- OVER");
    CHECK(n > 0);
    CHECK(c.nonfinite == 0);
    CHECK(c.max_abs_ref > 0.0f);
    CHECK(c.max_abs_err <= bound);
}

// ─── fused_resblock_forward ────────────────────────────────────────────────

void test_resblock(int C_in, int C_out, int H, int W, int num_groups,
                   const char* label) {
    const int spatial = H * W;
    const bool need_skip = (C_in != C_out);

    // Host fixtures. GN gammas sit near 1, betas near 0; conv weights small so
    // the synthetic resblock output magnitude stays O(1).
    const auto x   = gen(static_cast<std::size_t>(C_in) * spatial, 1, 1.0f);
    const auto g1  = gen(static_cast<std::size_t>(C_in),  2, 0.2f, 1.0f);
    const auto bt1 = gen(static_cast<std::size_t>(C_in),  3, 0.2f);
    const auto w1  = gen(static_cast<std::size_t>(C_out) * C_in * 9, 4, 0.1f);
    const auto b1  = gen(static_cast<std::size_t>(C_out), 5, 0.1f);
    const auto sh  = gen(static_cast<std::size_t>(C_out), 6, 0.2f);
    const auto g2  = gen(static_cast<std::size_t>(C_out), 7, 0.2f, 1.0f);
    const auto bt2 = gen(static_cast<std::size_t>(C_out), 8, 0.2f);
    const auto w2  = gen(static_cast<std::size_t>(C_out) * C_out * 9, 9, 0.1f);
    const auto b2  = gen(static_cast<std::size_t>(C_out), 10, 0.1f);
    const auto wsk = gen(static_cast<std::size_t>(C_out) * C_in, 11, 0.15f);
    const auto bsk = gen(static_cast<std::size_t>(C_out), 12, 0.1f);

    const float eps = 1e-5f;

    auto run = [&](bt::Device dev) {
        auto mk = [&](const std::vector<float>& v, int r, int c) {
            return dev == bt::Device::CPU ? cpu_t(v, r, c) : metal_t(v, r, c);
        };
        bt::Tensor X   = mk(x,   1, C_in * spatial);
        bt::Tensor G1  = mk(g1,  C_in, 1);
        bt::Tensor B1n = mk(bt1, C_in, 1);
        bt::Tensor W1  = mk(w1,  C_out, C_in * 9);
        bt::Tensor B1  = mk(b1,  C_out, 1);
        bt::Tensor SH  = mk(sh,  C_out, 1);
        bt::Tensor G2  = mk(g2,  C_out, 1);
        bt::Tensor B2n = mk(bt2, C_out, 1);
        bt::Tensor W2  = mk(w2,  C_out, C_out * 9);
        bt::Tensor B2  = mk(b2,  C_out, 1);
        bt::Tensor WSK, BSK;
        if (need_skip) {
            WSK = mk(wsk, C_out, C_in);
            BSK = mk(bsk, C_out, 1);
        }
        bt::Tensor Y;
        bd::fused_resblock_forward(
            X, G1, B1n, W1, B1, SH, G2, B2n, W2, B2,
            need_skip ? &WSK : nullptr, need_skip ? &BSK : nullptr,
            C_in, C_out, H, W, num_groups, eps, Y);
        return bdtest::bd_download(Y);
    };

    const auto cpu = run(bt::Device::CPU);
    const auto gpu = run(bt::Device::Metal);
    // Observed err ~0.001 (FP16 storage + intermediate rounding through GN,
    // SiLU and two convs); bound leaves ~20x headroom for cross-GPU ULP drift
    // in the transcendentals while still catching a genuinely broken kernel.
    expect_parity(label, compare(cpu, gpu), gpu.size(),
                  /*tol_rel=*/0.01f, /*tol_abs=*/0.008f);
}

// ─── fused_linear_geglu ────────────────────────────────────────────────────

void test_geglu(int B, int D_in, int D_out, const char* label) {
    const int two_D = 2 * D_out;
    const auto x = gen(static_cast<std::size_t>(B) * D_in, 21, 1.0f);
    const auto w = gen(static_cast<std::size_t>(two_D) * D_in, 22, 0.1f);
    const auto b = gen(static_cast<std::size_t>(two_D), 23, 0.2f);

    auto run = [&](bt::Device dev) {
        auto mk = [&](const std::vector<float>& v, int r, int c) {
            return dev == bt::Device::CPU ? cpu_t(v, r, c) : metal_t(v, r, c);
        };
        bt::Tensor X = mk(x, B, D_in);
        bt::Tensor Wt = mk(w, two_D, D_in);
        bt::Tensor Bt = mk(b, two_D, 1);
        bt::Tensor Y;
        bd::fused_linear_geglu(X, Wt, Bt, Y);
        return bdtest::bd_download(Y);
    };

    const auto cpu = run(bt::Device::CPU);
    const auto gpu = run(bt::Device::Metal);
    // Observed err ~0.0006 (FP16 GEMM accumulator downcast + erf approx).
    expect_parity(label, compare(cpu, gpu), gpu.size(),
                  /*tol_rel=*/0.01f, /*tol_abs=*/0.008f);
}

// ─── add_inplace_vec ───────────────────────────────────────────────────────

void test_add_vec(int rows, int cols, const char* label) {
    const auto y0 = gen(static_cast<std::size_t>(rows) * cols, 31, 1.0f);
    const auto x  = gen(static_cast<std::size_t>(rows) * cols, 32, 1.0f);

    auto run = [&](bt::Device dev) {
        bt::Tensor Y = dev == bt::Device::CPU ? cpu_t(y0, rows, cols)
                                              : metal_t(y0, rows, cols);
        bt::Tensor X = dev == bt::Device::CPU ? cpu_t(x, rows, cols)
                                              : metal_t(x, rows, cols);
        bd::add_inplace_vec(Y, X);
        return bdtest::bd_download(Y);
    };

    const auto cpu = run(bt::Device::CPU);
    const auto gpu = run(bt::Device::Metal);
    // Pure FP16 add: observed err ~0.0005 is one ULP at this magnitude.
    expect_parity(label, compare(cpu, gpu), gpu.size(),
                  /*tol_rel=*/0.005f, /*tol_abs=*/0.003f);
}

// ─── add_inplace_row_bias ──────────────────────────────────────────────────

void test_add_row_bias(int rows, int cols, const char* label) {
    const auto y0   = gen(static_cast<std::size_t>(rows) * cols, 41, 1.0f);
    const auto bias = gen(static_cast<std::size_t>(cols), 42, 1.0f);

    auto run = [&](bt::Device dev) {
        bt::Tensor Y = dev == bt::Device::CPU ? cpu_t(y0, rows, cols)
                                              : metal_t(y0, rows, cols);
        bt::Tensor Bs = dev == bt::Device::CPU ? cpu_t(bias, cols, 1)
                                               : metal_t(bias, cols, 1);
        bd::add_inplace_row_bias(Y, Bs);
        return bdtest::bd_download(Y);
    };

    const auto cpu = run(bt::Device::CPU);
    const auto gpu = run(bt::Device::Metal);
    // Pure FP16 add: observed err ~0.0005 is one ULP at this magnitude.
    expect_parity(label, compare(cpu, gpu), gpu.size(),
                  /*tol_rel=*/0.005f, /*tol_abs=*/0.003f);
}

}  // namespace

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    if (!bt::is_available(bt::Device::Metal)) {
        std::printf("fused_parity: no Metal backend — skipping\n");
        return 0;
    }

    std::printf("fused_parity: CPU vs Metal\n");

    // fused_resblock_forward — single output tile and multi-tile, with and
    // without the 1x1 skip conv.
    test_resblock(8,  8,  8,  8, 2, "resblock 8ch 8x8 no-skip");
    test_resblock(4,  8,  8,  8, 2, "resblock 4->8ch 8x8 skip");
    test_resblock(16, 16, 16, 16, 4, "resblock 16ch 16x16 no-skip");

    // fused_linear_geglu — single tile and multi-tile in both grid dims.
    test_geglu(40,  64,  48, "geglu B40 64->48");
    test_geglu(128, 128, 96, "geglu B128 128->96");

    // Elementwise epilogue helpers.
    test_add_vec(10, 37, "add_inplace_vec 10x37");
    test_add_vec(64, 64, "add_inplace_vec 64x64");
    test_add_row_bias(13, 29, "add_inplace_row_bias 13x29");
    test_add_row_bias(48, 80, "add_inplace_row_bias 48x80");

    if (g_failures == 0) std::printf("fused_parity: OK\n");
    else std::fprintf(stderr, "fused_parity: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
