// PyTorch-compatible noise test.
//
// Checks brodiffusion::detail::torch_randn_f32 against values captured from
// torch.randn(n, generator=torch.Generator().manual_seed(seed), dtype=float32)
// run with ATEN_CPU_CAPABILITY=default (the scalar normal kernel). Covers a
// single block (n=16), two blocks (n=32), and the non-multiple-of-16 tail
// recompute path (n=20). A matching MT19937 stream is the only way these
// land within tolerance — any RNG error perturbs every value by O(1).

#include "brodiffusion/detail/torch_rng.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// Compare against a baked torch.randn reference. Tolerance is loose enough to
// absorb libm ulp differences between the brodiffusion and PyTorch builds but
// far tighter than any algorithmic divergence (which is O(1)).
static void check_against(std::uint64_t seed, const std::vector<float>& ref) {
    const auto got = brodiffusion::detail::torch_randn_f32(seed, ref.size());
    CHECK(got.size() == ref.size());
    double max_abs = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) -
                                   static_cast<double>(ref[i]));
        if (d > max_abs) max_abs = d;
    }
    std::printf("seed=%llu n=%zu  max|diff|=%.3e\n",
                static_cast<unsigned long long>(seed), ref.size(), max_abs);
    CHECK(max_abs < 1e-4);
}

int main() {
    // torch.randn — single 16-element block.
    check_against(0, {
        -1.125839829e+00f, -1.152360201e+00f, -2.505785823e-01f,
        -4.338788390e-01f,  8.487103581e-01f,  6.920092106e-01f,
        -3.160127699e-01f, -2.115219593e+00f,  3.222749233e-01f,
        -1.263334751e+00f,  3.499831855e-01f,  3.081339002e-01f,
         1.198415086e-01f,  1.237657905e+00f,  1.116777182e+00f,
        -2.472776473e-01f });

    // n=20 — exercises the tail: elements [4,20) are recomputed with fresh
    // draws, so only [0,4) coincide with the n=16 case above.
    check_against(0, {
        -1.125839829e+00f, -1.152360201e+00f, -2.505785823e-01f,
        -4.338788390e-01f,  5.988394618e-01f, -1.555095077e+00f,
        -3.413603008e-01f,  1.853006124e+00f,  4.680964053e-01f,
        -1.577124447e-01f,  1.443660140e+00f,  2.660494149e-01f,
         1.389366150e+00f,  1.586334348e+00f,  9.462983608e-01f,
        -8.436768055e-01f,  9.318266511e-01f,  1.259009242e+00f,
         2.004980564e+00f,  5.373689905e-02f });

    check_against(12345, {
        -1.479807734e+00f,  4.873059988e-01f, -3.012793779e+00f,
         4.438551962e-01f,  3.597561121e-01f, -1.234805677e-02f,
         2.185224444e-01f, -1.281468391e+00f,  2.411195993e+00f,
         1.999127984e+00f,  7.847856879e-01f, -1.019471169e+00f,
        -2.105759680e-01f,  6.268384457e-01f,  9.317625761e-01f,
         1.867547333e-01f });

    // n=32 — two full blocks back to back.
    check_against(7, {
        -8.201345205e-01f,  3.956309259e-01f,  8.989079595e-01f,
        -1.388404012e+00f, -1.669960171e-01f,  2.851499617e-01f,
        -6.410915256e-01f, -8.936551213e-01f,  9.265430570e-01f,
        -5.355123878e-01f, -1.159720659e+00f, -4.601567984e-01f,
         7.085390687e-01f,  1.012755394e+00f,  2.303968668e-01f,
         1.090165377e+00f, -1.582664609e+00f, -3.245666921e-01f,
         1.926367283e+00f, -3.300117254e-01f,  1.984440535e-01f,
         7.820726633e-01f,  1.039107680e+00f, -7.245113254e-01f,
        -2.093404233e-01f, -2.153414488e-01f, -1.815729618e+00f,
        -3.452419639e-01f, -2.061477900e+00f,  6.741006970e-01f,
        -1.323345780e+00f, -1.359768510e+00f });

    if (g_failures == 0) std::printf("test_torch_rng: OK\n");
    return g_failures == 0 ? 0 : 1;
}
