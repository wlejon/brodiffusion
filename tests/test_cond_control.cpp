// CondControl stack-budget unit test.
//
// A stack of axes injects their SUM, and past a model-dependent length the
// injection outweighs the prompt. set_budget() caps that length in alpha units;
// over budget, apply() scales every active axis by ONE common factor.
//
// Verified here:
//   1. active_norm() is the injection's length in alpha units (one axis at
//      weight a -> a; two orthogonal axes at a, b -> hypot(a, b)).
//   2. Uncapped (budget 0), apply() adds the plain weighted sum.
//   3. Over budget, the applied vector keeps the stack's DIRECTION exactly and
//      has length budget * scale — the mix survives, the overdrive is shed.
//   4. Under budget, the budget is inert (bit-identical to uncapped).
//   5. Row discipline is unchanged by the clamp (row_start/row_end respected).

#include "brodiffusion/cond_control.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace bt = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

static bool near(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol * std::max(1.0f, std::fabs(b));
}

int main() {
    try { bt::init(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    constexpr int D = 8, ROWS = 4;
    constexpr float SCALE = 10.0f;

    // Two orthogonal unit axes in an 8-dim "encoder" space.
    std::vector<float> e0(D, 0.0f), e1(D, 0.0f);
    e0[0] = 1.0f;
    e1[1] = 1.0f;

    brodiffusion::CondControl cc;
    cc.set_vector("a", 0.0f, e0, SCALE);
    cc.set_vector("b", 0.0f, e1, SCALE);

    // 1. active_norm() in alpha units.
    CHECK(cc.active_norm() == 0.0f);              // nothing active
    cc.set("a", 3.0f);
    CHECK(near(cc.active_norm(), 3.0f));
    cc.set("b", 4.0f);
    CHECK(near(cc.active_norm(), 5.0f));          // hypot(3, 4)

    auto apply_to_zeros = [&](int row_start, int row_end) {
        std::vector<float> host(ROWS * D, 0.0f);
        bt::Tensor emb = bt::Tensor::from_host_on(bt::Device::CPU, host.data(), ROWS, D);
        cc.apply(emb, row_end, row_start);
        return emb.to_host_vector();
    };

    // 2. Uncapped: the plain weighted sum, on every row (row_start = 0).
    CHECK(cc.budget() == 0.0f);
    auto plain = apply_to_zeros(0, -1);
    for (int r = 0; r < ROWS; ++r) {
        CHECK(near(plain[r * D + 0], 3.0f * SCALE));
        CHECK(near(plain[r * D + 1], 4.0f * SCALE));
    }

    // 3. Over budget: same direction, length held to budget * scale.
    cc.set_budget(2.5f);                          // half of the stack's 5.0
    CHECK(near(cc.active_norm(), 5.0f));          // the STACK is unchanged...
    auto capped = apply_to_zeros(0, -1);          // ...only what apply() adds is
    const float ax = capped[0], ay = capped[1];
    CHECK(near(std::hypot(ax, ay), 2.5f * SCALE));
    CHECK(near(ax / ay, 3.0f / 4.0f));            // mix preserved exactly
    CHECK(near(ax, 0.5f * plain[0]));             // one common factor (0.5)
    CHECK(near(ay, 0.5f * plain[1]));

    // 4. Under budget: inert.
    cc.set_budget(50.0f);
    auto roomy = apply_to_zeros(0, -1);
    for (std::size_t i = 0; i < roomy.size(); ++i) CHECK(roomy[i] == plain[i]);
    cc.set_budget(0.0f);

    // 5. Row discipline survives the clamp: rows outside [row_start, row_end)
    //    stay untouched.
    cc.set_budget(1.0f);
    auto rows = apply_to_zeros(1, 3);
    for (int j = 0; j < D; ++j) {
        CHECK(rows[0 * D + j] == 0.0f);           // BOS row skipped
        CHECK(rows[3 * D + j] == 0.0f);           // past row_end
    }
    CHECK(near(std::hypot(rows[1 * D + 0], rows[1 * D + 1]), 1.0f * SCALE));
    CHECK(near(rows[2 * D + 0], rows[1 * D + 0]));

    if (g_failures == 0) std::printf("cond_control budget: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
