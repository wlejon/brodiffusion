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
//   6. load(path, merge) stacks banks of different provenance: axes append, a
//      same-named axis is overwritten (weight reset), a dim mismatch throws.

#include "brodiffusion/cond_control.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
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

// A minimal BCD1 dictionary on disk: magic, n_axes, dim, then per axis
// {name_len, name, scale, dim floats}. Axis k is the unit vector e_k.
static void write_bank(const std::string& path, const std::vector<std::string>& names,
                       int dim, float scale) {
    std::ofstream f(path, std::ios::binary);
    const std::int32_t n = static_cast<std::int32_t>(names.size());
    f.write("BCD1", 4);
    f.write(reinterpret_cast<const char*>(&n), sizeof n);
    f.write(reinterpret_cast<const char*>(&dim), sizeof dim);
    for (std::int32_t k = 0; k < n; ++k) {
        const std::int32_t len = static_cast<std::int32_t>(names[static_cast<std::size_t>(k)].size());
        f.write(reinterpret_cast<const char*>(&len), sizeof len);
        f.write(names[static_cast<std::size_t>(k)].data(), len);
        f.write(reinterpret_cast<const char*>(&scale), sizeof scale);
        std::vector<float> dir(static_cast<std::size_t>(dim), 0.0f);
        dir[static_cast<std::size_t>(k) % static_cast<std::size_t>(dim)] = 1.0f;
        f.write(reinterpret_cast<const char*>(dir.data()),
                static_cast<std::streamsize>(sizeof(float) * dir.size()));
    }
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

    // ── 6. load(path, merge) — banks of different provenance stack ───────────
    // A word-derived bank and an SAE-discovered one are separate files; a tool
    // that wants a slider for each must be able to hold both at once.
    {
        brodiffusion::CondControl m;
        const std::string a = "test_bank_a.bcd1", b = "test_bank_b.bcd1";
        write_bank(a, {"word.warm", "word.wide"}, D, 2.0f);
        write_bank(b, {"sae.4571", "word.warm"}, D, 7.0f);   // 2nd name COLLIDES

        m.load(a);
        CHECK(m.dim() == D);
        CHECK(m.names().size() == 2);

        m.load(b, /*merge=*/true);
        CHECK(m.names().size() == 3);                  // 2 + 2, one overwritten
        CHECK(m.axis_scale("word.wide") == 2.0f);      // bank A axis survives
        CHECK(m.axis_scale("sae.4571") == 7.0f);       // bank B axis appended
        CHECK(m.axis_scale("word.warm") == 7.0f);      // collision: B overwrites A
        m.set("sae.4571", 1.0f);                       // the appended axis is usable
        CHECK(m.active());

        // A merged axis's weight is reset — the direction it named is gone.
        m.clear();
        m.set("word.wide", 3.0f);
        m.load(b, /*merge=*/true);
        CHECK(m.active());                             // word.wide (not in B) keeps its weight
        m.clear();
        m.set("word.warm", 3.0f);
        m.load(b, /*merge=*/true);                     // B overwrites word.warm ...
        CHECK(!m.active());                            // ... so its weight is zeroed

        // Without merge, the second load replaces the first outright.
        m.load(a);
        m.load(b);
        CHECK(m.names().size() == 2);

        // A dim mismatch is a hard error, not a silent partial merge.
        const std::string c = "test_bank_c.bcd1";
        write_bank(c, {"other.dim"}, D + 1, 1.0f);
        m.load(a);
        bool threw = false;
        try { m.load(c, /*merge=*/true); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
        CHECK(m.names().size() == 2);                  // and the load left it untouched

        std::remove(a.c_str());
        std::remove(b.c_str());
        std::remove(c.c_str());
    }

    if (g_failures == 0) std::printf("cond_control budget + merge: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
