// ControlSchedule unit test — the between-step control schedule's bookkeeping.
//
// The expensive half of a schedule (re-running txt_in and dropping the prefix
// KV cache) needs a loaded model; the half that decides WHETHER to pay for it
// does not, and that is the half with all the edge cases. Verified here:
//
//   1. a slot's alpha is indexed by the ABSOLUTE step, and [lo, hi) is
//      half-open, with hi < 0 meaning "to the end".
//   2. advance() rebuilds only when the alpha stack MOVED. A flat schedule
//      pays once; a schedule that sits still pays nothing.
//   3. leaving the window is a rebuild with an inactive stack — the caller has
//      to put the base rows back, and advance() says so by returning true.
//   4. before anything is applied, an all-zero step is NOT a rebuild: the rows
//      are already the primed ones.
//   5. the stack composes: several slots at one step become one CondControl
//      whose injection is Σ alpha_k * scale_k * dir_k, and the budget holds
//      that sum exactly as it holds a prime-time stack.
//   6. arming a new slot forces the next step to rebuild even when every
//      other slot's alpha sat still.
//   7. width and emptiness are rejected at add() time, naming the mismatch.

#include "brodiffusion/control_schedule.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdio>
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

static bool near(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

// A slot over a one-hot direction, so an injected row reads back as the
// coefficient that produced it.
static bd::ControlScheduleSlot one_hot(int dim, int axis, float scale,
                                       std::vector<float> alpha, int lo,
                                       int hi) {
    bd::ControlScheduleSlot s;
    s.name = "e" + std::to_string(axis);
    s.dir.assign(static_cast<std::size_t>(dim), 0.0f);
    s.dir[static_cast<std::size_t>(axis)] = 1.0f;
    s.scale = scale;
    s.alpha = std::move(alpha);
    s.lo_step = lo;
    s.hi_step = hi;
    return s;
}

// Apply `cc` to a (rows, dim) zero tensor and read row `row` back.
static std::vector<float> injected(const bd::CondControl& cc, int rows,
                                   int dim, int row) {
    bt::Tensor t = bt::Tensor::zeros_on(bt::Device::CPU, rows, dim,
                                        bt::Dtype::FP32);
    cc.apply(t, /*row_end=*/-1, /*row_start=*/0);
    std::vector<float> h = t.to_host_vector();
    return std::vector<float>(
        h.begin() + static_cast<std::ptrdiff_t>(row) * dim,
        h.begin() + static_cast<std::ptrdiff_t>(row + 1) * dim);
}

int main() {
    bt::init();
    const int D = 8;

    // ── 1/2/3/4. one slot: window, absolute indexing, rebuild suppression ──
    {
        bd::ControlSchedule s;
        // alpha indexed absolutely; window [2, 5); steps 0..7.
        s.set(one_hot(D, 0, 2.0f, {9.0f, 9.0f, 1.0f, 1.0f, 0.5f, 9.0f, 9.0f,
                                   9.0f}, 2, 5));
        CHECK(s.count() == 1);
        CHECK(s.dim() == D);
        CHECK(s.at_base());

        bd::CondControl cc;
        // Steps 0 and 1 are outside the window: nothing applied, nothing to
        // undo, so no rebuild — and the 9.0 entries prove the window wins
        // over the curve.
        CHECK(!s.advance(0, 0.0f, cc));
        CHECK(!s.advance(1, 0.0f, cc));
        CHECK(s.at_base());

        // Step 2 enters the window at alpha 1: rebuild, injection 1*2*e0.
        CHECK(s.advance(2, 0.0f, cc));
        CHECK(!s.at_base());
        std::vector<float> r = injected(cc, 3, D, 1);
        CHECK(near(r[0], 2.0f));
        CHECK(near(r[1], 0.0f));

        // Step 3 has the same alpha: no rebuild, because the rows already
        // carry it and a rebuild costs a prefix re-extract.
        CHECK(!s.advance(3, 0.0f, cc));

        // Step 4 moves to 0.5: rebuild.
        CHECK(s.advance(4, 0.0f, cc));
        r = injected(cc, 3, D, 0);
        CHECK(near(r[0], 1.0f));

        // Step 5 leaves the window: a rebuild with an INACTIVE stack, which
        // is how the owner learns to put the base rows back.
        CHECK(s.advance(5, 0.0f, cc));
        CHECK(!cc.active());
        // ...and step 6 is the same nothing, so it costs nothing.
        CHECK(!s.advance(6, 0.0f, cc));

        // reset_applied() is what prime() calls: the rows are the base again.
        s.reset_applied();
        CHECK(s.at_base());
        CHECK(!s.advance(6, 0.0f, cc));
    }

    // An unbounded window (hi < 0) runs to the end of the curve, and a step
    // past the curve contributes nothing rather than throwing.
    {
        bd::ControlSchedule s;
        s.set(one_hot(D, 1, 1.0f, {1.0f, 1.0f}, 0, -1));
        bd::CondControl cc;
        CHECK(s.advance(0, 0.0f, cc));
        CHECK(!s.advance(1, 0.0f, cc));
        CHECK(s.advance(2, 0.0f, cc));       // off the end of alpha -> zero
        CHECK(!cc.active());
    }

    // ── 5. the stack composes, and the budget holds the sum ────────────────
    {
        bd::ControlSchedule s;
        s.add(one_hot(D, 0, 2.0f, {1.0f, 1.0f}, 0, -1));
        s.add(one_hot(D, 1, 3.0f, {0.0f, 1.0f}, 0, -1));
        CHECK(s.count() == 2);

        bd::CondControl cc;
        CHECK(s.advance(0, 0.0f, cc));
        std::vector<float> r = injected(cc, 2, D, 0);
        CHECK(near(r[0], 2.0f));
        CHECK(near(r[1], 0.0f));

        // Step 1 brings the second slot in: a different stack, so a rebuild.
        CHECK(s.advance(1, 0.0f, cc));
        r = injected(cc, 2, D, 0);
        CHECK(near(r[0], 2.0f));
        CHECK(near(r[1], 3.0f));

        // The same two slots under a budget of 1 alpha unit. Alpha units are
        // the stack's length over the active axes' MEAN scale, so this stack
        // measures hypot(2, 3) / 2.5 = 1.442 and apply() sheds the overdrive
        // with one common factor — the MIX survives exactly.
        s.reset_applied();
        bd::CondControl capped;
        CHECK(s.advance(1, 1.0f, capped));
        CHECK(near(capped.active_norm(), std::sqrt(13.0f) / 2.5f));
        std::vector<float> c = injected(capped, 2, D, 0);
        const float f = 2.5f / std::sqrt(13.0f);
        CHECK(near(c[0], 2.0f * f));
        CHECK(near(c[1], 3.0f * f));
        CHECK(near(c[1] / c[0], 1.5f));
    }

    // ── 6. a newly armed slot forces the next step to rebuild ──────────────
    {
        bd::ControlSchedule s;
        s.set(one_hot(D, 0, 1.0f, {1.0f, 1.0f, 1.0f}, 0, -1));
        bd::CondControl cc;
        CHECK(s.advance(0, 0.0f, cc));
        CHECK(!s.advance(1, 0.0f, cc));     // flat: nothing to do
        s.add(one_hot(D, 2, 1.0f, {1.0f, 1.0f, 1.0f}, 0, -1));
        // Slot 0's alpha has not moved, but the stack has.
        CHECK(s.advance(1, 0.0f, cc));
        std::vector<float> r = injected(cc, 2, D, 0);
        CHECK(near(r[0], 1.0f));
        CHECK(near(r[2], 1.0f));
    }

    // ── 7. malformed slots are rejected, not silently ignored ──────────────
    {
        bd::ControlSchedule s;
        bool threw = false;
        try { s.add(one_hot(D, 0, 1.0f, {}, 0, -1)); }
        catch (const std::exception&) { threw = true; }
        CHECK(threw);

        s.set(one_hot(D, 0, 1.0f, {1.0f}, 0, -1));
        threw = false;
        try { s.add(one_hot(D + 1, 0, 1.0f, {1.0f}, 0, -1)); }
        catch (const std::exception&) { threw = true; }
        CHECK(threw);
        CHECK(s.count() == 1);   // the bad slot did not land

        // set() replaces rather than appends, and clear() empties.
        s.set(one_hot(D, 3, 1.0f, {1.0f}, 0, -1));
        CHECK(s.count() == 1);
        s.clear();
        CHECK(s.empty());
    }

    if (g_failures == 0) {
        std::printf("ControlSchedule: OK\n");
        return 0;
    }
    std::fprintf(stderr, "ControlSchedule: %d failure(s)\n", g_failures);
    return 1;
}
