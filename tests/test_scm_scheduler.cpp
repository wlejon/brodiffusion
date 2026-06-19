// SCM (TrigFlow consistency-model) scheduler test — the Sana-Sprint sampler.
//
// Verifies the diffusers SCMScheduler behaviour brodiffusion relies on: the
// two-step default angle schedule [max, intermediate, 0] and the linspace
// schedule for other step counts, init_noise_sigma == 1, sigma_data == 0.5, the
// TrigFlow step() reconstruction (pred_x0 = cos(s)x - sin(s)out, then x_t =
// cos(t)pred_x0 + sin(t)*sigma_data*z), the deterministic final step (t == 0
// drops the noise term), and the misuse guards.

#include "brodiffusion/scm_scheduler.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
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

static bool approx(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

// Sana's latent is FP32 even on a GPU backend, so the SCM tensors are FP32.
// Upload verbatim (not at compute dtype, which would be FP16 on a GPU).
static bt::Tensor up_f32(const std::vector<float>& v, int rows, int cols) {
    return bt::Tensor::from_host(v.data(), rows, cols).to(bt::default_device());
}

int main() {
    // ── 1. Two-step default schedule: [max, intermediate, 0] ───────────────
    {
        sc::SCM s;
        s.set_timesteps(2);  // defaults max=1.57080, intermediate=1.3
        const auto& ts = s.timesteps();
        CHECK(ts.size() == 2);
        CHECK(s.num_inference_steps() == 2);
        CHECK(approx(ts[0], 1.57080f));
        CHECK(approx(ts[1], 1.3f));
        CHECK(s.init_noise_sigma() == 1.0f);
        CHECK(approx(s.sigma_data(), 0.5f));
    }

    // ── 2. Two-step with explicit max / intermediate overrides ─────────────
    {
        sc::SCM s;
        s.set_timesteps(2, /*max=*/1.4f, /*intermediate=*/0.9f);
        const auto& ts = s.timesteps();
        CHECK(ts.size() == 2);
        CHECK(approx(ts[0], 1.4f));
        CHECK(approx(ts[1], 0.9f));
    }

    // ── 3. N != 2: linspace(max, 0, N+1) minus trailing 0 ──────────────────
    {
        sc::SCM s;
        const float max_t = 1.57080f;
        s.set_timesteps(4);
        const auto& ts = s.timesteps();
        CHECK(ts.size() == 4);
        // i = 0..3 of linspace(max, 0, 5): max*(1 - i/4).
        for (int i = 0; i < 4; ++i) {
            CHECK(approx(ts[static_cast<std::size_t>(i)],
                         max_t * (1.0f - static_cast<float>(i) / 4.0f)));
        }
        // Strictly decreasing, all > 0 (the trailing 0 is not a run timestep).
        for (std::size_t i = 1; i < ts.size(); ++i) CHECK(ts[i] < ts[i - 1]);
        CHECK(ts.back() > 0.0f);
    }

    // ── runtime init for the tensor-level step() checks ────────────────────
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    const int n = 4 * 6 * 6;  // small C*H*W stand-in

    std::mt19937 rng(0x5C3D);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<float> sample0(n), out_vals(n), noise_vals(n);
    for (int i = 0; i < n; ++i) {
        sample0[i]    = nrm(rng);
        out_vals[i]   = nrm(rng);
        noise_vals[i] = nrm(rng);
    }

    // ── 4. Non-final step reconstruction matches the TrigFlow formula ──────
    {
        sc::SCM s;
        s.set_timesteps(2);
        const float sigma_data = s.sigma_data();
        const float angle_s = s.timesteps()[0];          // current
        const float angle_t = s.timesteps()[1];          // next (step 0 -> step 1)

        bt::Tensor sample = up_f32(sample0,    1, n);
        bt::Tensor out    = up_f32(out_vals,   1, n);
        bt::Tensor noise  = up_f32(noise_vals, 1, n);
        s.step(out, /*step_index=*/0, sample, noise);

        std::vector<float> got = bdtest::bd_download(sample);
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            const float pred_x0 =
                std::cos(angle_s) * sample0[i] - std::sin(angle_s) * out_vals[i];
            const float expect = std::cos(angle_t) * pred_x0 +
                                 std::sin(angle_t) * sigma_data * noise_vals[i];
            if (!approx(got[static_cast<std::size_t>(i)], expect, 1e-4f)) ++bad;
        }
        CHECK(bad == 0);
    }

    // ── 5. Final step (t == 0) is deterministic: result == pred_x0, the noise
    //       term vanishes (sin(0) == 0), independent of the noise tensor ─────
    {
        sc::SCM s;
        s.set_timesteps(2);
        const float angle_s = s.timesteps()[1];  // last run angle; next angle = 0

        // Run the last step with two different noise tensors.
        std::vector<float> noise_a(n), noise_b(n);
        for (int i = 0; i < n; ++i) { noise_a[i] = nrm(rng); noise_b[i] = nrm(rng); }

        bt::Tensor sa = up_f32(sample0, 1, n);
        bt::Tensor oa = up_f32(out_vals, 1, n);
        bt::Tensor na = up_f32(noise_a, 1, n);
        s.step(oa, /*step_index=*/1, sa, na);
        std::vector<float> got_a = bdtest::bd_download(sa);

        bt::Tensor sb = up_f32(sample0, 1, n);
        bt::Tensor ob = up_f32(out_vals, 1, n);
        bt::Tensor nb = up_f32(noise_b, 1, n);
        s.step(ob, /*step_index=*/1, sb, nb);
        std::vector<float> got_b = bdtest::bd_download(sb);

        // Identical regardless of noise, and equal to pred_x0.
        CHECK(got_a == got_b);
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            const float pred_x0 =
                std::cos(angle_s) * sample0[i] - std::sin(angle_s) * out_vals[i];
            if (!approx(got_a[static_cast<std::size_t>(i)], pred_x0, 1e-4f)) ++bad;
            if (!bdtest::bd_finite(got_a[static_cast<std::size_t>(i)])) ++bad;
        }
        CHECK(bad == 0);
    }

    // ── 6. Misuse guards ───────────────────────────────────────────────────
    {
        // step() before set_timesteps() throws.
        sc::SCM s;
        bt::Tensor a = up_f32(sample0, 1, n);
        bt::Tensor b = up_f32(out_vals, 1, n);
        bt::Tensor c = up_f32(noise_vals, 1, n);
        bool threw = false;
        try { s.step(b, 0, a, c); } catch (const std::exception&) { threw = true; }
        CHECK(threw);

        // step_index out of range throws.
        s.set_timesteps(2);
        threw = false;
        try { s.step(b, 2, a, c); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
        threw = false;
        try { s.step(b, -1, a, c); } catch (const std::exception&) { threw = true; }
        CHECK(threw);

        // set_timesteps(0) throws.
        threw = false;
        try { sc::SCM bad; bad.set_timesteps(0); }
        catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }

    if (g_failures == 0) std::fprintf(stderr, "test_scm_scheduler: OK\n");
    return g_failures == 0 ? 0 : 1;
}
