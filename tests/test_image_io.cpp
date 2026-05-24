// image_io smoke test.
//
// Encodes a synthetic 32x32 PNG via broimage and round-trips it through
// brodiffusion::load_image_as_latent_input, verifying:
//   - output shape (1, 3 * dst_h * dst_w),
//   - dtype matches the pipeline compute dtype,
//   - values are in [-1, 1] within tolerance,
//   - a constant-128 input maps near zero (128*2/255 - 1 ≈ 0.0039).

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/image_io.h"

#include "broimage/encode.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace bt = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    auto tmp = std::filesystem::temp_directory_path();
    auto cb_path  = tmp / "brodiffusion_imgio_checker.png";
    auto mid_path = tmp / "brodiffusion_imgio_mid.png";

    // ── Checkerboard 32x32 RGBA8 ──────────────────────────────────────────
    {
        const int w = 32, h = 32;
        std::vector<uint8_t> rgba(static_cast<std::size_t>(w * h * 4));
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                bool dark = ((x / 4) ^ (y / 4)) & 1;
                std::size_t i = static_cast<std::size_t>((y * w + x) * 4);
                uint8_t c = dark ? 32 : 224;
                rgba[i+0] = c; rgba[i+1] = c; rgba[i+2] = c; rgba[i+3] = 255;
            }
        }
        bool ok = broimage::encode_png_file(cb_path.string(), rgba.data(),
                                            w, h, /*channels=*/4);
        CHECK(ok);
    }

    // ── Mid-gray 32x32 RGBA8 (all 128) ────────────────────────────────────
    {
        const int w = 32, h = 32;
        std::vector<uint8_t> rgba(static_cast<std::size_t>(w * h * 4), 128);
        // Restore alpha to 255 so the rgba->rgb drop leaves the RGB at 128.
        for (std::size_t i = 3; i < rgba.size(); i += 4) rgba[i] = 255;
        bool ok = broimage::encode_png_file(mid_path.string(), rgba.data(),
                                            w, h, /*channels=*/4);
        CHECK(ok);
    }

    // ── Decode checkerboard → (1, 3, 16, 16) tensor in [-1, 1] ────────────
    {
        bt::Tensor t = brodiffusion::load_image_as_latent_input(
            cb_path.string(), /*dst_w=*/16, /*dst_h=*/16);
        bt::sync_all();
        CHECK(t.rows == 1);
        CHECK(t.cols == 3 * 16 * 16);
        CHECK(t.dtype == brodiffusion::compute_dtype());
        std::vector<float> vals = bdtest::bd_download(t);
        int oob = 0;
        for (float v : vals) {
            if (!bdtest::bd_finite(v)) ++oob;
            // Allow a touch of slop for FP16 round-trip on GPU builds.
            else if (v < -1.01f || v > 1.01f) ++oob;
        }
        CHECK(oob == 0);
    }

    // ── Decode mid-gray → values near 0 ───────────────────────────────────
    {
        bt::Tensor t = brodiffusion::load_image_as_latent_input(
            mid_path.string(), /*dst_w=*/16, /*dst_h=*/16);
        bt::sync_all();
        CHECK(t.rows == 1);
        CHECK(t.cols == 3 * 16 * 16);
        std::vector<float> vals = bdtest::bd_download(t);
        // 128 * 2/255 - 1 = 0.003921...  Allow a wide margin so this passes on
        // both FP32 CPU and FP16 GPU builds.
        const float tol = 0.02f;
        int bad = 0;
        for (float v : vals) {
            if (std::fabs(v - (128.0f * 2.0f / 255.0f - 1.0f)) > tol) ++bad;
        }
        CHECK(bad == 0);
    }

    std::error_code ec;
    std::filesystem::remove(cb_path,  ec);
    std::filesystem::remove(mid_path, ec);

    if (g_failures == 0) std::printf("image_io: OK\n");
    else std::fprintf(stderr, "image_io: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
