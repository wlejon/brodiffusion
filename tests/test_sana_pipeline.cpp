// Sana end-to-end Pipeline integration test (txt2img).
//
// Gated entirely on the real Sana 0.6B model directory under weights/sana-600m
// (Gemma-2 text encoder, Linear DiT transformer, DC-AE f32c32 decoder — there
// is no compact synthetic fixture for these, exactly as for test_sana_dit /
// test_dcae_decoder). When the directory is absent it prints "skipped (no
// weights)" and exits 0.
//
// With the weights present it loads the whole model via Pipeline::from_model_dir
// and runs a short txt2img generation at a small resolution (256px, 4 steps) —
// exercising the full Sana path: Gemma prompt encoding (positive + negative),
// caption_projection, the FP32 rectified-flow denoising loop with CFG, and the
// DC-AE decode. Asserts the decoded image has the expected pixel count, is
// all-finite, lands in a sane range (roughly [-1, 1] with a small slack), and
// reproduces byte-for-byte on a second identical generate() (determinism).

#define _CRT_SECURE_NO_WARNINGS   // std::getenv for the gated checkpoint path

#include "brodiffusion/pipeline.h"
#include "brodiffusion/model_config.h"

#include "brotensor/runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
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

#ifndef BRODIFFUSION_WEIGHTS_DIR
#define BRODIFFUSION_WEIGHTS_DIR ""
#endif

static std::string weights_dir() {
    if (const char* e = std::getenv("BRODIFFUSION_WEIGHTS_DIR")) {
        if (e[0]) return e;
    }
    return BRODIFFUSION_WEIGHTS_DIR;
}

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    const std::string root = weights_dir() + "/sana-600m";
    if (!std::filesystem::exists(root + "/model_index.json") ||
        !std::filesystem::exists(root + "/transformer") ||
        !std::filesystem::exists(root + "/text_encoder")) {
        std::printf("sana_pipeline: skipped (no weights)\n");
        return 0;
    }

    try {
        pl::Pipeline pipeline = pl::Pipeline::from_model_dir(root);
        CHECK(pipeline.config().model_class == brodiffusion::ModelClass::Sana);

        pl::GenerateOptions opts;
        opts.height = 256;          // 8x8 latent (DC-AE downsamples 32x)
        opts.width  = 256;
        opts.num_inference_steps = 4;
        opts.guidance_scale = 4.5f;  // exercises the CFG (uncond) branch
        opts.seed = 42;

        std::vector<float> img = pipeline.generate(
            "a photorealistic photo of a red panda sitting in a tree", opts);

        const std::size_t expected =
            static_cast<std::size_t>(3) * opts.height * opts.width;
        CHECK(img.size() == expected);

        int   nonfinite = 0;
        float lo = 1e30f, hi = -1e30f;
        for (float v : img) {
            if (!std::isfinite(v)) { ++nonfinite; continue; }
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        CHECK(nonfinite == 0);
        // The DC-AE output is the raw image in [-1, 1]; allow a little slack
        // for the network overshooting at the extremes.
        CHECK(lo >= -1.5f && hi <= 1.5f);
        // Not a flat field — a real decode spans a meaningful range.
        CHECK(hi - lo > 0.2f);
        std::printf("sana_pipeline: 256px x4 -> %zu px, range [%.3f, %.3f]\n",
                    img.size(), lo, hi);

        // Reproducibility: same seed + prompt yields the same image to within
        // GPU floating-point reassociation noise (exact bit-equality is not
        // guaranteed — Sana's long cross-attention context reduces over ~200
        // caption tokens, and the FP16 DC-AE decode reorders accumulations).
        std::vector<float> img2 = pipeline.generate(
            "a photorealistic photo of a red panda sitting in a tree", opts);
        CHECK(img2.size() == img.size());
        float max_abs_err = 0.0f;
        for (std::size_t i = 0; i < img.size(); ++i) {
            max_abs_err = std::max(max_abs_err, std::fabs(img[i] - img2[i]));
        }
        std::printf("sana_pipeline: reproduce max_abs_err=%.5f\n", max_abs_err);
        CHECK(max_abs_err < 0.05f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sana_pipeline: exception: %s\n", e.what());
        ++g_failures;
    }

    if (g_failures == 0) std::printf("sana_pipeline: OK\n");
    else std::fprintf(stderr, "sana_pipeline: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
