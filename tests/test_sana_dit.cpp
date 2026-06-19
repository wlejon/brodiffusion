// Sana Linear DiT (SanaTransformer2DModel) denoiser smoke test.
//
// Gated entirely on the real Sana 0.6B transformer checkpoint
// (weights/sana-600m/transformer/diffusion_pytorch_model.fp16.safetensors).
// When the checkpoint is absent the test prints "skipped (no weights)" and
// exits 0 — the Sana DiT has no compact synthetic-weight fixture (the 28-block
// linear-attn / cross-attn / GLU-MBConv tensor list is large and the exact
// channel arithmetic is the point of the test).
//
// With the checkpoint present it loads the real transformer, projects a
// deterministic synthetic caption sequence (16, 2304) through caption_projection
// + caption_norm, runs one forward over a deterministic synthetic latent
// (1, 32*32*32) at a mid timestep, and asserts: output shape (1, 32*32*32),
// dtype, all-finite, determinism (two forwards bit-identical), and that the
// velocity is neither all-zero nor constant (the network actually ran).

#include "brodiffusion/dit/sana.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace dit = brodiffusion::dit;
namespace st  = brotensor::safetensors;
namespace bt  = brotensor;

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

    const std::string ckpt = weights_dir() +
        "/sana-600m/transformer/diffusion_pytorch_model.fp16.safetensors";
    if (!std::filesystem::exists(ckpt)) {
        std::printf("sana_dit: skipped (no weights)\n");
        return 0;
    }

    const int H_lat = 32, W_lat = 32;        // 1024px latent
    const int L_cap = 16;                    // valid caption tokens (no padding)

    dit::SanaConfig cfg;                      // defaults match Sana 0.6B
    const int IC  = cfg.in_channels;          // 32
    const int CAP = cfg.caption_channels;     // 2304
    const int out_elems = IC * H_lat * W_lat;

    std::vector<float> v1, v2;
    try {
        auto file = st::File::open(ckpt);
        dit::SanaDenoiser dn(cfg);
        dn.load_weights(file, "");
        dn.finalize_weights();

        // Deterministic synthetic caption (L_cap, 2304).
        std::vector<float> cap_h(static_cast<std::size_t>(L_cap) * CAP);
        for (std::size_t i = 0; i < cap_h.size(); ++i) {
            cap_h[i] = 0.1f * std::sin(0.017f * static_cast<float>(i) +
                                       0.5f * static_cast<float>(i % 13));
        }
        brodiffusion::Conditioning cond;
        cond.text_embeddings = bdtest::bd_upload(cap_h, L_cap, CAP);
        cond.has_uncond = false;

        auto prepared = dn.prepare(cond);

        // Deterministic synthetic latent (1, 32*32*32).
        std::vector<float> lat_h(static_cast<std::size_t>(out_elems));
        for (std::size_t i = 0; i < lat_h.size(); ++i) {
            lat_h[i] = std::sin(0.03f * static_cast<float>(i) +
                                0.2f * static_cast<float>(i % 19)) * 0.5f;
        }
        bt::Tensor latent = bdtest::bd_upload(lat_h, 1, out_elems);

        bt::Tensor out;
        dn.forward(latent, H_lat, W_lat, /*timestep=*/500.0f, prepared,
                   brodiffusion::Branch::Cond, out);
        bt::sync_all();

        CHECK(out.rows == 1);
        CHECK(out.cols == out_elems);
        CHECK(out.dtype == dn.compute_dtype());   // FP32 — Sana needs the range

        v1 = bdtest::bd_download(out);
        CHECK(static_cast<int>(v1.size()) == out_elems);

        int nonfinite = 0;
        double sum = 0.0, sumsq = 0.0;
        float lo = 1e30f, hi = -1e30f;
        for (float v : v1) {
            if (!bdtest::bd_finite(v)) { ++nonfinite; continue; }
            sum += v; sumsq += static_cast<double>(v) * v;
            lo = std::min(lo, v); hi = std::max(hi, v);
        }
        CHECK(nonfinite == 0);
        const double mean = sum / out_elems;
        const double var  = sumsq / out_elems - mean * mean;
        CHECK(var > 1e-8);                    // not constant / not all-zero
        CHECK(hi > lo);
        std::printf("sana_dit: range [%.4f, %.4f] mean %.5f var %.5f\n",
                    lo, hi, mean, var);

        // Determinism: a second forward must be bit-identical.
        dn.forward(latent, H_lat, W_lat, 500.0f, prepared,
                   brodiffusion::Branch::Cond, out);
        bt::sync_all();
        v2 = bdtest::bd_download(out);
        CHECK(v1 == v2);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sana_dit: exception: %s\n", e.what());
        return 1;
    }

    if (g_failures == 0) std::printf("sana_dit: OK\n");
    else std::fprintf(stderr, "sana_dit: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
