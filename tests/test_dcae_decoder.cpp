// DC-AE (AutoencoderDC f32c32) decoder smoke test.
//
// Gated entirely on the real Sana VAE checkpoint
// (weights/sana-600m/vae/diffusion_pytorch_model.fp16.safetensors). When the
// checkpoint is absent the test prints "skipped (no weights)" and exits 0 — the
// DC-AE decoder has no compact synthetic-weight fixture (the multiscale-attn /
// GLU-MBConv tensor list is large and the exact channel arithmetic is the point
// of the test, so a shrunk fixture would not exercise it faithfully).
//
// With the checkpoint present it loads the real decoder, decodes a deterministic
// synthetic latent (1, 32, 8, 8) → image (1, 3, 256, 256), and asserts: output
// shape + dtype, all-finite (no NaN/Inf), values within a sane bounded range,
// and determinism (two decodes are bit-identical).

#include "brodiffusion/vae_dcae.h"
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

namespace dcae = brodiffusion::dcae;
namespace st   = brotensor::safetensors;
namespace bt   = brotensor;

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

    const std::string ckpt =
        weights_dir() + "/sana-600m/vae/diffusion_pytorch_model.fp16.safetensors";
    if (!std::filesystem::exists(ckpt)) {
        std::printf("dcae_decoder: skipped (no weights)\n");
        return 0;
    }

    const int H_lat = 8, W_lat = 8;            // 8×8 latent → 256×256 image
    const int H_out = H_lat * 32, W_out = W_lat * 32;

    dcae::DecoderConfig cfg;                    // defaults match dc-ae-f32c32
    const int latentC = cfg.latent_channels;   // 32
    const int imgC    = cfg.image_channels;    // 3
    const int out_elems = imgC * H_out * W_out;

    std::vector<float> vals1, vals2;
    try {
        auto file = st::File::open(ckpt);
        dcae::Decoder dec(cfg);
        dec.load_weights(file, "decoder.");

        // Deterministic synthetic latent — small, varied, channel-dependent.
        std::vector<float> latent_h(
            static_cast<std::size_t>(latentC) * H_lat * W_lat);
        for (std::size_t i = 0; i < latent_h.size(); ++i) {
            latent_h[i] = std::sin(0.05f * static_cast<float>(i) +
                                   0.3f * static_cast<float>(i % 17)) * 0.5f;
        }
        bt::Tensor latent =
            bdtest::bd_upload(latent_h, 1, latentC * H_lat * W_lat);

        bt::Tensor out;
        dec.decode(latent, H_lat, W_lat, out);
        bt::sync_all();

        CHECK(out.rows == 1);
        CHECK(out.cols == out_elems);
        CHECK(out.dtype == brodiffusion::compute_dtype());

        vals1 = bdtest::bd_download(out);
        CHECK(static_cast<int>(vals1.size()) == out_elems);

        int nonfinite = 0;
        float lo = 1e30f, hi = -1e30f;
        for (float v : vals1) {
            if (!bdtest::bd_finite(v)) ++nonfinite;
            else { lo = std::min(lo, v); hi = std::max(hi, v); }
        }
        CHECK(nonfinite == 0);
        // Raw decoder output (pre-clamp) should be in a sane bounded range.
        CHECK(lo > -4.0f && hi < 4.0f);
        std::printf("dcae_decoder: range [%.4f, %.4f]\n", lo, hi);

        // Determinism: a second decode must be bit-identical.
        dec.decode(latent, H_lat, W_lat, out);
        bt::sync_all();
        vals2 = bdtest::bd_download(out);
        CHECK(vals1 == vals2);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dcae_decoder: exception: %s\n", e.what());
        return 1;
    }

    if (g_failures == 0) std::printf("dcae_decoder: OK\n");
    else std::fprintf(stderr, "dcae_decoder: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
