// Flux.2 VAE image-encoder (TripoSplat feature2) numeric-parity test.
//
// Structural part: always runs — constructs the encoder and asserts the token
// geometry math (T = (H/16)*(W/16), D = 128) is consistent.
//
// Golden parity part: gated on the real Flux.2 VAE checkpoint
// (weights/triposplat/vae/flux2-vae.safetensors) and a golden dump
// (weights/triposplat/vae/golden/golden_flux2vae.bin) produced out-of-repo from
// the upstream reference model.py (never committed). Skips cleanly when either
// is absent. Loads the real weights, encodes the golden image (deterministic =
// posterior mean), and compares the (T, 128) tokens against the reference.

#include "brodiffusion/triposplat/vae_encoder.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace tsp = brodiffusion::triposplat;
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

namespace {

std::string weights_dir() {
    if (const char* e = std::getenv("BRODIFFUSION_WEIGHTS_DIR")) {
        if (e[0]) return e;
    }
    return BRODIFFUSION_WEIGHTS_DIR;
}

// Golden format BD2VAEG1: magic(8) version(i32) H,W,T,D(i32) image[3HW] tokens[TD].
struct Golden {
    int H = 0, W = 0, T = 0, D = 0;
    std::vector<float> image;   // (3, H, W) in [0, 1]
    std::vector<float> tokens;  // (T, D)
};

bool read_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "BD2VAEG1", 8) != 0) return false;
    int version = 0;
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&g.H), 4);
    f.read(reinterpret_cast<char*>(&g.W), 4);
    f.read(reinterpret_cast<char*>(&g.T), 4);
    f.read(reinterpret_cast<char*>(&g.D), 4);
    g.image.resize(static_cast<std::size_t>(3) * g.H * g.W);
    g.tokens.resize(static_cast<std::size_t>(g.T) * g.D);
    f.read(reinterpret_cast<char*>(g.image.data()),
           static_cast<std::streamsize>(g.image.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(g.tokens.data()),
           static_cast<std::streamsize>(g.tokens.size() * sizeof(float)));
    return static_cast<bool>(f);
}

}  // namespace

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    // ── structural: token-geometry math ───────────────────────────────────
    CHECK(tsp::Flux2VaeEncoder::kTokenDim == 128);
    CHECK(tsp::Flux2VaeEncoder::kTokenStride == 16);
    {
        const int H = 256, Wd = 256;
        const int T = (H / 16) * (Wd / 16);
        CHECK(T == 256);
    }

    // ── golden parity (gated) ─────────────────────────────────────────────
    const std::string wd = weights_dir();
    const std::string ckpt = wd + "/triposplat/vae/flux2-vae.safetensors";
    const std::string gpath = wd + "/triposplat/vae/golden/golden_flux2vae.bin";

    Golden g;
    const bool have_ckpt = !wd.empty() && std::filesystem::exists(ckpt);
    const bool have_gold = !wd.empty() && read_golden(gpath, g);

    if (!have_ckpt || !have_gold) {
        std::printf("test_flux2_vae_encoder: structural OK; "
                    "parity SKIPPED (ckpt=%d golden=%d)\n",
                    have_ckpt ? 1 : 0, have_gold ? 1 : 0);
        return g_failures ? 1 : 0;
    }

    try {
        st::File f = st::File::open(ckpt);
        tsp::Flux2VaeEncoder enc;
        enc.load_weights(f);

        // image: golden stores [0,1]; encoder wants [-1,1].
        std::vector<float> img(g.image.size());
        for (std::size_t i = 0; i < img.size(); ++i) img[i] = g.image[i] * 2.0f - 1.0f;
        bt::Tensor image = bdtest::bd_upload(img, 1, static_cast<int>(img.size()));

        bt::Tensor out;
        enc.encode(image, g.H, g.W, out);
        std::vector<float> got = bdtest::bd_download(out);

        CHECK(static_cast<int>(got.size()) == g.T * g.D);

        float max_diff = 0.0f, mean_diff = 0.0f;
        const std::size_t n = std::min(got.size(), g.tokens.size());
        for (std::size_t i = 0; i < n; ++i) {
            float d = std::fabs(got[i] - g.tokens[i]);
            max_diff = std::max(max_diff, d);
            mean_diff += d;
        }
        mean_diff /= static_cast<float>(n);

        // FP32 on CPU is tight; FP16 on a GPU accumulates conv error across the
        // deep encoder — allow more headroom there.
        const bool fp16 = brodiffusion::compute_dtype() == bt::Dtype::FP16;
        const float tol = fp16 ? 0.05f : 5e-3f;
        std::printf("test_flux2_vae_encoder: T=%d D=%d  max=%.4g mean=%.4g  "
                    "(tol=%.3g, %s)\n",
                    g.T, g.D, max_diff, mean_diff, tol, fp16 ? "fp16" : "fp32");
        CHECK(max_diff < tol);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL parity threw: %s\n", e.what());
        ++g_failures;
    }

    std::printf("test_flux2_vae_encoder: %s\n", g_failures ? "FAILED" : "OK");
    return g_failures ? 1 : 0;
}
