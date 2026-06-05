// TripoSplat OctreeGaussianDecoder parity test.
//
// Structural part: always runs — constructs the decoder and asserts the config
// geometry (channel counts, layout total, gaussians per point).
//
// Golden parity part: gated on the real decoder checkpoint
// (weights/triposplat/vae/triposplat_vae_decoder_fp16.safetensors) and a golden
// dump (weights/triposplat/vae/golden/golden_decoder.bin) produced out-of-repo
// from the upstream reference (FP32 CUDA; never committed). Skips cleanly when
// either is absent. The two deterministic forwards (octree occupancy logits and
// the elastic-Gaussian head + assembly) are compared; the stochastic resampler
// is exercised separately for shape/bounds/determinism, not golden-matched.

#include "brodiffusion/triposplat/octree_decoder.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <algorithm>
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

// Golden format BDOCTG1:
//   magic(8) version(i32) N,Mt,Kc,C,ng,res(i32)
//   coords[N*3] cond[Kc*C] logits[N*8] points[Mt*3]
//   positions[G*3] scales[G*3] rotations[G*4] opacities[G] sh[G*3]   (G = Mt*ng)
struct Golden {
    int N = 0, Mt = 0, Kc = 0, C = 0, ng = 0, res = 0;
    std::vector<float> coords, cond, logits, points;
    std::vector<float> positions, scales, rotations, opacities, sh;
};

bool read_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "BDOCTG1\0", 8) != 0) return false;
    int version = 0;
    f.read(reinterpret_cast<char*>(&version), 4);
    int dims[6];
    f.read(reinterpret_cast<char*>(dims), sizeof(dims));
    g.N = dims[0]; g.Mt = dims[1]; g.Kc = dims[2];
    g.C = dims[3]; g.ng = dims[4]; g.res = dims[5];
    const int G = g.Mt * g.ng;
    auto rd = [&](std::vector<float>& v, std::size_t n) {
        v.resize(n);
        f.read(reinterpret_cast<char*>(v.data()),
               static_cast<std::streamsize>(n * sizeof(float)));
    };
    rd(g.coords,    static_cast<std::size_t>(g.N) * 3);
    rd(g.cond,      static_cast<std::size_t>(g.Kc) * g.C);
    rd(g.logits,    static_cast<std::size_t>(g.N) * 8);
    rd(g.points,    static_cast<std::size_t>(g.Mt) * 3);
    rd(g.positions, static_cast<std::size_t>(G) * 3);
    rd(g.scales,    static_cast<std::size_t>(G) * 3);
    rd(g.rotations, static_cast<std::size_t>(G) * 4);
    rd(g.opacities, static_cast<std::size_t>(G));
    rd(g.sh,        static_cast<std::size_t>(G) * 3);
    return static_cast<bool>(f);
}

// FP16 inference vs the FP32 reference golden: a tiny mean error with a
// heavier-tailed max. mean is the correctness signal; max guards corruption.
void compare(const char* label, const std::vector<float>& got,
             const std::vector<float>& ref, float mean_tol, float max_tol) {
    const std::size_t n = std::min(got.size(), ref.size());
    float max_diff = 0.0f, mean_diff = 0.0f, max_ref = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::fabs(got[i] - ref[i]));
        mean_diff += std::fabs(got[i] - ref[i]);
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }
    mean_diff /= static_cast<float>(std::max<std::size_t>(n, 1));
    std::printf("  %-10s n=%zu  max=%.4g mean=%.4g  (|ref|max=%.4g, "
                "mean_tol=%.3g max_tol=%.3g)\n",
                label, n, max_diff, mean_diff, max_ref, mean_tol, max_tol);
    CHECK(got.size() == ref.size());
    CHECK(mean_diff < mean_tol);
    CHECK(max_diff < max_tol);
}

}  // namespace

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    // ── structural ─────────────────────────────────────────────────────────
    {
        tsp::OctreeGaussianDecoder d;
        CHECK(d.config().cond_channels == 16);
        CHECK(d.config().num_gaussians == 32);
        CHECK(d.config().num_heads * d.config().head_dim == d.config().model_channels);
        CHECK(d.config().gs_out_channels == d.config().num_gaussians * (3 + 3 + 3 + 4 + 1 + 1));
        CHECK(d.gaussians_per_point() == 32);
    }

    // ── golden parity (gated) ──────────────────────────────────────────────
    const std::string wd = weights_dir();
    const std::string ckpt = wd + "/triposplat/vae/triposplat_vae_decoder_fp16.safetensors";
    const std::string gpath = wd + "/triposplat/vae/golden/golden_decoder.bin";

    Golden g;
    const bool have_ckpt = !wd.empty() && std::filesystem::exists(ckpt);
    const bool have_gold = !wd.empty() && read_golden(gpath, g);
    if (!have_ckpt || !have_gold) {
        std::printf("test_octree_decoder: structural OK; parity SKIPPED "
                    "(ckpt=%d golden=%d)\n", have_ckpt ? 1 : 0, have_gold ? 1 : 0);
        return g_failures ? 1 : 0;
    }
    // The decoder's attention uses the GPU-only FP16 flash kernel; skip on CPU.
    if (brodiffusion::compute_dtype() != bt::Dtype::FP16) {
        std::printf("test_octree_decoder: structural OK; parity SKIPPED "
                    "(CPU backend — decoder is GPU/FP16 only)\n");
        return g_failures ? 1 : 0;
    }

    try {
        st::File f = st::File::open(ckpt);
        tsp::OctreeGaussianDecoder d;
        d.load_weights(f);

        bt::Tensor cond = bdtest::bd_upload(g.cond, g.Kc, g.C);

        // octree occupancy logits
        bt::Tensor logits_t;
        d.octree_logits(g.coords, g.N, g.res, cond, logits_t);
        bt::sync_all();
        std::vector<float> got_logits = bdtest::bd_download(logits_t);

        // gs head + gaussian assembly
        bt::Tensor feats_t;
        d.gs_features(g.points, g.Mt, cond, feats_t);
        bt::sync_all();
        tsp::GaussianSplats splats = d.build_gaussians(g.points, g.Mt, feats_t);

        std::printf("test_octree_decoder: (fp16 vs fp32 reference)\n");
        compare("logits",    got_logits,        g.logits,    /*mean*/0.03f, /*max*/0.5f);
        compare("positions", splats.positions,  g.positions, /*mean*/0.003f, /*max*/0.03f);
        compare("scales",    splats.scales,     g.scales,    /*mean*/0.002f, /*max*/0.02f);
        compare("rotations", splats.rotations,  g.rotations, /*mean*/0.02f, /*max*/0.2f);
        compare("opacities", splats.opacities,  g.opacities, /*mean*/0.002f, /*max*/0.02f);
        compare("sh",        splats.sh,         g.sh,        /*mean*/0.01f, /*max*/0.1f);

        // ── resampler: shape / bounds / determinism (not golden-matched) ────
        const int target = 64 * d.gaussians_per_point();  // 64 decoder tokens
        tsp::GaussianSplats a = d.decode(cond, target, /*seed=*/1234);
        tsp::GaussianSplats b = d.decode(cond, target, /*seed=*/1234);
        const std::size_t expect = static_cast<std::size_t>(64) * d.gaussians_per_point();
        CHECK(a.count() == expect);
        CHECK(a.positions.size() == b.positions.size());
        bool identical = a.positions.size() == b.positions.size();
        for (std::size_t i = 0; identical && i < a.positions.size(); ++i)
            if (a.positions[i] != b.positions[i]) identical = false;
        CHECK(identical);  // same seed => same cloud
        float lo = 1e9f, hi = -1e9f;
        for (float v : a.positions) { lo = std::min(lo, v); hi = std::max(hi, v); }
        std::printf("  decode: count=%zu pos in [%.3f, %.3f] (aabb-centered)\n",
                    a.count(), lo, hi);
        CHECK(lo >= -0.5001f && hi <= 0.5001f);  // aabb [-0.5,0.5]^3
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL parity threw: %s\n", e.what());
        ++g_failures;
    }

    std::printf("test_octree_decoder: %s\n", g_failures ? "FAILED" : "OK");
    return g_failures ? 1 : 0;
}
