// TripoSplat flow DiT (LatentSeqMMFlowModel) numeric-parity test.
//
// Structural part: always runs — constructs the model and asserts the config
// geometry (q_token_length, channel counts) is consistent.
//
// Golden parity part: gated on the real flow checkpoint
// (weights/triposplat/diffusion_models/triposplat_fp16.safetensors) and a golden
// dump (weights/triposplat/diffusion_models/golden/golden_flow.bin) produced
// out-of-repo from the upstream reference model.py (FP32 CUDA; never committed).
// Skips cleanly when either is absent. Loads weights, runs one forward over the
// golden's synthetic inputs, and compares the predicted latent/camera velocities.

#include "brodiffusion/triposplat/flow_model.h"
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

// Golden format BDFLOWG1:
//   magic(8) version(i32) L,Ci,Kc,Cc,C2,Cm(i32) t(f32)
//   latent_in[L*Ci] camera_in[Cm] feat1[Kc*Cc] feat2[Kc*C2]
//   latent_out[L*Ci] camera_out[Cm]
struct Golden {
    int L = 0, Ci = 0, Kc = 0, Cc = 0, C2 = 0, Cm = 0;
    float t = 0.0f;
    std::vector<float> latent_in, camera_in, feat1, feat2, latent_out, camera_out;
};

bool read_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "BDFLOWG1", 8) != 0) return false;
    int version = 0;
    f.read(reinterpret_cast<char*>(&version), 4);
    int dims[6];
    f.read(reinterpret_cast<char*>(dims), sizeof(dims));
    g.L = dims[0]; g.Ci = dims[1]; g.Kc = dims[2];
    g.Cc = dims[3]; g.C2 = dims[4]; g.Cm = dims[5];
    f.read(reinterpret_cast<char*>(&g.t), 4);
    auto rd = [&](std::vector<float>& v, std::size_t n) {
        v.resize(n);
        f.read(reinterpret_cast<char*>(v.data()),
               static_cast<std::streamsize>(n * sizeof(float)));
    };
    rd(g.latent_in,  static_cast<std::size_t>(g.L) * g.Ci);
    rd(g.camera_in,  static_cast<std::size_t>(g.Cm));
    rd(g.feat1,      static_cast<std::size_t>(g.Kc) * g.Cc);
    rd(g.feat2,      static_cast<std::size_t>(g.Kc) * g.C2);
    rd(g.latent_out, static_cast<std::size_t>(g.L) * g.Ci);
    rd(g.camera_out, static_cast<std::size_t>(g.Cm));
    return static_cast<bool>(f);
}

// FP16 inference vs the fully-FP32 reference golden: parity shows as a tiny
// mean error with a heavy-tailed max (FP16 activation rounding compounds over
// 28 blocks). The mean is the correctness signal; the max only guards against
// gross corruption.
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
    std::printf("  %-7s n=%zu  max=%.4g mean=%.4g  (|ref|max=%.4g, "
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
        tsp::FlowDiT m;
        CHECK(m.config().q_token_length == 8192);
        CHECK(m.config().in_channels == 16);
        CHECK(m.config().model_channels == 1024);
        CHECK(m.config().num_heads * m.config().head_dim == m.config().model_channels);
        CHECK(m.config().num_blocks == 24);
    }

    // ── golden parity (gated) ──────────────────────────────────────────────
    const std::string wd = weights_dir();
    const std::string ckpt = wd + "/triposplat/diffusion_models/triposplat_fp16.safetensors";
    const std::string gpath = wd + "/triposplat/diffusion_models/golden/golden_flow.bin";

    Golden g;
    const bool have_ckpt = !wd.empty() && std::filesystem::exists(ckpt);
    const bool have_gold = !wd.empty() && read_golden(gpath, g);
    if (!have_ckpt || !have_gold) {
        std::printf("test_flow_model: structural OK; parity SKIPPED "
                    "(ckpt=%d golden=%d)\n", have_ckpt ? 1 : 0, have_gold ? 1 : 0);
        return g_failures ? 1 : 0;
    }
    // The flow DiT's attention uses the GPU-only FP16 flash kernel; on a CPU
    // backend there is nothing to compare against, so skip the parity run.
    if (brodiffusion::compute_dtype() != bt::Dtype::FP16) {
        std::printf("test_flow_model: structural OK; parity SKIPPED "
                    "(CPU backend — flow DiT is GPU/FP16 only)\n");
        return g_failures ? 1 : 0;
    }

    try {
        st::File f = st::File::open(ckpt);
        tsp::FlowDiT m;
        m.load_weights(f);

        bt::Tensor latent = bdtest::bd_upload(g.latent_in, g.L, g.Ci);
        bt::Tensor camera = bdtest::bd_upload(g.camera_in, 1, g.Cm);
        bt::Tensor feat1  = bdtest::bd_upload(g.feat1, g.Kc, g.Cc);
        bt::Tensor feat2  = bdtest::bd_upload(g.feat2, g.Kc, g.C2);

        bt::Tensor out_lat, out_cam;
        m.forward(latent, camera, feat1, feat2, g.t, out_lat, out_cam);
        std::vector<float> got_lat = bdtest::bd_download(out_lat);
        std::vector<float> got_cam = bdtest::bd_download(out_cam);

        // FP16 inference vs the FP32 reference: mean ~2e-3 against |values| up to
        // ~3, with a heavy-tailed max from FP16 rounding over 28 blocks.
        std::printf("test_flow_model: (fp16 vs fp32 reference)\n");
        compare("latent", got_lat, g.latent_out, /*mean_tol=*/0.01f, /*max_tol=*/0.35f);
        compare("camera", got_cam, g.camera_out, /*mean_tol=*/0.01f, /*max_tol=*/0.05f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL parity threw: %s\n", e.what());
        ++g_failures;
    }

    std::printf("test_flow_model: %s\n", g_failures ? "FAILED" : "OK");
    return g_failures ? 1 : 0;
}
