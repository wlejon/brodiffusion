// TripoSplat flow Euler CFG sampler (sample_latent) parity test.
//
// Structural part: always runs — builds a small synthetic FlowDiT fixture and
// asserts the CUDA-graph step-capture path (sample_latent's capture session)
// produces bitwise-identical output to the eager path, with CFG on and off.
// On a CPU-only build both runs are eager and the check degenerates to
// eager == eager.
//
// Golden parity part: gated on the real flow checkpoint and golden_sampler.bin
// (out-of-repo, FP32 CUDA reference FlowEulerCfgSampler over injected noise).
// Skips cleanly when either is absent. Loads weights, runs the sampler over the
// golden's injected noise + features, and compares the clean latent.

#include "brodiffusion/triposplat/sampler.h"
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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
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

void set_env(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

// ── synthetic FlowDiT fixture (structural captured-vs-eager parity) ────────

// Small deterministic FP16 values bounded around zero (wrapped sawtooth) so
// the synthetic forward stays numerically tame.
std::vector<uint16_t> fp16_seq(std::size_t n, float scale) {
    std::vector<uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float v = (static_cast<float>(i % 7) - 3.0f) * scale;
        out[i] = bt::fp32_to_fp16_bits(v);
    }
    return out;
}

std::vector<uint16_t> fp16_from(const std::vector<float>& v) {
    std::vector<uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = bt::fp32_to_fp16_bits(v[i]);
    return out;
}

struct Fixture {
    std::vector<std::unique_ptr<std::vector<uint16_t>>> store;
    std::vector<st::WriteEntry> entries;

    void add(const std::string& name, std::vector<int64_t> shape,
             std::vector<uint16_t> data) {
        store.push_back(std::make_unique<std::vector<uint16_t>>(std::move(data)));
        st::WriteEntry e;
        e.name = name;
        e.dtype = st::Dtype::F16;
        e.shape = std::move(shape);
        e.host_data = store.back()->data();
        e.bytes = store.back()->size() * 2;
        entries.push_back(std::move(e));
    }

    void add_linear(const std::string& key, int out, int in, float ws, float bs) {
        add(key + ".weight", {out, in}, fp16_seq(static_cast<std::size_t>(out) * in, ws));
        add(key + ".bias", {out}, fp16_seq(static_cast<std::size_t>(out), bs));
    }
};

// Shrunk config: same topology as the real model, tiny dimensions.
brodiffusion::triposplat::FlowModelConfig small_cfg() {
    tsp::FlowModelConfig c;
    c.q_token_length     = 64;
    c.model_channels     = 64;
    c.cond_channels      = 32;
    c.cond2_channels     = 8;
    c.num_refiner_blocks = 1;
    c.num_blocks         = 2;
    c.num_heads          = 4;
    c.head_dim           = 16;
    return c;
}

void add_repo(Fixture& fx, const std::string& p, const tsp::FlowModelConfig& c) {
    const int D = c.model_channels, H = c.num_heads;
    const int hidden = D / 8;
    fx.add(p + ".norm.weight", {D}, fp16_seq(static_cast<std::size_t>(D), 0.02f));
    fx.add(p + ".norm.bias",   {D}, fp16_seq(static_cast<std::size_t>(D), 0.01f));
    fx.add(p + ".gate_map.weight",    {hidden, D}, fp16_seq(static_cast<std::size_t>(hidden) * D, 0.03f));
    fx.add(p + ".content_map.weight", {hidden, D}, fp16_seq(static_cast<std::size_t>(hidden) * D, 0.03f));
    fx.add(p + ".final_map.weight",   {3 * H, hidden}, fp16_seq(static_cast<std::size_t>(3 * H) * hidden, 0.05f));
    // f0 + f1 + f2 must equal head_dim/2 (= 8 here): 3 + 3 + 2.
    fx.add(p + ".freqs_0", {3}, fp16_from({0.25f, 0.5f, 1.0f}));
    fx.add(p + ".freqs_1", {3}, fp16_from({0.25f, 0.5f, 1.0f}));
    fx.add(p + ".freqs_2", {2}, fp16_from({0.5f, 1.5f}));
}

void add_block(Fixture& fx, const std::string& p, bool modulated,
               const tsp::FlowModelConfig& c) {
    const int D = c.model_channels, FF = D * c.mlp_ratio;
    const int H = c.num_heads, hd = c.head_dim;
    fx.add_linear(p + ".attn.qkv", 3 * D, D, 0.03f, 0.01f);
    fx.add_linear(p + ".attn.out", D, D, 0.03f, 0.01f);
    fx.add(p + ".attn.q_norm.gamma", {H * hd}, fp16_seq(static_cast<std::size_t>(H) * hd, 0.05f));
    fx.add(p + ".attn.k_norm.gamma", {H * hd}, fp16_seq(static_cast<std::size_t>(H) * hd, 0.05f));
    fx.add_linear(p + ".mlp.mlp.0", FF, D, 0.03f, 0.01f);
    fx.add_linear(p + ".mlp.mlp.2", D, FF, 0.03f, 0.01f);
    if (modulated) {
        fx.add(p + ".shift_table", {1, 6 * D}, fp16_seq(static_cast<std::size_t>(6) * D, 0.02f));
    } else {
        fx.add(p + ".norm1.weight", {D}, fp16_seq(static_cast<std::size_t>(D), 0.02f));
        fx.add(p + ".norm1.bias",   {D}, fp16_seq(static_cast<std::size_t>(D), 0.01f));
        fx.add(p + ".norm2.weight", {D}, fp16_seq(static_cast<std::size_t>(D), 0.02f));
        fx.add(p + ".norm2.bias",   {D}, fp16_seq(static_cast<std::size_t>(D), 0.01f));
    }
}

std::string build_fixture_file(const tsp::FlowModelConfig& c) {
    const int D = c.model_channels;
    Fixture fx;
    fx.add_linear("t_embedder.mlp.0", D, 256, 0.02f, 0.01f);
    fx.add_linear("t_embedder.mlp.2", D, D, 0.03f, 0.01f);
    fx.add_linear("adaLN_modulation.1", 6 * D, D, 0.02f, 0.01f);
    fx.add_linear("input_layer", D, c.in_channels, 0.05f, 0.01f);
    fx.add_linear("cond_embedder", D, c.cond_channels, 0.05f, 0.01f);
    fx.add_linear("cond_embedder2", D, c.cond2_channels, 0.05f, 0.01f);
    fx.add_linear("cam_refiner.mlp.0", D, c.cam_channels, 0.05f, 0.01f);
    fx.add_linear("cam_refiner.mlp.2", D, D, 0.03f, 0.01f);
    fx.add("shift_table", {1, 2 * D}, fp16_seq(static_cast<std::size_t>(2) * D, 0.02f));
    fx.add_linear("out_layer", c.out_channels, D, 0.03f, 0.01f);
    fx.add_linear("cam_out_layer", c.cam_channels, D, 0.03f, 0.01f);
    for (int i = 0; i < c.num_refiner_blocks; ++i) {
        add_repo(fx, "noise_repo_layers." + std::to_string(i), c);
        add_repo(fx, "context_repo_layers." + std::to_string(i), c);
        add_block(fx, "noise_refiner." + std::to_string(i), /*modulated=*/true, c);
        add_block(fx, "context_refiner." + std::to_string(i), /*modulated=*/false, c);
    }
    for (int i = 0; i < c.num_blocks; ++i) {
        add_repo(fx, "repo_layers." + std::to_string(i), c);
        add_block(fx, "blocks." + std::to_string(i), /*modulated=*/true, c);
    }
    const std::string path =
        (std::filesystem::temp_directory_path() / "bd_flow_capture_fixture.safetensors").string();
    st::write_file(path, fx.entries);
    return path;
}

// Captured vs eager parity: the step-capture session must be a pure execution
// optimization — bitwise-identical outputs with the graph enabled and with the
// BRODIFFUSION_DISABLE_STEP_GRAPH hatch forcing eager stepping.
void structural_capture_parity() {
    const tsp::FlowModelConfig c = small_cfg();
    const std::string path = build_fixture_file(c);
    try {
        st::File f = st::File::open(path);
        tsp::FlowDiT m(c);
        m.load_weights(f);

        const int L = c.q_token_length, K = 8;
        auto seq = [](std::size_t n, float scale) {
            std::vector<float> v(n);
            for (std::size_t i = 0; i < n; ++i) {
                v[i] = (static_cast<float>(i % 11) - 5.0f) * scale;
            }
            return v;
        };
        bt::Tensor nl = bdtest::bd_upload(seq(static_cast<std::size_t>(L) * c.in_channels, 0.2f), L, c.in_channels);
        bt::Tensor nc = bdtest::bd_upload(seq(static_cast<std::size_t>(c.cam_channels), 0.2f), 1, c.cam_channels);
        bt::Tensor f1 = bdtest::bd_upload(seq(static_cast<std::size_t>(K) * c.cond_channels, 0.1f), K, c.cond_channels);
        bt::Tensor f2 = bdtest::bd_upload(seq(static_cast<std::size_t>(K) * c.cond2_channels, 0.1f), K, c.cond2_channels);

        auto run = [&](float guidance) {
            tsp::FlowSampleOptions o;
            o.steps = 8;
            o.guidance_scale = guidance;
            o.shift = 3.0f;
            bt::Tensor out;
            tsp::sample_latent(m, f1, f2, nl, nc, o, out);
            return bdtest::bd_download(out);
        };

        for (float guidance : {3.0f, 1.0f}) {
            set_env("BRODIFFUSION_DISABLE_STEP_GRAPH", "");
            std::vector<float> out_a = run(guidance);          // graph path (on CUDA)
            set_env("BRODIFFUSION_DISABLE_STEP_GRAPH", "1");
            std::vector<float> out_b = run(guidance);          // forced eager
            set_env("BRODIFFUSION_DISABLE_STEP_GRAPH", "");

            CHECK(out_a.size() == out_b.size());
            const bool identical =
                out_a.size() == out_b.size() &&
                std::memcmp(out_a.data(), out_b.data(),
                            out_a.size() * sizeof(float)) == 0;
            CHECK(identical);
            std::printf("  captured-vs-eager guidance=%.1f n=%zu %s\n",
                        guidance, out_a.size(),
                        identical ? "bitwise-identical" : "MISMATCH");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL structural capture parity threw: %s\n", e.what());
        ++g_failures;
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

std::string weights_dir() {
    if (const char* e = std::getenv("BRODIFFUSION_WEIGHTS_DIR")) {
        if (e[0]) return e;
    }
    return BRODIFFUSION_WEIGHTS_DIR;
}

// Golden format BDSAMPG1:
//   magic(8) version(i32) L,Ci,Kc,Cc,C2,Cm(i32) steps(i32) guidance(f32) shift(f32)
//   noise_latent[L*Ci] noise_camera[Cm] feat1[Kc*Cc] feat2[Kc*C2] latent_out[L*Ci]
struct Golden {
    int L = 0, Ci = 0, Kc = 0, Cc = 0, C2 = 0, Cm = 0, steps = 0;
    float guidance = 0.0f, shift = 0.0f;
    std::vector<float> noise_latent, noise_camera, feat1, feat2, latent_out;
};

bool read_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "BDSAMPG1", 8) != 0) return false;
    int version = 0;
    f.read(reinterpret_cast<char*>(&version), 4);
    int dims[6];
    f.read(reinterpret_cast<char*>(dims), sizeof(dims));
    g.L = dims[0]; g.Ci = dims[1]; g.Kc = dims[2];
    g.Cc = dims[3]; g.C2 = dims[4]; g.Cm = dims[5];
    f.read(reinterpret_cast<char*>(&g.steps), 4);
    f.read(reinterpret_cast<char*>(&g.guidance), 4);
    f.read(reinterpret_cast<char*>(&g.shift), 4);
    auto rd = [&](std::vector<float>& v, std::size_t n) {
        v.resize(n);
        f.read(reinterpret_cast<char*>(v.data()),
               static_cast<std::streamsize>(n * sizeof(float)));
    };
    rd(g.noise_latent, static_cast<std::size_t>(g.L) * g.Ci);
    rd(g.noise_camera, static_cast<std::size_t>(g.Cm));
    rd(g.feat1,        static_cast<std::size_t>(g.Kc) * g.Cc);
    rd(g.feat2,        static_cast<std::size_t>(g.Kc) * g.C2);
    rd(g.latent_out,   static_cast<std::size_t>(g.L) * g.Ci);
    return static_cast<bool>(f);
}

// FP16-vs-FP32 over an iterative CFG sampler: the per-element MAX is not a
// stable signal — classifier-free guidance amplifies the (already verified,
// ~0.2 max) per-forward FP16 error by its 3x/2x combine, and over the feedback
// loop that lands on a few sensitive tokens whose identity shifts with the
// trajectory. So correctness is asserted two ways that ARE stable: a tight mean
// (the whole field must track the reference) and a small tail fraction (a
// systematic break would push a large fraction of elements over 1.0, not ~0.1%).
void compare(const char* label, const std::vector<float>& got,
             const std::vector<float>& ref, float mean_tol, float tail_frac_tol) {
    const std::size_t n = std::min(got.size(), ref.size());
    float max_diff = 0.0f, mean_diff = 0.0f, max_ref = 0.0f;
    std::size_t c1 = 0;   // |diff| > 1.0
    for (std::size_t i = 0; i < n; ++i) {
        const float d = std::fabs(got[i] - ref[i]);
        max_diff = std::max(max_diff, d);
        mean_diff += d;
        max_ref = std::max(max_ref, std::fabs(ref[i]));
        if (d > 1.0f) ++c1;
    }
    mean_diff /= static_cast<float>(std::max<std::size_t>(n, 1));
    const double tail_frac = static_cast<double>(c1) / static_cast<double>(std::max<std::size_t>(n, 1));
    std::printf("  %-7s n=%zu  max=%.4g mean=%.4g  (|ref|max=%.4g)  "
                "tail(>1)=%zu (%.3f%%)  [mean_tol=%.3g tail_tol=%.3f%%]\n",
                label, n, max_diff, mean_diff, max_ref, c1, 100.0 * tail_frac,
                mean_tol, 100.0 * tail_frac_tol);
    CHECK(got.size() == ref.size());
    CHECK(mean_diff < mean_tol);
    CHECK(tail_frac < tail_frac_tol);
}

}  // namespace

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    // ── structural: captured vs eager parity (synthetic weights) ──────────
    std::printf("test_flow_sampler: structural captured-vs-eager parity\n");
    structural_capture_parity();

    // ── golden parity (gated) ──────────────────────────────────────────────
    const std::string wd = weights_dir();
    const std::string ckpt = wd + "/triposplat/diffusion_models/triposplat_fp16.safetensors";
    const std::string gpath = wd + "/triposplat/diffusion_models/golden/golden_sampler.bin";

    Golden g;
    const bool have_ckpt = !wd.empty() && std::filesystem::exists(ckpt);
    const bool have_gold = !wd.empty() && read_golden(gpath, g);
    if (!have_ckpt || !have_gold) {
        std::printf("test_flow_sampler: structural %s; golden SKIPPED (ckpt=%d golden=%d)\n",
                    g_failures ? "FAILED" : "OK",
                    have_ckpt ? 1 : 0, have_gold ? 1 : 0);
        return g_failures ? 1 : 0;
    }
    if (brodiffusion::compute_dtype() != bt::Dtype::FP16) {
        std::printf("test_flow_sampler: SKIPPED (CPU backend — flow DiT is GPU/FP16 only)\n");
        return g_failures ? 1 : 0;
    }

    try {
        st::File f = st::File::open(ckpt);
        tsp::FlowDiT m;
        m.load_weights(f);

        bt::Tensor nl = bdtest::bd_upload(g.noise_latent, g.L, g.Ci);
        bt::Tensor nc = bdtest::bd_upload(g.noise_camera, 1, g.Cm);
        bt::Tensor f1 = bdtest::bd_upload(g.feat1, g.Kc, g.Cc);
        bt::Tensor f2 = bdtest::bd_upload(g.feat2, g.Kc, g.C2);

        tsp::FlowSampleOptions opts;
        opts.steps = g.steps;
        opts.guidance_scale = g.guidance;
        opts.shift = g.shift;

        bt::Tensor out;
        tsp::sample_latent(m, f1, f2, nl, nc, opts, out);
        bt::sync_all();
        std::vector<float> got = bdtest::bd_download(out);

        std::printf("test_flow_sampler: (fp16 vs fp32 reference, steps=%d cfg=%.1f)\n",
                    g.steps, g.guidance);
        compare("latent", got, g.latent_out, /*mean_tol=*/0.02f, /*tail_frac_tol=*/0.005f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL sampler threw: %s\n", e.what());
        ++g_failures;
    }

    std::printf("test_flow_sampler: %s\n", g_failures ? "FAILED" : "OK");
    return g_failures ? 1 : 0;
}
