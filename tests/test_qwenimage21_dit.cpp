// Qwen-Image 2.1 image DiT (dit::QwenImage21Transformer2DModel) smoke +
// real-weights test.
//
// Part 1 (synthetic, always runs): a scaled-down but architecturally complete
// checkpoint — 2 blocks, 2 heads of 64, context 128, axes_dims_rope {8,28,28} —
// small enough for an in-memory fixture. Checks output shape / all-finite /
// determinism, that a CACHED second step reproduces a cache-free full prefill
// of the same step (the prefix KV cache's whole contract), and that the
// general multi-segment entry point agrees with the text-only one.
//
// Part 2 (synthetic): the Denoiser wrapper's NCHW <-> token transpose and its
// per-branch prefix caches.
//
// Part 3 (cooperative cancel): the sharded loader polls should_cancel once per
// block and throws LoadCancelled.
//
// Part 4 (gated on BRODIFFUSION_QI21_DIT_REAL=1 AND the weights): loads the
// real 2-shard 7.1B transformer with INT8 weight-only quantisation (so it fits
// a 24 GB card) and runs one forward. Numerical parity vs diffusers is checked
// separately by scripts/qwenimage21_dit_parity.sh, not by ctest.

#define _CRT_SECURE_NO_WARNINGS

#include "brodiffusion/dit/qwenimage21.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace bt = brotensor;
namespace st = brotensor::safetensors;
namespace qd = brodiffusion::dit;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

namespace {

struct Builder {
    std::string entries;
    std::vector<uint8_t> payload;
    bool first = true;
    void add(const std::string& name, std::vector<int> shape,
             const std::vector<float>& f32) {
        std::size_t expected = 1;
        for (int d : shape) expected *= static_cast<std::size_t>(d);
        if (expected != f32.size()) std::abort();
        std::uint64_t start = payload.size();
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(f32.data());
        payload.insert(payload.end(), bytes, bytes + f32.size() * 4);
        std::uint64_t end = payload.size();
        if (!first) entries += ",";
        first = false;
        entries += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i) entries += ",";
            entries += std::to_string(shape[i]);
        }
        entries += "],\"data_offsets\":[" + std::to_string(start) + "," +
                   std::to_string(end) + "]}";
    }
    void write(const std::filesystem::path& path) const {
        std::string header = "{" + entries + "}";
        std::uint64_t hdr = header.size();
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(&hdr), 8);
        f.write(header.data(), header.size());
        f.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
    }
};

// Deterministic small values clustered near zero.
std::vector<float> rnd(std::size_t n, std::size_t salt) {
    std::vector<float> out(n);
    std::uint32_t s = static_cast<std::uint32_t>(salt * 2654435761u + 12345u);
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        out[i] = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.2f;
    }
    return out;
}

void emit_lin(Builder& b, const std::string& p, int out, int in) {
    b.add(p + ".weight", {out, in},
          rnd(static_cast<std::size_t>(out) * in, p.size() + out));
}
void emit_norm(Builder& b, const std::string& p, int dim) {
    b.add(p + ".weight", {dim}, rnd(static_cast<std::size_t>(dim), p.size() + 3));
}

qd::QwenImage21Config synth_cfg() {
    qd::QwenImage21Config c;
    c.patch_size = 1;
    c.in_channels = 64;
    c.out_channels = 64;
    c.num_layers = 2;
    c.attention_head_dim = 64;
    c.num_attention_heads = 2;      // hidden = 128
    c.context_in_dim = 128;
    c.mlp_ratio = 3;
    c.axes_dims_rope = {8, 28, 28};
    c.eps = 1e-6f;
    c.causal_condition = true;
    return c;
}

void build_fixture(Builder& b, const qd::QwenImage21Config& c) {
    const int H = c.hidden_size();
    const int MH = c.mlp_hidden_size();
    emit_lin(b, "img_in", H, c.in_channels);
    emit_lin(b, "time_text_embed.timestep_embedder.linear_1", H,
             c.timestep_embed_dim);
    emit_lin(b, "time_text_embed.timestep_embedder.linear_2", H, H);
    emit_lin(b, "modulation.1", 4 * H, H);
    emit_norm(b, "txt_in.text_norm", c.context_in_dim);
    emit_lin(b, "txt_in.in_layer", H, c.context_in_dim);
    emit_lin(b, "txt_in.out_layer", H, H);
    for (int i = 0; i < c.num_layers; ++i) {
        const std::string p = "transformer_blocks." + std::to_string(i) + ".";
        emit_lin(b, p + "attn.to_q", H, H);
        emit_lin(b, p + "attn.to_k", H, H);
        emit_lin(b, p + "attn.to_v", H, H);
        emit_lin(b, p + "attn.to_out.0", H, H);
        emit_norm(b, p + "attn.norm_q", c.attention_head_dim);
        emit_norm(b, p + "attn.norm_k", c.attention_head_dim);
        emit_lin(b, p + "img_mlp.gate_layer", MH, H);
        emit_lin(b, p + "img_mlp.proj", MH, H);
        emit_lin(b, p + "img_mlp.out", H, MH);
    }
    emit_lin(b, "norm_out.linear", H, H);
    emit_lin(b, "proj_out", c.out_channels, H);
}

std::vector<float> download_any(const bt::Tensor& t) {
    if (t.dtype == bt::Dtype::BF16) {
        bt::Tensor f32;
        bt::cast(t, f32, bt::Dtype::FP32);
        return bdtest::bd_download(f32);
    }
    return bdtest::bd_download(t);
}

// Max |a-b| / (1 + max|ref|).
double rel_maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 1e9;
    double mx = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        mx = std::max(mx, std::abs(static_cast<double>(a[i]) - b[i]));
        scale = std::max(scale, std::abs(static_cast<double>(a[i])));
    }
    return mx / (1.0 + scale);
}

std::filesystem::path write_fixture(const qd::QwenImage21Config& cfg,
                                    const char* stem) {
    Builder b;
    build_fixture(b, cfg);
    auto path = std::filesystem::temp_directory_path() / stem;
    b.write(path);
    return path;
}

}  // namespace

// ── shape / finite / determinism, and the prefix cache's contract ──────────
static void test_synthetic() {
    qd::QwenImage21Config cfg = synth_cfg();
    auto path = write_fixture(cfg, "brodiffusion_qi21_dit_test.safetensors");
    auto file = st::File::open(path.string());
    qd::QwenImage21Transformer2DModel model(cfg);
    model.load_weights(file, "");

    const int hp = 6, wp = 4, text_seq = 5;
    const int img_len = hp * wp;
    std::vector<float> lat_h =
        rnd(static_cast<std::size_t>(img_len) * cfg.in_channels, 1001);
    std::vector<float> emb_h =
        rnd(static_cast<std::size_t>(text_seq) * cfg.context_in_dim, 1002);

    bt::Tensor lat = bdtest::bd_upload(lat_h, img_len, cfg.in_channels);
    bt::Tensor emb = bdtest::bd_upload(emb_h, text_seq, cfg.context_in_dim);

    bt::Tensor txt;
    model.encode_text(emb, txt);
    bt::sync_all();
    CHECK(txt.rows == text_seq);
    CHECK(txt.cols == cfg.hidden_size());
    CHECK(txt.dtype == model.compute_dtype());

    // Step 0 with no cache at all.
    bt::Tensor out;
    model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
    bt::sync_all();
    CHECK(out.rows == img_len);
    CHECK(out.cols == cfg.out_channels);
    CHECK(out.dtype == model.compute_dtype());
    std::vector<float> v1 = download_any(out);
    int nonfinite = 0;
    for (float v : v1) if (!std::isfinite(v)) ++nonfinite;
    CHECK(nonfinite == 0);

    // Determinism.
    model.forward(lat, hp, wp, txt, 0.7f, nullptr, out);
    bt::sync_all();
    CHECK(v1 == download_any(out));

    // An "extract" step must equal the cache-free step exactly (same code
    // path plus a snapshot).
    qd::QwenImage21PrefixCache cache;
    model.forward(lat, hp, wp, txt, 0.7f, &cache, out);
    bt::sync_all();
    CHECK(cache.valid());
    CHECK(cache.prefix_len() == text_seq);
    CHECK(cache.target_hp() == hp && cache.target_wp() == wp);
    CHECK(v1 == download_any(out));

    // A "cached" step must reproduce the cache-free full prefill of the same
    // timestep — the whole point of the cache.
    model.forward(lat, hp, wp, txt, 0.4f, &cache, out);
    bt::sync_all();
    std::vector<float> cached = download_any(out);
    bt::Tensor out_ref;
    model.forward(lat, hp, wp, txt, 0.4f, nullptr, out_ref);
    bt::sync_all();
    std::vector<float> full = download_any(out_ref);
    const double d = rel_maxdiff(full, cached);
    CHECK(d < 2e-3);
    std::printf("qi21_dit: cached vs full-prefill rel maxdiff %.3e\n", d);

    // A grid change on a live cache is a caller error, not silent garbage.
    bool threw = false;
    try {
        model.forward(lat, wp, hp, txt, 0.4f, &cache, out);
    } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    cache.reset();
    CHECK(!cache.valid());

    // forward() is forward_joint() with a single Text segment — check the
    // general entry point agrees bit-for-bit.
    std::vector<qd::QwenImage21Segment> prefix(1);
    prefix[0].kind = qd::QwenImage21Segment::Kind::Text;
    prefix[0].n_tokens = text_seq;
    model.forward_joint(lat, hp, wp, txt, nullptr, prefix, 0.7f, nullptr, out);
    bt::sync_all();
    CHECK(v1 == download_any(out));

    // Multi-segment prefix (text / condition image / text) — the layout edit
    // mode uses. Only structural checks here; numerical parity for the T2I
    // layout is scripts/qwenimage21_dit_parity.sh.
    {
        const int ch = 2, cw = 2;
        std::vector<qd::QwenImage21Segment> segs(3);
        segs[0].kind = qd::QwenImage21Segment::Kind::Text;
        segs[0].n_tokens = 3;
        segs[1].kind = qd::QwenImage21Segment::Kind::Image;
        segs[1].n_tokens = ch * cw;
        segs[1].h = ch;
        segs[1].w = cw;
        segs[2].kind = qd::QwenImage21Segment::Kind::Text;
        segs[2].n_tokens = text_seq - 3;
        std::vector<float> cond_h =
            rnd(static_cast<std::size_t>(ch) * cw * cfg.in_channels, 1003);
        bt::Tensor cond = bdtest::bd_upload(cond_h, ch * cw, cfg.in_channels);

        bt::Tensor joint_out;
        model.forward_joint(lat, hp, wp, txt, &cond, segs, 0.7f, nullptr,
                            joint_out);
        bt::sync_all();
        CHECK(joint_out.rows == img_len);
        CHECK(joint_out.cols == cfg.out_channels);
        std::vector<float> jv = download_any(joint_out);
        int nf = 0;
        for (float v : jv) if (!std::isfinite(v)) ++nf;
        CHECK(nf == 0);
        // The condition image genuinely participates.
        CHECK(rel_maxdiff(v1, jv) > 1e-4);

        // ... and the cache reproduces it for the multi-segment layout too.
        qd::QwenImage21PrefixCache c2;
        model.forward_joint(lat, hp, wp, txt, &cond, segs, 0.7f, &c2, joint_out);
        bt::sync_all();
        CHECK(c2.prefix_len() == text_seq + ch * cw);
        CHECK(jv == download_any(joint_out));
        model.forward_joint(lat, hp, wp, txt, &cond, segs, 0.4f, &c2, joint_out);
        bt::sync_all();
        std::vector<float> jc = download_any(joint_out);
        model.forward_joint(lat, hp, wp, txt, &cond, segs, 0.4f, nullptr,
                            joint_out);
        bt::sync_all();
        const double d2 = rel_maxdiff(download_any(joint_out), jc);
        CHECK(d2 < 2e-3);
        std::printf("qi21_dit: multi-segment cached rel maxdiff %.3e\n", d2);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ── Denoiser wrapper: NCHW <-> token transpose, per-branch caches ──────────
static void test_denoiser() {
    qd::QwenImage21Config cfg = synth_cfg();
    auto path = write_fixture(cfg, "brodiffusion_qi21_den_test.safetensors");
    auto file = st::File::open(path.string());
    qd::QwenImage21Denoiser den(cfg);
    den.load_weights(file, "");
    CHECK(den.latent_channels() == cfg.in_channels);
    CHECK(den.prediction_type() == brodiffusion::PredictionType::Velocity);
    CHECK(den.uses_cfg());

    const int H_lat = 6, W_lat = 4, text_seq = 5;
    const std::size_t n_lat =
        static_cast<std::size_t>(cfg.in_channels) * H_lat * W_lat;

    brodiffusion::Conditioning cond;
    cond.text_embeddings = bdtest::bd_upload(
        rnd(static_cast<std::size_t>(text_seq) * cfg.context_in_dim, 2001),
        text_seq, cfg.context_in_dim);
    cond.uncond_embeddings = bdtest::bd_upload(
        rnd(static_cast<std::size_t>(text_seq) * cfg.context_in_dim, 2002),
        text_seq, cfg.context_in_dim);
    cond.has_uncond = true;
    brodiffusion::PreparedConditioning prep = den.prepare(cond);
    CHECK(static_cast<bool>(prep));

    bt::Tensor latent = bdtest::bd_upload(rnd(n_lat, 2003), 1,
                                          static_cast<int>(n_lat));
    bt::Tensor out;
    den.forward(latent, H_lat, W_lat, 700.0f, prep,
                brodiffusion::Branch::Cond, out);
    bt::sync_all();
    CHECK(out.rows == 1);
    CHECK(out.cols == static_cast<int>(n_lat));
    std::vector<float> v0 = bdtest::bd_download(out);
    int nonfinite = 0;
    for (float v : v0) if (!std::isfinite(v)) ++nonfinite;
    CHECK(nonfinite == 0);

    // Second step decodes from the cache the first one extracted; it must
    // match a forward off a freshly reset cache (a full prefill).
    den.forward(latent, H_lat, W_lat, 400.0f, prep,
                brodiffusion::Branch::Cond, out);
    bt::sync_all();
    std::vector<float> cached = bdtest::bd_download(out);
    den.reset_cache(prep);
    den.forward(latent, H_lat, W_lat, 400.0f, prep,
                brodiffusion::Branch::Cond, out);
    bt::sync_all();
    CHECK(rel_maxdiff(bdtest::bd_download(out), cached) < 2e-3);

    // The uncond branch has its own prompt and its own cache.
    bt::Tensor uout;
    den.forward(latent, H_lat, W_lat, 700.0f, prep,
                brodiffusion::Branch::Uncond, uout);
    bt::sync_all();
    CHECK(rel_maxdiff(v0, bdtest::bd_download(uout)) > 1e-4);

    // A grid change resets the cache rather than failing.
    den.forward(latent, W_lat, H_lat, 400.0f, prep,
                brodiffusion::Branch::Cond, out);
    bt::sync_all();
    CHECK(out.cols == static_cast<int>(n_lat));

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ── cooperative cancel ─────────────────────────────────────────────────────
static void test_load_cancel() {
    qd::QwenImage21Config cfg = synth_cfg();   // num_layers == 2
    auto path = write_fixture(cfg, "brodiffusion_qi21_cancel_test.safetensors");
    auto file = st::File::open(path.string());
    std::vector<const st::File*> shards{ &file };

    {
        int polls = 0;
        qd::QwenImage21Transformer2DModel model(cfg);
        bool threw = false;
        try {
            model.load_weights(shards, "", [&]() { ++polls; return true; });
        } catch (const brodiffusion::LoadCancelled&) { threw = true; }
        CHECK(threw);
        CHECK(polls == 1);
    }
    {
        int polls = 0;
        qd::QwenImage21Transformer2DModel model(cfg);
        bool threw = false;
        try {
            model.load_weights(shards, "", [&]() { ++polls; return false; });
        } catch (const brodiffusion::LoadCancelled&) { threw = true; }
        CHECK(!threw);
        CHECK(polls == cfg.num_layers);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

#ifndef BRODIFFUSION_WEIGHTS_DIR
#define BRODIFFUSION_WEIGHTS_DIR ""
#endif

static std::string weights_dir() {
    if (const char* e = std::getenv("BRODIFFUSION_WEIGHTS_DIR")) {
        if (e[0]) return e;
    }
    return BRODIFFUSION_WEIGHTS_DIR;
}

static void test_real_weights() {
    const char* opt = std::getenv("BRODIFFUSION_QI21_DIT_REAL");
    if (!opt || !opt[0] || opt[0] == '0') {
        std::printf("qi21_dit: real-weights skipped "
                    "(set BRODIFFUSION_QI21_DIT_REAL=1)\n");
        return;
    }
    const std::string tdir = weights_dir() + "/qwen-image-2.1/transformer";
    const std::string s1 =
        tdir + "/diffusion_pytorch_model-00001-of-00002.safetensors";
    if (!std::filesystem::exists(s1)) {
        std::printf("qi21_dit: real-weights skipped (no weights)\n");
        return;
    }
    std::vector<st::File> files;
    files.push_back(st::File::open(s1));
    files.push_back(st::File::open(
        tdir + "/diffusion_pytorch_model-00002-of-00002.safetensors"));
    std::vector<const st::File*> shards;
    for (const st::File& f : files) shards.push_back(&f);

    qd::QwenImage21Config cfg;   // real 2.1 config (header defaults)
    // INT8 weight-only: 7.2 GB instead of 14.2, so this fits a 24 GB card.
    cfg.quantize_weights = true;
    qd::QwenImage21Transformer2DModel model(cfg);
    model.load_weights(shards, "");

    const int hp = 16, wp = 16, text_seq = 24;
    const int img_len = hp * wp;
    std::vector<float> lat_h(
        static_cast<std::size_t>(img_len) * cfg.in_channels);
    for (std::size_t i = 0; i < lat_h.size(); ++i) {
        lat_h[i] = std::sin(0.01f * static_cast<float>(i)) * 0.5f;
    }
    std::vector<float> emb_h(
        static_cast<std::size_t>(text_seq) * cfg.context_in_dim);
    for (std::size_t i = 0; i < emb_h.size(); ++i) {
        emb_h[i] = std::sin(0.001f * static_cast<float>(i)) * 0.5f;
    }
    bt::Tensor lat = bdtest::bd_upload(lat_h, img_len, cfg.in_channels);
    bt::Tensor emb = bdtest::bd_upload(emb_h, text_seq, cfg.context_in_dim);

    bt::Tensor txt, out;
    model.encode_text(emb, txt);
    qd::QwenImage21PrefixCache cache;
    model.forward(lat, hp, wp, txt, 0.7f, &cache, out);
    bt::sync_all();
    CHECK(out.rows == img_len);
    CHECK(out.cols == cfg.out_channels);
    std::vector<float> v = download_any(out);
    int nonfinite = 0;
    double mean = 0.0;
    for (float f : v) { if (!std::isfinite(f)) ++nonfinite; else mean += f; }
    CHECK(nonfinite == 0);
    mean /= static_cast<double>(v.size());
    double var = 0.0;
    for (float f : v) var += (f - mean) * (f - mean);
    std::printf("qi21_dit: real-weights forward OK (mean %.5f std %.5f)\n",
                mean, std::sqrt(var / static_cast<double>(v.size())));

    // And the cached step runs on the real weights too.
    model.forward(lat, hp, wp, txt, 0.4f, &cache, out);
    bt::sync_all();
    std::vector<float> v2 = download_any(out);
    nonfinite = 0;
    for (float f : v2) if (!std::isfinite(f)) ++nonfinite;
    CHECK(nonfinite == 0);
}

int main() {
    try { bt::init(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what()); return 1;
    }
    try { test_synthetic(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "qi21_dit: synthetic exception: %s\n", e.what());
        return 1;
    }
    try { test_denoiser(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "qi21_dit: denoiser exception: %s\n", e.what());
        return 1;
    }
    try { test_load_cancel(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "qi21_dit: load-cancel exception: %s\n", e.what());
        return 1;
    }
    try { test_real_weights(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "qi21_dit: real-weights exception: %s\n", e.what());
        return 1;
    }
    if (g_failures == 0) std::printf("qi21_dit: OK\n");
    else std::fprintf(stderr, "qi21_dit: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
