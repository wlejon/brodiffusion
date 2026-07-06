// Krea 2 image DiT (dit::Krea2Transformer2DModel) smoke + real-weights test.
//
// Part 1 (synthetic, always runs): a scaled-down but architecturally complete
// checkpoint — 2 transformer blocks, 2+2 text-fusion blocks, GQA (4 q / 2 kv
// heads), 3 tapped text layers — small enough for an in-memory fixture. Runs
// one forward on random inputs (with a mid-sequence pad in the text mask) and
// checks output shape, all-finite, and determinism.
//
// Part 2 (gated on BRODIFFUSION_KREA2_DIT_REAL=1 AND the weights): loads the
// real sharded 12.9B transformer and runs one forward. Opt-in because the BF16
// model is ~24 GB — beyond a single 24 GB card — so it must run on a large-VRAM
// or CPU host. Numerical parity vs diffusers is checked separately by
// scripts/krea2_dit_parity.sh (FP32 both sides), not by ctest.

#define _CRT_SECURE_NO_WARNINGS

#include "brodiffusion/dit/krea2.h"
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
namespace kd = brodiffusion::dit;

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

void emit_lin(Builder& b, const std::string& p, int out, int in, bool bias) {
    b.add(p + ".weight", {out, in}, rnd(static_cast<std::size_t>(out) * in, p.size() + out));
    if (bias) b.add(p + ".bias", {out}, rnd(static_cast<std::size_t>(out), p.size() + 7));
}
void emit_norm(Builder& b, const std::string& p, int dim) {
    b.add(p + ".weight", {dim}, rnd(static_cast<std::size_t>(dim), p.size() + 3));
}
void emit_attn(Builder& b, const std::string& p, int dim, int hd, int nq, int nkv) {
    emit_lin(b, p + "to_q", hd * nq, dim, false);
    emit_lin(b, p + "to_k", hd * nkv, dim, false);
    emit_lin(b, p + "to_v", hd * nkv, dim, false);
    emit_lin(b, p + "to_gate", dim, dim, false);
    emit_lin(b, p + "to_out.0", dim, dim, false);
    emit_norm(b, p + "norm_q", hd);
    emit_norm(b, p + "norm_k", hd);
}
void emit_ff(Builder& b, const std::string& p, int dim, int inter) {
    emit_lin(b, p + "gate", inter, dim, false);
    emit_lin(b, p + "up", inter, dim, false);
    emit_lin(b, p + "down", dim, inter, false);
}
void emit_fusion(Builder& b, const std::string& p, int dim, int hd, int nq, int inter) {
    emit_norm(b, p + "norm1", dim);
    emit_norm(b, p + "norm2", dim);
    emit_attn(b, p + "attn.", dim, hd, nq, nq);
    emit_ff(b, p + "ff.", dim, inter);
}

kd::Krea2Config synth_cfg() {
    kd::Krea2Config c;
    c.in_channels = 64; c.num_layers = 2; c.attention_head_dim = 8;
    c.num_attention_heads = 4; c.num_key_value_heads = 2; c.intermediate_size = 64;
    c.timestep_embed_dim = 16; c.text_hidden_dim = 24; c.num_text_layers = 3;
    c.text_num_attention_heads = 2; c.text_num_key_value_heads = 2;
    c.text_intermediate_size = 48; c.num_layerwise_text_blocks = 2;
    c.num_refiner_text_blocks = 2; c.axes_dims_rope = {4, 2, 2};
    c.rope_theta = 1000.0f; c.norm_eps = 1e-5f;
    return c;
}

void build_fixture(Builder& b, const kd::Krea2Config& c) {
    const int H = c.hidden_size();
    const int hd_txt = c.text_hidden_dim / c.text_num_attention_heads;
    emit_lin(b, "img_in", H, c.in_channels, true);
    emit_lin(b, "time_embed.linear_1", H, c.timestep_embed_dim, true);
    emit_lin(b, "time_embed.linear_2", H, H, true);
    emit_lin(b, "time_mod_proj", 6 * H, H, true);
    for (int i = 0; i < c.num_layerwise_text_blocks; ++i)
        emit_fusion(b, "text_fusion.layerwise_blocks." + std::to_string(i) + ".",
                    c.text_hidden_dim, hd_txt, c.text_num_attention_heads,
                    c.text_intermediate_size);
    b.add("text_fusion.projector.weight", {1, c.num_text_layers},
          rnd(static_cast<std::size_t>(c.num_text_layers), 99));
    for (int i = 0; i < c.num_refiner_text_blocks; ++i)
        emit_fusion(b, "text_fusion.refiner_blocks." + std::to_string(i) + ".",
                    c.text_hidden_dim, hd_txt, c.text_num_attention_heads,
                    c.text_intermediate_size);
    emit_norm(b, "txt_in.norm", c.text_hidden_dim);
    emit_lin(b, "txt_in.linear_1", H, c.text_hidden_dim, true);
    emit_lin(b, "txt_in.linear_2", H, H, true);
    for (int i = 0; i < c.num_layers; ++i) {
        const std::string p = "transformer_blocks." + std::to_string(i) + ".";
        b.add(p + "scale_shift_table", {6, H}, rnd(static_cast<std::size_t>(6) * H, i + 41));
        emit_norm(b, p + "norm1", H);
        emit_norm(b, p + "norm2", H);
        emit_attn(b, p + "attn.", H, c.attention_head_dim, c.num_attention_heads,
                  c.num_key_value_heads);
        emit_ff(b, p + "ff.", H, c.intermediate_size);
    }
    b.add("final_layer.scale_shift_table", {2, H}, rnd(static_cast<std::size_t>(2) * H, 55));
    emit_norm(b, "final_layer.norm", H);
    emit_lin(b, "final_layer.linear", c.in_channels, H, true);
}

std::vector<float> download_any(const bt::Tensor& t) {
    if (t.dtype == bt::Dtype::BF16) {
        bt::Tensor f32; bt::cast(t, f32, bt::Dtype::FP32); return bdtest::bd_download(f32);
    }
    return bdtest::bd_download(t);
}

}  // namespace

static void test_synthetic() {
    kd::Krea2Config cfg = synth_cfg();
    Builder b;
    build_fixture(b, cfg);
    auto path = std::filesystem::temp_directory_path() / "brodiffusion_krea2_dit_test.safetensors";
    b.write(path);

    auto file = st::File::open(path.string());
    kd::Krea2Transformer2DModel model(cfg);
    model.load_weights(file, "");

    const int hp = 4, wp = 4, text_seq = 5;
    const int img_len = hp * wp;
    std::vector<float> lat_h = rnd(static_cast<std::size_t>(img_len) * cfg.in_channels, 1001);
    std::vector<float> emb_h =
        rnd(static_cast<std::size_t>(text_seq) * cfg.num_text_layers * cfg.text_hidden_dim, 1002);
    std::vector<float> msk_h = {1.0f, 1.0f, 1.0f, 0.0f, 1.0f};   // mid-sequence pad

    bt::Tensor lat = bdtest::bd_upload(lat_h, img_len, cfg.in_channels);
    bt::Tensor emb = bdtest::bd_upload(emb_h, text_seq * cfg.num_text_layers, cfg.text_hidden_dim);
    bt::Tensor msk = bt::Tensor::from_host(msk_h.data(), text_seq, 1).to(bt::default_device());

    bt::Tensor out;
    model.forward(lat, hp, wp, emb, msk, 0.7f, out);
    bt::sync_all();

    CHECK(out.rows == img_len);
    CHECK(out.cols == cfg.in_channels);
    CHECK(out.dtype == model.compute_dtype());
    std::vector<float> v1 = download_any(out);
    int nonfinite = 0;
    for (float v : v1) if (!std::isfinite(v)) ++nonfinite;
    CHECK(nonfinite == 0);

    model.forward(lat, hp, wp, emb, msk, 0.7f, out);
    bt::sync_all();
    std::vector<float> v2 = download_any(out);
    CHECK(v1 == v2);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// Cooperative-cancel wiring: the sharded load_weights polls should_cancel once
// per transformer block and throws LoadCancelled when it fires. Verifies both
// that a cancel is honored (and stops the load) and that a never-firing hook
// runs to completion — polled exactly num_layers times. Uses the synthetic
// fixture, so it needs no real weights.
static void test_load_cancel() {
    kd::Krea2Config cfg = synth_cfg();   // num_layers == 2
    Builder b;
    build_fixture(b, cfg);
    auto path = std::filesystem::temp_directory_path() / "brodiffusion_krea2_cancel_test.safetensors";
    b.write(path);
    auto file = st::File::open(path.string());
    std::vector<const st::File*> shards{ &file };

    // Cancel immediately: throws before the first block loads (poll #1).
    {
        int polls = 0;
        kd::Krea2Transformer2DModel model(cfg);
        bool threw = false;
        try {
            model.load_weights(shards, "", [&]() { ++polls; return true; });
        } catch (const brodiffusion::LoadCancelled&) { threw = true; }
        CHECK(threw);
        CHECK(polls == 1);
    }

    // Never cancel: completes, and the hook is polled once per block.
    {
        int polls = 0;
        kd::Krea2Transformer2DModel model(cfg);
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
    const char* opt = std::getenv("BRODIFFUSION_KREA2_DIT_REAL");
    if (!opt || !opt[0] || opt[0] == '0') {
        std::printf("krea2_dit: real-weights skipped (set BRODIFFUSION_KREA2_DIT_REAL=1)\n");
        return;
    }
    const std::string tdir = weights_dir() + "/krea-2-raw/transformer";
    const std::string s1 = tdir + "/diffusion_pytorch_model-00001-of-00003.safetensors";
    if (!std::filesystem::exists(s1)) {
        std::printf("krea2_dit: real-weights skipped (no weights)\n");
        return;
    }
    std::vector<st::File> files;
    files.push_back(st::File::open(tdir + "/diffusion_pytorch_model-00001-of-00003.safetensors"));
    files.push_back(st::File::open(tdir + "/diffusion_pytorch_model-00002-of-00003.safetensors"));
    files.push_back(st::File::open(tdir + "/diffusion_pytorch_model-00003-of-00003.safetensors"));
    std::vector<const st::File*> shards;
    for (const st::File& f : files) shards.push_back(&f);

    kd::Krea2Config cfg;   // real Krea 2 config (header defaults)
    kd::Krea2Transformer2DModel model(cfg);
    model.load_weights(shards, "");

    const int hp = 8, wp = 8, text_seq = 512;
    const int img_len = hp * wp;
    std::vector<float> lat_h(static_cast<std::size_t>(img_len) * cfg.in_channels);
    for (std::size_t i = 0; i < lat_h.size(); ++i) lat_h[i] = std::sin(0.01f * i) * 0.5f;
    std::vector<float> emb_h(static_cast<std::size_t>(text_seq) * cfg.num_text_layers * cfg.text_hidden_dim);
    for (std::size_t i = 0; i < emb_h.size(); ++i) emb_h[i] = std::sin(0.001f * i) * 0.5f;
    std::vector<float> msk_h(static_cast<std::size_t>(text_seq), 0.0f);
    for (int i = 0; i < 20; ++i) msk_h[static_cast<std::size_t>(i)] = 1.0f;
    for (int i = 507; i < 512; ++i) msk_h[static_cast<std::size_t>(i)] = 1.0f;

    bt::Tensor lat = bdtest::bd_upload(lat_h, img_len, cfg.in_channels);
    bt::Tensor emb = bdtest::bd_upload(emb_h, text_seq * cfg.num_text_layers, cfg.text_hidden_dim);
    bt::Tensor msk = bt::Tensor::from_host(msk_h.data(), text_seq, 1).to(bt::default_device());

    bt::Tensor out;
    model.forward(lat, hp, wp, emb, msk, 0.7f, out);
    bt::sync_all();
    CHECK(out.rows == img_len);
    CHECK(out.cols == cfg.in_channels);
    std::vector<float> v = download_any(out);
    int nonfinite = 0;
    double mean = 0.0;
    for (float f : v) { if (!std::isfinite(f)) ++nonfinite; else mean += f; }
    CHECK(nonfinite == 0);
    double var = 0.0;
    mean /= static_cast<double>(v.size());
    for (float f : v) var += (f - mean) * (f - mean);
    std::printf("krea2_dit: real-weights forward OK (mean %.5f std %.5f)\n",
                mean, std::sqrt(var / static_cast<double>(v.size())));
}

int main() {
    try { bt::init(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what()); return 1;
    }
    try { test_synthetic(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "krea2_dit: synthetic exception: %s\n", e.what()); return 1;
    }
    try { test_load_cancel(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "krea2_dit: load-cancel exception: %s\n", e.what()); return 1;
    }
    try { test_real_weights(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "krea2_dit: real-weights exception: %s\n", e.what()); return 1;
    }
    if (g_failures == 0) std::printf("krea2_dit: OK\n");
    else std::fprintf(stderr, "krea2_dit: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
