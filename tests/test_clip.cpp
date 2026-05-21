// CLIP text encoder smoke test.
//
// Builds a tiny encoder (vocab=8, max_pos=4, dim=4, heads=1, layers=1,
// ffn=8), synthesizes a safetensors fixture with the HF tensor names, loads
// it, and runs forward. Verifies output shape/dtype, that all FP16 outputs
// are finite (no NaN/Inf bit patterns), and that two consecutive forwards
// produce byte-identical results.
//
// Numerical accuracy against a reference is intentionally not checked here —
// that needs real CLIP weights and lives in a future integration test.

#include "brodiffusion/clip.h"
#include "brodiffusion/detail/compute.h"
#include "brotensor/safetensors.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace clip = brodiffusion::clip;
namespace st   = brotensor::safetensors;
namespace bt   = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// ─── safetensors fixture builder ───────────────────────────────────────────

namespace {

struct Builder {
    std::string entries;
    std::vector<uint8_t> payload;
    bool first = true;

    void add(const std::string& name, std::vector<int> shape,
             const std::vector<uint16_t>& fp16_bits) {
        std::size_t expected = 1;
        for (int d : shape) expected *= static_cast<std::size_t>(d);
        if (expected != fp16_bits.size()) {
            std::fprintf(stderr, "fixture: shape/data mismatch for %s\n", name.c_str());
            std::abort();
        }
        std::uint64_t start = payload.size();
        const std::uint8_t* bytes =
            reinterpret_cast<const std::uint8_t*>(fp16_bits.data());
        payload.insert(payload.end(), bytes, bytes + fp16_bits.size() * 2);
        std::uint64_t end = payload.size();

        if (!first) entries += ",";
        first = false;
        entries += "\"" + name + "\":{\"dtype\":\"F16\",\"shape\":[";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i) entries += ",";
            entries += std::to_string(shape[i]);
        }
        entries += "],\"data_offsets\":[" + std::to_string(start) + "," +
                   std::to_string(end) + "]}";
    }

    void write(const std::filesystem::path& path) const {
        std::string header = "{" + entries + "}";
        std::uint64_t hdr_size = header.size();
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) std::abort();
        f.write(reinterpret_cast<const char*>(&hdr_size), 8);
        f.write(header.data(), header.size());
        f.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
    }
};

// FP16 vector helpers.
std::vector<uint16_t> fp16_zeros(std::size_t n) {
    return std::vector<uint16_t>(n, 0);
}
std::vector<uint16_t> fp16_ones(std::size_t n) {
    return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(1.0f));
}
// Small deterministic values: 0.01, 0.02, 0.03, ... so each row/col differs.
std::vector<uint16_t> fp16_seq(std::size_t n, float scale = 0.01f, float offset = 0.0f) {
    std::vector<uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = bt::fp32_to_fp16_bits(offset + (static_cast<float>(i) + 1.0f) * scale);
    }
    return out;
}

}  // namespace

// ─── test ──────────────────────────────────────────────────────────────────

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }
    clip::TextEncoderConfig cfg;
    cfg.vocab_size       = 8;
    cfg.max_position     = 4;
    cfg.hidden_dim       = 4;
    cfg.num_heads        = 1;
    cfg.num_layers       = 1;
    cfg.intermediate_dim = 8;
    cfg.layer_norm_eps   = 1e-5f;

    const int V = cfg.vocab_size;
    const int P = cfg.max_position;
    const int D = cfg.hidden_dim;
    const int F = cfg.intermediate_dim;

    Builder b;
    const std::string p = "text_model.";

    b.add(p + "embeddings.token_embedding.weight",
          {V, D}, fp16_seq(static_cast<std::size_t>(V) * D, 0.05f));
    b.add(p + "embeddings.position_embedding.weight",
          {P, D}, fp16_seq(static_cast<std::size_t>(P) * D, 0.05f));

    const std::string lp = p + "encoder.layers.0.";
    b.add(lp + "layer_norm1.weight", {D}, fp16_ones(D));
    b.add(lp + "layer_norm1.bias",   {D}, fp16_zeros(D));

    // Attention projections: small varied values so output isn't trivially zero.
    auto W = [&](float s) { return fp16_seq(static_cast<std::size_t>(D) * D, s); };
    b.add(lp + "self_attn.q_proj.weight",   {D, D}, W(0.02f));
    b.add(lp + "self_attn.q_proj.bias",     {D},    fp16_zeros(D));
    b.add(lp + "self_attn.k_proj.weight",   {D, D}, W(0.03f));
    b.add(lp + "self_attn.k_proj.bias",     {D},    fp16_zeros(D));
    b.add(lp + "self_attn.v_proj.weight",   {D, D}, W(0.04f));
    b.add(lp + "self_attn.v_proj.bias",     {D},    fp16_zeros(D));
    b.add(lp + "self_attn.out_proj.weight", {D, D}, W(0.05f));
    b.add(lp + "self_attn.out_proj.bias",   {D},    fp16_zeros(D));

    b.add(lp + "layer_norm2.weight", {D}, fp16_ones(D));
    b.add(lp + "layer_norm2.bias",   {D}, fp16_zeros(D));

    b.add(lp + "mlp.fc1.weight", {F, D}, fp16_seq(static_cast<std::size_t>(F) * D, 0.01f));
    b.add(lp + "mlp.fc1.bias",   {F},    fp16_zeros(F));
    b.add(lp + "mlp.fc2.weight", {D, F}, fp16_seq(static_cast<std::size_t>(D) * F, 0.01f));
    b.add(lp + "mlp.fc2.bias",   {D},    fp16_zeros(D));

    b.add(p + "final_layer_norm.weight", {D}, fp16_ones(D));
    b.add(p + "final_layer_norm.bias",   {D}, fp16_zeros(D));

    auto path = std::filesystem::temp_directory_path() / "brodiffusion_clip_test.safetensors";
    b.write(path);

    std::vector<float> out_vals1, out_vals2;
    {
        auto file = st::File::open(path.string());
        clip::TextEncoder enc(cfg);
        enc.load_weights(file, "text_model.");

        std::vector<int32_t> ids = {1, 2, 3, 0};
        bt::Tensor out;
        enc.forward(ids.data(), out);
        bt::sync_all();

        CHECK(out.rows == P);
        CHECK(out.cols == D);
        CHECK(out.dtype == brodiffusion::compute_dtype());

        out_vals1 = bdtest::bd_download(out);

        int nonfinite = 0;
        for (float v : out_vals1) if (!bdtest::bd_finite(v)) ++nonfinite;
        CHECK(nonfinite == 0);

        // Second forward — must be bitwise-identical for a deterministic graph.
        enc.forward(ids.data(), out);
        bt::sync_all();
        out_vals2 = bdtest::bd_download(out);
        CHECK(out_vals1 == out_vals2);

        // Pooled output: no text_projection.weight in this fixture, so the
        // pooled vector is the raw final hidden state at the EOS position.
        // ids = {1,2,3,0}; with eos_token_id pointing at id 3 (index 2) we
        // get a deterministic (1, D) finite vector matching out row 2.
        clip::TextEncoderConfig cfg_pool = cfg;
        cfg_pool.eos_token_id = 3;
        clip::TextEncoder enc_pool(cfg_pool);
        enc_pool.load_weights(file, "text_model.");
        bt::Tensor out_pool, pooled;
        enc_pool.forward(ids.data(), out_pool, &pooled);
        bt::sync_all();
        CHECK(pooled.rows == 1);
        CHECK(pooled.cols == D);
        CHECK(pooled.dtype == brodiffusion::compute_dtype());
        std::vector<float> pooled_vals = bdtest::bd_download(pooled);
        int pooled_nonfinite = 0;
        for (float v : pooled_vals) if (!bdtest::bd_finite(v)) ++pooled_nonfinite;
        CHECK(pooled_nonfinite == 0);
        // Pooled vector must equal row 2 (EOS index) of the hidden states.
        std::vector<float> hidden = bdtest::bd_download(out_pool);
        for (int j = 0; j < D; ++j) {
            CHECK(pooled_vals[static_cast<std::size_t>(j)] ==
                  hidden[static_cast<std::size_t>(2 * D + j)]);
        }
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("clip: OK\n");
    else std::fprintf(stderr, "clip: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
