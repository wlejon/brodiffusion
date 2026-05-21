// T5 encoder smoke test.
//
// Builds a tiny T5 encoder (d_model=8, d_kv=4, num_heads=2, d_ff=16,
// num_layers=2, vocab=20, num_buckets=8, max_distance=16), synthesizes a
// safetensors fixture with the HF tensor names, loads it, and runs forward.
// Verifies output shape/dtype, that all outputs are finite (no NaN/Inf), and
// that two consecutive forwards produce identical results.
//
// Numerical accuracy against a reference is intentionally not checked here —
// that needs real T5 weights and lives in a future integration test.

#include "brodiffusion/t5.h"
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

namespace t5 = brodiffusion::t5;
namespace st = brotensor::safetensors;
namespace bt = brotensor;

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
            std::fprintf(stderr, "fixture: shape/data mismatch for %s\n",
                         name.c_str());
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

std::vector<uint16_t> fp16_ones(std::size_t n) {
    return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(1.0f));
}
// Small deterministic values: scale, 2*scale, 3*scale, ... so rows/cols differ.
std::vector<uint16_t> fp16_seq(std::size_t n, float scale) {
    std::vector<uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = bt::fp32_to_fp16_bits((static_cast<float>(i) + 1.0f) * scale);
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

    t5::T5Config cfg;
    cfg.vocab_size = 20;
    cfg.d_model    = 8;
    cfg.d_kv       = 4;
    cfg.num_heads  = 2;
    cfg.d_ff       = 16;
    cfg.num_layers = 2;
    cfg.relative_attention_num_buckets  = 8;
    cfg.relative_attention_max_distance = 16;
    cfg.layer_norm_eps = 1e-6f;

    const int V  = cfg.vocab_size;
    const int D  = cfg.d_model;
    const int FF = cfg.d_ff;
    const int NB = cfg.relative_attention_num_buckets;
    const int H  = cfg.num_heads;

    Builder b;
    const std::string prefix = "";  // standalone T5EncoderModel export

    b.add(prefix + "shared.weight", {V, D},
          fp16_seq(static_cast<std::size_t>(V) * D, 0.03f));

    auto WDD = [&](float s) { return fp16_seq(static_cast<std::size_t>(D) * D, s); };

    for (int i = 0; i < cfg.num_layers; ++i) {
        const std::string p =
            prefix + "encoder.block." + std::to_string(i) + ".";

        b.add(p + "layer.0.layer_norm.weight", {D}, fp16_ones(D));

        const std::string sa = p + "layer.0.SelfAttention.";
        b.add(sa + "q.weight", {D, D}, WDD(0.02f));
        b.add(sa + "k.weight", {D, D}, WDD(0.03f));
        b.add(sa + "v.weight", {D, D}, WDD(0.04f));
        b.add(sa + "o.weight", {D, D}, WDD(0.05f));

        // relative_attention_bias lives on block 0 only.
        if (i == 0) {
            b.add(sa + "relative_attention_bias.weight", {NB, H},
                  fp16_seq(static_cast<std::size_t>(NB) * H, 0.01f));
        }

        b.add(p + "layer.1.layer_norm.weight", {D}, fp16_ones(D));

        const std::string dr = p + "layer.1.DenseReluDense.";
        b.add(dr + "wi_0.weight", {FF, D},
              fp16_seq(static_cast<std::size_t>(FF) * D, 0.01f));
        b.add(dr + "wi_1.weight", {FF, D},
              fp16_seq(static_cast<std::size_t>(FF) * D, 0.015f));
        b.add(dr + "wo.weight", {D, FF},
              fp16_seq(static_cast<std::size_t>(D) * FF, 0.01f));
    }

    b.add(prefix + "encoder.final_layer_norm.weight", {D}, fp16_ones(D));

    auto path =
        std::filesystem::temp_directory_path() / "brodiffusion_t5_test.safetensors";
    b.write(path);

    std::vector<float> out_vals1, out_vals2;
    {
        auto file = st::File::open(path.string());
        t5::TextEncoder enc(cfg);
        enc.load_weights(file, "");

        const int L = 5;
        std::vector<int32_t> ids = {1, 2, 3, 4, 0};
        bt::Tensor out;
        enc.forward(ids.data(), L, out);
        bt::sync_all();

        CHECK(out.rows == L);
        CHECK(out.cols == D);
        CHECK(out.dtype == brodiffusion::compute_dtype());

        out_vals1 = bdtest::bd_download(out);
        int nonfinite = 0;
        for (float v : out_vals1) if (!bdtest::bd_finite(v)) ++nonfinite;
        CHECK(nonfinite == 0);

        // Second forward — must be identical for a deterministic graph.
        enc.forward(ids.data(), L, out);
        bt::sync_all();
        out_vals2 = bdtest::bd_download(out);
        CHECK(out_vals1 == out_vals2);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("t5: OK\n");
    else std::fprintf(stderr, "t5: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
