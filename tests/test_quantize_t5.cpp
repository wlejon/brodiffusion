// T5 encoder INT8 (W8A16) weight-only quantization correctness test.
//
// Builds the same tiny T5 encoder used by test_t5 (d_model=8, d_kv=4,
// num_heads=2, d_ff=16, num_layers=2), synthesizes a safetensors fixture with
// small random weights, and runs a forward pass twice:
//   1) FP16 baseline                  (T5Config::quantize_weights = false)
//   2) INT8 weight-only quantised T5   (T5Config::quantize_weights = true)
// then checks the INT8 output is finite and within a bounded max-abs error of
// the FP16 baseline.
//
// The INT8 path quantises all seven weight matmuls per block: the four
// attention projections (Wq/Wk/Wv/Wo, exercised through brotensor's
// self_attention_bias_int8w_fp16) and the three FFN linears (wi_0/wi_1/wo,
// through linear_forward_batched_int8w_fp16). INT8 (W8A16) quantization is
// GPU-only, so the test is a no-op on the CPU backend.

#include "brodiffusion/t5.h"
#include "brotensor/safetensors.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
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

// ─── safetensors fixture builder (FP16 tensors) ────────────────────────────

namespace {

struct Builder {
    std::string entries;
    std::vector<uint8_t> payload;
    bool first = true;

    void add(const std::string& name, std::vector<int> shape,
             const std::vector<uint16_t>& fp16_bits) {
        std::size_t expected = 1;
        for (int d : shape) expected *= static_cast<std::size_t>(d);
        if (expected != fp16_bits.size()) std::abort();
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
// Deterministic small Gaussian weights — enough per-row spread for INT8
// quantisation to produce non-trivial scales without saturating FP16.
std::vector<uint16_t> fp16_rand(std::size_t n, float scale, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = bt::fp32_to_fp16_bits(scale * nrm(rng));
    }
    return out;
}

t5::T5Config tiny_config() {
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
    return cfg;
}

void write_fixture(const std::filesystem::path& path, const t5::T5Config& cfg) {
    const int V  = cfg.vocab_size;
    const int D  = cfg.d_model;
    const int FF = cfg.d_ff;
    const int NB = cfg.relative_attention_num_buckets;
    const int H  = cfg.num_heads;

    Builder b;
    b.add("shared.weight", {V, D},
          fp16_rand(static_cast<std::size_t>(V) * D, 0.05f, 1));

    for (int i = 0; i < cfg.num_layers; ++i) {
        const std::string p = "encoder.block." + std::to_string(i) + ".";
        const auto seed = static_cast<std::uint64_t>(100 + i * 20);

        b.add(p + "layer.0.layer_norm.weight", {D}, fp16_ones(D));
        const std::string sa = p + "layer.0.SelfAttention.";
        b.add(sa + "q.weight", {D, D},
              fp16_rand(static_cast<std::size_t>(D) * D, 0.06f, seed + 0));
        b.add(sa + "k.weight", {D, D},
              fp16_rand(static_cast<std::size_t>(D) * D, 0.06f, seed + 1));
        b.add(sa + "v.weight", {D, D},
              fp16_rand(static_cast<std::size_t>(D) * D, 0.06f, seed + 2));
        b.add(sa + "o.weight", {D, D},
              fp16_rand(static_cast<std::size_t>(D) * D, 0.06f, seed + 3));
        if (i == 0) {
            b.add(sa + "relative_attention_bias.weight", {NB, H},
                  fp16_rand(static_cast<std::size_t>(NB) * H, 0.02f, seed + 4));
        }
        b.add(p + "layer.1.layer_norm.weight", {D}, fp16_ones(D));
        const std::string dr = p + "layer.1.DenseReluDense.";
        b.add(dr + "wi_0.weight", {FF, D},
              fp16_rand(static_cast<std::size_t>(FF) * D, 0.05f, seed + 5));
        b.add(dr + "wi_1.weight", {FF, D},
              fp16_rand(static_cast<std::size_t>(FF) * D, 0.05f, seed + 6));
        b.add(dr + "wo.weight", {D, FF},
              fp16_rand(static_cast<std::size_t>(D) * FF, 0.05f, seed + 7));
    }
    b.add("encoder.final_layer_norm.weight", {D}, fp16_ones(D));
    b.write(path);
}

std::vector<float> run_forward(const st::File& file, const t5::T5Config& cfg,
                               const std::vector<std::int32_t>& ids) {
    t5::TextEncoder enc(cfg);
    enc.load_weights(file, "");
    bt::Tensor out;
    enc.forward(ids.data(), static_cast<int>(ids.size()), out);
    bt::sync_all();
    const std::size_t n = static_cast<std::size_t>(out.size());
    if (out.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        out.copy_to_host_fp16(bits.data());
        bt::sync_all();
        std::vector<float> vals(n);
        for (std::size_t i = 0; i < n; ++i) {
            vals[i] = bt::fp16_bits_to_fp32(bits[i]);
        }
        return vals;
    }
    return out.to_host_vector();
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
    if (!bt::is_available(bt::Device::CUDA) &&
        !bt::is_available(bt::Device::Metal)) {
        std::fprintf(stderr,
                     "INT8 quantization is GPU-only — skipping\n");
        return 0;
    }

    const t5::T5Config cfg = tiny_config();
    auto path = std::filesystem::temp_directory_path() /
                "brodiffusion_quantize_t5_test.safetensors";
    write_fixture(path, cfg);

    const std::vector<std::int32_t> ids = {1, 2, 3, 4, 5, 0};

    std::vector<float> out_fp16, out_int8;
    {
        auto file = st::File::open(path.string());
        out_fp16 = run_forward(file, cfg, ids);
    }
    {
        t5::T5Config qcfg = cfg;
        qcfg.quantize_weights = true;
        auto file = st::File::open(path.string());
        out_int8 = run_forward(file, qcfg, ids);
    }

    CHECK(out_fp16.size() == out_int8.size());

    float max_abs_err = 0.0f;
    float max_abs_ref = 0.0f;
    int nonfinite = 0;
    if (out_fp16.size() == out_int8.size()) {
        for (std::size_t i = 0; i < out_fp16.size(); ++i) {
            const float a = out_fp16[i];
            const float b = out_int8[i];
            if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; continue; }
            max_abs_err = std::max(max_abs_err, std::fabs(a - b));
            max_abs_ref = std::max(max_abs_ref, std::fabs(a));
        }
    }
    CHECK(nonfinite == 0);
    std::printf("quantize_t5: max_abs_err=%.4f  max_abs_ref=%.4f\n",
                max_abs_err, max_abs_ref);
    // The output passes through a final RMSNorm, so reference magnitudes sit
    // near 1. INT8 W8A16 error on this synthetic two-layer fixture stays well
    // under 0.1; 0.15 leaves margin for FP16 rounding drift.
    CHECK(max_abs_err < 0.15f);

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("quantize_t5: OK\n");
    else std::fprintf(stderr, "quantize_t5: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
