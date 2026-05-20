// LoRA loader / merge tests.
//
// Three units:
//   1. Key translator round-trip: kohya <-> diffusers paths.
//   2. Format detection + alpha defaulting on synthetic safetensors files
//      that mimic both kohya and diffusers/PEFT layouts.
//   3. End-to-end merge math: build a 4x8 FP16 base weight on the GPU,
//      synthesize a rank-2 LoRA, run the same matmul+scale+add the loader
//      uses, and check `W' == W + (alpha/rank) * (up @ down) * user_scale`
//      to within FP16 tolerance.

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/lora_internal.h"
#include "brodiffusion/lora.h"
#include "brodiffusion/safetensors.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace lora = brodiffusion::lora;
namespace st   = brodiffusion::safetensors;
namespace bt   = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

namespace {

// Build a tiny on-disk safetensors fixture with the given entries.
// Each entry is {name, shape, raw bytes, dtype-name}.
struct Entry {
    std::string             name;
    std::vector<int>        shape;
    std::vector<uint8_t>    bytes;
    std::string             dtype;  // "F16" or "F32"
};

std::filesystem::path write_fixture(const std::string& tag,
                                    const std::vector<Entry>& entries) {
    auto path = std::filesystem::temp_directory_path() /
                ("brodiffusion_lora_test_" + tag + ".safetensors");

    std::string header = "{";
    std::vector<uint8_t> payload;
    bool first = true;
    for (const Entry& e : entries) {
        if (!first) header += ",";
        first = false;
        uint64_t s = payload.size();
        payload.insert(payload.end(), e.bytes.begin(), e.bytes.end());
        uint64_t en = payload.size();
        header += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype +
                  "\",\"shape\":[";
        for (std::size_t i = 0; i < e.shape.size(); ++i) {
            if (i) header += ",";
            header += std::to_string(e.shape[i]);
        }
        header += "],\"data_offsets\":[" + std::to_string(s) + "," +
                  std::to_string(en) + "]}";
    }
    header += "}";
    uint64_t hdr_size = header.size();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot create fixture: " + path.string());
    f.write(reinterpret_cast<const char*>(&hdr_size), 8);
    f.write(header.data(), header.size());
    f.write(reinterpret_cast<const char*>(payload.data()),
            static_cast<std::streamsize>(payload.size()));
    return path;
}

std::vector<uint8_t> fp32_bytes(const std::vector<float>& v) {
    std::vector<uint8_t> out(v.size() * sizeof(float));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}
std::vector<uint8_t> fp16_bytes(const std::vector<float>& v) {
    std::vector<uint8_t> out(v.size() * 2);
    auto* dst = reinterpret_cast<uint16_t*>(out.data());
    for (std::size_t i = 0; i < v.size(); ++i) dst[i] = bt::fp32_to_fp16_bits(v[i]);
    return out;
}

// ─── unit 1: key translator round-trip ─────────────────────────────────────

void test_key_translator() {
    // Spot-check a representative slice of the canonical SD1.5 targets and
    // verify diffusers -> kohya -> diffusers round-trips. This catches the
    // ambiguous-underscore failure mode end-to-end.
    static const char* unet_paths[] = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_k",
        "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj",
        "mid_block.attentions.0.transformer_blocks.0.ff.net.2",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0",
        "up_blocks.3.attentions.2.transformer_blocks.0.ff.net.0.proj",
    };
    for (const char* path : unet_paths) {
        std::string k = lora::diffusers_to_kohya_prefix("unet", path);
        CHECK(k.rfind("lora_unet_", 0) == 0);
        std::string dom, tail;
        bool ok = lora::kohya_to_diffusers(k, dom, tail);
        CHECK(ok);
        CHECK(dom == "unet");
        CHECK(tail == path);
    }

    static const char* clip_paths[] = {
        "text_model.encoder.layers.0.self_attn.q_proj",
        "text_model.encoder.layers.5.self_attn.k_proj",
        "text_model.encoder.layers.11.self_attn.out_proj",
    };
    for (const char* path : clip_paths) {
        std::string k = lora::diffusers_to_kohya_prefix("text_encoder", path);
        CHECK(k.rfind("lora_te_", 0) == 0);
        std::string dom, tail;
        bool ok = lora::kohya_to_diffusers(k, dom, tail);
        CHECK(ok);
        CHECK(dom == "text_encoder");
        CHECK(tail == path);
    }

    // Negative case: garbage kohya prefix should return false.
    std::string dom, tail;
    CHECK(!lora::kohya_to_diffusers("lora_unet_nonsense", dom, tail));
    CHECK(!lora::kohya_to_diffusers("not_a_lora_key", dom, tail));
}

// ─── unit 2: format detection + alpha defaulting ───────────────────────────

void test_format_and_alpha() {
    // (A) Build a minimal kohya fixture with one LoRA target and explicit alpha.
    // Target: lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q
    // rank=2, in=out=8. Bytes are placeholder; we only care about shapes and alpha.
    const int R = 2, D = 8;
    std::vector<float> down(R * D, 0.1f);
    std::vector<float> up(D * R, 0.2f);
    float alpha = 4.0f;

    const std::string p =
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q";
    std::vector<Entry> kohya = {
        {p + ".lora_down.weight", {R, D}, fp16_bytes(down), "F16"},
        {p + ".lora_up.weight",   {D, R}, fp16_bytes(up),   "F16"},
        {p + ".alpha",            {},     fp32_bytes({alpha}), "F32"},
    };
    // shape "[]" not supported by our parser — use shape "[1]".
    kohya.back().shape = {1};

    auto kpath = write_fixture("kohya", kohya);
    {
        auto f = st::File::open(kpath.string());
        CHECK(lora::detect_format(f) == lora::Format::Kohya);
        auto trips = lora::enumerate(f);
        CHECK(trips.size() == 1);
        if (trips.size() == 1) {
            CHECK(trips[0].domain == "unet");
            CHECK(trips[0].target_path ==
                  "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q");
            CHECK(trips[0].rank == R);
            CHECK(trips[0].alpha == alpha);
        }
    }

    // (B) Diffusers/PEFT layout, alpha absent — should default to rank.
    const std::string p2 =
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q";
    std::vector<Entry> diff = {
        {p2 + ".lora_A.weight", {R, D}, fp16_bytes(down), "F16"},
        {p2 + ".lora_B.weight", {D, R}, fp16_bytes(up),   "F16"},
    };
    auto dpath = write_fixture("diff", diff);
    {
        auto f = st::File::open(dpath.string());
        CHECK(lora::detect_format(f) == lora::Format::Diffusers);
        auto trips = lora::enumerate(f);
        CHECK(trips.size() == 1);
        if (trips.size() == 1) {
            CHECK(trips[0].rank == R);
            CHECK(trips[0].alpha == static_cast<float>(R));  // defaulted
        }
    }

    // (C) PEFT with .default. interposition + F16 alpha.
    const std::string p3 =
        "text_encoder.text_model.encoder.layers.0.self_attn.q_proj";
    std::vector<Entry> peft = {
        {p3 + ".lora_A.default.weight", {R, D}, fp16_bytes(down), "F16"},
        {p3 + ".lora_B.default.weight", {D, R}, fp16_bytes(up),   "F16"},
        {p3 + ".alpha",                 {1},     fp16_bytes({8.0f}), "F16"},
    };
    auto ppath = write_fixture("peft", peft);
    {
        auto f = st::File::open(ppath.string());
        CHECK(lora::detect_format(f) == lora::Format::Diffusers);
        auto trips = lora::enumerate(f);
        CHECK(trips.size() == 1);
        if (trips.size() == 1) {
            CHECK(trips[0].domain == "text_encoder");
            CHECK(trips[0].target_path ==
                  "text_model.encoder.layers.0.self_attn.q_proj");
            CHECK(trips[0].alpha == 8.0f);
        }
    }

    // Clean up.
    std::error_code ec;
    std::filesystem::remove(kpath, ec);
    std::filesystem::remove(dpath, ec);
    std::filesystem::remove(ppath, ec);
}

// ─── unit 3: end-to-end merge math ─────────────────────────────────────────
//
// Reproduces apply_lora_delta's GPU sequence (matmul -> scale_inplace ->
// add_inplace) on an isolated 4x8 base weight + rank-2 LoRA. The point is to
// confirm `W' == W + (alpha/rank) * (up @ down) * user_scale` within FP16
// tolerance.

void test_merge_math() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed in test_merge_math: %s\n", e.what());
        ++g_failures;
        return;
    }

    const int OUT = 4, IN = 8, RANK = 2;

    // Base weight: filled with a deterministic pattern.
    std::vector<float> W_base(OUT * IN);
    for (int i = 0; i < OUT * IN; ++i) {
        W_base[static_cast<std::size_t>(i)] = 0.1f * static_cast<float>((i % 5) - 2);
    }
    // LoRA: deterministic patterns.
    std::vector<float> down(RANK * IN);
    for (int i = 0; i < RANK * IN; ++i) {
        down[static_cast<std::size_t>(i)] = 0.05f * static_cast<float>((i % 3) - 1);
    }
    std::vector<float> up(OUT * RANK);
    for (int i = 0; i < OUT * RANK; ++i) {
        up[static_cast<std::size_t>(i)] = 0.07f * static_cast<float>((i % 4) - 1);
    }
    const float alpha = 4.0f;
    const float user_scale = 0.5f;
    const float scale_total =
        (alpha / static_cast<float>(RANK)) * user_scale;  // = 1.0

    // Upload at the pipeline compute dtype (FP32 on CPU, FP16 on GPU).
    bt::Tensor W, D, U, delta;
    W = bdtest::bd_upload(W_base, OUT,  IN);
    D = bdtest::bd_upload(down,   RANK, IN);
    U = bdtest::bd_upload(up,     OUT,  RANK);

    bt::matmul(U, D, delta);                  // (OUT, IN)
    bt::scale_inplace(delta, scale_total);
    bt::add_inplace(W, delta);

    // Read back W and compare to expectation.
    std::vector<float> W_out = bdtest::bd_download(W);

    // Reference (fp32 host).
    std::vector<float> ref(OUT * IN, 0.0f);
    for (int o = 0; o < OUT; ++o) {
        for (int i = 0; i < IN; ++i) {
            float v = 0.0f;
            for (int r = 0; r < RANK; ++r) {
                v += up[o * RANK + r] * down[r * IN + i];
            }
            ref[o * IN + i] = W_base[o * IN + i] + scale_total * v;
        }
    }

    float max_abs = 0.0f, max_diff = 0.0f;
    for (int i = 0; i < OUT * IN; ++i) {
        const float got = W_out[static_cast<std::size_t>(i)];
        const float exp = ref[static_cast<std::size_t>(i)];
        max_abs = std::max(max_abs, std::abs(exp));
        max_diff = std::max(max_diff, std::abs(got - exp));
    }
    // FP16 mantissa is 10 bits; relative ~1e-3 plus accumulation slack.
    const float tol = 5e-3f * std::max(max_abs, 1e-3f);
    if (max_diff > tol) {
        std::fprintf(stderr,
            "merge math: max_diff=%.4g tol=%.4g max_abs=%.4g\n",
            max_diff, tol, max_abs);
    }
    CHECK(max_diff <= tol);
}

}  // namespace

int main() {
    test_key_translator();
    test_format_and_alpha();
    test_merge_math();

    if (g_failures == 0) std::printf("lora: OK\n");
    else std::fprintf(stderr, "lora: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
