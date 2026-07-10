// LoRA loader / merge tests.
//
// Five units:
//   1. Key translator round-trip: kohya <-> diffusers paths.
//   2. Format detection + alpha defaulting on synthetic safetensors files
//      that mimic both kohya and diffusers/PEFT layouts.
//   3. End-to-end merge math: build a 4x8 FP16 base weight on the GPU,
//      synthesize a rank-2 LoRA, run the same matmul+scale+add the loader
//      uses, and check `W' == W + (alpha/rank) * (up @ down) * user_scale`
//      to within FP16 tolerance.
//   4. Krea2-style DiT key grammar: transformer-domain enumeration across
//      the diffusers (`transformer.`), ComfyUI (`diffusion_model.`), bare,
//      and kohya-mangled (`lora_unet_transformer_blocks_*`) conventions.
//   5. Krea2 runtime-adapter math: a tiny Krea2Transformer2DModel with a
//      LoRA attached via add_lora() must produce the same forward output as
//      the same model loaded from pre-merged weights; scale-0 and
//      clear_loras() must reproduce the base output.

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/dit/krea2.h"
#include "brodiffusion/detail/lora_internal.h"
#include "brodiffusion/lora.h"
#include "brotensor/safetensors.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
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
namespace st   = brotensor::safetensors;
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

// ─── unit 4: Krea2-style DiT key grammar ───────────────────────────────────

void test_krea2_key_grammar() {
    const int R = 2, IN = 16, OUT = 16;
    std::vector<float> down(R * IN, 0.1f);
    std::vector<float> up(OUT * R, 0.2f);

    // One file mixing every accepted DiT spelling: diffusers `transformer.`,
    // ComfyUI `diffusion_model.`, and a bare text_fusion path. All must land
    // in domain "transformer" with the canonical checkpoint path.
    const std::string p_diff = "transformer.transformer_blocks.0.attn.to_q";
    const std::string p_comfy = "diffusion_model.transformer_blocks.5.ff.gate";
    const std::string p_bare = "text_fusion.refiner_blocks.0.ff.up";
    std::vector<Entry> dit = {
        {p_diff + ".lora_A.weight",     {R, IN},  fp16_bytes(down), "F16"},
        {p_diff + ".lora_B.weight",     {OUT, R}, fp16_bytes(up),   "F16"},
        {p_diff + ".alpha",             {1},      fp32_bytes({4.0f}), "F32"},
        {p_comfy + ".lora_down.weight", {R, IN},  fp16_bytes(down), "F16"},
        {p_comfy + ".lora_up.weight",   {OUT, R}, fp16_bytes(up),   "F16"},
        {p_bare + ".lora_A.weight",     {R, IN},  fp16_bytes(down), "F16"},
        {p_bare + ".lora_B.weight",     {OUT, R}, fp16_bytes(up),   "F16"},
    };
    auto dpath = write_fixture("krea2_diff", dit);
    {
        auto f = st::File::open(dpath.string());
        CHECK(lora::detect_format(f) == lora::Format::Diffusers);
        auto trips = lora::enumerate(f);
        CHECK(trips.size() == 3);
        for (const auto& t : trips) CHECK(t.domain == "transformer");
        // Deterministic order: sorted by target_path.
        if (trips.size() == 3) {
            CHECK(trips[0].target_path == "text_fusion.refiner_blocks.0.ff.up");
            CHECK(trips[0].alpha == static_cast<float>(R));   // defaulted
            CHECK(trips[1].target_path == "transformer_blocks.0.attn.to_q");
            CHECK(trips[1].alpha == 4.0f);
            CHECK(trips[2].target_path == "transformer_blocks.5.ff.gate");
            CHECK(trips[2].rank == R);
        }
    }

    // Kohya-mangled spelling (musubi-style): lora_unet_transformer_blocks_*.
    const std::string p_kohya = "lora_unet_transformer_blocks_3_attn_to_out_0";
    std::vector<Entry> kohya = {
        {p_kohya + ".lora_down.weight", {R, IN},  fp16_bytes(down), "F16"},
        {p_kohya + ".lora_up.weight",   {OUT, R}, fp16_bytes(up),   "F16"},
        {p_kohya + ".alpha",            {1},      fp32_bytes({2.0f}), "F32"},
    };
    auto kpath = write_fixture("krea2_kohya", kohya);
    {
        auto f = st::File::open(kpath.string());
        CHECK(lora::detect_format(f) == lora::Format::Kohya);
        auto trips = lora::enumerate(f);
        CHECK(trips.size() == 1);
        if (trips.size() == 1) {
            CHECK(trips[0].domain == "transformer");
            CHECK(trips[0].target_path == "transformer_blocks.3.attn.to_out.0");
            CHECK(trips[0].alpha == 2.0f);
        }
    }

    // Translator: transformer domain round-trips through the kohya spelling.
    std::string k = lora::diffusers_to_kohya_prefix(
        "transformer", "transformer_blocks.3.attn.to_out.0");
    CHECK(k == "lora_unet_transformer_blocks_3_attn_to_out_0");
    std::string dom, tail;
    CHECK(lora::kohya_to_diffusers(k, dom, tail));
    CHECK(dom == "transformer");
    CHECK(tail == "transformer_blocks.3.attn.to_out.0");
    // lora_transformer_ head is accepted too.
    CHECK(lora::kohya_to_diffusers("lora_transformer_transformer_blocks_3_ff_down",
                                   dom, tail));
    CHECK(dom == "transformer");
    CHECK(tail == "transformer_blocks.3.ff.down");
    // Unknown DiT tails must NOT parse.
    CHECK(!lora::kohya_to_diffusers("lora_unet_transformer_blocks_3_attn_add_q_proj",
                                    dom, tail));

    std::error_code ec;
    std::filesystem::remove(dpath, ec);
    std::filesystem::remove(kpath, ec);
}

// ─── unit 5: Krea2 runtime-adapter math ────────────────────────────────────
//
// Build a tiny Krea2Transformer2DModel checkpoint fixture, run one forward as
// the base, attach a two-target LoRA via the same enumerate() -> add_lora()
// path Pipeline::apply_lora uses, and check:
//   - the adapted output matches a second model loaded from PRE-MERGED
//     weights (W' = W + eff * up @ down) to within compute-dtype tolerance,
//   - set_lora_scale(0) and clear_loras() reproduce the base output.

namespace k2 {

// Deterministic small pseudo-random weights (LCG), centred, ~N(0, 0.05).
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed) {}
    float next() {
        s = s * 1664525u + 1013904223u;
        return (static_cast<float>((s >> 8) & 0xFFFF) / 65535.0f - 0.5f) * 0.1f;
    }
    std::vector<float> vec(int n) {
        std::vector<float> v(static_cast<std::size_t>(n));
        for (float& x : v) x = next();
        return v;
    }
};

brodiffusion::dit::Krea2Config tiny_config() {
    brodiffusion::dit::Krea2Config c;
    c.in_channels = 8;               // latent_channels 2
    c.num_layers = 1;
    c.attention_head_dim = 8;
    c.num_attention_heads = 2;       // hidden = 16
    c.num_key_value_heads = 1;
    c.intermediate_size = 32;
    c.timestep_embed_dim = 8;
    c.text_hidden_dim = 8;
    c.num_text_layers = 2;
    c.text_num_attention_heads = 2;  // hd_txt = 4
    c.text_num_key_value_heads = 2;
    c.text_intermediate_size = 16;
    c.num_layerwise_text_blocks = 1;
    c.num_refiner_text_blocks = 1;
    c.axes_dims_rope = {4, 2, 2};
    return c;
}

// All tiny-checkpoint weights as name -> host FP32 values, in a stable order.
struct Weights {
    std::vector<std::pair<std::string, std::pair<std::vector<int>,
                                                 std::vector<float>>>> entries;
    std::vector<float>* find(const std::string& name) {
        for (auto& e : entries) if (e.first == name) return &e.second.second;
        return nullptr;
    }
};

Weights make_weights(const brodiffusion::dit::Krea2Config& c, uint32_t seed) {
    Rng rng(seed);
    const int H = c.hidden_size();
    const int IC = c.in_channels;
    const int TE = c.timestep_embed_dim;
    const int TH = c.text_hidden_dim;
    const int hd_txt = TH / c.text_num_attention_heads;
    Weights w;
    auto put = [&](const std::string& name, std::vector<int> shape) {
        int n = 1;
        for (int d : shape) n *= d;
        w.entries.push_back({name, {std::move(shape), rng.vec(n)}});
    };
    auto lin = [&](const std::string& key, int out, int in, bool bias) {
        put(key + ".weight", {out, in});
        if (bias) put(key + ".bias", {out});
    };
    auto attn = [&](const std::string& p, int dim, int hd, int nq, int nkv) {
        lin(p + "to_q", hd * nq, dim, false);
        lin(p + "to_k", hd * nkv, dim, false);
        lin(p + "to_v", hd * nkv, dim, false);
        lin(p + "to_gate", dim, dim, false);
        lin(p + "to_out.0", dim, dim, false);
        put(p + "norm_q.weight", {hd});
        put(p + "norm_k.weight", {hd});
    };
    auto ff = [&](const std::string& p, int dim, int inter) {
        lin(p + "gate", inter, dim, false);
        lin(p + "up", inter, dim, false);
        lin(p + "down", dim, inter, false);
    };
    auto fusion = [&](const std::string& p) {
        put(p + "norm1.weight", {TH});
        put(p + "norm2.weight", {TH});
        attn(p + "attn.", TH, hd_txt, c.text_num_attention_heads,
             c.text_num_key_value_heads);
        ff(p + "ff.", TH, c.text_intermediate_size);
    };

    lin("img_in", H, IC, true);
    lin("time_embed.linear_1", H, TE, true);
    lin("time_embed.linear_2", H, H, true);
    lin("time_mod_proj", 6 * H, H, true);
    for (int i = 0; i < c.num_layerwise_text_blocks; ++i) {
        fusion("text_fusion.layerwise_blocks." + std::to_string(i) + ".");
    }
    put("text_fusion.projector.weight", {1, c.num_text_layers});
    for (int i = 0; i < c.num_refiner_text_blocks; ++i) {
        fusion("text_fusion.refiner_blocks." + std::to_string(i) + ".");
    }
    put("txt_in.norm.weight", {TH});
    lin("txt_in.linear_1", H, TH, true);
    lin("txt_in.linear_2", H, H, true);
    for (int i = 0; i < c.num_layers; ++i) {
        const std::string p = "transformer_blocks." + std::to_string(i) + ".";
        put(p + "scale_shift_table", {6, H});
        put(p + "norm1.weight", {H});
        put(p + "norm2.weight", {H});
        attn(p + "attn.", H, c.attention_head_dim, c.num_attention_heads,
             c.num_key_value_heads);
        ff(p + "ff.", H, c.intermediate_size);
    }
    put("final_layer.scale_shift_table", {2, H});
    put("final_layer.norm.weight", {H});
    lin("final_layer.linear", IC, H, true);
    return w;
}

std::filesystem::path write_checkpoint(const std::string& tag,
                                       const Weights& w) {
    std::vector<Entry> entries;
    entries.reserve(w.entries.size());
    for (const auto& e : w.entries) {
        entries.push_back({e.first, e.second.first,
                           fp32_bytes(e.second.second), "F32"});
    }
    return write_fixture(tag, entries);
}

// Forward one deterministic (latent, text) pair and download FP32.
std::vector<float> run_forward(brodiffusion::dit::Krea2Transformer2DModel& m,
                               const brodiffusion::dit::Krea2Config& c) {
    const int hp = 2, wp = 2, text_seq = 3;
    Rng rng(99);   // same inputs every call
    std::vector<float> lat = rng.vec(hp * wp * c.in_channels);
    std::vector<float> emb =
        rng.vec(text_seq * c.num_text_layers * c.text_hidden_dim);
    std::vector<float> mask(text_seq, 1.0f);
    bt::Tensor latent = bt::Tensor::from_host(lat.data(), hp * wp, c.in_channels);
    bt::Tensor embeds = bt::Tensor::from_host(
        emb.data(), text_seq * c.num_text_layers, c.text_hidden_dim);
    bt::Tensor mvec = bt::Tensor::from_host(mask.data(), text_seq, 1);
    bt::Tensor out;
    m.forward(latent, hp, wp, embeds, mvec, 0.5f, out);
    bt::Tensor out32 = out;
    if (out.dtype != bt::Dtype::FP32) bt::cast(out, out32, bt::Dtype::FP32);
    bt::sync_all();
    return out32.to(bt::Device::CPU).to_host_vector();
}

float rel_rms(const std::vector<float>& a, const std::vector<float>& b) {
    double d2 = 0, m2 = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        d2 += d * d;
        m2 += static_cast<double>(a[i]) * a[i];
    }
    return static_cast<float>(std::sqrt(d2 / (m2 + 1e-20)));
}

}  // namespace k2

void test_krea2_runtime_adapter() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed in test_krea2_runtime_adapter: %s\n",
                     e.what());
        ++g_failures;
        return;
    }

    namespace dit = brodiffusion::dit;
    const dit::Krea2Config cfg = k2::tiny_config();
    const int H = cfg.hidden_size();
    const int TH = cfg.text_hidden_dim;

    // Two LoRA targets, chosen to cover a body block AND a text-fusion block:
    //   transformer_blocks.0.attn.to_q       (out 16, in 16), alpha 4, rank 2
    //   text_fusion.refiner_blocks.0.ff.up   (out 16, in 8),  alpha absent
    const int R = 2;
    const float user_scale = 0.5f;
    k2::Rng lrng(7);
    std::vector<float> toq_down = lrng.vec(R * H), toq_up = lrng.vec(H * R);
    std::vector<float> ffup_down = lrng.vec(R * TH);
    std::vector<float> ffup_up = lrng.vec(cfg.text_intermediate_size * R);
    // Make the low-rank delta LARGE relative to the tiny random base weights
    // (~10x), so the forward output visibly moves — with same-scale factors
    // the rank-2 delta on a random tiny model shifts the velocity by less
    // than measurement noise and the equivalence check would be vacuous.
    for (auto* v : {&toq_down, &toq_up, &ffup_down, &ffup_up}) {
        for (float& x : *v) x *= 10.0f;
    }
    const float toq_eff = (4.0f / R) * user_scale;   // explicit alpha 4
    const float ffup_eff = (static_cast<float>(R) / R) * user_scale;

    // Base checkpoint + LoRA file (diffusers keys, one with explicit alpha).
    k2::Weights base = k2::make_weights(cfg, 42);
    auto base_path = k2::write_checkpoint("krea2_base", base);
    std::vector<Entry> lora_entries = {
        {"transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
         {R, H}, fp32_bytes(toq_down), "F32"},
        {"transformer.transformer_blocks.0.attn.to_q.lora_B.weight",
         {H, R}, fp32_bytes(toq_up), "F32"},
        {"transformer.transformer_blocks.0.attn.to_q.alpha",
         {1}, fp32_bytes({4.0f}), "F32"},
        {"transformer.text_fusion.refiner_blocks.0.ff.up.lora_A.weight",
         {R, TH}, fp32_bytes(ffup_down), "F32"},
        {"transformer.text_fusion.refiner_blocks.0.ff.up.lora_B.weight",
         {cfg.text_intermediate_size, R}, fp32_bytes(ffup_up), "F32"},
    };
    auto lora_path = write_fixture("krea2_lora", lora_entries);

    // Pre-merged checkpoint: same base, W' = W + eff * (up @ down) on the two
    // targets — the ground truth the runtime adapters must reproduce.
    k2::Weights merged = k2::make_weights(cfg, 42);
    auto merge = [&](const std::string& name, const std::vector<float>& up,
                     const std::vector<float>& down, int out, int in,
                     float eff) {
        std::vector<float>* W = merged.find(name);
        CHECK(W != nullptr);
        if (!W) return;
        for (int o = 0; o < out; ++o) {
            for (int i = 0; i < in; ++i) {
                float v = 0.0f;
                for (int r = 0; r < R; ++r) v += up[o * R + r] * down[r * in + i];
                (*W)[static_cast<std::size_t>(o) * in + i] += eff * v;
            }
        }
    };
    merge("transformer_blocks.0.attn.to_q.weight", toq_up, toq_down, H, H,
          toq_eff);
    merge("text_fusion.refiner_blocks.0.ff.up.weight", ffup_up, ffup_down,
          cfg.text_intermediate_size, TH, ffup_eff);
    auto merged_path = k2::write_checkpoint("krea2_merged", merged);

    {
        dit::Krea2Transformer2DModel model(cfg);
        auto f = st::File::open(base_path.string());
        model.load_weights(f);
        const std::vector<float> out_base = k2::run_forward(model, cfg);

        // Attach the LoRA through the same enumerate() -> add_lora() glue
        // Pipeline::apply_lora uses.
        auto lf = st::File::open(lora_path.string());
        auto trips = lora::enumerate(lf);
        CHECK(trips.size() == 2);
        std::vector<dit::Krea2Transformer2DModel::LoraTarget> targets;
        for (const auto& t : trips) {
            CHECK(t.domain == "transformer");
            targets.push_back({t.target_path, &lf.get(t.down_key),
                               &lf.get(t.up_key),
                               t.alpha / static_cast<float>(t.rank)});
        }
        const int group = model.add_lora(targets, user_scale);
        CHECK(group == 0);
        CHECK(model.num_loras() == 1);
        const std::vector<float> out_lora = k2::run_forward(model, cfg);

        // The pre-merged ground truth must actually displace the output —
        // otherwise the equivalence check below would pass vacuously.
        dit::Krea2Transformer2DModel model_m(cfg);
        auto mf = st::File::open(merged_path.string());
        model_m.load_weights(mf);
        const std::vector<float> out_merged = k2::run_forward(model_m, cfg);
        CHECK(k2::rel_rms(out_base, out_merged) > 1e-3f);

        // The adapter path must match that ground truth. BF16 (CUDA) rounds
        // the merged weight differently from the adapter sum, so the
        // tolerance is compute-dtype-loose.
        const float rr = k2::rel_rms(out_merged, out_lora);
        if (rr > 2e-2f) {
            std::fprintf(stderr, "krea2 adapter-vs-merged rel_rms=%.4g\n", rr);
        }
        CHECK(rr <= 2e-2f);

        // scale 0 disables; restoring the scale re-enables; clear removes.
        model.set_lora_scale(0, 0.0f);
        CHECK(k2::rel_rms(out_base, k2::run_forward(model, cfg)) < 1e-4f);
        model.set_lora_scale(0, user_scale);
        CHECK(k2::rel_rms(out_lora, k2::run_forward(model, cfg)) < 1e-4f);
        model.clear_loras();
        CHECK(model.num_loras() == 0);
        CHECK(k2::rel_rms(out_base, k2::run_forward(model, cfg)) < 1e-4f);

        // Unknown target paths must be rejected without partial application.
        bool threw = false;
        std::vector<dit::Krea2Transformer2DModel::LoraTarget> bad = targets;
        bad[0].path = "transformer_blocks.9.attn.to_q";
        try {
            model.add_lora(bad, 1.0f);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
        CHECK(model.num_loras() == 0);
        CHECK(k2::rel_rms(out_base, k2::run_forward(model, cfg)) < 1e-4f);
    }

    std::error_code ec;
    std::filesystem::remove(base_path, ec);
    std::filesystem::remove(lora_path, ec);
    std::filesystem::remove(merged_path, ec);
}

}  // namespace

int main() {
    test_key_translator();
    test_format_and_alpha();
    test_merge_math();
    test_krea2_key_grammar();
    test_krea2_runtime_adapter();

    if (g_failures == 0) std::printf("lora: OK\n");
    else std::fprintf(stderr, "lora: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
