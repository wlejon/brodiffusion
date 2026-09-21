#pragma once

// A scaled-down but architecturally complete Qwen-Image 2.1 transformer
// checkpoint, written to a temp safetensors file — 2 blocks, 2 heads of 64,
// context 128, axes_dims_rope {8, 28, 28}. Small enough to build in memory
// and load in well under a second, which is what lets the hook tests run one
// forward per assertion.
//
// Shared by tests/test_qwenimage21_dit.cpp (shapes, the prefix cache, the
// single-binding hooks) and tests/test_qwenimage21_slots.cpp (the multi-slot
// binding lists and the prefix-KV dial), so the two cannot drift into testing
// differently-shaped models.

#include "brodiffusion/dit/qwenimage21.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace qi21fix {

namespace bt = ::brotensor;
namespace qd = ::brodiffusion::dit;

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
inline std::vector<float> rnd(std::size_t n, std::size_t salt) {
    std::vector<float> out(n);
    std::uint32_t s = static_cast<std::uint32_t>(salt * 2654435761u + 12345u);
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        out[i] = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.2f;
    }
    return out;
}

inline void emit_lin(Builder& b, const std::string& p, int out, int in) {
    b.add(p + ".weight", {out, in},
          rnd(static_cast<std::size_t>(out) * in, p.size() + out));
}

inline void emit_norm(Builder& b, const std::string& p, int dim) {
    b.add(p + ".weight", {dim}, rnd(static_cast<std::size_t>(dim), p.size() + 3));
}

inline qd::QwenImage21Config synth_cfg() {
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

inline void build_fixture(Builder& b, const qd::QwenImage21Config& c) {
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

inline std::vector<float> download_any(const bt::Tensor& t) {
    if (t.dtype != bt::Dtype::FP32) {
        bt::Tensor f32;
        bt::cast(t, f32, bt::Dtype::FP32);
        return bdtest::bd_download(f32);
    }
    return bdtest::bd_download(t);
}

// Max |a-b| / (1 + max|ref|).
inline double rel_maxdiff(const std::vector<float>& a,
                          const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 1e9;
    double mx = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        mx = std::max(mx, std::abs(static_cast<double>(a[i]) - b[i]));
        scale = std::max(scale, std::abs(static_cast<double>(a[i])));
    }
    return mx / (1.0 + scale);
}

inline std::filesystem::path write_fixture(const qd::QwenImage21Config& cfg,
                                           const char* stem) {
    Builder b;
    build_fixture(b, cfg);
    auto path = std::filesystem::temp_directory_path() / stem;
    b.write(path);
    return path;
}

}  // namespace qi21fix
