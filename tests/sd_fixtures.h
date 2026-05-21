#pragma once
//
// Shared synthetic-weight fixtures for brodiffusion's parity / smoke tests.
//
// Each build_* function writes a scaled-down but architecturally complete set
// of SD1.5 sub-module weights into a safetensors Builder, using deterministic
// FP16 values so the same fixture loads identically on every backend. The
// module configs are shrunk (small channel counts) so a whole module's weight
// list fits in one in-memory fixture, while keeping the real layer topology.
//
// The fixture payload is FP16; brodiffusion's loaders convert to the active
// compute dtype (FP32 on the CPU backend, FP16 on a GPU backend), so a single
// fixture file drives a CPU↔GPU parity comparison.

#include "brodiffusion/clip.h"
#include "brodiffusion/unet.h"
#include "brodiffusion/vae.h"

#include "brotensor/tensor.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace bdfix {

namespace bt = brotensor;

// ─── safetensors fixture builder (FP16 payload) ────────────────────────────

struct Builder {
    std::string entries;
    std::vector<std::uint8_t> payload;
    bool first = true;

    void add(const std::string& name, std::vector<int> shape,
             const std::vector<std::uint16_t>& fp16_bits) {
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

// ─── deterministic FP16 vector helpers ─────────────────────────────────────

inline std::vector<std::uint16_t> fp16_zeros(std::size_t n) {
    return std::vector<std::uint16_t>(n, 0);
}
inline std::vector<std::uint16_t> fp16_ones(std::size_t n) {
    return std::vector<std::uint16_t>(n, bt::fp32_to_fp16_bits(1.0f));
}
// Deterministic small Gaussian weights. A parity fixture must use *random*
// (decorrelated) weights: a structured/repetitive pattern makes every conv
// filter nearly identical, which drives the GroupNorm per-group variance
// towards zero — the resulting huge `rstd` amplifies FP16 rounding so much
// that a CPU(FP32)↔Metal(FP16) comparison fails on conditioning alone, not
// on any kernel defect. Gaussian weights keep the groups well-conditioned.
inline std::vector<std::uint16_t> fp16_rand(std::size_t n, float scale,
                                            std::size_t seed = 0) {
    std::mt19937_64 rng(0x9E3779B97F4A7C15ull ^ seed);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<std::uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = bt::fp32_to_fp16_bits(scale * nrm(rng));
    }
    return out;
}

// ─── UNet fixture ──────────────────────────────────────────────────────────

inline void emit_unet_resnet(Builder& b, const std::string& p, int C_in,
                             int C_out, int temb_dim, std::size_t salt) {
    b.add(p + "norm1.weight", {C_in},  fp16_ones(C_in));
    b.add(p + "norm1.bias",   {C_in},  fp16_zeros(C_in));
    b.add(p + "conv1.weight", {C_out, C_in, 3, 3},
          fp16_rand(static_cast<std::size_t>(C_out) * C_in * 9, 0.02f, salt));
    b.add(p + "conv1.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "time_emb_proj.weight", {C_out, temb_dim},
          fp16_rand(static_cast<std::size_t>(C_out) * temb_dim, 0.02f, salt + 1));
    b.add(p + "time_emb_proj.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "norm2.weight", {C_out}, fp16_ones(C_out));
    b.add(p + "norm2.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "conv2.weight", {C_out, C_out, 3, 3},
          fp16_rand(static_cast<std::size_t>(C_out) * C_out * 9, 0.02f, salt + 2));
    b.add(p + "conv2.bias",   {C_out}, fp16_zeros(C_out));
    if (C_in != C_out) {
        b.add(p + "conv_shortcut.weight", {C_out, C_in, 1, 1},
              fp16_rand(static_cast<std::size_t>(C_out) * C_in, 0.05f, salt + 3));
        b.add(p + "conv_shortcut.bias",   {C_out}, fp16_zeros(C_out));
    }
}

inline void emit_unet_transformer(Builder& b, const std::string& p, int C,
                                  int ctx_dim, std::size_t salt) {
    const int ff_inner = 4 * C;
    b.add(p + "norm.weight", {C}, fp16_ones(C));
    b.add(p + "norm.bias",   {C}, fp16_zeros(C));
    b.add(p + "proj_in.weight",  {C, C, 1, 1},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.03f, salt));
    b.add(p + "proj_in.bias",    {C}, fp16_zeros(C));
    b.add(p + "proj_out.weight", {C, C, 1, 1},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.03f, salt + 1));
    b.add(p + "proj_out.bias",   {C}, fp16_zeros(C));

    const std::string bp = p + "transformer_blocks.0.";
    b.add(bp + "norm1.weight", {C}, fp16_ones(C));
    b.add(bp + "norm1.bias",   {C}, fp16_zeros(C));
    b.add(bp + "attn1.to_q.weight", {C, C},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.03f, salt + 2));
    b.add(bp + "attn1.to_k.weight", {C, C},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.03f, salt + 3));
    b.add(bp + "attn1.to_v.weight", {C, C},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.03f, salt + 4));
    b.add(bp + "attn1.to_out.0.weight", {C, C},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.03f, salt + 5));
    b.add(bp + "attn1.to_out.0.bias",   {C}, fp16_zeros(C));

    b.add(bp + "norm2.weight", {C}, fp16_ones(C));
    b.add(bp + "norm2.bias",   {C}, fp16_zeros(C));
    b.add(bp + "attn2.to_q.weight", {C, C},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.03f, salt + 6));
    b.add(bp + "attn2.to_k.weight", {C, ctx_dim},
          fp16_rand(static_cast<std::size_t>(C) * ctx_dim, 0.03f, salt + 7));
    b.add(bp + "attn2.to_v.weight", {C, ctx_dim},
          fp16_rand(static_cast<std::size_t>(C) * ctx_dim, 0.03f, salt + 8));
    b.add(bp + "attn2.to_out.0.weight", {C, C},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.03f, salt + 9));
    b.add(bp + "attn2.to_out.0.bias",   {C}, fp16_zeros(C));

    b.add(bp + "norm3.weight", {C}, fp16_ones(C));
    b.add(bp + "norm3.bias",   {C}, fp16_zeros(C));
    b.add(bp + "ff.net.0.proj.weight", {2 * ff_inner, C},
          fp16_rand(static_cast<std::size_t>(2 * ff_inner) * C, 0.02f, salt + 10));
    b.add(bp + "ff.net.0.proj.bias",   {2 * ff_inner}, fp16_zeros(2 * ff_inner));
    b.add(bp + "ff.net.2.weight", {C, ff_inner},
          fp16_rand(static_cast<std::size_t>(C) * ff_inner, 0.02f, salt + 11));
    b.add(bp + "ff.net.2.bias",   {C}, fp16_zeros(C));
}

// Emit a complete UNet2DConditionModel weight set under `prefix`.
inline void build_unet(Builder& b, const brodiffusion::unet::UNetConfig& cfg,
                       const std::string& prefix) {
    const int nb       = static_cast<int>(cfg.block_out_channels.size());
    const int first_C  = cfg.block_out_channels.front();
    const int mid_C    = cfg.block_out_channels.back();
    const int temb_dim = first_C * cfg.time_embed_dim_mult;
    const int freq_dim = first_C;
    const int ctx_dim  = cfg.cross_attention_dim;

    b.add(prefix + "conv_in.weight", {first_C, cfg.in_channels, 3, 3},
          fp16_rand(static_cast<std::size_t>(first_C) * cfg.in_channels * 9, 0.05f));
    b.add(prefix + "conv_in.bias",   {first_C}, fp16_zeros(first_C));

    b.add(prefix + "time_embedding.linear_1.weight", {temb_dim, freq_dim},
          fp16_rand(static_cast<std::size_t>(temb_dim) * freq_dim, 0.05f, 100));
    b.add(prefix + "time_embedding.linear_1.bias",   {temb_dim}, fp16_zeros(temb_dim));
    b.add(prefix + "time_embedding.linear_2.weight", {temb_dim, temb_dim},
          fp16_rand(static_cast<std::size_t>(temb_dim) * temb_dim, 0.05f, 101));
    b.add(prefix + "time_embedding.linear_2.bias",   {temb_dim}, fp16_zeros(temb_dim));

    int C_prev = first_C;
    for (int i = 0; i < nb; ++i) {
        const int C_out = cfg.block_out_channels[static_cast<std::size_t>(i)];
        const bool has_attn   = (i < nb - 1);
        const bool has_downsm = (i < nb - 1);
        for (int j = 0; j < cfg.layers_per_block; ++j) {
            const int Ci = (j == 0) ? C_prev : C_out;
            const std::string rp = prefix + "down_blocks." + std::to_string(i) +
                                   ".resnets." + std::to_string(j) + ".";
            emit_unet_resnet(b, rp, Ci, C_out, temb_dim,
                             static_cast<std::size_t>(1000 + i * 10 + j));
            if (has_attn) {
                const std::string tp = prefix + "down_blocks." +
                                       std::to_string(i) + ".attentions." +
                                       std::to_string(j) + ".";
                emit_unet_transformer(b, tp, C_out, ctx_dim,
                                      static_cast<std::size_t>(2000 + i * 10 + j));
            }
        }
        if (has_downsm) {
            const std::string sp = prefix + "down_blocks." + std::to_string(i) +
                                   ".downsamplers.0.conv.";
            b.add(sp + "weight", {C_out, C_out, 3, 3},
                  fp16_rand(static_cast<std::size_t>(C_out) * C_out * 9, 0.02f,
                           static_cast<std::size_t>(3000 + i)));
            b.add(sp + "bias",   {C_out}, fp16_zeros(C_out));
        }
        C_prev = C_out;
    }

    emit_unet_resnet(b, prefix + "mid_block.resnets.0.", mid_C, mid_C, temb_dim, 4000);
    emit_unet_transformer(b, prefix + "mid_block.attentions.0.", mid_C, ctx_dim, 4100);
    emit_unet_resnet(b, prefix + "mid_block.resnets.1.", mid_C, mid_C, temb_dim, 4200);

    std::vector<int> skip_stack;
    skip_stack.push_back(first_C);
    for (int i = 0; i < nb; ++i) {
        const int Cb = cfg.block_out_channels[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg.layers_per_block; ++j) skip_stack.push_back(Cb);
        if (i < nb - 1) skip_stack.push_back(Cb);
    }
    int C_up_prev = mid_C;
    for (int i = 0; i < nb; ++i) {
        const int C_out =
            cfg.block_out_channels[static_cast<std::size_t>(nb - 1 - i)];
        const bool has_attn = (i > 0);
        const bool has_upsm = (i < nb - 1);
        const int layers = cfg.layers_per_block + 1;
        for (int j = 0; j < layers; ++j) {
            const int Cskip = skip_stack.back();
            skip_stack.pop_back();
            const int C_h = (j == 0) ? C_up_prev : C_out;
            const int Ci  = C_h + Cskip;
            const std::string rp = prefix + "up_blocks." + std::to_string(i) +
                                   ".resnets." + std::to_string(j) + ".";
            emit_unet_resnet(b, rp, Ci, C_out, temb_dim,
                             static_cast<std::size_t>(5000 + i * 10 + j));
            if (has_attn) {
                const std::string tp = prefix + "up_blocks." +
                                       std::to_string(i) + ".attentions." +
                                       std::to_string(j) + ".";
                emit_unet_transformer(b, tp, C_out, ctx_dim,
                                      static_cast<std::size_t>(6000 + i * 10 + j));
            }
        }
        if (has_upsm) {
            const std::string sp = prefix + "up_blocks." + std::to_string(i) +
                                   ".upsamplers.0.conv.";
            b.add(sp + "weight", {C_out, C_out, 3, 3},
                  fp16_rand(static_cast<std::size_t>(C_out) * C_out * 9, 0.02f,
                           static_cast<std::size_t>(7000 + i)));
            b.add(sp + "bias",   {C_out}, fp16_zeros(C_out));
        }
        C_up_prev = C_out;
    }

    b.add(prefix + "conv_norm_out.weight", {first_C}, fp16_ones(first_C));
    b.add(prefix + "conv_norm_out.bias",   {first_C}, fp16_zeros(first_C));
    b.add(prefix + "conv_out.weight", {cfg.out_channels, first_C, 3, 3},
          fp16_rand(static_cast<std::size_t>(cfg.out_channels) * first_C * 9, 0.04f));
    b.add(prefix + "conv_out.bias",   {cfg.out_channels}, fp16_zeros(cfg.out_channels));
}

// ─── VAE decoder fixture ───────────────────────────────────────────────────

inline void emit_vae_resnet(Builder& b, const std::string& p, int C_in,
                            int C_out) {
    b.add(p + "norm1.weight", {C_in},  fp16_ones(C_in));
    b.add(p + "norm1.bias",   {C_in},  fp16_zeros(C_in));
    b.add(p + "conv1.weight", {C_out, C_in, 3, 3},
          fp16_rand(static_cast<std::size_t>(C_out) * C_in * 9, 0.02f, p.size()));
    b.add(p + "conv1.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "norm2.weight", {C_out}, fp16_ones(C_out));
    b.add(p + "norm2.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "conv2.weight", {C_out, C_out, 3, 3},
          fp16_rand(static_cast<std::size_t>(C_out) * C_out * 9, 0.02f, p.size() + 1));
    b.add(p + "conv2.bias",   {C_out}, fp16_zeros(C_out));
    if (C_in != C_out) {
        b.add(p + "conv_shortcut.weight", {C_out, C_in, 1, 1},
              fp16_rand(static_cast<std::size_t>(C_out) * C_in, 0.05f, p.size() + 2));
        b.add(p + "conv_shortcut.bias",   {C_out}, fp16_zeros(C_out));
    }
}

// Emit a complete AutoencoderKL decoder weight set under `prefix`.
inline void build_vae(Builder& b, const brodiffusion::vae::DecoderConfig& cfg,
                      const std::string& prefix) {
    const int mid_C   = cfg.block_out_channels.back();
    const int first_C = cfg.block_out_channels.front();
    const int nb      = static_cast<int>(cfg.block_out_channels.size());

    b.add(prefix + "conv_in.weight", {mid_C, cfg.in_channels, 3, 3},
          fp16_rand(static_cast<std::size_t>(mid_C) * cfg.in_channels * 9, 0.05f));
    b.add(prefix + "conv_in.bias",   {mid_C}, fp16_zeros(mid_C));

    emit_vae_resnet(b, prefix + "mid_block.resnets.0.", mid_C, mid_C);
    emit_vae_resnet(b, prefix + "mid_block.resnets.1.", mid_C, mid_C);

    const std::string ap = prefix + "mid_block.attentions.0.";
    b.add(ap + "group_norm.weight", {mid_C}, fp16_ones(mid_C));
    b.add(ap + "group_norm.bias",   {mid_C}, fp16_zeros(mid_C));
    b.add(ap + "query.weight", {mid_C, mid_C},
          fp16_rand(static_cast<std::size_t>(mid_C) * mid_C, 0.03f, 11));
    b.add(ap + "query.bias",   {mid_C}, fp16_zeros(mid_C));
    b.add(ap + "key.weight",   {mid_C, mid_C},
          fp16_rand(static_cast<std::size_t>(mid_C) * mid_C, 0.03f, 13));
    b.add(ap + "key.bias",     {mid_C}, fp16_zeros(mid_C));
    b.add(ap + "value.weight", {mid_C, mid_C},
          fp16_rand(static_cast<std::size_t>(mid_C) * mid_C, 0.03f, 17));
    b.add(ap + "value.bias",   {mid_C}, fp16_zeros(mid_C));
    b.add(ap + "proj_attn.weight", {mid_C, mid_C},
          fp16_rand(static_cast<std::size_t>(mid_C) * mid_C, 0.03f, 19));
    b.add(ap + "proj_attn.bias",   {mid_C}, fp16_zeros(mid_C));

    int C_prev = mid_C;
    for (int i = 0; i < nb; ++i) {
        const int C_block =
            cfg.block_out_channels[static_cast<std::size_t>(nb - 1 - i)];
        for (int j = 0; j <= cfg.layers_per_block; ++j) {
            const int Ci = (j == 0) ? C_prev : C_block;
            emit_vae_resnet(b, prefix + "up_blocks." + std::to_string(i) +
                                ".resnets." + std::to_string(j) + ".",
                            Ci, C_block);
        }
        if (i + 1 < nb) {
            const std::string up = prefix + "up_blocks." + std::to_string(i) +
                                   ".upsamplers.0.conv.";
            b.add(up + "weight", {C_block, C_block, 3, 3},
                  fp16_rand(static_cast<std::size_t>(C_block) * C_block * 9, 0.02f,
                           static_cast<std::size_t>(i + 23)));
            b.add(up + "bias",   {C_block}, fp16_zeros(C_block));
        }
        C_prev = C_block;
    }

    b.add(prefix + "conv_norm_out.weight", {first_C}, fp16_ones(first_C));
    b.add(prefix + "conv_norm_out.bias",   {first_C}, fp16_zeros(first_C));
    b.add(prefix + "conv_out.weight", {cfg.out_channels, first_C, 3, 3},
          fp16_rand(static_cast<std::size_t>(cfg.out_channels) * first_C * 9, 0.04f));
    b.add(prefix + "conv_out.bias",   {cfg.out_channels}, fp16_zeros(cfg.out_channels));
}

// ─── CLIP text-encoder fixture ─────────────────────────────────────────────

// Emit a complete CLIP text-encoder weight set under `prefix`.
inline void build_clip(Builder& b,
                       const brodiffusion::clip::TextEncoderConfig& cfg,
                       const std::string& prefix) {
    const int V = cfg.vocab_size;
    const int P = cfg.max_position;
    const int D = cfg.hidden_dim;
    const int F = cfg.intermediate_dim;

    b.add(prefix + "embeddings.token_embedding.weight", {V, D},
          fp16_rand(static_cast<std::size_t>(V) * D, 0.05f));
    b.add(prefix + "embeddings.position_embedding.weight", {P, D},
          fp16_rand(static_cast<std::size_t>(P) * D, 0.05f, 3));

    for (int i = 0; i < cfg.num_layers; ++i) {
        const std::string lp =
            prefix + "encoder.layers." + std::to_string(i) + ".";
        const std::size_t salt = static_cast<std::size_t>(i) * 17;
        b.add(lp + "layer_norm1.weight", {D}, fp16_ones(D));
        b.add(lp + "layer_norm1.bias",   {D}, fp16_zeros(D));
        auto W = [&](float s, std::size_t k) {
            return fp16_rand(static_cast<std::size_t>(D) * D, s, salt + k);
        };
        b.add(lp + "self_attn.q_proj.weight",   {D, D}, W(0.02f, 1));
        b.add(lp + "self_attn.q_proj.bias",     {D},    fp16_zeros(D));
        b.add(lp + "self_attn.k_proj.weight",   {D, D}, W(0.03f, 2));
        b.add(lp + "self_attn.k_proj.bias",     {D},    fp16_zeros(D));
        b.add(lp + "self_attn.v_proj.weight",   {D, D}, W(0.04f, 3));
        b.add(lp + "self_attn.v_proj.bias",     {D},    fp16_zeros(D));
        b.add(lp + "self_attn.out_proj.weight", {D, D}, W(0.05f, 4));
        b.add(lp + "self_attn.out_proj.bias",   {D},    fp16_zeros(D));
        b.add(lp + "layer_norm2.weight", {D}, fp16_ones(D));
        b.add(lp + "layer_norm2.bias",   {D}, fp16_zeros(D));
        b.add(lp + "mlp.fc1.weight", {F, D},
              fp16_rand(static_cast<std::size_t>(F) * D, 0.01f, salt + 5));
        b.add(lp + "mlp.fc1.bias",   {F}, fp16_zeros(F));
        b.add(lp + "mlp.fc2.weight", {D, F},
              fp16_rand(static_cast<std::size_t>(D) * F, 0.01f, salt + 6));
        b.add(lp + "mlp.fc2.bias",   {D}, fp16_zeros(D));
    }

    b.add(prefix + "final_layer_norm.weight", {D}, fp16_ones(D));
    b.add(prefix + "final_layer_norm.bias",   {D}, fp16_zeros(D));
}

}  // namespace bdfix
