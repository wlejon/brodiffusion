// Phase D2 — ControlNet standalone tests.
//
// Builds a small synthetic ControlNet via safetensors::Builder and exercises:
//   1. forward_shapes:           output residual count + per-tensor shapes.
//   2. zero_weights:             zero zero-conv weights → zero residuals.
//   3. finite_output:            random weights produce finite outputs.
//   4. conditioning_scale:       cs=2 doubles every residual vs cs=1.

#include "brodiffusion/controlnet.h"
#include "brodiffusion/detail/compute.h"
#include "brotensor/safetensors.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace cn = brodiffusion::controlnet;
namespace st = brotensor::safetensors;
namespace bt = brotensor;

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

std::vector<uint16_t> fp16_zeros(std::size_t n) { return std::vector<uint16_t>(n, 0); }
std::vector<uint16_t> fp16_ones(std::size_t n) {
    return std::vector<uint16_t>(n, bt::fp32_to_fp16_bits(1.0f));
}
std::vector<uint16_t> fp16_seq(std::size_t n, float scale, std::size_t salt = 0) {
    std::vector<uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        float s = (static_cast<float>((i + salt) % 7) - 3.0f) * scale;
        out[i] = bt::fp32_to_fp16_bits(s);
    }
    return out;
}

void emit_resnet(Builder& b, const std::string& p, int C_in, int C_out,
                 int temb_dim, std::size_t salt) {
    b.add(p + "norm1.weight", {C_in},  fp16_ones(C_in));
    b.add(p + "norm1.bias",   {C_in},  fp16_zeros(C_in));
    b.add(p + "conv1.weight", {C_out, C_in, 3, 3},
          fp16_seq(static_cast<std::size_t>(C_out) * C_in * 9, 0.02f, salt));
    b.add(p + "conv1.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "time_emb_proj.weight", {C_out, temb_dim},
          fp16_seq(static_cast<std::size_t>(C_out) * temb_dim, 0.02f, salt + 1));
    b.add(p + "time_emb_proj.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "norm2.weight", {C_out}, fp16_ones(C_out));
    b.add(p + "norm2.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "conv2.weight", {C_out, C_out, 3, 3},
          fp16_seq(static_cast<std::size_t>(C_out) * C_out * 9, 0.02f, salt + 2));
    b.add(p + "conv2.bias",   {C_out}, fp16_zeros(C_out));
    if (C_in != C_out) {
        b.add(p + "conv_shortcut.weight", {C_out, C_in, 1, 1},
              fp16_seq(static_cast<std::size_t>(C_out) * C_in, 0.05f, salt + 3));
        b.add(p + "conv_shortcut.bias",   {C_out}, fp16_zeros(C_out));
    }
}

void emit_transformer(Builder& b, const std::string& p, int C, int ctx_dim,
                      std::size_t salt) {
    const int ff_inner = 4 * C;
    b.add(p + "norm.weight", {C}, fp16_ones(C));
    b.add(p + "norm.bias",   {C}, fp16_zeros(C));
    b.add(p + "proj_in.weight",  {C, C, 1, 1},
          fp16_seq(static_cast<std::size_t>(C) * C, 0.03f, salt));
    b.add(p + "proj_in.bias",    {C}, fp16_zeros(C));
    b.add(p + "proj_out.weight", {C, C, 1, 1},
          fp16_seq(static_cast<std::size_t>(C) * C, 0.03f, salt + 1));
    b.add(p + "proj_out.bias",   {C}, fp16_zeros(C));
    const std::string bp = p + "transformer_blocks.0.";
    b.add(bp + "norm1.weight", {C}, fp16_ones(C));
    b.add(bp + "norm1.bias",   {C}, fp16_zeros(C));
    b.add(bp + "attn1.to_q.weight", {C, C},
          fp16_seq(static_cast<std::size_t>(C) * C, 0.03f, salt + 2));
    b.add(bp + "attn1.to_k.weight", {C, C},
          fp16_seq(static_cast<std::size_t>(C) * C, 0.03f, salt + 3));
    b.add(bp + "attn1.to_v.weight", {C, C},
          fp16_seq(static_cast<std::size_t>(C) * C, 0.03f, salt + 4));
    b.add(bp + "attn1.to_out.0.weight", {C, C},
          fp16_seq(static_cast<std::size_t>(C) * C, 0.03f, salt + 5));
    b.add(bp + "attn1.to_out.0.bias",   {C}, fp16_zeros(C));
    b.add(bp + "norm2.weight", {C}, fp16_ones(C));
    b.add(bp + "norm2.bias",   {C}, fp16_zeros(C));
    b.add(bp + "attn2.to_q.weight", {C, C},
          fp16_seq(static_cast<std::size_t>(C) * C, 0.03f, salt + 6));
    b.add(bp + "attn2.to_k.weight", {C, ctx_dim},
          fp16_seq(static_cast<std::size_t>(C) * ctx_dim, 0.03f, salt + 7));
    b.add(bp + "attn2.to_v.weight", {C, ctx_dim},
          fp16_seq(static_cast<std::size_t>(C) * ctx_dim, 0.03f, salt + 8));
    b.add(bp + "attn2.to_out.0.weight", {C, C},
          fp16_seq(static_cast<std::size_t>(C) * C, 0.03f, salt + 9));
    b.add(bp + "attn2.to_out.0.bias",   {C}, fp16_zeros(C));
    b.add(bp + "norm3.weight", {C}, fp16_ones(C));
    b.add(bp + "norm3.bias",   {C}, fp16_zeros(C));
    b.add(bp + "ff.net.0.proj.weight", {2 * ff_inner, C},
          fp16_seq(static_cast<std::size_t>(2 * ff_inner) * C, 0.02f, salt + 10));
    b.add(bp + "ff.net.0.proj.bias",   {2 * ff_inner}, fp16_zeros(2 * ff_inner));
    b.add(bp + "ff.net.2.weight", {C, ff_inner},
          fp16_seq(static_cast<std::size_t>(C) * ff_inner, 0.02f, salt + 11));
    b.add(bp + "ff.net.2.bias",   {C}, fp16_zeros(C));
}

// Build the safetensors file for the fixture. `zero_zero_convs` controls
// whether the controlnet_down_blocks.* / controlnet_mid_block.* weights are
// zeros (per ControlNet training init) or pseudo-random.
void build_fixture(const std::filesystem::path& path,
                   const cn::ControlNetConfig& cfg,
                   bool zero_zero_convs) {
    const int nb       = static_cast<int>(cfg.block_out_channels.size());
    const int first_C  = cfg.block_out_channels.front();
    const int mid_C    = cfg.block_out_channels.back();
    const int temb_dim = cfg.time_embed_dim;
    const int freq_dim = cfg.freq_dim;
    const int ctx_dim  = cfg.cross_attention_dim;

    Builder b;
    b.add("conv_in.weight", {first_C, cfg.in_channels, 3, 3},
          fp16_seq(static_cast<std::size_t>(first_C) * cfg.in_channels * 9, 0.05f));
    b.add("conv_in.bias",   {first_C}, fp16_zeros(first_C));
    b.add("time_embedding.linear_1.weight", {temb_dim, freq_dim},
          fp16_seq(static_cast<std::size_t>(temb_dim) * freq_dim, 0.05f, 100));
    b.add("time_embedding.linear_1.bias",   {temb_dim}, fp16_zeros(temb_dim));
    b.add("time_embedding.linear_2.weight", {temb_dim, temb_dim},
          fp16_seq(static_cast<std::size_t>(temb_dim) * temb_dim, 0.05f, 101));
    b.add("time_embedding.linear_2.bias",   {temb_dim}, fp16_zeros(temb_dim));

    // Conditioning embedding CNN.
    {
        const std::string cp = "controlnet_cond_embedding.";
        const int first_ce = cfg.conditioning_embedding_channels.front();
        b.add(cp + "conv_in.weight", {first_ce, cfg.control_channels, 3, 3},
              fp16_seq(static_cast<std::size_t>(first_ce) * cfg.control_channels * 9, 0.04f, 200));
        b.add(cp + "conv_in.bias",   {first_ce}, fp16_zeros(first_ce));

        const int n_pairs = static_cast<int>(cfg.conditioning_embedding_channels.size()) - 1;
        for (int k = 0; k < n_pairs; ++k) {
            const int Ck   = cfg.conditioning_embedding_channels[static_cast<std::size_t>(k)];
            const int Cnxt = cfg.conditioning_embedding_channels[static_cast<std::size_t>(k + 1)];
            const std::string ap = cp + "blocks." + std::to_string(2 * k) + ".";
            b.add(ap + "weight", {Ck, Ck, 3, 3},
                  fp16_seq(static_cast<std::size_t>(Ck) * Ck * 9, 0.03f, 210 + 10 * k));
            b.add(ap + "bias",   {Ck}, fp16_zeros(Ck));
            const std::string sp = cp + "blocks." + std::to_string(2 * k + 1) + ".";
            b.add(sp + "weight", {Cnxt, Ck, 3, 3},
                  fp16_seq(static_cast<std::size_t>(Cnxt) * Ck * 9, 0.03f, 215 + 10 * k));
            b.add(sp + "bias",   {Cnxt}, fp16_zeros(Cnxt));
        }

        const int last_ce = cfg.conditioning_embedding_channels.back();
        b.add(cp + "conv_out.weight", {first_C, last_ce, 3, 3},
              fp16_seq(static_cast<std::size_t>(first_C) * last_ce * 9, 0.03f, 299));
        b.add(cp + "conv_out.bias",   {first_C}, fp16_zeros(first_C));
    }

    // down_blocks (same-as-UNet flag rule).
    int C_prev = first_C;
    for (int i = 0; i < nb; ++i) {
        const int C_out = cfg.block_out_channels[static_cast<std::size_t>(i)];
        const bool has_attn   = (i < nb - 1);
        const bool has_downsm = (i < nb - 1);
        for (int j = 0; j < cfg.layers_per_block; ++j) {
            const int Ci = (j == 0) ? C_prev : C_out;
            const std::string rp = "down_blocks." + std::to_string(i) +
                                   ".resnets." + std::to_string(j) + ".";
            emit_resnet(b, rp, Ci, C_out, temb_dim,
                        static_cast<std::size_t>(1000 + i * 10 + j));
            if (has_attn) {
                const std::string tp = "down_blocks." + std::to_string(i) +
                                       ".attentions." + std::to_string(j) + ".";
                emit_transformer(b, tp, C_out, ctx_dim,
                                 static_cast<std::size_t>(2000 + i * 10 + j));
            }
        }
        if (has_downsm) {
            const std::string sp = "down_blocks." + std::to_string(i) +
                                   ".downsamplers.0.conv.";
            b.add(sp + "weight", {C_out, C_out, 3, 3},
                  fp16_seq(static_cast<std::size_t>(C_out) * C_out * 9, 0.02f,
                           static_cast<std::size_t>(3000 + i)));
            b.add(sp + "bias",   {C_out}, fp16_zeros(C_out));
        }
        C_prev = C_out;
    }

    emit_resnet(b, "mid_block.resnets.0.", mid_C, mid_C, temb_dim, 4000);
    emit_transformer(b, "mid_block.attentions.0.", mid_C, ctx_dim, 4100);
    emit_resnet(b, "mid_block.resnets.1.", mid_C, mid_C, temb_dim, 4200);

    // Per-skip channels for zero-convs: replays the forward push order.
    std::vector<int> skip_channels;
    skip_channels.push_back(first_C);  // conv_in
    for (int i = 0; i < nb; ++i) {
        const int Cb = cfg.block_out_channels[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg.layers_per_block; ++j) skip_channels.push_back(Cb);
        if (i < nb - 1) skip_channels.push_back(Cb);
    }

    for (std::size_t i = 0; i < skip_channels.size(); ++i) {
        const int C = skip_channels[i];
        const std::string zp = "controlnet_down_blocks." + std::to_string(i) + ".";
        const std::size_t n = static_cast<std::size_t>(C) * C;
        const std::vector<uint16_t> Wbits = zero_zero_convs
            ? fp16_zeros(n)
            : fp16_seq(n, 0.03f, 5000 + i);
        b.add(zp + "weight", {C, C, 1, 1}, Wbits);
        b.add(zp + "bias",   {C}, fp16_zeros(C));
    }
    {
        const std::size_t n = static_cast<std::size_t>(mid_C) * mid_C;
        const std::vector<uint16_t> Wbits = zero_zero_convs
            ? fp16_zeros(n)
            : fp16_seq(n, 0.03f, 6000);
        b.add("controlnet_mid_block.weight", {mid_C, mid_C, 1, 1}, Wbits);
        b.add("controlnet_mid_block.bias",   {mid_C}, fp16_zeros(mid_C));
    }

    b.write(path);
}

}  // namespace

int run_tests();

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }
    try {
        return run_tests();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_controlnet: uncaught exception: %s\n", e.what());
        std::fflush(stderr);
        return 1;
    } catch (...) {
        std::fprintf(stderr, "test_controlnet: uncaught unknown exception\n");
        std::fflush(stderr);
        return 1;
    }
}

int run_tests() {

    cn::ControlNetConfig cfg;
    cfg.in_channels       = 4;
    cfg.control_channels  = 3;
    cfg.block_out_channels = {8, 16, 32, 32};
    cfg.layers_per_block  = 1;
    cfg.norm_num_groups   = 2;
    cfg.eps               = 1e-5f;
    cfg.freq_dim          = 8;
    cfg.time_embed_dim    = 16;
    cfg.cross_attention_dim   = 16;
    cfg.transformer_num_heads = 2;
    cfg.conditioning_embedding_channels = {4, 8, 12, 16};  // 3 stride-2 → /8

    const int H_lat = 8, W_lat = 8;
    const int L_text = 4;

    // Synthesize a latent, ctx, control image.
    std::vector<float> latent_h(static_cast<std::size_t>(cfg.in_channels) * H_lat * W_lat);
    for (std::size_t i = 0; i < latent_h.size(); ++i) {
        latent_h[i] = (static_cast<float>(i % 5) - 2.0f) * 0.1f;
    }
    bt::Tensor latent = bdtest::bd_upload(latent_h, 1, cfg.in_channels * H_lat * W_lat);

    std::vector<float> ctx_h(static_cast<std::size_t>(L_text) * cfg.cross_attention_dim);
    for (std::size_t i = 0; i < ctx_h.size(); ++i) {
        ctx_h[i] = (static_cast<float>(i % 7) - 3.0f) * 0.05f;
    }
    bt::Tensor ctx = bdtest::bd_upload(ctx_h, L_text, cfg.cross_attention_dim);

    const int H_img = H_lat * 8, W_img = W_lat * 8;
    std::vector<float> ctrl_h(static_cast<std::size_t>(cfg.control_channels) * H_img * W_img);
    for (std::size_t i = 0; i < ctrl_h.size(); ++i) {
        ctrl_h[i] = (static_cast<float>(i % 11) - 5.0f) * 0.05f;
    }
    bt::Tensor ctrl_img = bdtest::bd_upload(ctrl_h, 1,
                                            cfg.control_channels * H_img * W_img);

    // Expected push count: 1 (conv_in) + 1*4 + 3 = 8 (with layers_per_block=1).
    const int nb = static_cast<int>(cfg.block_out_channels.size());
    int expected_skips = 1;
    for (int i = 0; i < nb; ++i) {
        expected_skips += cfg.layers_per_block;
        if (i < nb - 1) ++expected_skips;
    }
    CHECK(expected_skips == 8);

    // ── Test 1: forward_shapes (random zero-convs) ────────────────────────
    {
        auto path = std::filesystem::temp_directory_path() /
                    "brodiffusion_controlnet_rand.safetensors";
        build_fixture(path, cfg, /*zero_zero_convs=*/false);
        auto file = st::File::open(path.string());
        cn::ControlNet net(cfg);
        net.load_weights(file, "");
        CHECK(net.num_down_residuals() == expected_skips);

        std::vector<bt::Tensor> down_out;
        bt::Tensor mid_out;
        net.forward(latent, H_lat, W_lat, /*timestep=*/500.0f, ctx, ctrl_img,
                    /*conditioning_scale=*/1.0f, down_out, mid_out);
        bt::sync_all();

        CHECK(static_cast<int>(down_out.size()) == expected_skips);
        // Expected per-skip (channels, Hc, Wc) following the push order.
        std::vector<std::tuple<int, int, int>> expected;
        int Hc = H_lat, Wc = W_lat;
        expected.emplace_back(cfg.block_out_channels.front(), Hc, Wc);
        for (int i = 0; i < nb; ++i) {
            const int Cb = cfg.block_out_channels[static_cast<std::size_t>(i)];
            for (int j = 0; j < cfg.layers_per_block; ++j) {
                expected.emplace_back(Cb, Hc, Wc);
            }
            if (i < nb - 1) {
                Hc /= 2; Wc /= 2;
                expected.emplace_back(Cb, Hc, Wc);
            }
        }
        for (int i = 0; i < expected_skips; ++i) {
            auto [C, H, W] = expected[static_cast<std::size_t>(i)];
            CHECK(down_out[static_cast<std::size_t>(i)].rows == 1);
            CHECK(down_out[static_cast<std::size_t>(i)].cols == C * H * W);
        }
        // mid at bottom resolution = H_lat / 2^(nb-2) (mid follows the last
        // down_block which has no downsampler — bottom is set by the
        // previous downsampler). With H_lat=4, nb=4 → Hc=1.
        CHECK(mid_out.rows == 1);
        CHECK(mid_out.cols == cfg.block_out_channels.back() * Hc * Wc);

        // Finite output.
        std::vector<float> v = bdtest::bd_download(mid_out);
        bool all_finite = true;
        for (float x : v) if (!std::isfinite(x)) { all_finite = false; break; }
        CHECK(all_finite);
        for (auto& t : down_out) {
            std::vector<float> w = bdtest::bd_download(t);
            for (float x : w) if (!std::isfinite(x)) { all_finite = false; break; }
        }
        CHECK(all_finite);
    }

    // ── Test 2: zero zero-conv weights → zero residuals ───────────────────
    {
        auto path = std::filesystem::temp_directory_path() /
                    "brodiffusion_controlnet_zero.safetensors";
        build_fixture(path, cfg, /*zero_zero_convs=*/true);
        auto file = st::File::open(path.string());
        cn::ControlNet net(cfg);
        net.load_weights(file, "");

        std::vector<bt::Tensor> down_out;
        bt::Tensor mid_out;
        net.forward(latent, H_lat, W_lat, 500.0f, ctx, ctrl_img,
                    /*conditioning_scale=*/1.0f, down_out, mid_out);
        bt::sync_all();

        bool all_zero = true;
        for (auto& t : down_out) {
            std::vector<float> v = bdtest::bd_download(t);
            for (float x : v) {
                if (x != 0.0f) { all_zero = false; break; }
            }
        }
        CHECK(all_zero);
        std::vector<float> mid_v = bdtest::bd_download(mid_out);
        for (float x : mid_v) {
            if (x != 0.0f) { all_zero = false; break; }
        }
        CHECK(all_zero);
    }

    // ── Test 3 & 4: conditioning_scale doubles every residual ─────────────
    {
        auto path = std::filesystem::temp_directory_path() /
                    "brodiffusion_controlnet_scale.safetensors";
        build_fixture(path, cfg, /*zero_zero_convs=*/false);
        auto file = st::File::open(path.string());
        cn::ControlNet net(cfg);
        net.load_weights(file, "");

        std::vector<bt::Tensor> down1;
        bt::Tensor mid1;
        net.forward(latent, H_lat, W_lat, 500.0f, ctx, ctrl_img,
                    /*conditioning_scale=*/1.0f, down1, mid1);
        bt::sync_all();
        std::vector<std::vector<float>> down1_h;
        for (auto& t : down1) down1_h.push_back(bdtest::bd_download(t));
        std::vector<float> mid1_h = bdtest::bd_download(mid1);

        std::vector<bt::Tensor> down2;
        bt::Tensor mid2;
        net.forward(latent, H_lat, W_lat, 500.0f, ctx, ctrl_img,
                    /*conditioning_scale=*/2.0f, down2, mid2);
        bt::sync_all();
        // Per-element 2x comparison (FP16 round-trip tolerance).
        auto close_to_2x = [](float a, float b) {
            // a is from cs=1, b is from cs=2, expect b ≈ 2a.
            float diff = std::fabs(b - 2.0f * a);
            float tol = 1e-3f + 1e-3f * std::fabs(a);
            return diff <= tol;
        };
        bool scale_ok = true;
        for (std::size_t i = 0; i < down2.size(); ++i) {
            std::vector<float> w = bdtest::bd_download(down2[i]);
            const std::vector<float>& v = down1_h[i];
            CHECK(w.size() == v.size());
            for (std::size_t k = 0; k < w.size(); ++k) {
                if (!close_to_2x(v[k], w[k])) { scale_ok = false; break; }
            }
            if (!scale_ok) break;
        }
        CHECK(scale_ok);
        std::vector<float> mid2_h = bdtest::bd_download(mid2);
        CHECK(mid2_h.size() == mid1_h.size());
        for (std::size_t k = 0; k < mid2_h.size(); ++k) {
            if (!close_to_2x(mid1_h[k], mid2_h[k])) { scale_ok = false; break; }
        }
        CHECK(scale_ok);
    }

    if (g_failures) {
        std::fprintf(stderr, "test_controlnet: %d failure(s)\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "test_controlnet: ok\n");
    return 0;
}
