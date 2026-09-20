// Qwen-Image 2.1 VAE (brodiffusion::vae_qwenimage21) smoke + real-weights test.
//
// Part 1 (unconditional): a scaled-down synthetic checkpoint — base_dim=4,
// decoder_base_dim=6, dim_mult={1,2,4,4}, temperal_downsample={F,T,T,T},
// num_res_blocks=1, z_dim=4, 4-channel I/O — chosen so every residual
// shortcut reduction the real model uses appears at least once:
//   encoder: AvgDown3D plain avg-pool (stage 0), zero-even/odd-avg (1, 2),
//            identity (3)
//   decoder: DupUp3D nearest (0, 1), odd-channel nearest (2),
//            pixel-shuffle (3)
// plus the dead time_conv keys the loader must ignore. Checks shape, dtype,
// finiteness, the decoder's [-1,1] clamp and determinism.
//
// Part 2 (gated): if weights/qwen-image-2.1/vae/diffusion_pytorch_model.safetensors
// is present (BRODIFFUSION_WEIGHTS_DIR overrides the "weights" root), loads
// the real VAE and round-trips a synthetic RGBA image (encode -> decode),
// checking the reconstruction correlates with the input. Numerical parity vs
// diffusers main is scripts/qwenimage21_vae_parity.sh (not part of ctest).

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/vae_qwenimage21.h"
#include "brotensor/safetensors.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace vq = brodiffusion::vae_qwenimage21;
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
             const std::vector<float>& f32) {
        std::size_t expected = 1;
        for (int d : shape) expected *= static_cast<std::size_t>(d);
        if (expected != f32.size()) {
            std::fprintf(stderr, "fixture: shape/data mismatch for %s (expected %zu, got %zu)\n",
                         name.c_str(), expected, f32.size());
            std::abort();
        }
        std::uint64_t start = payload.size();
        const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(f32.data());
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
        std::uint64_t hdr_size = header.size();
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) std::abort();
        f.write(reinterpret_cast<const char*>(&hdr_size), 8);
        f.write(header.data(), header.size());
        f.write(reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size()));
    }
};

std::vector<float> f32_ones(std::size_t n) { return std::vector<float>(n, 1.0f); }

std::vector<float> f32_seq(std::size_t n, float scale, std::size_t salt = 0) {
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (static_cast<float>((i + salt) % 7) - 3.0f) * scale;
    }
    return out;
}

// 2D conv weights (the 2.1 checkpoint stores plain [Co,Ci,kh,kw]).
void emit_resnet(Builder& b, const std::string& p, int C_in, int C_out) {
    b.add(p + "norm1.gamma", {C_in, 1, 1, 1}, f32_ones(static_cast<std::size_t>(C_in)));
    b.add(p + "conv1.weight", {C_out, C_in, 3, 3},
          f32_seq(static_cast<std::size_t>(C_out) * C_in * 9, 0.02f, p.size()));
    b.add(p + "conv1.bias", {C_out}, f32_seq(static_cast<std::size_t>(C_out), 0.001f, p.size() + 100));
    b.add(p + "norm2.gamma", {C_out, 1, 1, 1}, f32_ones(static_cast<std::size_t>(C_out)));
    b.add(p + "conv2.weight", {C_out, C_out, 3, 3},
          f32_seq(static_cast<std::size_t>(C_out) * C_out * 9, 0.02f, p.size() + 1));
    b.add(p + "conv2.bias", {C_out}, f32_seq(static_cast<std::size_t>(C_out), 0.001f, p.size() + 101));
    if (C_in != C_out) {
        b.add(p + "conv_shortcut.weight", {C_out, C_in, 1, 1},
              f32_seq(static_cast<std::size_t>(C_out) * C_in, 0.05f, p.size() + 2));
        b.add(p + "conv_shortcut.bias", {C_out}, f32_seq(static_cast<std::size_t>(C_out), 0.001f, p.size() + 102));
    }
}

void emit_attention(Builder& b, const std::string& ap, int C) {
    b.add(ap + "norm.gamma", {C, 1, 1}, f32_ones(static_cast<std::size_t>(C)));
    b.add(ap + "to_qkv.weight", {3 * C, C, 1, 1},
          f32_seq(static_cast<std::size_t>(3 * C) * C, 0.03f, 11));
    b.add(ap + "to_qkv.bias", {3 * C}, f32_seq(static_cast<std::size_t>(3 * C), 0.001f, 12));
    b.add(ap + "proj.weight", {C, C, 1, 1}, f32_seq(static_cast<std::size_t>(C) * C, 0.03f, 19));
    b.add(ap + "proj.bias", {C}, f32_seq(static_cast<std::size_t>(C), 0.001f, 20));
}

vq::Config make_synth_config() {
    vq::Config cfg;
    cfg.base_dim = 4;
    cfg.decoder_base_dim = 6;
    cfg.z_dim = 4;
    cfg.dim_mult = {1, 2, 4, 4};
    cfg.num_res_blocks = 1;
    cfg.attn_scales = {};
    cfg.temperal_downsample = {false, true, true, true};
    cfg.in_channels = 4;
    cfg.out_channels = 4;
    cfg.latents_mean = std::vector<float>(4, 0.0f);
    cfg.latents_std  = std::vector<float>(4, 1.0f);
    cfg.force_upcast = false;
    return cfg;
}

void build_decoder_fixture(Builder& b, const vq::Config& cfg) {
    const int nb = static_cast<int>(cfg.dim_mult.size());
    std::vector<int> dims;
    dims.push_back(cfg.decoder_base_dim * cfg.dim_mult.back());
    for (int i = nb - 1; i >= 0; --i) dims.push_back(cfg.decoder_base_dim * cfg.dim_mult[static_cast<std::size_t>(i)]);

    b.add("post_quant_conv.weight", {cfg.z_dim, cfg.z_dim, 1, 1},
          f32_seq(static_cast<std::size_t>(cfg.z_dim) * cfg.z_dim, 0.1f, 1));
    b.add("post_quant_conv.bias", {cfg.z_dim}, f32_seq(static_cast<std::size_t>(cfg.z_dim), 0.001f, 2));

    b.add("decoder.conv_in.weight", {dims[0], cfg.z_dim, 3, 3},
          f32_seq(static_cast<std::size_t>(dims[0]) * cfg.z_dim * 9, 0.05f, 3));
    b.add("decoder.conv_in.bias", {dims[0]}, f32_seq(static_cast<std::size_t>(dims[0]), 0.001f, 4));

    emit_resnet(b, "decoder.mid_block.resnets.0.", dims[0], dims[0]);
    emit_resnet(b, "decoder.mid_block.resnets.1.", dims[0], dims[0]);
    emit_attention(b, "decoder.mid_block.attentions.0.", dims[0]);

    for (int i = 0; i < nb; ++i) {
        const int in_dim = dims[static_cast<std::size_t>(i)];   // is_residual: not halved
        const int out_dim = dims[static_cast<std::size_t>(i) + 1];
        for (int j = 0; j <= cfg.num_res_blocks; ++j) {
            const int Ci = (j == 0) ? in_dim : out_dim;
            emit_resnet(b, "decoder.up_blocks." + std::to_string(i) + ".resnets." +
                          std::to_string(j) + ".", Ci, out_dim);
        }
        if (i < nb - 1) {
            const std::string up = "decoder.up_blocks." + std::to_string(i) + ".upsampler.";
            b.add(up + "resample.1.weight", {out_dim, out_dim, 3, 3},
                  f32_seq(static_cast<std::size_t>(out_dim) * out_dim * 9, 0.02f, i + 23));
            b.add(up + "resample.1.bias", {out_dim}, f32_seq(static_cast<std::size_t>(out_dim), 0.001f, i + 24));
            if (cfg.temperal_downsample[static_cast<std::size_t>(nb - 2 - i)]) {
                // Dead for a single frame — the loader must ignore it.
                b.add(up + "time_conv.weight", {2 * out_dim, out_dim, 1, 1},
                      f32_seq(static_cast<std::size_t>(2 * out_dim) * out_dim, 1.0f, 99));
                b.add(up + "time_conv.bias", {2 * out_dim}, f32_ones(static_cast<std::size_t>(2 * out_dim)));
            }
        }
    }

    const int firstC = dims.back();
    b.add("decoder.norm_out.gamma", {firstC, 1, 1, 1}, f32_ones(static_cast<std::size_t>(firstC)));
    b.add("decoder.conv_out.weight", {cfg.out_channels, firstC, 3, 3},
          f32_seq(static_cast<std::size_t>(cfg.out_channels) * firstC * 9, 0.04f, 31));
    b.add("decoder.conv_out.bias", {cfg.out_channels},
          f32_seq(static_cast<std::size_t>(cfg.out_channels), 0.001f, 32));
}

void build_encoder_fixture(Builder& b, const vq::Config& cfg) {
    const int nb = static_cast<int>(cfg.dim_mult.size());
    std::vector<int> dims;
    dims.push_back(cfg.base_dim);
    for (int i = 0; i < nb; ++i) dims.push_back(cfg.base_dim * cfg.dim_mult[static_cast<std::size_t>(i)]);

    b.add("encoder.conv_in.weight", {dims[0], cfg.in_channels, 3, 3},
          f32_seq(static_cast<std::size_t>(dims[0]) * cfg.in_channels * 9, 0.05f, 41));
    b.add("encoder.conv_in.bias", {dims[0]}, f32_seq(static_cast<std::size_t>(dims[0]), 0.001f, 42));

    for (int i = 0; i < nb; ++i) {
        const int in_dim = dims[static_cast<std::size_t>(i)];
        const int out_dim = dims[static_cast<std::size_t>(i) + 1];
        for (int j = 0; j < cfg.num_res_blocks; ++j) {
            const int Ci = (j == 0) ? in_dim : out_dim;
            emit_resnet(b, "encoder.down_blocks." + std::to_string(i) + ".resnets." +
                          std::to_string(j) + ".", Ci, out_dim);
        }
        if (i < nb - 1) {
            const std::string dp = "encoder.down_blocks." + std::to_string(i) + ".downsampler.";
            b.add(dp + "resample.1.weight", {out_dim, out_dim, 3, 3},
                  f32_seq(static_cast<std::size_t>(out_dim) * out_dim * 9, 0.02f, i + 51));
            b.add(dp + "resample.1.bias", {out_dim}, f32_seq(static_cast<std::size_t>(out_dim), 0.001f, i + 52));
            if (cfg.temperal_downsample[static_cast<std::size_t>(i)]) {
                b.add(dp + "time_conv.weight", {out_dim, out_dim, 1, 1},
                      f32_seq(static_cast<std::size_t>(out_dim) * out_dim, 1.0f, 98));
                b.add(dp + "time_conv.bias", {out_dim}, f32_ones(static_cast<std::size_t>(out_dim)));
            }
        }
    }

    const int mid_C = dims.back();
    emit_resnet(b, "encoder.mid_block.resnets.0.", mid_C, mid_C);
    emit_resnet(b, "encoder.mid_block.resnets.1.", mid_C, mid_C);
    emit_attention(b, "encoder.mid_block.attentions.0.", mid_C);

    b.add("encoder.norm_out.gamma", {mid_C, 1, 1, 1}, f32_ones(static_cast<std::size_t>(mid_C)));
    const int twoZ = 2 * cfg.z_dim;
    b.add("encoder.conv_out.weight", {twoZ, mid_C, 3, 3},
          f32_seq(static_cast<std::size_t>(twoZ) * mid_C * 9, 0.04f, 61));
    b.add("encoder.conv_out.bias", {twoZ}, f32_seq(static_cast<std::size_t>(twoZ), 0.001f, 62));

    b.add("quant_conv.weight", {twoZ, twoZ, 1, 1},
          f32_seq(static_cast<std::size_t>(twoZ) * twoZ, 0.1f, 71));
    b.add("quant_conv.bias", {twoZ}, f32_seq(static_cast<std::size_t>(twoZ), 0.001f, 72));
}

std::vector<float> download_any(const bt::Tensor& t) {
    if (t.dtype == bt::Dtype::BF16) {
        bt::Tensor f32;
        bt::cast(t, f32, bt::Dtype::FP32);
        return bdtest::bd_download(f32);
    }
    return bdtest::bd_download(t);
}

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-20));
}

}  // namespace

// ─── Part 1: synthetic-weights smoke test ──────────────────────────────────

static void test_synthetic() {
    vq::Config cfg = make_synth_config();
    CHECK(cfg.spatial_scale() == 8);

    Builder db;
    build_decoder_fixture(db, cfg);
    auto dpath = std::filesystem::temp_directory_path() / "brodiffusion_qi21_vae_decoder_test.safetensors";
    db.write(dpath);

    Builder eb;
    build_encoder_fixture(eb, cfg);
    auto epath = std::filesystem::temp_directory_path() / "brodiffusion_qi21_vae_encoder_test.safetensors";
    eb.write(epath);

    const int H_lat = 2, W_lat = 3;
    const int S = cfg.spatial_scale();
    const int H_out = H_lat * S, W_out = W_lat * S;
    const int out_elems = cfg.out_channels * H_out * W_out;

    {
        auto file = st::File::open(dpath.string());
        vq::Decoder dec(cfg);
        dec.load_weights(file, "");

        std::vector<float> latent_h(static_cast<std::size_t>(cfg.z_dim) * H_lat * W_lat);
        for (std::size_t i = 0; i < latent_h.size(); ++i) {
            latent_h[i] = (static_cast<float>(i % 5) - 2.0f) * 0.1f;
        }
        bt::Tensor latent = bdtest::bd_upload(latent_h, 1, cfg.z_dim * H_lat * W_lat);

        bt::Tensor out;
        dec.decode(latent, H_lat, W_lat, out);
        bt::sync_all();

        CHECK(out.rows == 1);
        CHECK(out.cols == out_elems);

        std::vector<float> vals1 = download_any(out);
        CHECK(static_cast<int>(vals1.size()) == out_elems);
        int nonfinite = 0, out_of_range = 0;
        for (float v : vals1) {
            if (!bdtest::bd_finite(v)) ++nonfinite;
            else if (v < -1.0f || v > 1.0f) ++out_of_range;
        }
        CHECK(nonfinite == 0);
        CHECK(out_of_range == 0);   // decoder clamps to [-1, 1]

        dec.decode(latent, H_lat, W_lat, out);
        bt::sync_all();
        std::vector<float> vals2 = download_any(out);
        CHECK(vals1 == vals2);
    }

    {
        auto file = st::File::open(epath.string());
        vq::Encoder enc(cfg);
        enc.load_weights(file, "");

        const int H = 16, W = 24;
        std::vector<float> img_h(static_cast<std::size_t>(cfg.in_channels) * H * W);
        for (std::size_t i = 0; i < img_h.size(); ++i) {
            img_h[i] = (static_cast<float>(i % 9) - 4.0f) * 0.1f;
        }
        bt::Tensor img = bdtest::bd_upload(img_h, 1, cfg.in_channels * H * W);

        bt::Tensor lat_out;
        enc.encode(img, H, W, nullptr, lat_out);
        bt::sync_all();

        const int lat_elems = cfg.z_dim * (H / S) * (W / S);
        CHECK(lat_out.rows == 1);
        CHECK(lat_out.cols == lat_elems);
        CHECK(lat_out.dtype == brodiffusion::compute_dtype());

        std::vector<float> vals1 = download_any(lat_out);
        int nonfinite = 0;
        for (float v : vals1) if (!bdtest::bd_finite(v)) ++nonfinite;
        CHECK(nonfinite == 0);

        enc.encode(img, H, W, nullptr, lat_out);
        bt::sync_all();
        std::vector<float> vals2 = download_any(lat_out);
        CHECK(vals1 == vals2);

        // Stochastic sample: eps shifts the output by exp(0.5*logvar)*eps.
        std::vector<float> eps_h(static_cast<std::size_t>(lat_elems), 1.0f);
        bt::Tensor eps = bdtest::bd_upload(eps_h, 1, lat_elems);
        bt::Tensor lat_s;
        enc.encode(img, H, W, &eps, lat_s);
        bt::sync_all();
        std::vector<float> vals3 = download_any(lat_s);
        CHECK(vals3.size() == vals1.size());
        CHECK(vals3 != vals1);
        for (float v : vals3) CHECK(bdtest::bd_finite(v));
    }

    std::error_code ec;
    std::filesystem::remove(dpath, ec);
    std::filesystem::remove(epath, ec);
}

// ─── Part 2: real-weights round trip (gated) ───────────────────────────────

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
    const std::string ckpt =
        weights_dir() + "/qwen-image-2.1/vae/diffusion_pytorch_model.safetensors";
    if (!std::filesystem::exists(ckpt)) {
        std::printf("qwenimage21_vae: skipped (no weights)\n");
        return;
    }

    vq::Config cfg;   // header defaults == vae/config.json
    const int S = cfg.spatial_scale();
    CHECK(S == 16);
    const int H = 128, W = 128;
    const int H_lat = H / S, W_lat = W / S;

    auto file = st::File::open(ckpt);
    vq::Encoder enc(cfg);
    enc.load_weights(file, "");
    vq::Decoder dec(cfg);
    dec.load_weights(file, "");

    // Smooth RGBA test image, opaque alpha.
    std::vector<float> img_h(static_cast<std::size_t>(cfg.in_channels) * H * W);
    for (int c = 0; c < cfg.in_channels; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const float fx = static_cast<float>(x) / W, fy = static_cast<float>(y) / H;
                float v;
                if (c == 0)      v = std::sin(6.0f * fx + 2.0f * fy);
                else if (c == 1) v = std::cos(4.0f * fy - 3.0f * fx);
                else if (c == 2) v = std::sin(9.0f * fx * fy);
                else             v = 1.0f;
                img_h[(static_cast<std::size_t>(c) * H + y) * W + x] = 0.9f * v;
            }
        }
    }
    bt::Tensor img = bdtest::bd_upload(img_h, 1, cfg.in_channels * H * W);

    bt::Tensor lat;
    enc.encode(img, H, W, nullptr, lat);
    bt::sync_all();
    CHECK(lat.cols == cfg.z_dim * H_lat * W_lat);
    std::vector<float> lat_h = download_any(lat);
    float lo = 1e30f, hi = -1e30f;
    int nonfinite = 0;
    for (float v : lat_h) {
        if (!bdtest::bd_finite(v)) ++nonfinite;
        else { lo = std::min(lo, v); hi = std::max(hi, v); }
    }
    CHECK(nonfinite == 0);
    CHECK(lo > -12.0f && hi < 12.0f);   // normalised latent stays O(1)

    bt::Tensor rec;
    dec.decode(lat, H_lat, W_lat, rec);
    bt::sync_all();
    CHECK(rec.cols == cfg.out_channels * H * W);
    std::vector<float> rec_h = download_any(rec);
    nonfinite = 0;
    for (float v : rec_h) if (!bdtest::bd_finite(v)) ++nonfinite;
    CHECK(nonfinite == 0);

    // Round trip should reconstruct the colour planes well (per-channel
    // cosine); the alpha plane is constant so it is skipped.
    for (int c = 0; c < 3; ++c) {
        const std::size_t off = static_cast<std::size_t>(c) * H * W;
        std::vector<float> a(img_h.begin() + off, img_h.begin() + off + H * W);
        std::vector<float> b(rec_h.begin() + off, rec_h.begin() + off + H * W);
        const float cs = cosine(a, b);
        std::printf("qwenimage21_vae: round-trip channel %d cosine %.4f\n", c, cs);
        CHECK(cs > 0.95f);
    }
}

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    try {
        test_synthetic();
        test_real_weights();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 1;
    }

    if (g_failures == 0) std::printf("qwenimage21_vae: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
