// Qwen-Image VAE (brodiffusion::vae_qwenimage) smoke + real-weights test.
//
// Part 1 (unconditional): builds a scaled-down synthetic checkpoint —
// base_dim=4, dim_mult={1,2,2}, num_res_blocks=1, z_dim=4 — keeping the
// architecture intact (mid-block attention, a channel-changing resnet with
// conv_shortcut, one upsample/downsample transition that halves/keeps
// channels) but small enough to fit in an in-memory fixture. Exercises both
// Decoder::decode and Encoder::encode; verifies shape, dtype, all-finite
// output, and determinism across two calls.
//
// Part 2 (gated): if weights/krea-2-raw/vae/diffusion_pytorch_model.safetensors
// is present (BRODIFFUSION_WEIGHTS_DIR overrides the "weights" root), loads
// the real Krea 2 / Qwen-Image VAE decoder and decodes a deterministic
// synthetic latent, checking shape/finite/range/determinism. Numerical parity
// against the diffusers reference is checked separately by
// scripts/krea2_vae_parity.sh (not part of ctest).

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/vae_qwenimage.h"
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

namespace vq = brodiffusion::vae_qwenimage;
namespace st = brotensor::safetensors;
namespace bt = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// ─── safetensors fixture builder (F32) ─────────────────────────────────────

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

// Small values clustered around zero so the cascade of convs doesn't blow up.
std::vector<float> f32_seq(std::size_t n, float scale, std::size_t salt = 0) {
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = (static_cast<float>((i + salt) % 7) - 3.0f) * scale;
    }
    return out;
}

// One resnet's tensors under prefix p (already ending with "."). kT=3 for the
// main convs (causal Conv3d), matching the real checkpoint's 3x3x3 filters —
// load_conv_lasttap only ever reads the last of the kT taps, but the fixture
// must still ship all of them.
void emit_resnet(Builder& b, const std::string& p, int C_in, int C_out) {
    b.add(p + "norm1.gamma", {C_in, 1, 1, 1}, f32_ones(static_cast<std::size_t>(C_in)));
    b.add(p + "conv1.weight", {C_out, C_in, 3, 3, 3},
          f32_seq(static_cast<std::size_t>(C_out) * C_in * 27, 0.02f, p.size()));
    b.add(p + "conv1.bias", {C_out}, f32_seq(static_cast<std::size_t>(C_out), 0.001f, p.size() + 100));
    b.add(p + "norm2.gamma", {C_out, 1, 1, 1}, f32_ones(static_cast<std::size_t>(C_out)));
    b.add(p + "conv2.weight", {C_out, C_out, 3, 3, 3},
          f32_seq(static_cast<std::size_t>(C_out) * C_out * 27, 0.02f, p.size() + 1));
    b.add(p + "conv2.bias", {C_out}, f32_seq(static_cast<std::size_t>(C_out), 0.001f, p.size() + 101));
    if (C_in != C_out) {
        b.add(p + "conv_shortcut.weight", {C_out, C_in, 1, 1, 1},
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
    cfg.z_dim = 4;
    cfg.dim_mult = {1, 2, 2};
    cfg.num_res_blocks = 1;
    cfg.attn_scales = {};
    cfg.temperal_downsample = {false, true};
    cfg.input_channels = 3;
    cfg.latents_mean = std::vector<float>(4, 0.0f);
    cfg.latents_std  = std::vector<float>(4, 1.0f);
    cfg.force_upcast = false;   // keep the synthetic-data check dtype-simple
    return cfg;
}

void build_decoder_fixture(Builder& b, const vq::Config& cfg) {
    const int nb = static_cast<int>(cfg.dim_mult.size());
    std::vector<int> dims;
    dims.push_back(cfg.base_dim * cfg.dim_mult.back());
    for (int i = nb - 1; i >= 0; --i) dims.push_back(cfg.base_dim * cfg.dim_mult[static_cast<std::size_t>(i)]);

    b.add("post_quant_conv.weight", {cfg.z_dim, cfg.z_dim, 1, 1, 1},
          f32_seq(static_cast<std::size_t>(cfg.z_dim) * cfg.z_dim, 0.1f, 1));
    b.add("post_quant_conv.bias", {cfg.z_dim}, f32_seq(static_cast<std::size_t>(cfg.z_dim), 0.001f, 2));

    b.add("decoder.conv_in.weight", {dims[0], cfg.z_dim, 3, 3, 3},
          f32_seq(static_cast<std::size_t>(dims[0]) * cfg.z_dim * 27, 0.05f, 3));
    b.add("decoder.conv_in.bias", {dims[0]}, f32_seq(static_cast<std::size_t>(dims[0]), 0.001f, 4));

    emit_resnet(b, "decoder.mid_block.resnets.0.", dims[0], dims[0]);
    emit_resnet(b, "decoder.mid_block.resnets.1.", dims[0], dims[0]);
    emit_attention(b, "decoder.mid_block.attentions.0.", dims[0]);

    for (int i = 0; i < nb; ++i) {
        int in_dim = dims[static_cast<std::size_t>(i)];
        const int out_dim = dims[static_cast<std::size_t>(i) + 1];
        if (i > 0) in_dim /= 2;
        for (int j = 0; j <= cfg.num_res_blocks; ++j) {
            const int Ci = (j == 0) ? in_dim : out_dim;
            emit_resnet(b, "decoder.up_blocks." + std::to_string(i) + ".resnets." +
                          std::to_string(j) + ".", Ci, out_dim);
        }
        if (i < nb - 1) {
            const std::string up = "decoder.up_blocks." + std::to_string(i) + ".upsamplers.0.resample.1.";
            b.add(up + "weight", {out_dim / 2, out_dim, 3, 3},
                  f32_seq(static_cast<std::size_t>(out_dim / 2) * out_dim * 9, 0.02f, i + 23));
            b.add(up + "bias", {out_dim / 2}, f32_seq(static_cast<std::size_t>(out_dim / 2), 0.001f, i + 24));
        }
    }

    const int firstC = dims.back();
    b.add("decoder.norm_out.gamma", {firstC, 1, 1, 1}, f32_ones(static_cast<std::size_t>(firstC)));
    b.add("decoder.conv_out.weight", {cfg.input_channels, firstC, 3, 3, 3},
          f32_seq(static_cast<std::size_t>(cfg.input_channels) * firstC * 27, 0.04f, 31));
    b.add("decoder.conv_out.bias", {cfg.input_channels},
          f32_seq(static_cast<std::size_t>(cfg.input_channels), 0.001f, 32));
}

void build_encoder_fixture(Builder& b, const vq::Config& cfg) {
    const int nb = static_cast<int>(cfg.dim_mult.size());
    std::vector<int> dims;
    dims.push_back(cfg.base_dim);
    for (int i = 0; i < nb; ++i) dims.push_back(cfg.base_dim * cfg.dim_mult[static_cast<std::size_t>(i)]);

    b.add("encoder.conv_in.weight", {dims[0], cfg.input_channels, 3, 3, 3},
          f32_seq(static_cast<std::size_t>(dims[0]) * cfg.input_channels * 27, 0.05f, 41));
    b.add("encoder.conv_in.bias", {dims[0]}, f32_seq(static_cast<std::size_t>(dims[0]), 0.001f, 42));

    for (int i = 0; i < nb; ++i) {
        const int in_dim = dims[static_cast<std::size_t>(i)];
        const int out_dim = dims[static_cast<std::size_t>(i) + 1];
        for (int j = 0; j < cfg.num_res_blocks; ++j) {
            const int Ci = (j == 0) ? in_dim : out_dim;
            emit_resnet(b, "encoder.down_blocks." + std::to_string(i * (cfg.num_res_blocks + 1) + j) + ".",
                       Ci, out_dim);
        }
        if (i < nb - 1) {
            const std::string dp = "encoder.down_blocks." +
                std::to_string(i * (cfg.num_res_blocks + 1) + cfg.num_res_blocks) + ".resample.1.";
            b.add(dp + "weight", {out_dim, out_dim, 3, 3},
                  f32_seq(static_cast<std::size_t>(out_dim) * out_dim * 9, 0.02f, i + 51));
            b.add(dp + "bias", {out_dim}, f32_seq(static_cast<std::size_t>(out_dim), 0.001f, i + 52));
        }
    }

    const int mid_C = dims.back();
    emit_resnet(b, "encoder.mid_block.resnets.0.", mid_C, mid_C);
    emit_resnet(b, "encoder.mid_block.resnets.1.", mid_C, mid_C);
    emit_attention(b, "encoder.mid_block.attentions.0.", mid_C);

    b.add("encoder.norm_out.gamma", {mid_C, 1, 1, 1}, f32_ones(static_cast<std::size_t>(mid_C)));
    const int twoZ = 2 * cfg.z_dim;
    b.add("encoder.conv_out.weight", {twoZ, mid_C, 3, 3, 3},
          f32_seq(static_cast<std::size_t>(twoZ) * mid_C * 27, 0.04f, 61));
    b.add("encoder.conv_out.bias", {twoZ}, f32_seq(static_cast<std::size_t>(twoZ), 0.001f, 62));

    b.add("quant_conv.weight", {twoZ, twoZ, 1, 1, 1},
          f32_seq(static_cast<std::size_t>(twoZ) * twoZ, 0.1f, 71));
    b.add("quant_conv.bias", {twoZ}, f32_seq(static_cast<std::size_t>(twoZ), 0.001f, 72));
}

// Download a tensor to host FP32 regardless of storage dtype (FP32/FP16/
// BF16) — bdtest::bd_download only special-cases FP16.
std::vector<float> download_any(const bt::Tensor& t) {
    if (t.dtype == bt::Dtype::BF16) {
        bt::Tensor f32;
        bt::cast(t, f32, bt::Dtype::FP32);
        return bdtest::bd_download(f32);
    }
    return bdtest::bd_download(t);
}

}  // namespace

// ─── Part 1: synthetic-weights smoke test ──────────────────────────────────

static void test_synthetic() {
    vq::Config cfg = make_synth_config();

    Builder db;
    build_decoder_fixture(db, cfg);
    auto dpath = std::filesystem::temp_directory_path() / "brodiffusion_krea2_vae_decoder_test.safetensors";
    db.write(dpath);

    Builder eb;
    build_encoder_fixture(eb, cfg);
    auto epath = std::filesystem::temp_directory_path() / "brodiffusion_krea2_vae_encoder_test.safetensors";
    eb.write(epath);

    const int H_lat = 2, W_lat = 2;
    const int total_ds = 1 << (static_cast<int>(cfg.dim_mult.size()) - 1);   // 4 (2 transitions)
    const int H_out = H_lat * total_ds, W_out = W_lat * total_ds;
    const int out_elems = cfg.input_channels * H_out * W_out;

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
        CHECK(out.dtype == brodiffusion::compute_dtype());

        std::vector<float> vals1 = download_any(out);
        CHECK(static_cast<int>(vals1.size()) == out_elems);
        int nonfinite = 0;
        for (float v : vals1) if (!bdtest::bd_finite(v)) ++nonfinite;
        CHECK(nonfinite == 0);

        dec.decode(latent, H_lat, W_lat, out);
        bt::sync_all();
        std::vector<float> vals2 = download_any(out);
        CHECK(vals1 == vals2);
    }

    {
        auto file = st::File::open(epath.string());
        vq::Encoder enc(cfg);
        enc.load_weights(file, "");

        const int H = 8, W = 8;   // multiple of total_ds(4)
        std::vector<float> img_h(static_cast<std::size_t>(cfg.input_channels) * H * W);
        for (std::size_t i = 0; i < img_h.size(); ++i) {
            img_h[i] = (static_cast<float>(i % 9) - 4.0f) * 0.1f;
        }
        bt::Tensor img = bdtest::bd_upload(img_h, 1, cfg.input_channels * H * W);

        bt::Tensor lat_out;
        enc.encode(img, H, W, nullptr, lat_out);
        bt::sync_all();

        const int lat_elems = cfg.z_dim * (H / total_ds) * (W / total_ds);
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
    }

    std::error_code ec;
    std::filesystem::remove(dpath, ec);
    std::filesystem::remove(epath, ec);
}

// ─── Part 2: real-weights decode (gated) ───────────────────────────────────

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
        weights_dir() + "/krea-2-raw/vae/diffusion_pytorch_model.safetensors";
    if (!std::filesystem::exists(ckpt)) {
        std::printf("krea2_vae: skipped (no weights)\n");
        return;
    }

    vq::Config cfg;   // real Krea 2 / Qwen-Image config (header defaults)
    const int H_lat = 8, W_lat = 8;
    const int H_out = H_lat * 8, W_out = W_lat * 8;   // f8 spatial downsample
    const int out_elems = cfg.input_channels * H_out * W_out;

    auto file = st::File::open(ckpt);
    vq::Decoder dec(cfg);
    dec.load_weights(file, "");

    std::vector<float> latent_h(static_cast<std::size_t>(cfg.z_dim) * H_lat * W_lat);
    for (std::size_t i = 0; i < latent_h.size(); ++i) {
        latent_h[i] = std::sin(0.05f * static_cast<float>(i) + 0.3f * static_cast<float>(i % 17)) * 0.5f;
    }
    bt::Tensor latent = bdtest::bd_upload(latent_h, 1, cfg.z_dim * H_lat * W_lat);

    bt::Tensor out;
    dec.decode(latent, H_lat, W_lat, out);
    bt::sync_all();

    CHECK(out.rows == 1);
    CHECK(out.cols == out_elems);

    std::vector<float> vals1 = download_any(out);
    CHECK(static_cast<int>(vals1.size()) == out_elems);

    int nonfinite = 0;
    float lo = 1e30f, hi = -1e30f;
    for (float v : vals1) {
        if (!bdtest::bd_finite(v)) ++nonfinite;
        else { lo = std::min(lo, v); hi = std::max(hi, v); }
    }
    CHECK(nonfinite == 0);
    CHECK(lo > -4.0f && hi < 4.0f);
    std::printf("krea2_vae: real-weights range [%.4f, %.4f]\n", lo, hi);

    dec.decode(latent, H_lat, W_lat, out);
    bt::sync_all();
    std::vector<float> vals2 = download_any(out);
    CHECK(vals1 == vals2);
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
    } catch (const std::exception& e) {
        std::fprintf(stderr, "krea2_vae: synthetic test exception: %s\n", e.what());
        return 1;
    }

    try {
        test_real_weights();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "krea2_vae: real-weights test exception: %s\n", e.what());
        return 1;
    }

    if (g_failures == 0) std::printf("krea2_vae: OK\n");
    else std::fprintf(stderr, "krea2_vae: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
