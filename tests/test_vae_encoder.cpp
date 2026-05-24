// VAE encoder smoke test.
//
// Mirrors test_vae.cpp's scaled-down architecture (block_out_channels=[4,8,16,16],
// norm_num_groups=2, layers_per_block=2) keeping the SD1.5 encoder structure
// intact but with channels shrunk enough to enumerate the full tensor list.
// Image input (1, 3, 16, 16); latent output (1, 4, 2, 2).

#include "brodiffusion/detail/compute.h"
#include "brotensor/safetensors.h"
#include "brodiffusion/vae.h"

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

namespace vae = brodiffusion::vae;
namespace st  = brotensor::safetensors;
namespace bt  = brotensor;

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

std::vector<uint16_t> fp16_zeros(std::size_t n) {
    return std::vector<uint16_t>(n, 0);
}
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

void emit_resnet(Builder& b, const std::string& p, int C_in, int C_out) {
    b.add(p + "norm1.weight", {C_in},  fp16_ones(C_in));
    b.add(p + "norm1.bias",   {C_in},  fp16_zeros(C_in));
    b.add(p + "conv1.weight", {C_out, C_in, 3, 3},
          fp16_seq(static_cast<std::size_t>(C_out) * C_in * 9, 0.02f, p.size()));
    b.add(p + "conv1.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "norm2.weight", {C_out}, fp16_ones(C_out));
    b.add(p + "norm2.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "conv2.weight", {C_out, C_out, 3, 3},
          fp16_seq(static_cast<std::size_t>(C_out) * C_out * 9, 0.02f, p.size() + 1));
    b.add(p + "conv2.bias",   {C_out}, fp16_zeros(C_out));
    if (C_in != C_out) {
        b.add(p + "conv_shortcut.weight", {C_out, C_in, 1, 1},
              fp16_seq(static_cast<std::size_t>(C_out) * C_in, 0.05f, p.size() + 2));
        b.add(p + "conv_shortcut.bias",   {C_out}, fp16_zeros(C_out));
    }
}

}  // namespace

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    vae::EncoderConfig cfg;
    cfg.in_channels        = 4;
    cfg.out_channels       = 3;
    cfg.block_out_channels = {4, 8, 16, 16};
    cfg.layers_per_block   = 2;
    cfg.norm_num_groups    = 2;
    cfg.scaling_factor     = 1.0f;
    cfg.shift_factor       = 0.0f;
    cfg.eps                = 1e-6f;
    cfg.num_attention_heads = 1;

    const int first_C = cfg.block_out_channels.front();
    const int mid_C   = cfg.block_out_channels.back();
    const int nb      = static_cast<int>(cfg.block_out_channels.size());
    const int twoC    = 2 * cfg.in_channels;

    Builder b;
    const std::string p = "encoder.";

    // conv_in: (first_C, out_channels, 3, 3)
    b.add(p + "conv_in.weight", {first_C, cfg.out_channels, 3, 3},
          fp16_seq(static_cast<std::size_t>(first_C) * cfg.out_channels * 9, 0.05f));
    b.add(p + "conv_in.bias",   {first_C}, fp16_zeros(first_C));

    // down_blocks
    int C_prev = first_C;
    for (int i = 0; i < nb; ++i) {
        const int C_block = cfg.block_out_channels[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg.layers_per_block; ++j) {
            const int Ci = (j == 0) ? C_prev : C_block;
            emit_resnet(b,
                p + "down_blocks." + std::to_string(i) + ".resnets." + std::to_string(j) + ".",
                Ci, C_block);
        }
        if (i + 1 < nb) {
            const std::string dp = p + "down_blocks." + std::to_string(i) + ".downsamplers.0.conv.";
            b.add(dp + "weight", {C_block, C_block, 3, 3},
                  fp16_seq(static_cast<std::size_t>(C_block) * C_block * 9, 0.02f, i + 23));
            b.add(dp + "bias",   {C_block}, fp16_zeros(C_block));
        }
        C_prev = C_block;
    }

    // mid_block resnets + attention.
    emit_resnet(b, p + "mid_block.resnets.0.", mid_C, mid_C);
    emit_resnet(b, p + "mid_block.resnets.1.", mid_C, mid_C);

    const std::string ap = p + "mid_block.attentions.0.";
    b.add(ap + "group_norm.weight", {mid_C}, fp16_ones(mid_C));
    b.add(ap + "group_norm.bias",   {mid_C}, fp16_zeros(mid_C));
    b.add(ap + "query.weight",      {mid_C, mid_C},
          fp16_seq(static_cast<std::size_t>(mid_C) * mid_C, 0.03f, 11));
    b.add(ap + "query.bias",        {mid_C}, fp16_zeros(mid_C));
    b.add(ap + "key.weight",        {mid_C, mid_C},
          fp16_seq(static_cast<std::size_t>(mid_C) * mid_C, 0.03f, 13));
    b.add(ap + "key.bias",          {mid_C}, fp16_zeros(mid_C));
    b.add(ap + "value.weight",      {mid_C, mid_C},
          fp16_seq(static_cast<std::size_t>(mid_C) * mid_C, 0.03f, 17));
    b.add(ap + "value.bias",        {mid_C}, fp16_zeros(mid_C));
    b.add(ap + "proj_attn.weight",  {mid_C, mid_C},
          fp16_seq(static_cast<std::size_t>(mid_C) * mid_C, 0.03f, 19));
    b.add(ap + "proj_attn.bias",    {mid_C}, fp16_zeros(mid_C));

    b.add(p + "conv_norm_out.weight", {mid_C}, fp16_ones(mid_C));
    b.add(p + "conv_norm_out.bias",   {mid_C}, fp16_zeros(mid_C));
    b.add(p + "conv_out.weight", {twoC, mid_C, 3, 3},
          fp16_seq(static_cast<std::size_t>(twoC) * mid_C * 9, 0.04f));
    b.add(p + "conv_out.bias",   {twoC}, fp16_zeros(twoC));

    // quant_conv lives at root (sibling of "encoder.").
    b.add("quant_conv.weight", {twoC, twoC, 1, 1},
          fp16_seq(static_cast<std::size_t>(twoC) * twoC, 0.05f, 31));
    b.add("quant_conv.bias",   {twoC}, fp16_zeros(twoC));

    auto path = std::filesystem::temp_directory_path() /
                "brodiffusion_vae_encoder_test.safetensors";
    b.write(path);

    const int H = 16, W = 16;
    const int H_lat = H / 8, W_lat = W / 8;
    const int img_elems = cfg.out_channels * H * W;
    const int lat_elems = cfg.in_channels  * H_lat * W_lat;

    // Synthesize an image in [-1, 1] with small varied values.
    std::vector<float> img_h(static_cast<std::size_t>(img_elems));
    for (std::size_t i = 0; i < img_h.size(); ++i) {
        float v = (static_cast<float>(i % 11) - 5.0f) * 0.1f;
        if (v >  1.0f) v =  1.0f;
        if (v < -1.0f) v = -1.0f;
        img_h[i] = v;
    }

    std::vector<float> det1, det2;
    {
        auto file = st::File::open(path.string());
        vae::Encoder enc(cfg);
        enc.load_weights(file, "encoder.");

        bt::Tensor image = bdtest::bd_upload(img_h, 1, img_elems);

        // ── 1. encode_shape_and_finite (deterministic) ────────────────────
        bt::Tensor out;
        enc.encode(image, H, W, /*eps=*/nullptr, out);
        bt::sync_all();
        CHECK(out.rows == 1);
        CHECK(out.cols == lat_elems);
        CHECK(out.dtype == brodiffusion::compute_dtype());

        det1 = bdtest::bd_download(out);
        int nonfinite = 0;
        for (float v : det1) if (!bdtest::bd_finite(v)) ++nonfinite;
        CHECK(nonfinite == 0);

        // ── 2. encode_determinism ─────────────────────────────────────────
        bt::Tensor out2;
        enc.encode(image, H, W, /*eps=*/nullptr, out2);
        bt::sync_all();
        det2 = bdtest::bd_download(out2);
        CHECK(det1 == det2);

        // ── 3. encode_sampling_diff ───────────────────────────────────────
        // eps = zeros → output should equal deterministic mode.
        std::vector<float> zeros(static_cast<std::size_t>(lat_elems), 0.0f);
        std::vector<float> ones(static_cast<std::size_t>(lat_elems),  1.0f);
        bt::Tensor eps_zero = bdtest::bd_upload(zeros, 1, lat_elems);
        bt::Tensor eps_one  = bdtest::bd_upload(ones,  1, lat_elems);

        bt::Tensor out_z, out_o;
        enc.encode(image, H, W, &eps_zero, out_z);
        bt::sync_all();
        std::vector<float> vals_z = bdtest::bd_download(out_z);
        CHECK(vals_z == det1);

        enc.encode(image, H, W, &eps_one, out_o);
        bt::sync_all();
        std::vector<float> vals_o = bdtest::bd_download(out_o);
        CHECK(vals_o != vals_z);
        int o_bad = 0;
        for (float v : vals_o) if (!bdtest::bd_finite(v)) ++o_bad;
        CHECK(o_bad == 0);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("vae_encoder: OK\n");
    else std::fprintf(stderr, "vae_encoder: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
