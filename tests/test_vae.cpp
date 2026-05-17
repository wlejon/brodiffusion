// VAE decoder smoke test.
//
// Builds a scaled-down VAE (block_out_channels=[4,8,16,16], norm_num_groups=2,
// layers_per_block=2) keeping the SD1.5 architecture intact but with channel
// counts shrunk enough that the full tensor list fits in a fixture. Latent is
// (1, 4, 2, 2); output (1, 3, 16, 16) after three 2x upsamples.
//
// Verifies: shape + dtype of the decoded image, no Inf/NaN in any output bit
// pattern, and determinism across two consecutive decodes. Numerical accuracy
// vs the reference VAE is left to a future real-weights integration test.

#include "brodiffusion/safetensors.h"
#include "brodiffusion/vae.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace vae = brodiffusion::vae;
namespace st  = brodiffusion::safetensors;
namespace bt  = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// ─── safetensors fixture builder (FP16-only) ───────────────────────────────

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
// Small values clustered around zero so the cascade of convs doesn't blow up.
std::vector<uint16_t> fp16_seq(std::size_t n, float scale, std::size_t salt = 0) {
    std::vector<uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        float s = (static_cast<float>((i + salt) % 7) - 3.0f) * scale;
        out[i] = bt::fp32_to_fp16_bits(s);
    }
    return out;
}

bool is_finite_fp16(uint16_t bits) {
    return ((bits >> 10) & 0x1F) != 0x1F;
}

// Emit one resnet's tensors under prefix p (already ending with ".").
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

// ─── test ──────────────────────────────────────────────────────────────────

int main() {
    try {
        bt::cuda_init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cuda_init failed: %s\n", e.what());
        return 1;
    }

    vae::DecoderConfig cfg;
    cfg.in_channels       = 4;
    cfg.out_channels      = 3;
    cfg.block_out_channels = {4, 8, 16, 16};
    cfg.layers_per_block  = 2;
    cfg.norm_num_groups   = 2;
    cfg.scaling_factor    = 1.0f;   // disable; we're feeding synthetic data
    cfg.eps               = 1e-6f;
    cfg.num_attention_heads = 1;

    const int mid_C  = cfg.block_out_channels.back();
    const int first_C = cfg.block_out_channels.front();
    const int nb     = static_cast<int>(cfg.block_out_channels.size());

    Builder b;
    const std::string p = "decoder.";

    // conv_in: (mid_C, in_channels, 3, 3)
    b.add(p + "conv_in.weight", {mid_C, cfg.in_channels, 3, 3},
          fp16_seq(static_cast<std::size_t>(mid_C) * cfg.in_channels * 9, 0.05f));
    b.add(p + "conv_in.bias",   {mid_C}, fp16_zeros(mid_C));

    // mid_block resnets + attention
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

    // up_blocks
    int C_prev = mid_C;
    for (int i = 0; i < nb; ++i) {
        int C_block = cfg.block_out_channels[static_cast<std::size_t>(nb - 1 - i)];
        for (int j = 0; j <= cfg.layers_per_block; ++j) {
            int Ci = (j == 0) ? C_prev : C_block;
            emit_resnet(b,
                p + "up_blocks." + std::to_string(i) + ".resnets." + std::to_string(j) + ".",
                Ci, C_block);
        }
        if (i + 1 < nb) {
            const std::string up = p + "up_blocks." + std::to_string(i) + ".upsamplers.0.conv.";
            b.add(up + "weight", {C_block, C_block, 3, 3},
                  fp16_seq(static_cast<std::size_t>(C_block) * C_block * 9, 0.02f, i + 23));
            b.add(up + "bias",   {C_block}, fp16_zeros(C_block));
        }
        C_prev = C_block;
    }

    b.add(p + "conv_norm_out.weight", {first_C}, fp16_ones(first_C));
    b.add(p + "conv_norm_out.bias",   {first_C}, fp16_zeros(first_C));
    b.add(p + "conv_out.weight", {cfg.out_channels, first_C, 3, 3},
          fp16_seq(static_cast<std::size_t>(cfg.out_channels) * first_C * 9, 0.04f));
    b.add(p + "conv_out.bias",   {cfg.out_channels}, fp16_zeros(cfg.out_channels));

    auto path = std::filesystem::temp_directory_path() / "brodiffusion_vae_test.safetensors";
    b.write(path);

    const int H_lat = 2, W_lat = 2;
    const int H_out = H_lat * 8;          // 16; three 2x upsamples
    const int W_out = W_lat * 8;
    const int out_elems = cfg.out_channels * H_out * W_out;

    std::vector<uint16_t> bits1, bits2;
    {
        auto file = st::File::open(path.string());
        vae::Decoder dec(cfg);
        dec.load_weights(file, "decoder.");

        // Synthesize a latent (1, 4, 2, 2) with small varied values.
        std::vector<uint16_t> latent_h(
            static_cast<std::size_t>(cfg.in_channels) * H_lat * W_lat);
        for (std::size_t i = 0; i < latent_h.size(); ++i) {
            float v = (static_cast<float>(i % 5) - 2.0f) * 0.1f;
            latent_h[i] = bt::fp32_to_fp16_bits(v);
        }
        bt::GpuTensor latent;
        bt::upload_fp16(latent_h.data(),
                        1, cfg.in_channels * H_lat * W_lat, latent);

        bt::GpuTensor out;
        dec.decode(latent, H_lat, W_lat, out);
        bt::cuda_sync();

        CHECK(out.rows == 1);
        CHECK(out.cols == out_elems);
        CHECK(out.dtype == bt::Dtype::FP16);

        bits1.resize(static_cast<std::size_t>(out_elems));
        bt::download_fp16(out, bits1.data());
        bt::cuda_sync();

        int nonfinite = 0;
        for (uint16_t v : bits1) if (!is_finite_fp16(v)) ++nonfinite;
        CHECK(nonfinite == 0);

        dec.decode(latent, H_lat, W_lat, out);
        bt::cuda_sync();
        bits2.resize(bits1.size());
        bt::download_fp16(out, bits2.data());
        bt::cuda_sync();
        CHECK(std::memcmp(bits1.data(), bits2.data(), bits1.size() * 2) == 0);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("vae: OK\n");
    else std::fprintf(stderr, "vae: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
