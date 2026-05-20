// W8A16 INT8 weight-only quantization correctness test.
//
// Builds the same scaled-down SD1.5 U-Net used by test_unet (block_out_channels
// = {8, 16, 32, 32}, layers_per_block=1, attention_head_dim=4,
// cross_attention_dim=8, norm_num_groups=2), runs a forward pass twice:
//   1) FP16 baseline
//   2) INT8 weight-only quantised UNet (UNetConfig::quantize_weights = true)
// and checks that the max-abs error is bounded.
//
// The two UNets share the same safetensors fixture; only the config flag and
// the call to finalize_weights() differ. Also exercises the lora-after-finalize
// guard and idempotent finalize_weights().

#include "brodiffusion/safetensors.h"
#include "brodiffusion/unet.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace un = brodiffusion::unet;
namespace st = brodiffusion::safetensors;
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
// Deterministic small Gaussian-ish weights — sufficient signal for quant to
// produce non-trivial scales without saturating the FP16 forward path.
std::vector<uint16_t> fp16_rand(std::size_t n, float scale, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    std::vector<uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = bt::fp32_to_fp16_bits(scale * nrm(rng));
    }
    return out;
}

void emit_resnet(Builder& b, const std::string& p, int C_in, int C_out,
                 int temb_dim, std::uint64_t seed) {
    b.add(p + "norm1.weight", {C_in},  fp16_ones(C_in));
    b.add(p + "norm1.bias",   {C_in},  fp16_zeros(C_in));
    b.add(p + "conv1.weight", {C_out, C_in, 3, 3},
          fp16_rand(static_cast<std::size_t>(C_out) * C_in * 9, 0.05f, seed));
    b.add(p + "conv1.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "time_emb_proj.weight", {C_out, temb_dim},
          fp16_rand(static_cast<std::size_t>(C_out) * temb_dim, 0.05f, seed + 1));
    b.add(p + "time_emb_proj.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "norm2.weight", {C_out}, fp16_ones(C_out));
    b.add(p + "norm2.bias",   {C_out}, fp16_zeros(C_out));
    b.add(p + "conv2.weight", {C_out, C_out, 3, 3},
          fp16_rand(static_cast<std::size_t>(C_out) * C_out * 9, 0.05f, seed + 2));
    b.add(p + "conv2.bias",   {C_out}, fp16_zeros(C_out));
    if (C_in != C_out) {
        b.add(p + "conv_shortcut.weight", {C_out, C_in, 1, 1},
              fp16_rand(static_cast<std::size_t>(C_out) * C_in, 0.1f, seed + 3));
        b.add(p + "conv_shortcut.bias",   {C_out}, fp16_zeros(C_out));
    }
}

void emit_transformer(Builder& b, const std::string& p, int C, int ctx_dim,
                      std::uint64_t seed) {
    const int ff_inner = 4 * C;
    b.add(p + "norm.weight", {C}, fp16_ones(C));
    b.add(p + "norm.bias",   {C}, fp16_zeros(C));
    b.add(p + "proj_in.weight",  {C, C, 1, 1},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.05f, seed));
    b.add(p + "proj_in.bias",    {C}, fp16_zeros(C));
    b.add(p + "proj_out.weight", {C, C, 1, 1},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.05f, seed + 1));
    b.add(p + "proj_out.bias",   {C}, fp16_zeros(C));
    const std::string bp = p + "transformer_blocks.0.";
    b.add(bp + "norm1.weight", {C}, fp16_ones(C));
    b.add(bp + "norm1.bias",   {C}, fp16_zeros(C));
    b.add(bp + "attn1.to_q.weight", {C, C},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.05f, seed + 2));
    b.add(bp + "attn1.to_k.weight", {C, C},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.05f, seed + 3));
    b.add(bp + "attn1.to_v.weight", {C, C},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.05f, seed + 4));
    b.add(bp + "attn1.to_out.0.weight", {C, C},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.05f, seed + 5));
    b.add(bp + "attn1.to_out.0.bias",   {C}, fp16_zeros(C));
    b.add(bp + "norm2.weight", {C}, fp16_ones(C));
    b.add(bp + "norm2.bias",   {C}, fp16_zeros(C));
    b.add(bp + "attn2.to_q.weight", {C, C},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.05f, seed + 6));
    b.add(bp + "attn2.to_k.weight", {C, ctx_dim},
          fp16_rand(static_cast<std::size_t>(C) * ctx_dim, 0.05f, seed + 7));
    b.add(bp + "attn2.to_v.weight", {C, ctx_dim},
          fp16_rand(static_cast<std::size_t>(C) * ctx_dim, 0.05f, seed + 8));
    b.add(bp + "attn2.to_out.0.weight", {C, C},
          fp16_rand(static_cast<std::size_t>(C) * C, 0.05f, seed + 9));
    b.add(bp + "attn2.to_out.0.bias",   {C}, fp16_zeros(C));
    b.add(bp + "norm3.weight", {C}, fp16_ones(C));
    b.add(bp + "norm3.bias",   {C}, fp16_zeros(C));
    b.add(bp + "ff.net.0.proj.weight", {2 * ff_inner, C},
          fp16_rand(static_cast<std::size_t>(2 * ff_inner) * C, 0.04f, seed + 10));
    b.add(bp + "ff.net.0.proj.bias",   {2 * ff_inner}, fp16_zeros(2 * ff_inner));
    b.add(bp + "ff.net.2.weight", {C, ff_inner},
          fp16_rand(static_cast<std::size_t>(C) * ff_inner, 0.04f, seed + 11));
    b.add(bp + "ff.net.2.bias",   {C}, fp16_zeros(C));
}

void write_fixture(const std::filesystem::path& path,
                   const un::UNetConfig& cfg) {
    const int nb        = static_cast<int>(cfg.block_out_channels.size());
    const int first_C   = cfg.block_out_channels.front();
    const int mid_C     = cfg.block_out_channels.back();
    const int temb_dim  = first_C * cfg.time_embed_dim_mult;
    const int freq_dim  = first_C;
    const int ctx_dim   = cfg.cross_attention_dim;

    Builder b;
    b.add("conv_in.weight", {first_C, cfg.in_channels, 3, 3},
          fp16_rand(static_cast<std::size_t>(first_C) * cfg.in_channels * 9, 0.1f, 1));
    b.add("conv_in.bias",   {first_C}, fp16_zeros(first_C));
    b.add("time_embedding.linear_1.weight", {temb_dim, freq_dim},
          fp16_rand(static_cast<std::size_t>(temb_dim) * freq_dim, 0.1f, 100));
    b.add("time_embedding.linear_1.bias",   {temb_dim}, fp16_zeros(temb_dim));
    b.add("time_embedding.linear_2.weight", {temb_dim, temb_dim},
          fp16_rand(static_cast<std::size_t>(temb_dim) * temb_dim, 0.1f, 101));
    b.add("time_embedding.linear_2.bias",   {temb_dim}, fp16_zeros(temb_dim));

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
                        static_cast<std::uint64_t>(1000 + i * 10 + j));
            if (has_attn) {
                const std::string tp = "down_blocks." + std::to_string(i) +
                                       ".attentions." + std::to_string(j) + ".";
                emit_transformer(b, tp, C_out, ctx_dim,
                                 static_cast<std::uint64_t>(2000 + i * 10 + j));
            }
        }
        if (has_downsm) {
            const std::string sp = "down_blocks." + std::to_string(i) +
                                   ".downsamplers.0.conv.";
            b.add(sp + "weight", {C_out, C_out, 3, 3},
                  fp16_rand(static_cast<std::size_t>(C_out) * C_out * 9, 0.05f,
                            static_cast<std::uint64_t>(3000 + i)));
            b.add(sp + "bias", {C_out}, fp16_zeros(C_out));
        }
        C_prev = C_out;
    }
    emit_resnet(b, "mid_block.resnets.0.", mid_C, mid_C, temb_dim, 4000);
    emit_transformer(b, "mid_block.attentions.0.", mid_C, ctx_dim, 4100);
    emit_resnet(b, "mid_block.resnets.1.", mid_C, mid_C, temb_dim, 4200);

    std::vector<int> skip_stack;
    skip_stack.push_back(first_C);
    for (int i = 0; i < nb; ++i) {
        const int Cb = cfg.block_out_channels[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg.layers_per_block; ++j) skip_stack.push_back(Cb);
        if (i < nb - 1) skip_stack.push_back(Cb);
    }
    int C_up_prev = mid_C;
    for (int i = 0; i < nb; ++i) {
        const int C_out = cfg.block_out_channels[static_cast<std::size_t>(nb - 1 - i)];
        const bool has_attn = (i > 0);
        const bool has_upsm = (i < nb - 1);
        const int layers = cfg.layers_per_block + 1;
        for (int j = 0; j < layers; ++j) {
            const int Cskip = skip_stack.back();
            skip_stack.pop_back();
            const int C_h = (j == 0) ? C_up_prev : C_out;
            const int Ci  = C_h + Cskip;
            const std::string rp = "up_blocks." + std::to_string(i) +
                                   ".resnets." + std::to_string(j) + ".";
            emit_resnet(b, rp, Ci, C_out, temb_dim,
                        static_cast<std::uint64_t>(5000 + i * 10 + j));
            if (has_attn) {
                const std::string tp = "up_blocks." + std::to_string(i) +
                                       ".attentions." + std::to_string(j) + ".";
                emit_transformer(b, tp, C_out, ctx_dim,
                                 static_cast<std::uint64_t>(6000 + i * 10 + j));
            }
        }
        if (has_upsm) {
            const std::string sp = "up_blocks." + std::to_string(i) +
                                   ".upsamplers.0.conv.";
            b.add(sp + "weight", {C_out, C_out, 3, 3},
                  fp16_rand(static_cast<std::size_t>(C_out) * C_out * 9, 0.05f,
                            static_cast<std::uint64_t>(7000 + i)));
            b.add(sp + "bias", {C_out}, fp16_zeros(C_out));
        }
        C_up_prev = C_out;
    }
    b.add("conv_norm_out.weight", {first_C}, fp16_ones(first_C));
    b.add("conv_norm_out.bias",   {first_C}, fp16_zeros(first_C));
    b.add("conv_out.weight", {cfg.out_channels, first_C, 3, 3},
          fp16_rand(static_cast<std::size_t>(cfg.out_channels) * first_C * 9, 0.1f, 8000));
    b.add("conv_out.bias",   {cfg.out_channels}, fp16_zeros(cfg.out_channels));
    b.write(path);
}

}  // namespace

int main() {
    try { bt::init(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }
    if (!bt::is_available(bt::Device::CUDA) &&
        !bt::is_available(bt::Device::Metal)) {
        std::fprintf(stderr,
                     "INT8 quantization is GPU-only — skipping\n");
        return 0;
    }

    un::UNetConfig cfg;
    cfg.in_channels = 4;
    cfg.out_channels = 4;
    cfg.block_out_channels = {8, 16, 32, 32};
    cfg.layers_per_block = 1;
    cfg.norm_num_groups = 2;
    cfg.cross_attention_dim = 8;
    cfg.attention_head_dim = 4;
    cfg.time_embed_dim_mult = 2;

    auto path = std::filesystem::temp_directory_path() /
                "brodiffusion_quantize_unet_test.safetensors";
    write_fixture(path, cfg);

    const int H = 8, W = 8;
    const int L_text = 4;
    const int ctx_dim = cfg.cross_attention_dim;
    const int out_elems = cfg.out_channels * H * W;

    // Prepare shared inputs.
    std::vector<uint16_t> latent_h(
        static_cast<std::size_t>(cfg.in_channels) * H * W);
    for (std::size_t i = 0; i < latent_h.size(); ++i) {
        float v = (static_cast<float>(i % 7) - 3.0f) * 0.1f;
        latent_h[i] = bt::fp32_to_fp16_bits(v);
    }
    std::vector<uint16_t> ctx_h(
        static_cast<std::size_t>(L_text) * ctx_dim);
    for (std::size_t i = 0; i < ctx_h.size(); ++i) {
        float v = (static_cast<float>(i % 5) - 2.0f) * 0.05f;
        ctx_h[i] = bt::fp32_to_fp16_bits(v);
    }

    auto file = st::File::open(path.string());

    std::vector<uint16_t> bits_fp16(static_cast<std::size_t>(out_elems));
    std::vector<uint16_t> bits_int8(static_cast<std::size_t>(out_elems));

    // ── FP16 baseline ─────────────────────────────────────────────────────
    {
        un::UNet net(cfg);
        net.load_weights(file, "");
        net.finalize_weights();              // no-op when quantize_weights=false
        CHECK(net.is_finalized());
        net.finalize_weights();              // idempotent

        bt::Tensor latent, ctx, out;
        latent = brotensor::Tensor::from_host_fp16(latent_h.data(), 1, cfg.in_channels * H * W);
        ctx = brotensor::Tensor::from_host_fp16(ctx_h.data(), L_text, ctx_dim);
        net.forward(latent, H, W, 500.0f, ctx, out);
        bt::sync_all();
        out.copy_to_host_fp16(bits_fp16.data());
        bt::sync_all();
    }

    // ── INT8 (W8A16) quantised ─────────────────────────────────────────────
    {
        un::UNetConfig qcfg = cfg;
        qcfg.quantize_weights = true;
        un::UNet net(qcfg);
        net.load_weights(file, "");

        // Verify the lora-after-finalize guard.
        net.finalize_weights();
        CHECK(net.is_finalized());

        bt::Tensor latent, ctx, out;
        latent = brotensor::Tensor::from_host_fp16(latent_h.data(), 1, cfg.in_channels * H * W);
        ctx = brotensor::Tensor::from_host_fp16(ctx_h.data(), L_text, ctx_dim);
        net.forward(latent, H, W, 500.0f, ctx, out);
        bt::sync_all();
        out.copy_to_host_fp16(bits_int8.data());
        bt::sync_all();
    }

    // ── Compare ───────────────────────────────────────────────────────────
    float max_abs_err = 0.0f;
    float max_abs_ref = 0.0f;
    int nonfinite = 0;
    for (int i = 0; i < out_elems; ++i) {
        float a = bt::fp16_bits_to_fp32(bits_fp16[i]);
        float b = bt::fp16_bits_to_fp32(bits_int8[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; continue; }
        float e = std::fabs(a - b);
        if (e > max_abs_err) max_abs_err = e;
        if (std::fabs(a) > max_abs_ref) max_abs_ref = std::fabs(a);
    }
    CHECK(nonfinite == 0);
    std::printf("quantize_unet: max_abs_err=%.4f  max_abs_ref=%.4f\n",
                max_abs_err, max_abs_ref);
    // Sanity bound. Per-element absolute error on this synthetic fixture sits
    // around 0.05; we use a slightly larger threshold (0.08) because the
    // synthetic fixture has output magnitudes ~2x what SD1.5 typically
    // produces, so 0.05 / 2 ≈ 0.025 is the comparable per-magnitude bound
    // and 0.08 / 2 = 0.04 leaves a small margin for FP16 rounding drift.
    CHECK(max_abs_err < 0.08f);

    // Lora-after-finalize guard.
    {
        un::UNetConfig qcfg = cfg;
        qcfg.quantize_weights = true;
        un::UNet net(qcfg);
        net.load_weights(file, "");
        net.finalize_weights();
        bool threw = false;
        try {
            // Construct dummy LoRA views — should fail before touching them.
            // The guard is the first check in apply_lora_delta, so any path
            // suffices.
            st::TensorView dummy_down, dummy_up;
            net.apply_lora_delta("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q",
                                 dummy_down, dummy_up, 1.0f);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("quantize_unet: OK\n");
    else std::fprintf(stderr, "quantize_unet: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
