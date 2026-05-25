// Phase D1 — UNet ControlNet residual hook test.
//
// Verifies that the new residual-aware forward() overload is wired correctly:
//   1. Empty `down_residuals` + null `mid_residual` is bit-identical to the
//      legacy cached forward.
//   2. All-zero residual tensors (12 down + mid) are bit-identical to the
//      legacy cached forward (FP zero-add is exact).
//   3. A non-null residual changes the output (the wiring is real).
//
// Reuses the scaled-down UNet fixture from test_unet.cpp, copy-pasted here
// for self-contained linking. Channel counts and block configuration are
// SD1.5's (nb=4, layers_per_block=2-ish; here layers_per_block=1 because
// the fixture mirrors test_unet.cpp). For layers_per_block=1 the expected
// skip count is 1 + 1 + 1 + 1 + 1 + 3 downsamplers = 1 + 4 + 3 = 8.
// (Stock SD1.5 with layers_per_block=2 has 12 skips: 1 + 4*2 + 3 = 12. Both
// counts are computed inside forward_impl_.)

#include "brodiffusion/detail/compute.h"
#include "brotensor/safetensors.h"
#include "brodiffusion/unet.h"

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

namespace un = brodiffusion::unet;
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

}  // namespace

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    un::UNetConfig cfg;
    cfg.in_channels         = 4;
    cfg.out_channels        = 4;
    cfg.block_out_channels  = {8, 16, 32, 32};
    cfg.layers_per_block    = 1;
    cfg.norm_num_groups     = 2;
    cfg.eps                 = 1e-5f;
    cfg.cross_attention_dim = 8;
    cfg.attention_head_dim  = 4;
    cfg.time_embed_dim_mult = 2;

    const int nb        = static_cast<int>(cfg.block_out_channels.size());
    const int first_C   = cfg.block_out_channels.front();
    const int mid_C     = cfg.block_out_channels.back();
    const int temb_dim  = first_C * cfg.time_embed_dim_mult;
    const int freq_dim  = first_C;
    const int ctx_dim   = cfg.cross_attention_dim;

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
                        static_cast<std::size_t>(5000 + i * 10 + j));
            if (has_attn) {
                const std::string tp = "up_blocks." + std::to_string(i) +
                                       ".attentions." + std::to_string(j) + ".";
                emit_transformer(b, tp, C_out, ctx_dim,
                                 static_cast<std::size_t>(6000 + i * 10 + j));
            }
        }
        if (has_upsm) {
            const std::string sp = "up_blocks." + std::to_string(i) +
                                   ".upsamplers.0.conv.";
            b.add(sp + "weight", {C_out, C_out, 3, 3},
                  fp16_seq(static_cast<std::size_t>(C_out) * C_out * 9, 0.02f,
                           static_cast<std::size_t>(7000 + i)));
            b.add(sp + "bias",   {C_out}, fp16_zeros(C_out));
        }
        C_up_prev = C_out;
    }
    b.add("conv_norm_out.weight", {first_C}, fp16_ones(first_C));
    b.add("conv_norm_out.bias",   {first_C}, fp16_zeros(first_C));
    b.add("conv_out.weight", {cfg.out_channels, first_C, 3, 3},
          fp16_seq(static_cast<std::size_t>(cfg.out_channels) * first_C * 9, 0.04f));
    b.add("conv_out.bias",   {cfg.out_channels}, fp16_zeros(cfg.out_channels));

    auto path = std::filesystem::temp_directory_path() /
                "brodiffusion_unet_residuals_test.safetensors";
    b.write(path);

    const int H = 8, W = 8;
    const int L_text = 4;

    auto file = st::File::open(path.string());
    un::UNet net(cfg);
    net.load_weights(file, "");

    // Synthesize a latent and a text context.
    std::vector<float> latent_h(static_cast<std::size_t>(cfg.in_channels) * H * W);
    for (std::size_t i = 0; i < latent_h.size(); ++i) {
        latent_h[i] = (static_cast<float>(i % 5) - 2.0f) * 0.1f;
    }
    bt::Tensor latent = bdtest::bd_upload(latent_h, 1, cfg.in_channels * H * W);

    std::vector<float> ctx_h(static_cast<std::size_t>(L_text) * ctx_dim);
    for (std::size_t i = 0; i < ctx_h.size(); ++i) {
        ctx_h[i] = (static_cast<float>(i % 7) - 3.0f) * 0.05f;
    }
    bt::Tensor ctx = bdtest::bd_upload(ctx_h, L_text, ctx_dim);

    // Prime the K/V cache once; the residual-aware forward uses it.
    un::UNet::CrossAttnKVCache cache;
    net.prime_xattn_cache(ctx, cache);

    // Baseline: cached forward.
    bt::Tensor base_out;
    net.forward(latent, H, W, /*timestep=*/500.0f, ctx, cache, base_out);
    bt::sync_all();
    std::vector<float> base = bdtest::bd_download(base_out);

    // Compute the expected skip count for the model: 1 (conv_in) +
    // layers_per_block per stage + 1 per downsampler. For this fixture:
    // 1 + 1*4 + 3 = 8. (Stock SD1.5 with layers_per_block=2 is 12.)
    int expected_skips = 1;
    for (int i = 0; i < nb; ++i) {
        expected_skips += cfg.layers_per_block;
        if (i < nb - 1) ++expected_skips;
    }
    CHECK(expected_skips == 8);

    // ── 1. null residuals: empty vector + null mid → bit-identical ──────
    {
        bt::Tensor out;
        std::vector<const bt::Tensor*> down;  // empty → forward treats as nullptr
        net.forward(latent, H, W, 500.0f, ctx, cache, down,
                    /*mid_residual=*/nullptr, out);
        bt::sync_all();
        std::vector<float> v = bdtest::bd_download(out);
        CHECK(v == base);
    }

    // ── 2. zero residuals (all 8 down + mid) → bit-identical ────────────
    // Build per-skip zero tensors of the right shapes. The shapes mirror
    // the running NCHW stream at each push: conv_in pushes at first_C×H×W,
    // then each stage pushes (resnets) at C_out_i × Hc × Wc, plus downsampler.
    std::vector<bt::Tensor> zero_storage;
    zero_storage.reserve(static_cast<std::size_t>(expected_skips));
    auto push_zero = [&](int C, int Hh, int Ww) {
        const int cols = C * Hh * Ww;
        std::vector<float> z(static_cast<std::size_t>(cols), 0.0f);
        zero_storage.push_back(bdtest::bd_upload(z, 1, cols));
    };
    int Hc = H, Wc = W;
    push_zero(first_C, Hc, Wc);  // conv_in skip
    for (int i = 0; i < nb; ++i) {
        const int C_out = cfg.block_out_channels[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg.layers_per_block; ++j) {
            push_zero(C_out, Hc, Wc);
        }
        if (i < nb - 1) {
            Hc /= 2; Wc /= 2;
            push_zero(C_out, Hc, Wc);
        }
    }
    CHECK(static_cast<int>(zero_storage.size()) == expected_skips);

    // Mid residual: at this point Hc, Wc are the bottom resolution and the
    // channel count is mid_C.
    std::vector<float> mid_zero_h(static_cast<std::size_t>(mid_C) * Hc * Wc, 0.0f);
    bt::Tensor mid_zero = bdtest::bd_upload(mid_zero_h, 1, mid_C * Hc * Wc);

    {
        std::vector<const bt::Tensor*> down(static_cast<std::size_t>(expected_skips));
        for (int i = 0; i < expected_skips; ++i) {
            down[static_cast<std::size_t>(i)] = &zero_storage[static_cast<std::size_t>(i)];
        }
        bt::Tensor out;
        net.forward(latent, H, W, 500.0f, ctx, cache, down, &mid_zero, out);
        bt::sync_all();
        std::vector<float> v = bdtest::bd_download(out);
        CHECK(v == base);
    }

    // ── 3. one nonzero residual changes the output ──────────────────────
    {
        // Replace the conv_in skip residual (index 0) with a small nonzero
        // pattern. Everything else stays zero / null.
        std::vector<float> rh(static_cast<std::size_t>(first_C) * H * W);
        for (std::size_t i = 0; i < rh.size(); ++i) {
            rh[i] = 0.05f * (static_cast<float>(i % 11) - 5.0f);
        }
        bt::Tensor r = bdtest::bd_upload(rh, 1, first_C * H * W);
        std::vector<const bt::Tensor*> down(static_cast<std::size_t>(expected_skips),
                                            nullptr);
        down[0] = &r;
        bt::Tensor out;
        net.forward(latent, H, W, 500.0f, ctx, cache, down,
                    /*mid_residual=*/nullptr, out);
        bt::sync_all();
        std::vector<float> v = bdtest::bd_download(out);
        CHECK(v.size() == base.size());
        bool any_diff = false;
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (v[i] != base[i]) { any_diff = true; break; }
        }
        CHECK(any_diff);
    }

    if (g_failures) {
        std::fprintf(stderr, "test_unet_residuals: %d failure(s)\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "test_unet_residuals: ok\n");
    return 0;
}
