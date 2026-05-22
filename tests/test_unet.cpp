// UNet2DConditionModel smoke test.
//
// Builds a scaled-down SD1.5 U-Net (block_out_channels=[8,16,32,32],
// layers_per_block=1, attention_head_dim=4, cross_attention_dim=8,
// norm_num_groups=2) preserving the full architecture but shrinking channel
// counts so the entire weight fixture fits in one file.
//
// Latent: (1, 4, 8, 8); text context: (4, 8). Output: (1, 4, 8, 8).
//
// Verifies: shape + dtype of the noise prediction, no Inf/NaN in any output
// bit pattern, and determinism across two consecutive forwards. Numerical
// fidelity vs the reference UNet is left to a future real-weights test.

#include "brodiffusion/detail/compute.h"
#include "brotensor/safetensors.h"
#include "brodiffusion/unet.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
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

    // conv_in
    b.add("conv_in.weight", {first_C, cfg.in_channels, 3, 3},
          fp16_seq(static_cast<std::size_t>(first_C) * cfg.in_channels * 9, 0.05f));
    b.add("conv_in.bias",   {first_C}, fp16_zeros(first_C));

    // time_embedding
    b.add("time_embedding.linear_1.weight", {temb_dim, freq_dim},
          fp16_seq(static_cast<std::size_t>(temb_dim) * freq_dim, 0.05f, 100));
    b.add("time_embedding.linear_1.bias",   {temb_dim}, fp16_zeros(temb_dim));
    b.add("time_embedding.linear_2.weight", {temb_dim, temb_dim},
          fp16_seq(static_cast<std::size_t>(temb_dim) * temb_dim, 0.05f, 101));
    b.add("time_embedding.linear_2.bias",   {temb_dim}, fp16_zeros(temb_dim));

    // LCM guidance-scale cond_proj — shape (freq_dim, cond_proj_dim), no bias.
    // Unused by the vanilla UNet (its load_weights only requests cond_proj
    // when time_cond_proj_dim > 0); the LCM UNet built from this same fixture
    // below does load it.
    const int cond_proj_dim = 4;
    b.add("time_embedding.cond_proj.weight", {freq_dim, cond_proj_dim},
          fp16_seq(static_cast<std::size_t>(freq_dim) * cond_proj_dim, 0.05f, 102));

    // down_blocks
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

    // mid_block
    emit_resnet(b, "mid_block.resnets.0.", mid_C, mid_C, temb_dim, 4000);
    emit_transformer(b, "mid_block.attentions.0.", mid_C, ctx_dim, 4100);
    emit_resnet(b, "mid_block.resnets.1.", mid_C, mid_C, temb_dim, 4200);

    // up_blocks (replay the skip stack to derive per-layer C_in).
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

    auto path = std::filesystem::temp_directory_path() / "brodiffusion_unet_test.safetensors";
    b.write(path);

    const int H = 8, W = 8;
    const int L_text = 4;
    const int out_elems = cfg.out_channels * H * W;

    std::vector<float> vals1, vals2;
    {
        auto file = st::File::open(path.string());
        un::UNet net(cfg);
        net.load_weights(file, "");

        // Synthesize a noisy latent (1, 4, 8, 8) with small varied values.
        std::vector<float> latent_h(
            static_cast<std::size_t>(cfg.in_channels) * H * W);
        for (std::size_t i = 0; i < latent_h.size(); ++i) {
            latent_h[i] = (static_cast<float>(i % 5) - 2.0f) * 0.1f;
        }
        bt::Tensor latent =
            bdtest::bd_upload(latent_h, 1, cfg.in_channels * H * W);

        // Synthesize a text context (L_text, cross_attention_dim).
        std::vector<float> ctx_h(static_cast<std::size_t>(L_text) * ctx_dim);
        for (std::size_t i = 0; i < ctx_h.size(); ++i) {
            ctx_h[i] = (static_cast<float>(i % 7) - 3.0f) * 0.05f;
        }
        bt::Tensor ctx = bdtest::bd_upload(ctx_h, L_text, ctx_dim);

        bt::Tensor out;
        net.forward(latent, H, W, /*timestep=*/500.0f, ctx, out);
        bt::sync_all();

        CHECK(out.rows == 1);
        CHECK(out.cols == out_elems);
        CHECK(out.dtype == brodiffusion::compute_dtype());

        vals1 = bdtest::bd_download(out);

        int nonfinite = 0;
        for (float v : vals1) if (!bdtest::bd_finite(v)) ++nonfinite;
        CHECK(nonfinite == 0);

        net.forward(latent, H, W, 500.0f, ctx, out);
        bt::sync_all();
        vals2 = bdtest::bd_download(out);
        CHECK(vals1 == vals2);

        // ── trace-mode forward ────────────────────────────────────────────
        // Same inputs, same UNet — exercises forward_trace, asserts the
        // CrossAttnTrace is populated with (Lq, Lk) softmax maps (rows sum
        // to ~1.0), and confirms the trace-path output matches the fast
        // path within FP16 numerical noise (different attn kernels = small
        // rounding drift even though the math is identical).
        un::UNet::CrossAttnTrace trace;
        net.forward_trace(latent, H, W, 500.0f, /*guidance_scale=*/nullptr, ctx,
                          /*attn_logit_biases=*/nullptr, &trace, out);
        bt::sync_all();

        const int n_xattn = net.num_xattn_blocks();
        CHECK(static_cast<int>(trace.size()) == n_xattn);

        // Verify each trace entry: (Lq, Lk=L_text), softmax row-sum ~= 1.
        for (int i = 0; i < n_xattn; ++i) {
            const bt::Tensor& m = trace[static_cast<std::size_t>(i)];
            CHECK(m.dtype == brodiffusion::compute_dtype());
            CHECK(m.cols == L_text);
            CHECK(m.rows > 0);
            std::vector<float> hb = bdtest::bd_download(m);
            // Each row of (Lq, Lk) is a softmax over Lk keys.
            for (int r = 0; r < m.rows; ++r) {
                float s = 0.0f;
                for (int c = 0; c < m.cols; ++c) {
                    s += hb[static_cast<std::size_t>(r) * m.cols + c];
                }
                CHECK(s > 0.95f && s < 1.05f);
            }
        }

        // Trace-path output should match the fast-path output closely.
        std::vector<float> vals_trace = bdtest::bd_download(out);
        int nonfinite_trace = 0;
        float max_abs_diff = 0.0f;
        for (std::size_t k = 0; k < vals1.size(); ++k) {
            if (!bdtest::bd_finite(vals_trace[k])) ++nonfinite_trace;
            const float d = std::fabs(vals1[k] - vals_trace[k]);
            if (d > max_abs_diff) max_abs_diff = d;
        }
        CHECK(nonfinite_trace == 0);
        // Flash vs non-flash kernels drift in low FP16 bits; the small
        // synthetic UNet has activation magnitudes O(1), so 0.05 is a
        // generous-but-meaningful bound.
        CHECK(max_abs_diff < 0.05f);

        // A vanilla UNet (time_cond_proj_dim == 0) must reject a trace call
        // that supplies a guidance scale — same contract the forward()
        // overloads enforce.
        bool threw_vanilla = false;
        try {
            un::UNet::CrossAttnTrace t2;
            bt::Tensor o2;
            const float g = 7.5f;
            net.forward_trace(latent, H, W, 500.0f, &g, ctx,
                              /*attn_logit_biases=*/nullptr, &t2, o2);
        } catch (const std::exception&) { threw_vanilla = true; }
        CHECK(threw_vanilla);
    }

    // ── LCM (time_cond_proj_dim > 0) trace mode ──────────────────────────────
    // An LCM-distilled U-Net routes a guidance-scale embedding through
    // cond_proj. forward_trace must support that path — the cond_proj plumbing
    // and the attention trace are independent. Build an LCM U-Net from the same
    // fixture (it carries the cond_proj weight), then check trace mode runs and
    // matches the LCM fast path.
    {
        un::UNetConfig lcm_cfg = cfg;
        lcm_cfg.time_cond_proj_dim = cond_proj_dim;

        auto file = st::File::open(path.string());
        un::UNet lcm(lcm_cfg);
        lcm.load_weights(file, "");

        std::vector<float> latent_h(
            static_cast<std::size_t>(cfg.in_channels) * H * W);
        for (std::size_t i = 0; i < latent_h.size(); ++i) {
            latent_h[i] = (static_cast<float>(i % 5) - 2.0f) * 0.1f;
        }
        bt::Tensor latent =
            bdtest::bd_upload(latent_h, 1, cfg.in_channels * H * W);

        std::vector<float> ctx_h(static_cast<std::size_t>(L_text) * ctx_dim);
        for (std::size_t i = 0; i < ctx_h.size(); ++i) {
            ctx_h[i] = (static_cast<float>(i % 7) - 3.0f) * 0.05f;
        }
        bt::Tensor ctx = bdtest::bd_upload(ctx_h, L_text, ctx_dim);

        const float guidance = 7.5f;

        // LCM fast path: prime the K/V cache, run the guidance-scale forward.
        un::UNet::CrossAttnKVCache cache;
        lcm.prime_xattn_cache(ctx, cache);
        bt::Tensor out_fast;
        lcm.forward(latent, H, W, 500.0f, guidance, ctx, cache, out_fast);
        bt::sync_all();
        std::vector<float> vals_fast = bdtest::bd_download(out_fast);

        // LCM trace path: forward_trace with the guidance scale supplied.
        un::UNet::CrossAttnTrace trace;
        bt::Tensor out_trace;
        lcm.forward_trace(latent, H, W, 500.0f, &guidance, ctx,
                          /*attn_logit_biases=*/nullptr, &trace, out_trace);
        bt::sync_all();

        const int n_xattn = lcm.num_xattn_blocks();
        CHECK(static_cast<int>(trace.size()) == n_xattn);
        for (int i = 0; i < n_xattn; ++i) {
            const bt::Tensor& m = trace[static_cast<std::size_t>(i)];
            CHECK(m.cols == L_text);
            CHECK(m.rows > 0);
            std::vector<float> hb = bdtest::bd_download(m);
            for (int r = 0; r < m.rows; ++r) {
                float s = 0.0f;
                for (int c = 0; c < m.cols; ++c) {
                    s += hb[static_cast<std::size_t>(r) * m.cols + c];
                }
                CHECK(s > 0.95f && s < 1.05f);
            }
        }

        // The trace output must match the LCM fast path within FP16 kernel
        // drift — proof the cond_proj guidance contribution is applied in
        // trace mode, not silently dropped.
        std::vector<float> vals_trace = bdtest::bd_download(out_trace);
        CHECK(vals_trace.size() == vals_fast.size());
        int nonfinite = 0;
        float max_abs_diff = 0.0f;
        for (std::size_t k = 0; k < vals_fast.size(); ++k) {
            if (!bdtest::bd_finite(vals_trace[k])) ++nonfinite;
            const float d = std::fabs(vals_fast[k] - vals_trace[k]);
            if (d > max_abs_diff) max_abs_diff = d;
        }
        CHECK(nonfinite == 0);
        CHECK(max_abs_diff < 0.05f);

        // An LCM U-Net must reject a trace call with no guidance scale.
        bool threw_missing = false;
        try {
            un::UNet::CrossAttnTrace t2;
            bt::Tensor o2;
            lcm.forward_trace(latent, H, W, 500.0f, /*guidance_scale=*/nullptr,
                              ctx, /*attn_logit_biases=*/nullptr, &t2, o2);
        } catch (const std::exception&) { threw_missing = true; }
        CHECK(threw_missing);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("unet: OK\n");
    else std::fprintf(stderr, "unet: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
