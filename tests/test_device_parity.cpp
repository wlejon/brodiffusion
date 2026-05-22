// End-to-end CPU↔Metal parity for brodiffusion's sub-modules.
//
// tests/test_fused_parity.cpp covers the individual fused kernels. This test
// covers the *compositions*: a whole CLIP text encoder, a whole UNet forward,
// and a whole VAE decode, each run once on the CPU backend (FP32) and once on
// the Metal backend (FP16) from the same synthetic weight fixture, with the
// outputs compared within an FP16-appropriate tolerance.
//
// It also exercises error accumulation: a UNet output fed back as its own
// input over several steps (a denoising-style feedback loop) is run on both
// backends, and the CPU↔Metal divergence is tracked per step to confirm it
// stays bounded rather than compounding catastrophically.
//
// The two backends are selected with brotensor::DeviceScope, which overrides
// the thread-local default device for tensor construction and compute_dtype()
// — so wrapping a module's construction + load + forward in a scope pins the
// whole graph to that backend.
//
// Skips cleanly (exit 0) when no Metal backend is available, so the same
// source builds and passes in the CPU-only configuration.

#include "brolm/clip.h"
#include "brodiffusion/unet.h"
#include "brodiffusion/vae.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "sd_fixtures.h"
#include "test_compute.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace bt   = brotensor;
namespace un   = brodiffusion::unet;
namespace vae  = brodiffusion::vae;
namespace clip = brolm::clip;
namespace st   = brotensor::safetensors;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

namespace {

// ─── module configs (shrunk, full topology) ────────────────────────────────

un::UNetConfig unet_cfg() {
    un::UNetConfig c;
    c.in_channels         = 4;
    c.out_channels        = 4;
    c.block_out_channels  = {8, 16, 32, 32};
    c.layers_per_block    = 1;
    c.norm_num_groups     = 2;
    c.eps                 = 1e-5f;
    c.cross_attention_dim = 8;
    c.attention_head_dim  = 8;   // SD1.5's real head_dim; flash-attn kernels
                                 // are not tuned for head_dim < 8.
    c.time_embed_dim_mult = 2;
    return c;
}

vae::DecoderConfig vae_cfg() {
    vae::DecoderConfig c;
    c.in_channels         = 4;
    c.out_channels        = 3;
    c.block_out_channels  = {4, 8, 16, 16};
    c.layers_per_block    = 2;
    c.norm_num_groups     = 2;
    c.scaling_factor      = 1.0f;   // synthetic latent — no SD rescale
    c.eps                 = 1e-6f;
    c.num_attention_heads = 1;
    return c;
}

clip::TextEncoderConfig clip_cfg() {
    clip::TextEncoderConfig c;
    c.vocab_size       = 16;
    c.max_position     = 8;
    c.hidden_dim       = 16;
    c.num_heads        = 2;
    c.num_layers       = 2;
    c.intermediate_dim = 32;
    c.layer_norm_eps   = 1e-5f;
    return c;
}

// ─── comparison ────────────────────────────────────────────────────────────

struct Cmp {
    float max_abs_err = 0.0f;
    float max_abs_ref = 0.0f;
    int   nonfinite   = 0;
    bool  size_match  = true;
};

Cmp compare(const std::vector<float>& cpu, const std::vector<float>& gpu) {
    Cmp c;
    if (cpu.size() != gpu.size()) { c.size_match = false; return c; }
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        if (!std::isfinite(cpu[i]) || !std::isfinite(gpu[i])) {
            ++c.nonfinite;
            continue;
        }
        c.max_abs_err = std::max(c.max_abs_err, std::fabs(cpu[i] - gpu[i]));
        c.max_abs_ref = std::max(c.max_abs_ref, std::fabs(cpu[i]));
    }
    return c;
}

void expect_parity(const char* label, const Cmp& c, std::size_t n,
                   float tol_rel, float tol_abs) {
    const float bound = tol_rel * c.max_abs_ref + tol_abs;
    std::printf("  %-28s err=%.5f ref=%.4f bound=%.5f%s\n",
                label, c.max_abs_err, c.max_abs_ref, bound,
                (c.size_match && c.max_abs_err <= bound) ? "" : "  <-- OVER");
    CHECK(n > 0);
    CHECK(c.size_match);
    CHECK(c.nonfinite == 0);
    CHECK(c.max_abs_ref > 0.0f);
    CHECK(c.max_abs_err <= bound);
}

// ─── per-module runners (one backend each) ─────────────────────────────────

// CLIP text encoder: returns the (P, D) last hidden state as host FP32.
std::vector<float> run_clip(const st::File& file, bt::Device dev) {
    bt::DeviceScope scope(dev);
    clip::TextEncoder enc(clip_cfg());
    enc.load_weights(file, "text_model.");
    std::vector<std::int32_t> ids;
    for (int i = 0; i < clip_cfg().max_position; ++i) {
        ids.push_back((i * 3 + 1) % clip_cfg().vocab_size);
    }
    bt::Tensor out;
    enc.forward(ids.data(), out);
    return bdtest::bd_download(out);
}

// UNet noise prediction for one forward, returned as host FP32.
std::vector<float> run_unet(const st::File& file, bt::Device dev,
                            int H, int W) {
    bt::DeviceScope scope(dev);
    const auto cfg = unet_cfg();
    un::UNet net(cfg);
    net.load_weights(file, "");

    std::vector<float> latent_h(
        static_cast<std::size_t>(cfg.in_channels) * H * W);
    for (std::size_t i = 0; i < latent_h.size(); ++i) {
        latent_h[i] = (static_cast<float>(i % 5) - 2.0f) * 0.1f;
    }
    bt::Tensor latent = bdtest::bd_upload(latent_h, 1, cfg.in_channels * H * W);

    const int L_text = 4;
    std::vector<float> ctx_h(
        static_cast<std::size_t>(L_text) * cfg.cross_attention_dim);
    for (std::size_t i = 0; i < ctx_h.size(); ++i) {
        ctx_h[i] = (static_cast<float>(i % 7) - 3.0f) * 0.05f;
    }
    bt::Tensor ctx = bdtest::bd_upload(ctx_h, L_text, cfg.cross_attention_dim);

    bt::Tensor out;
    net.forward(latent, H, W, /*timestep=*/500.0f, ctx, out);
    return bdtest::bd_download(out);
}

// VAE decode for one latent, returned as host FP32.
std::vector<float> run_vae(const st::File& file, bt::Device dev,
                           int H_lat, int W_lat) {
    bt::DeviceScope scope(dev);
    const auto cfg = vae_cfg();
    vae::Decoder dec(cfg);
    dec.load_weights(file, "decoder.");

    std::vector<float> latent_h(
        static_cast<std::size_t>(cfg.in_channels) * H_lat * W_lat);
    for (std::size_t i = 0; i < latent_h.size(); ++i) {
        latent_h[i] = (static_cast<float>(i % 5) - 2.0f) * 0.1f;
    }
    bt::Tensor latent =
        bdtest::bd_upload(latent_h, 1, cfg.in_channels * H_lat * W_lat);

    bt::Tensor out;
    dec.decode(latent, H_lat, W_lat, out);
    return bdtest::bd_download(out);
}

// Denoising-style feedback loop: each step's UNet output is fed back as the
// next step's input. Returns the latent after every step as host FP32.
// in_channels == out_channels for SD1.5, so the output is shape-compatible
// as the next input. GroupNorm normalises the activation at every resblock,
// so the loop is numerically stable regardless of input scale.
std::vector<std::vector<float>> run_unet_loop(const st::File& file,
                                              bt::Device dev,
                                              int H, int W, int n_steps) {
    bt::DeviceScope scope(dev);
    const auto cfg = unet_cfg();
    un::UNet net(cfg);
    net.load_weights(file, "");

    std::vector<float> latent_h(
        static_cast<std::size_t>(cfg.in_channels) * H * W);
    for (std::size_t i = 0; i < latent_h.size(); ++i) {
        latent_h[i] = (static_cast<float>(i % 5) - 2.0f) * 0.1f;
    }
    bt::Tensor latent = bdtest::bd_upload(latent_h, 1, cfg.in_channels * H * W);

    const int L_text = 4;
    std::vector<float> ctx_h(
        static_cast<std::size_t>(L_text) * cfg.cross_attention_dim);
    for (std::size_t i = 0; i < ctx_h.size(); ++i) {
        ctx_h[i] = (static_cast<float>(i % 7) - 3.0f) * 0.05f;
    }
    bt::Tensor ctx = bdtest::bd_upload(ctx_h, L_text, cfg.cross_attention_dim);

    // A descending timestep schedule, like a real inference run.
    const float timesteps[] = {900.0f, 720.0f, 540.0f, 360.0f, 180.0f, 20.0f};

    std::vector<std::vector<float>> snapshots;
    bt::Tensor out;
    for (int s = 0; s < n_steps; ++s) {
        const float t = timesteps[s % 6];
        net.forward(latent, H, W, t, ctx, out);
        snapshots.push_back(bdtest::bd_download(out));
        latent = out;   // deep copy on the active device — feed back
    }
    return snapshots;
}

}  // namespace

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    if (!bt::is_available(bt::Device::Metal)) {
        std::printf("device_parity: no Metal backend — skipping\n");
        return 0;
    }

    // ── Build one combined fixture holding all three sub-modules ────────────
    bdfix::Builder b;
    bdfix::build_clip(b, clip_cfg(), "text_model.");
    bdfix::build_unet(b, unet_cfg(), "");
    bdfix::build_vae(b, vae_cfg(), "decoder.");
    auto path = std::filesystem::temp_directory_path() /
                "brodiffusion_device_parity.safetensors";
    b.write(path);
    auto file = st::File::open(path.string());

    std::printf("device_parity: CPU (FP32) vs Metal (FP16)\n");

    // ── Single-forward parity per module ───────────────────────────────────
    {
        const auto cpu = run_clip(file, bt::Device::CPU);
        const auto gpu = run_clip(file, bt::Device::Metal);
        // 2-layer transformer: FP16 storage + accumulation drift end to end.
        // Observed err ~0.0017; bound leaves generous cross-GPU headroom.
        expect_parity("clip text encoder", compare(cpu, gpu), gpu.size(),
                      /*tol_rel=*/0.01f, /*tol_abs=*/0.01f);
    }
    {
        const auto cpu  = run_unet(file, bt::Device::CPU, 8, 8);
        const auto gpu  = run_unet(file, bt::Device::Metal, 8, 8);
        const auto gpu2 = run_unet(file, bt::Device::Metal, 8, 8);
        CHECK(gpu == gpu2);   // Metal forward must be deterministic
        // Full U-Net: conv_in, down/mid/up blocks (incl. the up-path skip
        // concats), conv_out — the deepest graph under test. Observed err
        // ~0.004; the bound stays well below the ~0.5 a concat-class
        // regression would produce, while tolerating cross-GPU FP16 drift.
        expect_parity("unet forward 8x8", compare(cpu, gpu), gpu.size(),
                      /*tol_rel=*/0.04f, /*tol_abs=*/0.02f);
    }
    {
        const auto cpu = run_unet(file, bt::Device::CPU, 16, 16);
        const auto gpu = run_unet(file, bt::Device::Metal, 16, 16);
        // Larger spatial — exercises the multi-tile conv path end to end.
        expect_parity("unet forward 16x16", compare(cpu, gpu), gpu.size(),
                      /*tol_rel=*/0.04f, /*tol_abs=*/0.02f);
    }
    {
        const auto cpu = run_vae(file, bt::Device::CPU, 2, 2);
        const auto gpu = run_vae(file, bt::Device::Metal, 2, 2);
        // VAE decoder: three 2x upsamples + a self-attention mid block.
        expect_parity("vae decode 2x2->16x16", compare(cpu, gpu), gpu.size(),
                      /*tol_rel=*/0.03f, /*tol_abs=*/0.015f);
    }

    // ── Multi-step accumulation ────────────────────────────────────────────
    {
        const int n_steps = 6;
        const auto cpu = run_unet_loop(file, bt::Device::CPU,   8, 8, n_steps);
        const auto gpu = run_unet_loop(file, bt::Device::Metal, 8, 8, n_steps);
        CHECK(static_cast<int>(cpu.size()) == n_steps);
        CHECK(static_cast<int>(gpu.size()) == n_steps);

        std::printf("  unet feedback loop (%d steps): per-step CPU<->Metal drift\n",
                    n_steps);
        float final_err = 0.0f, final_ref = 1.0f;
        for (int s = 0; s < n_steps && s < static_cast<int>(gpu.size()); ++s) {
            const Cmp c = compare(cpu[static_cast<std::size_t>(s)],
                                  gpu[static_cast<std::size_t>(s)]);
            std::printf("    step %d  err=%.5f  ref=%.4f\n",
                        s, c.max_abs_err, c.max_abs_ref);
            CHECK(c.size_match);
            CHECK(c.nonfinite == 0);
            final_err = c.max_abs_err;
            final_ref = c.max_abs_ref;
        }
        // The feedback loop must not let FP16 drift explode: after 6 steps the
        // CPU and Metal trajectories should still agree to well within the
        // activation magnitude. Observed final drift ~0.014; a genuine
        // regression (a kernel that diverges) blows past this bound, while
        // benign per-step FP16 rounding accumulates far below it.
        const float bound = 0.08f * final_ref + 0.04f;
        std::printf("    final drift %.5f  bound %.5f%s\n",
                    final_err, bound, final_err <= bound ? "" : "  <-- OVER");
        CHECK(final_err <= bound);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("device_parity: OK\n");
    else std::fprintf(stderr, "device_parity: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
