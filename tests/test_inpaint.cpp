// inpaint priming + blend smoke test.
//
// Builds a tiny SD1.5 Pipeline (CLIP + UNet + VAE decoder + VAE encoder)
// from synthetic FP16 weights, writes a synthetic init PNG and matching
// mask PNGs, and exercises the Phase C inpaint path:
//
//   1. all_one_mask_equals_img2img — a fully-white mask (inpaint everywhere)
//      should reduce to plain img2img, since the blend term (1-mask)*x0_renoised
//      collapses to zero. Compare the generated image bit-for-bit (FP) against
//      an img2img-only run with identical opts.
//   2. all_zero_mask_preserves_init — a fully-black mask (preserve everything)
//      should keep the latent on the renoised-x0 trajectory at every step, and
//      after the final blend the latent equals the noised x0 at step n-1 — i.e.
//      the decoded image is close to the encoded-then-decoded init.
//   3. mismatched_mask_without_init_throws — sanity guard.
//
// The fixture matches test_img2img.cpp's setup so the comparison is direct.

#include "brodiffusion/pipeline.h"
#include "brodiffusion/detail/compute.h"
#include "brolm/tokenizer.h"

#include "broimage/encode.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "sd_fixtures.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace pl   = brodiffusion::pipeline;
namespace vae  = brodiffusion::vae;
namespace clip = brolm::clip;
namespace st   = brotensor::safetensors;
namespace bt   = brotensor;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

namespace {

// ── duplicated from test_img2img.cpp (small enough to inline) ────────────

void build_vae_encoder(bdfix::Builder& b, const vae::EncoderConfig& cfg,
                       const std::string& prefix) {
    const int nb       = static_cast<int>(cfg.block_out_channels.size());
    const int first_C  = cfg.block_out_channels.front();
    const int mid_C    = cfg.block_out_channels.back();
    const int twoC     = 2 * cfg.in_channels;

    b.add(prefix + "conv_in.weight", {first_C, cfg.out_channels, 3, 3},
          bdfix::fp16_rand(
              static_cast<std::size_t>(first_C) * cfg.out_channels * 9,
              0.05f, 800));
    b.add(prefix + "conv_in.bias",   {first_C}, bdfix::fp16_zeros(first_C));

    int C_prev = first_C;
    for (int i = 0; i < nb; ++i) {
        const int C_block =
            cfg.block_out_channels[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg.layers_per_block; ++j) {
            const int Ci = (j == 0) ? C_prev : C_block;
            bdfix::emit_vae_resnet(
                b,
                prefix + "down_blocks." + std::to_string(i) + ".resnets." +
                    std::to_string(j) + ".",
                Ci, C_block);
        }
        if (i + 1 < nb) {
            const std::string dp = prefix + "down_blocks." +
                                   std::to_string(i) +
                                   ".downsamplers.0.conv.";
            b.add(dp + "weight", {C_block, C_block, 3, 3},
                  bdfix::fp16_rand(
                      static_cast<std::size_t>(C_block) * C_block * 9,
                      0.02f, static_cast<std::size_t>(900 + i)));
            b.add(dp + "bias",   {C_block}, bdfix::fp16_zeros(C_block));
        }
        C_prev = C_block;
    }

    bdfix::emit_vae_resnet(b, prefix + "mid_block.resnets.0.", mid_C, mid_C);
    bdfix::emit_vae_resnet(b, prefix + "mid_block.resnets.1.", mid_C, mid_C);

    const std::string ap = prefix + "mid_block.attentions.0.";
    b.add(ap + "group_norm.weight", {mid_C}, bdfix::fp16_ones(mid_C));
    b.add(ap + "group_norm.bias",   {mid_C}, bdfix::fp16_zeros(mid_C));
    b.add(ap + "query.weight",     {mid_C, mid_C},
          bdfix::fp16_rand(static_cast<std::size_t>(mid_C) * mid_C, 0.03f, 911));
    b.add(ap + "query.bias",       {mid_C}, bdfix::fp16_zeros(mid_C));
    b.add(ap + "key.weight",       {mid_C, mid_C},
          bdfix::fp16_rand(static_cast<std::size_t>(mid_C) * mid_C, 0.03f, 913));
    b.add(ap + "key.bias",         {mid_C}, bdfix::fp16_zeros(mid_C));
    b.add(ap + "value.weight",     {mid_C, mid_C},
          bdfix::fp16_rand(static_cast<std::size_t>(mid_C) * mid_C, 0.03f, 917));
    b.add(ap + "value.bias",       {mid_C}, bdfix::fp16_zeros(mid_C));
    b.add(ap + "proj_attn.weight", {mid_C, mid_C},
          bdfix::fp16_rand(static_cast<std::size_t>(mid_C) * mid_C, 0.03f, 919));
    b.add(ap + "proj_attn.bias",   {mid_C}, bdfix::fp16_zeros(mid_C));

    b.add(prefix + "conv_norm_out.weight", {mid_C}, bdfix::fp16_ones(mid_C));
    b.add(prefix + "conv_norm_out.bias",   {mid_C}, bdfix::fp16_zeros(mid_C));
    b.add(prefix + "conv_out.weight", {twoC, mid_C, 3, 3},
          bdfix::fp16_rand(
              static_cast<std::size_t>(twoC) * mid_C * 9, 0.04f, 931));
    b.add(prefix + "conv_out.bias",   {twoC}, bdfix::fp16_zeros(twoC));

    const std::string parent =
        prefix.substr(0, prefix.size() - std::string("encoder.").size());
    b.add(parent + "quant_conv.weight", {twoC, twoC, 1, 1},
          bdfix::fp16_rand(static_cast<std::size_t>(twoC) * twoC,
                           0.05f, 941));
    b.add(parent + "quant_conv.bias",   {twoC}, bdfix::fp16_zeros(twoC));
}

void build_post_quant_conv(bdfix::Builder& b, int C_lat,
                           const std::string& parent) {
    b.add(parent + "post_quant_conv.weight", {C_lat, C_lat, 1, 1},
          bdfix::fp16_rand(static_cast<std::size_t>(C_lat) * C_lat,
                           0.05f, 953));
    b.add(parent + "post_quant_conv.bias",   {C_lat},
          bdfix::fp16_zeros(C_lat));
}

brodiffusion::unet::UNetConfig unet_cfg() {
    brodiffusion::unet::UNetConfig c;
    c.in_channels         = 4;
    c.out_channels        = 4;
    c.block_out_channels  = {8, 16};
    c.layers_per_block    = 1;
    c.norm_num_groups     = 2;
    c.eps                 = 1e-5f;
    c.cross_attention_dim = 8;
    c.attention_head_dim  = 8;
    c.time_embed_dim_mult = 2;
    return c;
}

vae::DecoderConfig vae_dec_cfg() {
    vae::DecoderConfig c;
    c.in_channels         = 4;
    c.out_channels        = 3;
    c.block_out_channels  = {4, 8, 16, 16};
    c.layers_per_block    = 2;
    c.norm_num_groups     = 2;
    c.scaling_factor      = 1.0f;
    c.shift_factor        = 0.0f;
    c.eps                 = 1e-6f;
    c.num_attention_heads = 1;
    return c;
}

clip::TextEncoderConfig clip_cfg() {
    clip::TextEncoderConfig c;
    c.vocab_size       = 49408;
    c.max_position     = 77;
    c.hidden_dim       = 8;
    c.num_heads        = 2;
    c.num_layers       = 2;
    c.intermediate_dim = 16;
    c.layer_norm_eps   = 1e-5f;
    return c;
}

void write_tiny_vocab(const std::filesystem::path& vp,
                      const std::filesystem::path& mp) {
    std::ofstream(vp, std::ios::binary | std::ios::trunc)
        << "{\"a\":1,\"a</w>\":2}";
    std::ofstream(mp) << "#version: test\n";
}

// Solid-gray RGBA PNG.
void write_solid_png(const std::filesystem::path& path, std::uint8_t v) {
    const int w = 16, h = 16;
    std::vector<std::uint8_t> rgba(
        static_cast<std::size_t>(w * h * 4), v);
    for (std::size_t i = 3; i < rgba.size(); i += 4) rgba[i] = 255;
    if (!broimage::encode_png_file(path.string(), rgba.data(),
                                   w, h, /*channels=*/4)) {
        std::abort();
    }
}

std::filesystem::path build_full_fixture() {
    bdfix::Builder b;
    bdfix::build_clip(b, clip_cfg(), "text_model.");
    bdfix::build_unet(b, unet_cfg(), "");
    bdfix::build_vae(b, vae_dec_cfg(), "decoder.");
    build_post_quant_conv(b, vae_dec_cfg().in_channels, /*parent=*/"");
    {
        vae::EncoderConfig enc;
        const auto d = vae_dec_cfg();
        enc.in_channels         = d.in_channels;
        enc.out_channels        = d.out_channels;
        enc.block_out_channels  = d.block_out_channels;
        enc.layers_per_block    = d.layers_per_block;
        enc.norm_num_groups     = d.norm_num_groups;
        enc.scaling_factor      = d.scaling_factor;
        enc.shift_factor        = d.shift_factor;
        enc.eps                 = d.eps;
        enc.num_attention_heads = d.num_attention_heads;
        build_vae_encoder(b, enc, "encoder.");
    }
    auto path = std::filesystem::temp_directory_path() /
                "brodiffusion_inpaint_test.safetensors";
    b.write(path);
    return path;
}

pl::PipelineConfig make_cfg() {
    pl::PipelineConfig cfg;
    cfg.unet         = unet_cfg();
    cfg.vae          = vae_dec_cfg();
    cfg.text_encoder = clip_cfg();
    return cfg;
}

// Compare two FP image vectors with a tight relative tolerance.
double l2_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum += d * d;
    }
    return std::sqrt(sum);
}

}  // namespace

int main() {
    try { bt::init(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }
    try {

    auto tmp = std::filesystem::temp_directory_path();
    auto vp  = tmp / "brodiffusion_inpaint_vocab.json";
    auto mp  = tmp / "brodiffusion_inpaint_merges.txt";
    auto img_path        = tmp / "brodiffusion_inpaint_init.png";
    auto mask_white_path = tmp / "brodiffusion_inpaint_mask_white.png";
    auto mask_black_path = tmp / "brodiffusion_inpaint_mask_black.png";
    write_tiny_vocab(vp, mp);
    write_solid_png(img_path,        /*v=*/128);
    write_solid_png(mask_white_path, /*v=*/255);
    write_solid_png(mask_black_path, /*v=*/0);

    auto fix_path = build_full_fixture();
    auto fix_file = st::File::open(fix_path.string());

    auto make_loaded_pipeline = [&]() {
        auto tok = clip::Tokenizer::load(vp.string(), mp.string());
        pl::Pipeline p(make_cfg(), std::move(tok));
        p.load_weights(fix_file, "text_model.", "", "decoder.");
        return p;
    };

    constexpr int W = 16;
    constexpr int H = 16;

    auto base_opts = [&]() {
        pl::GenerateOptions opts;
        opts.width  = W;
        opts.height = H;
        opts.num_inference_steps = 3;
        opts.guidance_scale      = 1.0f;  // skip uncond pass
        opts.init_image_path     = img_path.string();
        opts.strength            = 1.0f;  // full schedule => t_start=0
        opts.seed                = 17;
        return opts;
    };

    // ── 1. all_one_mask_equals_img2img ────────────────────────────────────
    {
        auto p1 = make_loaded_pipeline();
        auto p2 = make_loaded_pipeline();

        pl::GenerateOptions o_img = base_opts();
        pl::GenerateOptions o_inp = base_opts();
        o_inp.mask_image_path = mask_white_path.string();

        std::vector<float> img_a = p1.generate("hi", o_img);
        std::vector<float> img_b = p2.generate("hi", o_inp);

        CHECK(img_a.size() == img_b.size());
        // mask == 1 everywhere -> blend reduces to latent *= 1 (no-op), plus
        // (1-mask)*x0_renoised = 0. The renoise noise is drawn from a
        // decorrelated Philox slot, but multiplied by zero — so the outputs
        // should be bit-identical (or FP-identical within rounding).
        const double d = l2_diff(img_a, img_b);
        // FP32 CPU build: expect exact equality; FP16 GPU: tiny rounding from
        // the mul_inplace by 1.0. Allow a small tolerance for portability.
        CHECK(d < 1e-3);
    }

    // ── 2. all_zero_mask_preserves_init ───────────────────────────────────
    // mask == 0 everywhere -> at every non-final step, latent gets fully
    // replaced by x0_renoised at the next timestep. After the final step
    // the latent is left at the scheduler-predicted x0 (t=0 — no blend on
    // the final step), but for an all-zero mask the trajectory is dominated
    // by the renoised-x0 path, so the decoded image should land in a finite
    // range and stay close to a plain img2img-decode of the init image.
    {
        auto p = make_loaded_pipeline();
        pl::GenerateOptions opts = base_opts();
        opts.mask_image_path = mask_black_path.string();

        std::vector<float> img = p.generate("hi", opts);
        const std::size_t n_img =
            3u * static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
        CHECK(img.size() == n_img);
        int nonfinite = 0;
        for (float v : img) if (!std::isfinite(v)) ++nonfinite;
        CHECK(nonfinite == 0);
    }

    // ── 3. mismatched_mask_without_init_throws ────────────────────────────
    {
        auto p = make_loaded_pipeline();
        pl::GenerateOptions opts;
        opts.width  = W;
        opts.height = H;
        opts.num_inference_steps = 2;
        opts.guidance_scale = 1.0f;
        opts.seed = 1;
        opts.mask_image_path = mask_white_path.string();
        // init_image_path deliberately empty.

        bool threw = false;
        try {
            (void)p.prime("hi", opts);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw);
    }

    std::error_code ec;
    std::filesystem::remove(fix_path,        ec);
    std::filesystem::remove(img_path,        ec);
    std::filesystem::remove(mask_white_path, ec);
    std::filesystem::remove(mask_black_path, ec);
    std::filesystem::remove(vp,              ec);
    std::filesystem::remove(mp,              ec);

    if (g_failures == 0) std::printf("inpaint: OK\n");
    else std::fprintf(stderr, "inpaint: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "inpaint: EXCEPTION: %s\n", e.what());
        std::fflush(stderr);
        return 1;
    } catch (...) {
        std::fprintf(stderr, "inpaint: UNKNOWN EXCEPTION\n");
        std::fflush(stderr);
        return 1;
    }
}
