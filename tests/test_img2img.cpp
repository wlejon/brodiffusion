// img2img priming smoke test.
//
// Builds a tiny but architecturally complete SD1.5 Pipeline (CLIP + UNet +
// VAE decoder + VAE encoder) from synthetic FP16 weights, writes a synthetic
// init PNG, and exercises the new Pipeline img2img options:
//
//   1. prime_with_init_skips_steps  — init_image_path + strength=0.5 against
//      num_inference_steps=4 puts the first denoising step at step_index=2.
//   2. strength_one_runs_full_schedule — strength=1.0 starts at step_index=0
//      and a full generate() produces a finite, in-shape image.
//   3. empty_path_is_txt2img — empty init_image_path leaves step_index=0
//      and the generated output depends only on the txt2img seed, not on
//      anything the encoder would have done.
//
// The fixture uses tiny channel counts (block_out_channels = {4,8,16,16},
// norm_num_groups = 2) so the entire safetensors payload fits in memory.

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

// Emit a VAE-encoder weight set under `prefix` (must end in "encoder."). The
// encoder mirrors the decoder topology with channels in *forward* order:
//   conv_in: (first_C, out_channels, 3, 3)
//   down_blocks[i] for i in [0, nb):
//     layers_per_block resnets (first j=0 widens C_prev->C_block)
//     downsamplers.0.conv : (C_block, C_block, 3, 3)   for i < nb-1
//   mid_block: resnets.0, attentions.0, resnets.1   (all at mid_C)
//   conv_norm_out, conv_out: (2*in_channels, mid_C, 3, 3)
//   sibling quant_conv: (2*in_channels, 2*in_channels, 1, 1) at parent.
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

    // quant_conv lives one level above "encoder." (sibling of post_quant_conv).
    // prefix ends in "encoder.", so the parent is prefix.size()-8 chars.
    const std::string parent =
        prefix.substr(0, prefix.size() - std::string("encoder.").size());
    b.add(parent + "quant_conv.weight", {twoC, twoC, 1, 1},
          bdfix::fp16_rand(static_cast<std::size_t>(twoC) * twoC,
                           0.05f, 941));
    b.add(parent + "quant_conv.bias",   {twoC}, bdfix::fp16_zeros(twoC));
}

// Emit a single-pixel post_quant_conv to keep the decoder load happy on
// fixtures that exercise the encoder too. Sibling of "decoder." prefix.
void build_post_quant_conv(bdfix::Builder& b, int C_lat,
                           const std::string& parent) {
    b.add(parent + "post_quant_conv.weight", {C_lat, C_lat, 1, 1},
          bdfix::fp16_rand(static_cast<std::size_t>(C_lat) * C_lat,
                           0.05f, 953));
    b.add(parent + "post_quant_conv.bias",   {C_lat},
          bdfix::fp16_zeros(C_lat));
}

// ── module configs (shrunk, full topology) ────────────────────────────────

brodiffusion::unet::UNetConfig unet_cfg() {
    brodiffusion::unet::UNetConfig c;
    c.in_channels         = 4;
    c.out_channels        = 4;
    // 2 blocks -> downsample factor 2: a 2x2 latent (from a 16x16 image
    // through the 8x VAE) is the smallest legal input.
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
    // brolm's CLIP tokenizer always pads to its compile-time 77 tokens, so
    // max_position must match for encode_prompt_'s shape check to pass.
    c.vocab_size       = 49408;     // standard CLIP vocab size — must be >=
                                    // the bos/eos token ids the tokenizer emits.
    c.max_position     = 77;
    c.hidden_dim       = 8;         // must match unet cross_attention_dim
    c.num_heads        = 2;
    c.num_layers       = 2;
    c.intermediate_dim = 16;
    c.layer_norm_eps   = 1e-5f;
    return c;
}

// Write a tiny CLIP-style vocab + merges that always returns a deterministic
// padded sequence of `max_position` ids. The Pipeline uses the tokenizer
// only for prompt encoding; correctness of the ids doesn't matter for these
// shape/control-flow tests.
void write_tiny_vocab(const std::filesystem::path& vp,
                      const std::filesystem::path& mp) {
    std::ofstream(vp, std::ios::binary | std::ios::trunc)
        << "{\"a\":1,\"a</w>\":2}";
    std::ofstream(mp) << "#version: test\n";
}

// Write a 16x16 solid-gray RGBA PNG to a temp file. Caller deletes it.
void write_solid_png(const std::filesystem::path& path, std::uint8_t v) {
    const int w = 16, h = 16;
    std::vector<std::uint8_t> rgba(
        static_cast<std::size_t>(w * h * 4), v);
    for (std::size_t i = 3; i < rgba.size(); i += 4) rgba[i] = 255;
    bool ok = broimage::encode_png_file(path.string(), rgba.data(),
                                        w, h, /*channels=*/4);
    if (!ok) {
        std::fprintf(stderr, "write_solid_png: encode failed for %s\n",
                     path.string().c_str());
        std::abort();
    }
}

// Build a complete CLIP+UNet+VAE(decoder+encoder) safetensors fixture.
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
                "brodiffusion_img2img_test.safetensors";
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

}  // namespace

int main() {
    try { bt::init(); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }
    try {

    // ── shared scratch paths ──────────────────────────────────────────────
    auto tmp = std::filesystem::temp_directory_path();
    auto vp  = tmp / "brodiffusion_img2img_vocab.json";
    auto mp  = tmp / "brodiffusion_img2img_merges.txt";
    auto img_path = tmp / "brodiffusion_img2img_init.png";
    write_tiny_vocab(vp, mp);
    write_solid_png(img_path, /*v=*/128);

    auto fix_path = build_full_fixture();
    auto fix_file = st::File::open(fix_path.string());

    // Pipelines share the same fixture file. Construct fresh per test so
    // any in-flight scratch is isolated.
    auto make_loaded_pipeline = [&]() {
        auto tok = clip::Tokenizer::load(vp.string(), mp.string());
        pl::Pipeline p(make_cfg(), std::move(tok));
        // 3-file path mirrors the diffusers-default prefixes ("text_model.",
        // "", "decoder." + sibling "encoder."). The single-file load_weights
        // overload uses the same encoder-prefix derivation.
        p.load_weights(fix_file, "text_model.", "", "decoder.");
        return p;
    };

    // The fixture's VAE is 8x downsample like real SD1.5. The UNet has 2
    // blocks (downsample factor 2), so the smallest legal latent is 2x2,
    // i.e. a 16x16 image. We use 16x16 throughout.
    constexpr int W = 16;
    constexpr int H = 16;

    // ── 1. prime_with_init_skips_steps ────────────────────────────────────
    {
        auto p = make_loaded_pipeline();
        pl::GenerateOptions opts;
        opts.width  = W;
        opts.height = H;
        opts.num_inference_steps = 4;
        opts.guidance_scale = 1.0f;   // skip uncond pass to keep test cheap
        opts.init_image_path = img_path.string();
        opts.strength = 0.5f;
        opts.seed = 7;

        pl::PipelineState state = p.prime("hi", opts);
        CHECK(state.n_steps == 4);
        // init_timestep = floor(4 * 0.5) = 2 -> t_start = 4 - 2 = 2.
        CHECK(state.step_index == 2);
        CHECK(state.H_lat == H / 8);
        CHECK(state.W_lat == W / 8);
        CHECK(state.latent.rows == 1);
        CHECK(state.latent.cols ==
              p.config().unet.in_channels * (H / 8) * (W / 8));
    }

    // ── 2. strength_one_runs_full_schedule ────────────────────────────────
    {
        auto p = make_loaded_pipeline();
        pl::GenerateOptions opts;
        opts.width  = W;
        opts.height = H;
        opts.num_inference_steps = 3;
        opts.guidance_scale = 1.0f;
        opts.init_image_path = img_path.string();
        opts.strength = 1.0f;
        opts.seed = 11;

        pl::PipelineState state = p.prime("hi", opts);
        // strength=1.0 -> init_timestep = 3 -> t_start = 0 (full schedule).
        CHECK(state.step_index == 0);

        std::vector<float> img = p.generate("hi", opts);
        const std::size_t n_img =
            3u * static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
        CHECK(img.size() == n_img);
        int nonfinite = 0;
        for (float v : img) if (!std::isfinite(v)) ++nonfinite;
        CHECK(nonfinite == 0);
    }

    // ── 3. empty_path_is_txt2img ──────────────────────────────────────────
    {
        auto p = make_loaded_pipeline();
        pl::GenerateOptions opts;
        opts.width  = W;
        opts.height = H;
        opts.num_inference_steps = 2;
        opts.guidance_scale = 1.0f;
        opts.seed = 23;
        // init_image_path stays empty -> txt2img.

        pl::PipelineState state = p.prime("hi", opts);
        CHECK(state.step_index == 0);
        CHECK(state.n_steps == 2);

        // Same seed, different prompt-token sequence not exercised — just
        // confirm two different seeds produce different latents (the encoder
        // path is NOT taken in this branch).
        opts.seed = 24;
        pl::PipelineState s2 = p.prime("hi", opts);
        std::vector<float> a(static_cast<std::size_t>(state.latent.cols));
        std::vector<float> bv(static_cast<std::size_t>(s2.latent.cols));
        if (state.latent.dtype == bt::Dtype::FP16) {
            std::vector<std::uint16_t> ab(a.size()), bb(bv.size());
            state.latent.copy_to_host_fp16(ab.data());
            s2.latent.copy_to_host_fp16(bb.data());
            bt::sync_all();
            for (std::size_t i = 0; i < a.size(); ++i) {
                a[i]  = bt::fp16_bits_to_fp32(ab[i]);
                bv[i] = bt::fp16_bits_to_fp32(bb[i]);
            }
        } else {
            a  = state.latent.to_host_vector();
            bv = s2.latent.to_host_vector();
        }
        CHECK(a != bv);
    }

    std::error_code ec;
    std::filesystem::remove(fix_path,  ec);
    std::filesystem::remove(img_path,  ec);
    std::filesystem::remove(vp,        ec);
    std::filesystem::remove(mp,        ec);

    if (g_failures == 0) std::printf("img2img: OK\n");
    else std::fprintf(stderr, "img2img: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "img2img: EXCEPTION: %s\n", e.what());
        std::fflush(stderr);
        return 1;
    } catch (...) {
        std::fprintf(stderr, "img2img: UNKNOWN EXCEPTION\n");
        std::fflush(stderr);
        return 1;
    }
}
