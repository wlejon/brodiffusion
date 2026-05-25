// Phase D3 — ControlNet end-to-end Pipeline tests.
//
// Builds a tiny SD1.5 Pipeline (CLIP + UNet + VAE decoder + VAE encoder) plus
// a matching ControlNet, both from synthetic FP16 weight fixtures, and checks
// that ControlNet is correctly wired into Pipeline::step_once:
//
//   1. zero_scale_equals_no_controlnet  — control_scale=0 → byte-identical to
//      the no-ControlNet baseline (the residuals are zeroed out before they
//      reach the UNet skips).
//   2. nonzero_scale_changes_output     — control_scale=1 with a non-trivial
//      control image moves the generated image away from the baseline.
//   3. finite_output                    — 2-step generate() with ControlNet
//      yields finite, in-shape pixels.
//   4. missing_controlnet_throws        — generate() with control_image_path
//      set but no apply_controlnet() call throws.
//
// The Flux-throws guard in apply_controlnet is exercised by inspection of the
// SD1.5-only check in src/pipeline.cpp; standing up a Flux pipeline just to
// hit that branch would dwarf the rest of this test, so it's skipped here
// (see the Phase D3 report).
//
// The UNet/VAE/CLIP shapes mirror tests/test_img2img.cpp so that the fixture
// stays tiny. The ControlNet matches the UNet's 2-stage block topology and
// keeps a 4-entry conditioning_embedding_channels ladder so that a 16x16
// control image downsamples to the 2x2 latent shape (/8 ladder).

#include "brodiffusion/controlnet.h"
#include "brodiffusion/pipeline.h"
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
namespace cn   = brodiffusion::controlnet;
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

// ── VAE encoder fixture (copied from test_img2img.cpp / test_inpaint.cpp) ──

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
          bdfix::fp16_rand(static_cast<std::size_t>(twoC) * twoC, 0.05f, 941));
    b.add(parent + "quant_conv.bias",   {twoC}, bdfix::fp16_zeros(twoC));
}

void build_post_quant_conv(bdfix::Builder& b, int C_lat,
                           const std::string& parent) {
    b.add(parent + "post_quant_conv.weight", {C_lat, C_lat, 1, 1},
          bdfix::fp16_rand(static_cast<std::size_t>(C_lat) * C_lat, 0.05f, 953));
    b.add(parent + "post_quant_conv.bias",   {C_lat},
          bdfix::fp16_zeros(C_lat));
}

// ── module configs ────────────────────────────────────────────────────────

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

cn::ControlNetConfig controlnet_cfg() {
    cn::ControlNetConfig c;
    c.in_channels             = 4;
    c.control_channels        = 3;
    // Mirror the UNet's 2-stage layout. ControlNet's down-stages must match
    // UNet's down-pass skip shapes one-for-one.
    c.block_out_channels      = {8, 16};
    c.layers_per_block        = 1;
    c.norm_num_groups         = 2;
    c.eps                     = 1e-5f;
    c.freq_dim                = 8;
    c.time_embed_dim          = 16;
    c.cross_attention_dim     = 8;
    c.transformer_num_heads   = 2;
    // 3 stride-2 pairs → /8 downsample (16x16 control image → 2x2 latent).
    c.conditioning_embedding_channels = {4, 8, 12, 16};
    return c;
}

// ── tokenizer ─────────────────────────────────────────────────────────────

void write_tiny_vocab(const std::filesystem::path& vp,
                      const std::filesystem::path& mp) {
    std::ofstream(vp, std::ios::binary | std::ios::trunc)
        << "{\"a\":1,\"a</w>\":2}";
    std::ofstream(mp) << "#version: test\n";
}

// Write a 16x16 RGBA PNG. `pattern` picks a deterministic gradient so two
// different calls produce two visibly different control inputs.
void write_png(const std::filesystem::path& path, int pattern) {
    const int w = 16, h = 16;
    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(w * h * 4), 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t i = static_cast<std::size_t>((y * w + x) * 4);
            // Use the pattern bit to flip the gradient direction.
            std::uint8_t r = static_cast<std::uint8_t>(x * 16);
            std::uint8_t g = static_cast<std::uint8_t>(y * 16);
            std::uint8_t bl =
                static_cast<std::uint8_t>((x + y) * 8);
            if (pattern & 1) { r = static_cast<std::uint8_t>(255 - r); }
            rgba[i + 0] = r;
            rgba[i + 1] = g;
            rgba[i + 2] = bl;
            rgba[i + 3] = 255;
        }
    }
    if (!broimage::encode_png_file(path.string(), rgba.data(), w, h, 4)) {
        std::fprintf(stderr, "write_png: failed for %s\n", path.string().c_str());
        std::abort();
    }
}

// ── fixture builders ──────────────────────────────────────────────────────

std::filesystem::path build_sd_fixture() {
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
                "brodiffusion_controlnet_pipeline_sd.safetensors";
    b.write(path);
    return path;
}

// ControlNet fixture using sd_fixtures helpers (resnets / transformers are
// the same Phase D1 building blocks the UNet uses).
void emit_resnet_cn(bdfix::Builder& b, const std::string& p, int C_in,
                    int C_out, int temb_dim, std::size_t salt) {
    bdfix::emit_unet_resnet(b, p, C_in, C_out, temb_dim, salt);
}
void emit_transformer_cn(bdfix::Builder& b, const std::string& p, int C,
                         int ctx_dim, std::size_t salt) {
    bdfix::emit_unet_transformer(b, p, C, ctx_dim, salt);
}

std::filesystem::path build_controlnet_fixture(const cn::ControlNetConfig& cfg) {
    const int nb       = static_cast<int>(cfg.block_out_channels.size());
    const int first_C  = cfg.block_out_channels.front();
    const int mid_C    = cfg.block_out_channels.back();
    const int temb_dim = cfg.time_embed_dim;
    const int freq_dim = cfg.freq_dim;
    const int ctx_dim  = cfg.cross_attention_dim;

    bdfix::Builder b;
    b.add("conv_in.weight", {first_C, cfg.in_channels, 3, 3},
          bdfix::fp16_rand(
              static_cast<std::size_t>(first_C) * cfg.in_channels * 9,
              0.05f, 50));
    b.add("conv_in.bias",   {first_C}, bdfix::fp16_zeros(first_C));
    b.add("time_embedding.linear_1.weight", {temb_dim, freq_dim},
          bdfix::fp16_rand(static_cast<std::size_t>(temb_dim) * freq_dim,
                           0.05f, 100));
    b.add("time_embedding.linear_1.bias",   {temb_dim},
          bdfix::fp16_zeros(temb_dim));
    b.add("time_embedding.linear_2.weight", {temb_dim, temb_dim},
          bdfix::fp16_rand(static_cast<std::size_t>(temb_dim) * temb_dim,
                           0.05f, 101));
    b.add("time_embedding.linear_2.bias",   {temb_dim},
          bdfix::fp16_zeros(temb_dim));

    // Conditioning embedding CNN.
    {
        const std::string cp = "controlnet_cond_embedding.";
        const int first_ce = cfg.conditioning_embedding_channels.front();
        b.add(cp + "conv_in.weight",
              {first_ce, cfg.control_channels, 3, 3},
              bdfix::fp16_rand(
                  static_cast<std::size_t>(first_ce) * cfg.control_channels * 9,
                  0.04f, 200));
        b.add(cp + "conv_in.bias", {first_ce}, bdfix::fp16_zeros(first_ce));

        const int n_pairs =
            static_cast<int>(cfg.conditioning_embedding_channels.size()) - 1;
        for (int k = 0; k < n_pairs; ++k) {
            const int Ck =
                cfg.conditioning_embedding_channels[static_cast<std::size_t>(k)];
            const int Cnxt =
                cfg.conditioning_embedding_channels[static_cast<std::size_t>(k + 1)];
            const std::string ap = cp + "blocks." + std::to_string(2 * k) + ".";
            b.add(ap + "weight", {Ck, Ck, 3, 3},
                  bdfix::fp16_rand(static_cast<std::size_t>(Ck) * Ck * 9, 0.03f,
                                   static_cast<std::size_t>(210 + 10 * k)));
            b.add(ap + "bias",   {Ck}, bdfix::fp16_zeros(Ck));
            const std::string sp = cp + "blocks." + std::to_string(2 * k + 1) + ".";
            b.add(sp + "weight", {Cnxt, Ck, 3, 3},
                  bdfix::fp16_rand(static_cast<std::size_t>(Cnxt) * Ck * 9, 0.03f,
                                   static_cast<std::size_t>(215 + 10 * k)));
            b.add(sp + "bias",   {Cnxt}, bdfix::fp16_zeros(Cnxt));
        }

        const int last_ce = cfg.conditioning_embedding_channels.back();
        b.add(cp + "conv_out.weight", {first_C, last_ce, 3, 3},
              bdfix::fp16_rand(
                  static_cast<std::size_t>(first_C) * last_ce * 9, 0.03f, 299));
        b.add(cp + "conv_out.bias", {first_C}, bdfix::fp16_zeros(first_C));
    }

    // Down blocks (HF flag rule mirrors UNet: attn + downsampler EXCEPT last).
    int C_prev = first_C;
    for (int i = 0; i < nb; ++i) {
        const int C_out = cfg.block_out_channels[static_cast<std::size_t>(i)];
        const bool has_attn   = (i < nb - 1);
        const bool has_downsm = (i < nb - 1);
        for (int j = 0; j < cfg.layers_per_block; ++j) {
            const int Ci = (j == 0) ? C_prev : C_out;
            emit_resnet_cn(b,
                "down_blocks." + std::to_string(i) +
                ".resnets." + std::to_string(j) + ".",
                Ci, C_out, temb_dim,
                static_cast<std::size_t>(1000 + i * 10 + j));
            if (has_attn) {
                emit_transformer_cn(b,
                    "down_blocks." + std::to_string(i) +
                    ".attentions." + std::to_string(j) + ".",
                    C_out, ctx_dim,
                    static_cast<std::size_t>(2000 + i * 10 + j));
            }
        }
        if (has_downsm) {
            const std::string sp = "down_blocks." + std::to_string(i) +
                                   ".downsamplers.0.conv.";
            b.add(sp + "weight", {C_out, C_out, 3, 3},
                  bdfix::fp16_rand(
                      static_cast<std::size_t>(C_out) * C_out * 9, 0.02f,
                      static_cast<std::size_t>(3000 + i)));
            b.add(sp + "bias",   {C_out}, bdfix::fp16_zeros(C_out));
        }
        C_prev = C_out;
    }

    emit_resnet_cn(b, "mid_block.resnets.0.", mid_C, mid_C, temb_dim, 4000);
    emit_transformer_cn(b, "mid_block.attentions.0.", mid_C, ctx_dim, 4100);
    emit_resnet_cn(b, "mid_block.resnets.1.", mid_C, mid_C, temb_dim, 4200);

    // Zero-conv outputs: NON-ZERO so the residuals are observable. Zero-init
    // training is irrelevant for a fixture; we just need a deterministic
    // non-trivial signal to wire through the UNet skips.
    std::vector<int> skip_channels;
    skip_channels.push_back(first_C);
    for (int i = 0; i < nb; ++i) {
        const int Cb = cfg.block_out_channels[static_cast<std::size_t>(i)];
        for (int j = 0; j < cfg.layers_per_block; ++j) skip_channels.push_back(Cb);
        if (i < nb - 1) skip_channels.push_back(Cb);
    }
    for (std::size_t i = 0; i < skip_channels.size(); ++i) {
        const int C = skip_channels[i];
        const std::string zp = "controlnet_down_blocks." + std::to_string(i) + ".";
        b.add(zp + "weight", {C, C, 1, 1},
              bdfix::fp16_rand(static_cast<std::size_t>(C) * C, 0.03f,
                               5000 + i));
        b.add(zp + "bias",   {C}, bdfix::fp16_zeros(C));
    }
    b.add("controlnet_mid_block.weight", {mid_C, mid_C, 1, 1},
          bdfix::fp16_rand(static_cast<std::size_t>(mid_C) * mid_C, 0.03f, 6000));
    b.add("controlnet_mid_block.bias",   {mid_C}, bdfix::fp16_zeros(mid_C));

    auto path = std::filesystem::temp_directory_path() /
                "brodiffusion_controlnet_pipeline_cn.safetensors";
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

// Download a (possibly FP16) tensor of known size to host floats.
std::vector<float> dl(const bt::Tensor& t) {
    const std::size_t n = static_cast<std::size_t>(t.rows) * t.cols;
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        t.copy_to_host_fp16(bits.data());
        bt::sync_all();
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = bt::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return t.to_host_vector();
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
    auto vp  = tmp / "brodiffusion_cnp_vocab.json";
    auto mp  = tmp / "brodiffusion_cnp_merges.txt";
    auto ctrl_path = tmp / "brodiffusion_cnp_ctrl.png";
    write_tiny_vocab(vp, mp);
    write_png(ctrl_path, /*pattern=*/1);

    auto sd_path = build_sd_fixture();
    auto sd_file = st::File::open(sd_path.string());
    auto cn_cfg  = controlnet_cfg();
    auto cn_path = build_controlnet_fixture(cn_cfg);
    auto cn_file = st::File::open(cn_path.string());

    auto make_pipeline = [&](bool with_controlnet) {
        auto tok = clip::Tokenizer::load(vp.string(), mp.string());
        pl::Pipeline p(make_cfg(), std::move(tok));
        p.load_weights(sd_file, "text_model.", "", "decoder.");
        if (with_controlnet) p.apply_controlnet(cn_file, cn_cfg);
        return p;
    };

    constexpr int W = 16;
    constexpr int H = 16;

    auto base_opts = []() {
        pl::GenerateOptions o;
        o.width  = W;
        o.height = H;
        o.num_inference_steps = 2;
        o.guidance_scale = 1.0f;  // skip uncond pass to keep test cheap
        o.seed = 7;
        return o;
    };

    // ── 1. zero_scale_equals_no_controlnet ───────────────────────────────
    {
        auto p_no = make_pipeline(/*with_controlnet=*/false);
        std::vector<float> img_no = p_no.generate("hi", base_opts());

        auto p_cn = make_pipeline(/*with_controlnet=*/true);
        auto opts = base_opts();
        opts.control_image_path = ctrl_path.string();
        opts.control_scale      = 0.0f;
        std::vector<float> img_cn0 = p_cn.generate("hi", opts);

        CHECK(img_no.size() == img_cn0.size());
        // FP16 backend: any rounding noise inside a multiplied-by-zero residual
        // could in principle perturb the result, but the ControlNet path adds
        // 0.0f directly, which round-trips through cast() exactly. Expect
        // bit-identical output (FP32 compare).
        bool exact = (img_no == img_cn0);
        if (!exact) {
            // Diagnostics on mismatch — print max abs delta.
            float max_d = 0.0f;
            for (std::size_t i = 0; i < img_no.size(); ++i) {
                float d = std::fabs(img_no[i] - img_cn0[i]);
                if (d > max_d) max_d = d;
            }
            std::fprintf(stderr,
                "zero_scale: not bit-identical, max abs delta = %g\n",
                static_cast<double>(max_d));
        }
        CHECK(exact);
    }

    // ── 2. nonzero_scale_changes_output ──────────────────────────────────
    {
        auto p_no = make_pipeline(/*with_controlnet=*/false);
        std::vector<float> img_no = p_no.generate("hi", base_opts());

        auto p_cn = make_pipeline(/*with_controlnet=*/true);
        auto opts = base_opts();
        opts.control_image_path = ctrl_path.string();
        opts.control_scale      = 1.0f;
        std::vector<float> img_cn = p_cn.generate("hi", opts);

        CHECK(img_no.size() == img_cn.size());
        bool any_diff = false;
        float max_d = 0.0f;
        for (std::size_t i = 0; i < img_no.size(); ++i) {
            float d = std::fabs(img_no[i] - img_cn[i]);
            if (d > 1e-5f) any_diff = true;
            if (d > max_d) max_d = d;
        }
        if (!any_diff) {
            std::fprintf(stderr,
                "nonzero_scale: outputs match too closely, max abs delta = %g\n",
                static_cast<double>(max_d));
        }
        CHECK(any_diff);
    }

    // ── 3. finite_output ─────────────────────────────────────────────────
    {
        auto p_cn = make_pipeline(/*with_controlnet=*/true);
        auto opts = base_opts();
        opts.control_image_path = ctrl_path.string();
        opts.control_scale      = 1.0f;
        std::vector<float> img = p_cn.generate("hi", opts);
        const std::size_t n_img =
            3u * static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
        CHECK(img.size() == n_img);
        int nonfinite = 0;
        for (float v : img) if (!std::isfinite(v)) ++nonfinite;
        CHECK(nonfinite == 0);
    }

    // ── 4. missing_controlnet_throws ─────────────────────────────────────
    {
        auto p_no = make_pipeline(/*with_controlnet=*/false);
        auto opts = base_opts();
        opts.control_image_path = ctrl_path.string();
        bool threw = false;
        try {
            (void)p_no.generate("hi", opts);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
    }

    // ── cleanup ──────────────────────────────────────────────────────────
    std::error_code ec;
    std::filesystem::remove(sd_path,   ec);
    std::filesystem::remove(cn_path,   ec);
    std::filesystem::remove(ctrl_path, ec);
    std::filesystem::remove(vp,        ec);
    std::filesystem::remove(mp,        ec);

    if (g_failures == 0) std::printf("controlnet_pipeline: OK\n");
    else std::fprintf(stderr, "controlnet_pipeline: %d failure(s)\n",
                      g_failures);
    return g_failures ? 1 : 0;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "controlnet_pipeline: EXCEPTION: %s\n", e.what());
        std::fflush(stderr);
        return 1;
    } catch (...) {
        std::fprintf(stderr, "controlnet_pipeline: UNKNOWN EXCEPTION\n");
        std::fflush(stderr);
        return 1;
    }
}
