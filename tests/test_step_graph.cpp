// CUDA-graph denoising-step parity test.
//
// Builds a tiny but architecturally complete SD1.5 Pipeline (CLIP + UNet +
// VAE decoder) from synthetic FP16 weights and generates the same image
// twice with the same seed:
//
//   run A — step graph enabled (the default on a CUDA backend): steps 0-1
//           run eagerly through the capture seam, the cond(+uncond) forward
//           bodies are captured at the end of step 1, steps 2+ replay the
//           graph.
//   run B — BRODIFFUSION_DISABLE_STEP_GRAPH=1: every step runs the classic
//           eager denoiser_->forward path.
//
// The two runs must be BIT-IDENTICAL: a graph replay re-executes exactly the
// kernel sequence the eager body launches, on the same buffers, with the
// same inputs. Any divergence means the captured body depends on state the
// replay path failed to refresh (a real bug), not acceptable numerical
// noise.
//
// Covered: CFG on (guidance 7.5 — cond + uncond bodies in one graph) and
// CFG off (guidance 1.0 — single-branch graph). On a CPU-only build the
// graph path is ineligible and both runs are eager — the comparison is then
// trivially equal and the test still passes.

#include "brodiffusion/pipeline.h"
#include "brodiffusion/detail/compute.h"
#include "brolm/tokenizer.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "sd_fixtures.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
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

void set_env(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

// ── module configs (shrunk, full topology — mirrors test_img2img) ─────────

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
    c.hidden_dim       = 8;          // must match unet cross_attention_dim
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

// VAE-encoder + quant-conv fixture emitters, copied verbatim from
// test_img2img.cpp (the Pipeline loader always loads the encoder).
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

std::filesystem::path build_fixture() {
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
                "brodiffusion_step_graph_test.safetensors";
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

    auto tmp = std::filesystem::temp_directory_path();
    auto vp  = tmp / "brodiffusion_step_graph_vocab.json";
    auto mp  = tmp / "brodiffusion_step_graph_merges.txt";
    write_tiny_vocab(vp, mp);

    auto fix_path = build_fixture();
    auto fix_file = st::File::open(fix_path.string());

    auto make_loaded_pipeline = [&]() {
        auto tok = clip::Tokenizer::load(vp.string(), mp.string());
        pl::Pipeline p(make_cfg(), std::move(tok));
        p.load_weights(fix_file, "text_model.", "", "decoder.");
        return p;
    };

    // 16x16 image -> 2x2 latent (8x VAE), the smallest legal input for the
    // 2-block UNet. 8 steps: 2 eager warm-ups + capture, 6 graph replays.
    auto run_generate_prompt = [&](const char* prompt, float guidance,
                                   bool disable_graph) -> std::vector<float> {
        set_env("BRODIFFUSION_DISABLE_STEP_GRAPH", disable_graph ? "1" : "0");
        auto p = make_loaded_pipeline();
        pl::GenerateOptions opts;
        opts.width  = 16;
        opts.height = 16;
        opts.num_inference_steps = 8;
        opts.guidance_scale = guidance;
        opts.seed = 1234;
        return p.generate(prompt, opts);
    };
    auto run_generate = [&](float guidance, bool disable_graph)
            -> std::vector<float> {
        return run_generate_prompt("hi", guidance, disable_graph);
    };

    // ── 1. CFG on: cond + uncond bodies in one graph ──────────────────────
    {
        std::vector<float> graph_img = run_generate(7.5f, /*disable=*/false);
        std::vector<float> eager_img = run_generate(7.5f, /*disable=*/true);
        CHECK(graph_img.size() == eager_img.size());
        CHECK(!graph_img.empty());
        int nonfinite = 0;
        for (float v : graph_img) if (!std::isfinite(v)) ++nonfinite;
        CHECK(nonfinite == 0);
        bool identical =
            graph_img.size() == eager_img.size() &&
            std::memcmp(graph_img.data(), eager_img.data(),
                        graph_img.size() * sizeof(float)) == 0;
        CHECK(identical);
        if (!identical) {
            float max_diff = 0.0f;
            for (std::size_t i = 0; i < graph_img.size(); ++i) {
                max_diff = std::max(max_diff,
                                    std::fabs(graph_img[i] - eager_img[i]));
            }
            std::fprintf(stderr, "  CFG-on graph vs eager max |diff| = %g\n",
                         max_diff);
        }
    }

    // ── 2. CFG off: single-branch graph ───────────────────────────────────
    {
        std::vector<float> graph_img = run_generate(1.0f, /*disable=*/false);
        std::vector<float> eager_img = run_generate(1.0f, /*disable=*/true);
        bool identical =
            graph_img.size() == eager_img.size() &&
            std::memcmp(graph_img.data(), eager_img.data(),
                        graph_img.size() * sizeof(float)) == 0;
        CHECK(identical);
        if (!identical) {
            float max_diff = 0.0f;
            for (std::size_t i = 0; i < graph_img.size(); ++i) {
                max_diff = std::max(max_diff,
                                    std::fabs(graph_img[i] - eager_img[i]));
            }
            std::fprintf(stderr, "  CFG-off graph vs eager max |diff| = %g\n",
                         max_diff);
        }
    }

    // ── 3. Many generations on ONE pipeline ───────────────────────────────
    //
    // Everything above builds a fresh Pipeline per image, so the graph session is
    // always new and the interesting case never runs. A real caller generates over
    // and over on one pipeline, and that is where the session key mattered: it
    // used to be the ADDRESS of the latent and of the prepared conditioning, both
    // of which are freed and re-allocated at the same size by the next generate().
    // A pooled allocator hands back the same address, so the key compared equal
    // across two different generations and the graph captured for the first was
    // replayed for the second — carrying the first one's baked buffer pointers.
    //
    // It survives on the accident that everything else lands back at its old
    // address too. So this perturbs the allocator between generations, the way any
    // real workload does (the sweep that found this ran a CLIP encode between
    // renders): churn some device memory, then generate again and demand the same
    // image the eager path produces. A stale replay shows up either as CUDA error
    // 700 or as a quietly wrong image, and both fail this check.
    {
        auto p = make_loaded_pipeline();
        set_env("BRODIFFUSION_DISABLE_STEP_GRAPH", "0");

        const char* prompts[] = {"hi", "hello there", "hi"};
        for (int i = 0; i < 3; ++i) {
            // Churn the device allocator between generations, the way a real
            // workload does. This has to be BIG and ragged to matter: the sweep
            // that found the bug ran a CLIP image encode between renders, which
            // allocates and frees megabytes and leaves the pool's free lists
            // rearranged. Kilobyte-sized churn is not enough — the pool hands
            // everything straight back to its old address and the stale graph
            // replays harmlessly.

            // Churn the device allocator between generations, the way a real
            // workload does: the sweep that found this ran a CLIP image encode
            // between renders, which allocates and frees megabytes and leaves the
            // pool's free lists rearranged. This is what forces the session key to
            // be re-examined, and with it the recapture path that never ran when
            // every image got a fresh Pipeline.
            for (int k = 1; k <= 6; ++k) {
                bt::Tensor big = bt::Tensor::zeros_on(
                    bt::default_device(), 1, 997 * 1024 * k, bt::Dtype::FP32);
                bt::Tensor odd = bt::Tensor::zeros_on(
                    bt::default_device(), 1, 65537 * k, bt::Dtype::FP16);
                (void)big.size();
                (void)odd.size();
            }

            pl::GenerateOptions opts;
            opts.width  = 16;
            opts.height = 16;
            opts.num_inference_steps = 8;
            opts.guidance_scale = 1.0f;
            opts.seed = 1234;
            std::vector<float> got = p.generate(prompts[i], opts);

            std::vector<float> want = run_generate_prompt(prompts[i], 1.0f,
                                                          /*disable=*/true);
            set_env("BRODIFFUSION_DISABLE_STEP_GRAPH", "0");

            CHECK(got.size() == want.size());
            bool identical = got.size() == want.size() &&
                             std::memcmp(got.data(), want.data(),
                                         got.size() * sizeof(float)) == 0;
            CHECK(identical);
            if (!identical) {
                float max_diff = 0.0f;
                for (std::size_t j = 0; j < got.size() && j < want.size(); ++j)
                    max_diff = std::max(max_diff, std::fabs(got[j] - want[j]));
                std::fprintf(stderr,
                             "  generation %d (\"%s\") on a reused pipeline "
                             "differs from eager: max |diff| = %g\n",
                             i, prompts[i], max_diff);
            }
        }
    }

    // ── 4. State identity is never recycled ───────────────────────────────
    //
    // The invariant the graph key now rests on. An address can be handed out
    // twice; a counter cannot.
    {
        auto p = make_loaded_pipeline();
        pl::GenerateOptions opts;
        opts.width  = 16;
        opts.height = 16;
        opts.num_inference_steps = 8;
        opts.seed = 1234;

        pl::PipelineState a = p.prime("hi", opts);
        const std::uint64_t a_id = a.id;
        pl::PipelineState b = p.prime("hi", opts);   // same prompt, same size
        CHECK(a.id != b.id);
        CHECK(a.clone().id != a_id);

        // And the states really can land on the same addresses — which is what
        // made the old address key unsound. Not a requirement, just the reason.
        if (a.latent.data == b.latent.data) {
            std::fprintf(stderr,
                         "  (note: two live states share a latent address — "
                         "exactly the collision the id key now prevents)\n");
        }
    }

    set_env("BRODIFFUSION_DISABLE_STEP_GRAPH", "");

    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 1;
    }
    return g_failures == 0 ? 0 : 1;
}
