#include "brodiffusion/cifar_pipeline.h"
#if BRODIFFUSION_HAS_BROGAMEAGENT
#include "brodiffusion/cifar_mcts.h"
#include "brodiffusion/clip_image.h"
#include "brodiffusion/clip_score.h"
#include "brodiffusion/sd_mcts.h"
#endif
#include "brodiffusion/pipeline.h"
#include "brodiffusion/safetensors.h"
#include "brodiffusion/tokenizer.h"
#include "brodiffusion/version.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pl   = brodiffusion::pipeline;
namespace st   = brodiffusion::safetensors;
namespace clip = brodiffusion::clip;

static int usage() {
    std::printf(
        "brodiffusion %s\n"
        "\n"
        "Usage:\n"
        "  brodiffusion --version\n"
        "  brodiffusion txt2img --text <st> --unet <st> --vae <st>\n"
        "                       --vocab <vocab.json> --merges <merges.txt>\n"
        "                       --prompt <text> --out <ppm>\n"
        "                       [--negative <text>] [--steps N] [--cfg F]\n"
        "                       [--width N] [--height N] [--seed N]\n"
        "                       [--scheduler ddim|lcm]\n"
        "                       [--lora <path>[:<scale>]]... [--lcm-lora <path>]\n"
        "                       [--quantize-unet]\n"
        "\n"
        "  --scheduler lcm  selects the LCM (Latent Consistency Model) scheduler;\n"
        "                   requires an LCM-distilled UNet checkpoint (e.g.\n"
        "                   SimianLuo/LCM_Dreamshaper_v7). When set, default --steps\n"
        "                   becomes 4 and the uncond pass is skipped; --cfg is reused\n"
        "                   as the guidance-scale embedding `w`.\n"
        "\n"
        "  --lora <path>[:<scale>]  merge a LoRA file into the loaded weights\n"
        "                   before generation. Repeatable; scale defaults to 1.0\n"
        "                   and may be negative. Supports both kohya-ss/A1111 and\n"
        "                   diffusers/PEFT key conventions (auto-detected).\n"
        "\n"
        "  --lcm-lora <path>  sugar for '--scheduler lcm --steps 4 --cfg 1.0\n"
        "                   --lora <path>' against a vanilla SD1.5 UNet (no\n"
        "                   cond_proj; LCM-LoRA on top of stock SD1.5).\n"
        "\n"
        "Writes an uncompressed PPM (P6) — a dev convenience for sanity-checking\n"
        "the generation pipeline. Proper PNG/JPEG encoding lives outside this\n"
        "library (bro's image-api on integration).\n",
        brodiffusion::version_string());
    return 0;
}

namespace {

const char* arg_after(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    }
    return nullptr;
}

struct LoraSpec {
    std::string path;
    float       scale = 1.0f;
};

// Collect every "--lora <path>[:<scale>]" from argv. Repeatable.
std::vector<LoraSpec> collect_loras(int argc, char** argv) {
    std::vector<LoraSpec> out;
    for (int i = 1; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], "--lora") != 0) continue;
        std::string raw = argv[i + 1];
        // Split on the LAST ':' so Windows drive letters ("D:\foo:0.8") parse
        // correctly — the path may contain colons. A trailing
        // ":<number>" is the scale; anything else is part of the path.
        LoraSpec s;
        auto pos = raw.rfind(':');
        bool has_scale = false;
        if (pos != std::string::npos && pos != 0 && pos > 1) {
            const std::string tail = raw.substr(pos + 1);
            // Heuristic: a scale is a parseable float of strlen() > 0.
            char* endp = nullptr;
            float v = std::strtof(tail.c_str(), &endp);
            if (endp && *endp == '\0' && !tail.empty()) {
                s.scale = v;
                s.path  = raw.substr(0, pos);
                has_scale = true;
            }
        }
        if (!has_scale) s.path = raw;
        out.push_back(std::move(s));
    }
    return out;
}

int run_txt2img(int argc, char** argv) {
    const char* text_path   = arg_after(argc, argv, "--text");
    const char* unet_path   = arg_after(argc, argv, "--unet");
    const char* vae_path    = arg_after(argc, argv, "--vae");
    const char* vocab_path  = arg_after(argc, argv, "--vocab");
    const char* merges_path = arg_after(argc, argv, "--merges");
    const char* prompt      = arg_after(argc, argv, "--prompt");
    const char* out_path    = arg_after(argc, argv, "--out");
    const char* neg         = arg_after(argc, argv, "--negative");
    const char* steps_s     = arg_after(argc, argv, "--steps");
    const char* cfg_s       = arg_after(argc, argv, "--cfg");
    const char* width_s     = arg_after(argc, argv, "--width");
    const char* height_s    = arg_after(argc, argv, "--height");
    const char* seed_s      = arg_after(argc, argv, "--seed");
    const char* sched_s     = arg_after(argc, argv, "--scheduler");
    const char* lcm_lora    = arg_after(argc, argv, "--lcm-lora");

    bool quantize_unet = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quantize-unet") == 0) quantize_unet = true;
    }

    if (!text_path || !unet_path || !vae_path ||
        !vocab_path || !merges_path || !prompt || !out_path) {
        std::fprintf(stderr,
            "txt2img: --text, --unet, --vae, --vocab, --merges, --prompt, --out are required\n");
        return 2;
    }

    bool use_lcm = false;
    if (sched_s) {
        if (std::strcmp(sched_s, "lcm") == 0) use_lcm = true;
        else if (std::strcmp(sched_s, "ddim") != 0) {
            std::fprintf(stderr, "txt2img: --scheduler must be 'ddim' or 'lcm'\n");
            return 2;
        }
    }

    // Collect explicit --lora flags. --lcm-lora is sugar for "LCM scheduler
    // on a vanilla SD1.5 UNet + this LoRA at scale 1.0"; the LoRA is appended
    // to the list and the scheduler/steps/cfg defaults are flipped.
    auto loras = collect_loras(argc, argv);
    bool lcm_lora_mode = false;
    if (lcm_lora) {
        use_lcm = true;
        lcm_lora_mode = true;
        LoraSpec s;
        s.path  = lcm_lora;
        s.scale = 1.0f;
        loras.push_back(std::move(s));
    }

    pl::GenerateOptions opts;
    if (neg)     opts.negative_prompt = neg;
    if (steps_s) opts.num_inference_steps = std::atoi(steps_s);
    else if (use_lcm) opts.num_inference_steps = 4;
    if (cfg_s)   opts.guidance_scale = static_cast<float>(std::atof(cfg_s));
    else if (lcm_lora_mode) opts.guidance_scale = 1.0f;
    if (width_s) opts.width  = std::atoi(width_s);
    if (height_s)opts.height = std::atoi(height_s);
    if (seed_s)  opts.seed = static_cast<std::uint64_t>(std::strtoull(seed_s, nullptr, 10));

    brotensor::cuda_init();

    auto tok = clip::Tokenizer::load(vocab_path, merges_path);

    pl::PipelineConfig cfg;
    if (use_lcm) {
        cfg.scheduler = brodiffusion::scheduler::LCMConfig{};
        // LCM-LoRA runs on a vanilla SD1.5 UNet (no cond_proj weight); only
        // a distilled LCM checkpoint has time_cond_proj_dim=256.
        if (!lcm_lora_mode) cfg.unet.time_cond_proj_dim = 256;
    }
    cfg.unet.quantize_weights = quantize_unet;
    pl::Pipeline pipeline(cfg, std::move(tok));

    std::printf("Loading weights:\n  text: %s\n  unet: %s\n  vae:  %s\n",
                text_path, unet_path, vae_path);
    auto text_file = st::File::open(text_path);
    auto unet_file = st::File::open(unet_path);
    auto vae_file  = st::File::open(vae_path);
    pipeline.load_weights(text_file, unet_file, vae_file);

    // Merge LoRAs in command-line order. Each apply_lora call mutates the
    // underlying UNet/CLIP weights in place; later calls stack on earlier
    // ones, so the order matches the user's argv order.
    for (const auto& spec : loras) {
        std::printf("Applying LoRA: %s (scale=%.3f)\n", spec.path.c_str(),
                    static_cast<double>(spec.scale));
        auto lora_file = st::File::open(spec.path);
        pipeline.apply_lora(lora_file, spec.scale);
    }

    std::printf("Generating %dx%d, %d steps, CFG=%.1f, seed=%llu\n",
                opts.width, opts.height, opts.num_inference_steps,
                static_cast<double>(opts.guidance_scale),
                static_cast<unsigned long long>(opts.seed));

    auto img = pipeline.generate(prompt, opts);  // (3*H*W) NCHW, FP32 in [-1, 1]

    // Convert planar [-1,1] FP32 → interleaved RGB8.
    const int H = opts.height, W = opts.width;
    const int plane = H * W;
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(3) * plane);
    for (int i = 0; i < plane; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = (img[c * plane + i] + 1.0f) * 127.5f;
            v = std::clamp(v, 0.0f, 255.0f);
            rgb[3 * i + c] = static_cast<std::uint8_t>(v + 0.5f);
        }
    }

    std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::fprintf(stderr, "txt2img: cannot open output %s\n", out_path);
        return 1;
    }
    f << "P6\n" << W << " " << H << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()),
            static_cast<std::streamsize>(rgb.size()));
    std::printf("Wrote %s\n", out_path);
    return 0;
}


int run_cifar_sample(int argc, char** argv) {
    const char* unet_path = arg_after(argc, argv, "--unet");
    const char* out_path  = arg_after(argc, argv, "--out");
    const char* steps_s   = arg_after(argc, argv, "--steps");
    const char* seed_s    = arg_after(argc, argv, "--seed");
    const char* size_s    = arg_after(argc, argv, "--size");

    if (!unet_path || !out_path) {
        std::fprintf(stderr,
            "cifar-sample: --unet <safetensors> --out <ppm> are required\n"
            "             [--steps N] [--seed N] [--size N]\n");
        return 2;
    }

    namespace cif = brodiffusion::cifar_pipeline;
    cif::GenerateOptions opts;
    if (steps_s) opts.num_inference_steps = std::atoi(steps_s);
    if (seed_s)  opts.seed = static_cast<std::uint64_t>(std::strtoull(seed_s, nullptr, 10));
    if (size_s) {
        opts.height = std::atoi(size_s);
        opts.width  = std::atoi(size_s);
    }

    brotensor::cuda_init();

    cif::PipelineConfig cfg;  // defaults match google/ddpm-cifar10-32
    cif::Pipeline pipeline(cfg);

    std::printf("Loading UNet: %s\n", unet_path);
    auto unet_file = st::File::open(unet_path);
    pipeline.load_weights(unet_file);

    std::printf("Sampling %dx%d, %d steps, seed=%llu\n",
                opts.width, opts.height, opts.num_inference_steps,
                static_cast<unsigned long long>(opts.seed));

    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    auto img = pipeline.generate(opts);  // (3 * H * W) NCHW FP32 in [-1, 1]
    auto t1 = clk::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Done in %.1f ms (%.2f ms/step)\n", ms, ms / opts.num_inference_steps);

    const int H = opts.height, W = opts.width;
    const int plane = H * W;
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(3) * plane);
    for (int i = 0; i < plane; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = (img[c * plane + i] + 1.0f) * 127.5f;
            v = std::clamp(v, 0.0f, 255.0f);
            rgb[3 * i + c] = static_cast<std::uint8_t>(v + 0.5f);
        }
    }

    std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::fprintf(stderr, "cifar-sample: cannot open output %s\n", out_path);
        return 1;
    }
    f << "P6\n" << W << " " << H << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()),
            static_cast<std::streamsize>(rgb.size()));
    std::printf("Wrote %s\n", out_path);
    return 0;
}

#if BRODIFFUSION_HAS_BROGAMEAGENT
int run_cifar_mcts(int argc, char** argv) {
    const char* unet_path = arg_after(argc, argv, "--unet");
    const char* out_path  = arg_after(argc, argv, "--out");
    const char* steps_s   = arg_after(argc, argv, "--steps");
    const char* seed_s    = arg_after(argc, argv, "--seed");
    const char* size_s    = arg_after(argc, argv, "--size");
    const char* branch_s  = arg_after(argc, argv, "--branching");
    const char* iters_s   = arg_after(argc, argv, "--iters");
    const char* di_s      = arg_after(argc, argv, "--decision-interval");

    if (!unet_path || !out_path) {
        std::fprintf(stderr,
            "cifar-mcts: --unet <safetensors> --out <ppm> are required\n"
            "           [--steps N] [--seed N] [--size N]\n"
            "           [--branching B] [--iters N] [--decision-interval N]\n"
            "\n"
            "Default scorer is mean luminance (smoke-test only — override in\n"
            "code with Sampler::set_scorer for real experiments).\n");
        return 2;
    }

    namespace cif = brodiffusion::cifar_pipeline;
    namespace cm  = brodiffusion::cifar_mcts;

    cif::GenerateOptions opts;
    if (steps_s) opts.num_inference_steps = std::atoi(steps_s);
    if (seed_s)  opts.seed = static_cast<std::uint64_t>(std::strtoull(seed_s, nullptr, 10));
    if (size_s) {
        opts.height = std::atoi(size_s);
        opts.width  = std::atoi(size_s);
    }

    cm::Config mcfg;
    if (branch_s) mcfg.branching_factor  = std::atoi(branch_s);
    if (iters_s)  mcfg.iterations        = std::atoi(iters_s);
    if (di_s)     mcfg.decision_interval = std::atoi(di_s);
    mcfg.seed = opts.seed;

    brotensor::cuda_init();

    cif::PipelineConfig cfg;
    cif::Pipeline pipeline(cfg);

    std::printf("Loading UNet: %s\n", unet_path);
    auto unet_file = st::File::open(unet_path);
    pipeline.load_weights(unet_file);

    cm::Sampler sampler(pipeline, mcfg);

    const int decisions = (opts.num_inference_steps + mcfg.decision_interval - 1)
                          / mcfg.decision_interval;
    std::printf("MCTS-guided sample: %dx%d, %d steps, B=%d iters=%d, "
                "decision-interval=%d (≈%d decisions), seed=%llu\n",
                opts.width, opts.height, opts.num_inference_steps,
                mcfg.branching_factor, mcfg.iterations, mcfg.decision_interval,
                decisions, static_cast<unsigned long long>(opts.seed));

    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    auto img = sampler.generate(opts);
    auto t1 = clk::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Done in %.1f ms (%.2f ms/decision avg)\n",
                ms, ms / std::max(1, static_cast<int>(sampler.last_decisions().size())));

    // Per-decision summary.
    for (std::size_t i = 0; i < sampler.last_decisions().size(); ++i) {
        const auto& d = sampler.last_decisions()[i];
        std::printf("  decision %zu @step=%d: action=%d visits=%d tree=%d\n",
                    i, d.step_index_before, d.best_action, d.best_visits, d.tree_size);
    }

    const int H = opts.height, W = opts.width;
    const int plane = H * W;
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(3) * plane);
    for (int i = 0; i < plane; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = (img[c * plane + i] + 1.0f) * 127.5f;
            v = std::clamp(v, 0.0f, 255.0f);
            rgb[3 * i + c] = static_cast<std::uint8_t>(v + 0.5f);
        }
    }
    std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::fprintf(stderr, "cifar-mcts: cannot open output %s\n", out_path);
        return 1;
    }
    f << "P6\n" << W << " " << H << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()),
            static_cast<std::streamsize>(rgb.size()));
    std::printf("Wrote %s\n", out_path);
    return 0;
}

int run_sd_mcts(int argc, char** argv) {
    const char* text_path   = arg_after(argc, argv, "--text");
    const char* unet_path   = arg_after(argc, argv, "--unet");
    const char* vae_path    = arg_after(argc, argv, "--vae");
    const char* vocab_path  = arg_after(argc, argv, "--vocab");
    const char* merges_path = arg_after(argc, argv, "--merges");
    const char* prompt      = arg_after(argc, argv, "--prompt");
    const char* out_path    = arg_after(argc, argv, "--out");
    const char* neg         = arg_after(argc, argv, "--negative");
    const char* steps_s     = arg_after(argc, argv, "--steps");
    const char* cfg_s       = arg_after(argc, argv, "--cfg");
    const char* width_s     = arg_after(argc, argv, "--width");
    const char* height_s    = arg_after(argc, argv, "--height");
    const char* seed_s      = arg_after(argc, argv, "--seed");
    const char* branch_s    = arg_after(argc, argv, "--branching");
    const char* iters_s     = arg_after(argc, argv, "--iters");
    const char* di_s        = arg_after(argc, argv, "--decision-interval");
    const char* bias_s      = arg_after(argc, argv, "--bias-magnitude");
    const char* cpuct_s     = arg_after(argc, argv, "--c-puct");
    const char* clip_path   = arg_after(argc, argv, "--clip");
    const char* trace_dir   = arg_after(argc, argv, "--enumerate-trace-out");
    bool enumerate_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--enumerate") == 0) enumerate_only = true;
    }
    if (trace_dir) enumerate_only = true;

    if (!text_path || !unet_path || !vae_path ||
        !vocab_path || !merges_path || !prompt || !out_path) {
        std::fprintf(stderr,
            "sd-mcts: --text, --unet, --vae, --vocab, --merges, --prompt, --out are required\n"
            "         [--clip <openai-clip-vit-l-14.safetensors>]\n"
            "         [--negative <text>] [--steps N] [--cfg F] [--width N] [--height N]\n"
            "         [--seed N] [--branching B] [--iters N] [--decision-interval N]\n"
            "         [--bias-magnitude F] [--c-puct F] [--enumerate]\n"
            "         [--enumerate-trace-out <dir>]\n"
            "\n"
            "  --enumerate  Skip MCTS. Run B trajectories (one per action) to\n"
            "               terminal, score each, write all B images with their\n"
            "               scores. Diagnostic for whether the action space\n"
            "               actually steers and whether the scorer discriminates.\n"
            "  --enumerate-trace-out <dir>\n"
            "               Implies --enumerate. Also writes per-action binary\n"
            "               traces to <dir>/trace_a<N>.bin. Format (little-endian):\n"
            "                 i32 magic = 0x42445354 ('BDST')\n"
            "                 i32 version = 1\n"
            "                 i32 action, i32 D, i32 C_lat, i32 H_lat, i32 W_lat\n"
            "                 f32 score, u64 seed\n"
            "                 D × { i32 step_index; (C_lat*H_lat*W_lat) f32 latent }\n"
            "               Bias pattern per action is deterministic and can be\n"
            "               regenerated host-side from (seed, action) by drawing\n"
            "               Lq*Lk samples from N(0, bias_magnitude) using mt19937_64\n"
            "               seeded with (seed XOR (0xA77B1A5 * (action+1))).\n"
            "\n"
            "If --clip is supplied, the scorer is CLIP score (cosine similarity\n"
            "between the projected image embedding and the projected prompt\n"
            "embedding). Without it the default is mean luminance — a smoke\n"
            "test only; meaningless on 512x512 SD outputs.\n");
        return 2;
    }

    pl::GenerateOptions opts;
    if (neg)      opts.negative_prompt = neg;
    if (steps_s)  opts.num_inference_steps = std::atoi(steps_s);
    if (cfg_s)    opts.guidance_scale = static_cast<float>(std::atof(cfg_s));
    if (width_s)  opts.width  = std::atoi(width_s);
    if (height_s) opts.height = std::atoi(height_s);
    if (seed_s)   opts.seed = static_cast<std::uint64_t>(std::strtoull(seed_s, nullptr, 10));

    namespace sm = brodiffusion::sd_mcts;
    sm::Config mcfg;
    if (branch_s) mcfg.branching_factor  = std::atoi(branch_s);
    if (iters_s)  mcfg.iterations        = std::atoi(iters_s);
    if (di_s)     mcfg.decision_interval = std::atoi(di_s);
    if (bias_s)   mcfg.bias_magnitude    = static_cast<float>(std::atof(bias_s));
    if (cpuct_s)  mcfg.c_puct            = static_cast<float>(std::atof(cpuct_s));
    mcfg.seed = opts.seed;

    brotensor::cuda_init();

    auto tok = clip::Tokenizer::load(vocab_path, merges_path);
    pl::PipelineConfig cfg;  // defaults: DDIM scheduler, vanilla SD1.5 UNet.
    pl::Pipeline pipeline(cfg, std::move(tok));

    std::printf("Loading weights:\n  text: %s\n  unet: %s\n  vae:  %s\n",
                text_path, unet_path, vae_path);
    auto text_file = st::File::open(text_path);
    auto unet_file = st::File::open(unet_path);
    auto vae_file  = st::File::open(vae_path);
    pipeline.load_weights(text_file, unet_file, vae_file);

    sm::Sampler sampler(pipeline, mcfg);

    // Optional CLIP-score scorer. We give the scorer its OWN tokenizer +
    // text encoder so the pipeline's text encoder is undisturbed (it's
    // mid-generation across MCTS branches; reusing it for prompt scoring
    // would race with the cross-attn cache). Memory cost: ~250 MB extra
    // FP16 weights, fine on a 24 GB card.
    std::unique_ptr<clip::Tokenizer>             clip_tok;
    std::unique_ptr<clip::TextEncoder>           clip_text;
    std::unique_ptr<brodiffusion::clip_image::ImageEncoder> clip_img;
    std::unique_ptr<brodiffusion::clip_score::CLIPScorer>   clip_scr;
    if (clip_path) {
        std::printf("Loading CLIP weights: %s\n", clip_path);
        clip_tok  = std::make_unique<clip::Tokenizer>(
            clip::Tokenizer::load(vocab_path, merges_path));
        clip_text = std::make_unique<clip::TextEncoder>(clip::TextEncoderConfig{});
        clip_img  = std::make_unique<brodiffusion::clip_image::ImageEncoder>(
            brodiffusion::clip_image::ImageEncoderConfig{});
        auto clip_file = st::File::open(clip_path);
        clip_text->load_weights(clip_file, "text_model.");
        clip_img->load_weights(clip_file);
        clip_scr = std::make_unique<brodiffusion::clip_score::CLIPScorer>(
            *clip_tok, *clip_text, *clip_img);
        clip_scr->load_projections(clip_file);
        clip_scr->set_prompt(prompt);
        sampler.set_scorer([&](const std::vector<float>& img, int H, int W) {
            return clip_scr->score(img, H, W);
        });
        std::printf("Scorer: CLIP score (active prompt cached)\n");
    } else {
        std::printf("Scorer: mean luminance (smoke test — NOT a real reward)\n");
    }

    const int decisions = (opts.num_inference_steps + mcfg.decision_interval - 1)
                          / mcfg.decision_interval;
    std::printf("SD MCTS-guided sample: %dx%d, %d steps, CFG=%.1f, "
                "B=%d iters=%d, decision-interval=%d (=~%d decisions), seed=%llu\n",
                opts.width, opts.height, opts.num_inference_steps,
                static_cast<double>(opts.guidance_scale),
                mcfg.branching_factor, mcfg.iterations, mcfg.decision_interval,
                decisions, static_cast<unsigned long long>(opts.seed));
    std::printf("Prompt: %s\n", prompt);

    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    if (enumerate_only) {
        auto rollouts = sampler.enumerate_actions(prompt, opts, /*capture_latents=*/trace_dir != nullptr);
        auto t1 = clk::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("Enumeration done in %.1f ms (%.1f ms/action)\n",
                    ms, ms / std::max(1, static_cast<int>(rollouts.size())));
        const int H = opts.height, W = opts.width;
        const int plane = H * W;
        const std::string base = out_path;
        // Strip trailing .ppm if present so we can insert "_aN" before it.
        std::string stem = base;
        if (stem.size() > 4 && stem.compare(stem.size() - 4, 4, ".ppm") == 0) {
            stem.resize(stem.size() - 4);
        }
        for (const auto& r : rollouts) {
            std::printf("  action %d: score=%.6f\n", r.action, r.score);
            std::vector<std::uint8_t> rgb(static_cast<std::size_t>(3) * plane);
            for (int i = 0; i < plane; ++i) {
                for (int c = 0; c < 3; ++c) {
                    float v = (r.image[c * plane + i] + 1.0f) * 127.5f;
                    v = std::clamp(v, 0.0f, 255.0f);
                    rgb[3 * i + c] = static_cast<std::uint8_t>(v + 0.5f);
                }
            }
            const std::string fn = stem + "_a" + std::to_string(r.action) + ".ppm";
            std::ofstream f(fn, std::ios::binary | std::ios::trunc);
            if (!f) {
                std::fprintf(stderr, "sd-mcts: cannot open output %s\n", fn.c_str());
                return 1;
            }
            f << "P6\n" << W << " " << H << "\n255\n";
            f.write(reinterpret_cast<const char*>(rgb.data()),
                    static_cast<std::streamsize>(rgb.size()));
            std::printf("  wrote %s\n", fn.c_str());

            if (trace_dir && !r.decision_latents.empty()) {
                const int D = static_cast<int>(r.decision_latents.size());
                const int C_lat = 4;
                const int H_lat = opts.height / 8;
                const int W_lat = opts.width  / 8;
                const std::size_t per_dec = static_cast<std::size_t>(C_lat) *
                                            H_lat * W_lat;
                if (r.decision_latents.front().size() != per_dec) {
                    std::fprintf(stderr,
                        "sd-mcts: latent snapshot size %zu != expected %zu\n",
                        r.decision_latents.front().size(), per_dec);
                    return 1;
                }
                const std::string tfn = std::string(trace_dir) +
                                        "/trace_a" + std::to_string(r.action) + ".bin";
                std::ofstream tf(tfn, std::ios::binary | std::ios::trunc);
                if (!tf) {
                    std::fprintf(stderr, "sd-mcts: cannot open trace output %s "
                                 "(does dir exist?)\n", tfn.c_str());
                    return 1;
                }
                auto w_i32 = [&](std::int32_t v) {
                    tf.write(reinterpret_cast<const char*>(&v), sizeof(v));
                };
                auto w_u64 = [&](std::uint64_t v) {
                    tf.write(reinterpret_cast<const char*>(&v), sizeof(v));
                };
                auto w_f32 = [&](float v) {
                    tf.write(reinterpret_cast<const char*>(&v), sizeof(v));
                };
                w_i32(0x42445354);   // 'BDST' little-endian
                w_i32(1);
                w_i32(r.action);
                w_i32(D);
                w_i32(C_lat);
                w_i32(H_lat);
                w_i32(W_lat);
                w_f32(r.score);
                w_u64(opts.seed);
                for (int d = 0; d < D; ++d) {
                    w_i32(r.decision_step_indices[static_cast<std::size_t>(d)]);
                    tf.write(reinterpret_cast<const char*>(
                                 r.decision_latents[static_cast<std::size_t>(d)].data()),
                             static_cast<std::streamsize>(per_dec * sizeof(float)));
                }
                std::printf("  wrote %s (%d decisions, %.1f KB)\n",
                            tfn.c_str(), D,
                            static_cast<double>(per_dec * sizeof(float) * D) / 1024.0);
            }
        }
        return 0;
    }
    auto img = sampler.generate(prompt, opts);
    auto t1 = clk::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Done in %.1f ms (%.2f ms/decision avg)\n",
                ms, ms / std::max(1, static_cast<int>(sampler.last_decisions().size())));

    for (std::size_t i = 0; i < sampler.last_decisions().size(); ++i) {
        const auto& d = sampler.last_decisions()[i];
        std::printf("  decision %zu @step=%d: action=%d visits=%d tree=%d  visit_dist=[",
                    i, d.step_index_before, d.best_action, d.best_visits, d.tree_size);
        for (std::size_t j = 0; j < d.root_visits.size(); ++j) {
            std::printf("%s%.2f", j == 0 ? "" : ",", d.root_visits[j]);
        }
        std::printf("]\n");
    }

    const int H = opts.height, W = opts.width;
    const int plane = H * W;
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(3) * plane);
    for (int i = 0; i < plane; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = (img[c * plane + i] + 1.0f) * 127.5f;
            v = std::clamp(v, 0.0f, 255.0f);
            rgb[3 * i + c] = static_cast<std::uint8_t>(v + 0.5f);
        }
    }
    std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::fprintf(stderr, "sd-mcts: cannot open output %s\n", out_path);
        return 1;
    }
    f << "P6\n" << W << " " << H << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()),
            static_cast<std::streamsize>(rgb.size()));
    std::printf("Wrote %s\n", out_path);
    return 0;
}
#endif  // BRODIFFUSION_HAS_BROGAMEAGENT

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    if (std::strcmp(argv[1], "--version") == 0 || std::strcmp(argv[1], "-v") == 0) {
        std::printf("brodiffusion %s\n", brodiffusion::version_string());
        return 0;
    }
#if BRODIFFUSION_HAS_BROGAMEAGENT
    if (std::strcmp(argv[1], "cifar-mcts") == 0) {
        try {
            return run_cifar_mcts(argc, argv);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "cifar-mcts: %s\n", e.what());
            return 1;
        }
    }
    if (std::strcmp(argv[1], "sd-mcts") == 0) {
        try {
            return run_sd_mcts(argc, argv);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "sd-mcts: %s\n", e.what());
            return 1;
        }
    }
#endif
    if (std::strcmp(argv[1], "cifar-sample") == 0) {
        try {
            return run_cifar_sample(argc, argv);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "cifar-sample: %s\n", e.what());
            return 1;
        }
    }
    if (std::strcmp(argv[1], "txt2img") == 0) {
        try {
            return run_txt2img(argc, argv);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "txt2img: %s\n", e.what());
            return 1;
        }
    }
    if (std::strcmp(argv[1], "bench") == 0) {
        try {
            const char* text_path   = arg_after(argc, argv, "--text");
            const char* unet_path   = arg_after(argc, argv, "--unet");
            const char* vae_path    = arg_after(argc, argv, "--vae");
            const char* vocab_path  = arg_after(argc, argv, "--vocab");
            const char* merges_path = arg_after(argc, argv, "--merges");
            const char* steps_s     = arg_after(argc, argv, "--steps");
            const char* iters_s     = arg_after(argc, argv, "--iters");
            const char* warmup_s    = arg_after(argc, argv, "--warmup");
            const char* sched_s     = arg_after(argc, argv, "--scheduler");
            if (!text_path || !unet_path || !vae_path || !vocab_path || !merges_path) {
                std::fprintf(stderr, "bench: --text, --unet, --vae, --vocab, --merges required\n");
                return 2;
            }
            bool use_lcm = false;
            if (sched_s) {
                if (std::strcmp(sched_s, "lcm") == 0) use_lcm = true;
                else if (std::strcmp(sched_s, "ddim") != 0) {
                    std::fprintf(stderr, "bench: --scheduler must be 'ddim' or 'lcm'\n");
                    return 2;
                }
            }
            const int steps  = steps_s  ? std::atoi(steps_s)  : (use_lcm ? 4 : 5);
            const int iters  = iters_s  ? std::atoi(iters_s)  : 5;
            const int warmup = warmup_s ? std::atoi(warmup_s) : 1;

            brotensor::cuda_init();
            auto tok = clip::Tokenizer::load(vocab_path, merges_path);
            pl::PipelineConfig cfg;
            if (use_lcm) {
                cfg.scheduler = brodiffusion::scheduler::LCMConfig{};
                cfg.unet.time_cond_proj_dim = 256;
            }
            pl::Pipeline pipeline(cfg, std::move(tok));
            pipeline.load_weights(st::File::open(text_path),
                                  st::File::open(unet_path),
                                  st::File::open(vae_path));
            for (const auto& spec : collect_loras(argc, argv)) {
                std::printf("Applying LoRA: %s (scale=%.3f)\n", spec.path.c_str(),
                            static_cast<double>(spec.scale));
                auto lora_file = st::File::open(spec.path);
                pipeline.apply_lora(lora_file, spec.scale);
            }
            pl::GenerateOptions opts;
            opts.num_inference_steps = steps;
            opts.guidance_scale = 1.0f;   // skip uncond pass — bench unet+vae core
            opts.width = 512; opts.height = 512; opts.seed = 1;

            for (int i = 0; i < warmup; ++i) (void)pipeline.generate("an astronaut", opts);
            using clk = std::chrono::high_resolution_clock;
            double sum_ms = 0.0, mn = 1e30, mx = 0.0;
            for (int i = 0; i < iters; ++i) {
                auto t0 = clk::now();
                (void)pipeline.generate("an astronaut", opts);
                auto t1 = clk::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                sum_ms += ms; mn = std::min(mn, ms); mx = std::max(mx, ms);
                std::printf("  iter %d: %.2f ms (%.2f steps/s)\n", i, ms, 1000.0 * steps / ms);
            }
            const double avg = sum_ms / iters;
            std::printf("bench: steps=%d iters=%d  avg=%.2f ms  min=%.2f  max=%.2f\n",
                        steps, iters, avg, mn, mx);
            std::printf("       per-step=%.2f ms  throughput=%.2f (steps=%d)/s  (target 24/s @ %d steps)\n",
                        avg / steps, 1000.0 / avg, steps, steps);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "bench: %s\n", e.what());
            return 1;
        }
    }
    return usage();
}
