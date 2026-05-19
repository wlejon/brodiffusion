#include "brodiffusion/cifar_pipeline.h"
#if BRODIFFUSION_HAS_BROGAMEAGENT
#include "brodiffusion/cifar_mcts.h"
#include "brodiffusion/clip_image.h"
#include "brodiffusion/clip_score.h"
#include "brodiffusion/sd_mcts.h"
#include "brodiffusion/value_head.h"
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
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
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

// Write one trace file in the BDST/v1 format documented in the sd-mcts
// --enumerate-trace-out help text. Returns false on I/O failure (and prints
// the error to stderr). `path` is the full destination filename.
bool write_trace_file(const std::string& path,
                      int action, int D, int C_lat, int H_lat, int W_lat,
                      float score, std::uint64_t seed,
                      const std::vector<int>& step_indices,
                      const std::vector<std::vector<float>>& latents) {
    std::ofstream tf(path, std::ios::binary | std::ios::trunc);
    if (!tf) {
        std::fprintf(stderr, "trace write: cannot open %s (does dir exist?)\n",
                     path.c_str());
        return false;
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
    const std::size_t per_dec =
        static_cast<std::size_t>(C_lat) * H_lat * W_lat;
    w_i32(0x42445354);   // 'BDST'
    w_i32(1);
    w_i32(action);
    w_i32(D);
    w_i32(C_lat);
    w_i32(H_lat);
    w_i32(W_lat);
    w_f32(score);
    w_u64(seed);
    for (int d = 0; d < D; ++d) {
        w_i32(step_indices[static_cast<std::size_t>(d)]);
        tf.write(reinterpret_cast<const char*>(
                     latents[static_cast<std::size_t>(d)].data()),
                 static_cast<std::streamsize>(per_dec * sizeof(float)));
    }
    return static_cast<bool>(tf);
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
                const std::string tfn = std::string(trace_dir) +
                                        "/trace_a" + std::to_string(r.action) + ".bin";
                if (!write_trace_file(tfn, r.action, D, C_lat, H_lat, W_lat,
                                      r.score, opts.seed,
                                      r.decision_step_indices,
                                      r.decision_latents)) {
                    return 1;
                }
                const std::size_t per_dec =
                    static_cast<std::size_t>(C_lat) * H_lat * W_lat;
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
int run_sd_mcts_collect(int argc, char** argv) {
    namespace sm = brodiffusion::sd_mcts;

    const char* text_path    = arg_after(argc, argv, "--text");
    const char* unet_path    = arg_after(argc, argv, "--unet");
    const char* vae_path     = arg_after(argc, argv, "--vae");
    const char* vocab_path   = arg_after(argc, argv, "--vocab");
    const char* merges_path  = arg_after(argc, argv, "--merges");
    const char* clip_path    = arg_after(argc, argv, "--clip");
    const char* prompts_file = arg_after(argc, argv, "--prompts-file");
    const char* out_dir      = arg_after(argc, argv, "--out-dir");
    const char* seeds_s      = arg_after(argc, argv, "--seeds");
    const char* start_seed_s = arg_after(argc, argv, "--start-seed");
    const char* steps_s      = arg_after(argc, argv, "--steps");
    const char* width_s      = arg_after(argc, argv, "--width");
    const char* height_s     = arg_after(argc, argv, "--height");
    const char* branch_s     = arg_after(argc, argv, "--branching");
    const char* di_s         = arg_after(argc, argv, "--decision-interval");
    const char* bias_s       = arg_after(argc, argv, "--bias-magnitude");

    if (!text_path || !unet_path || !vae_path || !vocab_path || !merges_path ||
        !clip_path || !prompts_file || !out_dir) {
        std::fprintf(stderr,
            "sd-mcts-collect: --text, --unet, --vae, --vocab, --merges, --clip,\n"
            "                 --prompts-file, --out-dir are required\n"
            "                 [--seeds N=3] [--start-seed N=0]\n"
            "                 [--steps N] [--width N] [--height N]\n"
            "                 [--branching B] [--decision-interval N]\n"
            "                 [--bias-magnitude F]\n"
            "\n"
            "Loads SD1.5 + CLIP once, then for each (prompt, seed) pair calls\n"
            "sd-mcts enumerate with latent capture and writes one trace file per\n"
            "action to <out-dir>/trace_p<promptIdx>_s<seed>_a<action>.bin in the\n"
            "same BDST/v1 format as sd-mcts --enumerate-trace-out.\n");
        return 2;
    }

    // ── Read prompts file (one prompt per non-empty, non-# line). ────────
    std::vector<std::string> prompts;
    {
        std::ifstream pf(prompts_file);
        if (!pf) {
            std::fprintf(stderr, "cannot open prompts file: %s\n", prompts_file);
            return 1;
        }
        std::string line;
        while (std::getline(pf, line)) {
            // Trim CR (Windows line endings) and leading/trailing whitespace.
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ' ||
                                     line.back() == '\t')) line.pop_back();
            std::size_t start = 0;
            while (start < line.size() && (line[start] == ' ' || line[start] == '\t'))
                ++start;
            if (start) line.erase(0, start);
            if (line.empty() || line[0] == '#') continue;
            prompts.push_back(line);
        }
    }
    if (prompts.empty()) {
        std::fprintf(stderr, "no prompts in %s\n", prompts_file);
        return 1;
    }

    pl::GenerateOptions opts_base;
    if (steps_s)  opts_base.num_inference_steps = std::atoi(steps_s);
    if (width_s)  opts_base.width  = std::atoi(width_s);
    if (height_s) opts_base.height = std::atoi(height_s);

    sm::Config mcfg;
    if (branch_s) mcfg.branching_factor  = std::atoi(branch_s);
    if (di_s)     mcfg.decision_interval = std::atoi(di_s);
    if (bias_s)   mcfg.bias_magnitude    = static_cast<float>(std::atof(bias_s));

    const int n_seeds = seeds_s ? std::atoi(seeds_s) : 3;
    const std::uint64_t start_seed = start_seed_s
        ? static_cast<std::uint64_t>(std::strtoull(start_seed_s, nullptr, 10)) : 0;

    if (!std::filesystem::exists(out_dir)) {
        std::filesystem::create_directories(out_dir);
    }

    brotensor::cuda_init();

    auto tok = clip::Tokenizer::load(vocab_path, merges_path);
    pl::PipelineConfig cfg;
    pl::Pipeline pipeline(cfg, std::move(tok));
    std::printf("Loading SD1.5 weights …\n");
    auto text_file = st::File::open(text_path);
    auto unet_file = st::File::open(unet_path);
    auto vae_file  = st::File::open(vae_path);
    pipeline.load_weights(text_file, unet_file, vae_file);

    sm::Sampler sampler(pipeline, mcfg);

    // CLIP scorer shares the same vocab/merges as the pipeline tokenizer.
    std::printf("Loading CLIP weights …\n");
    clip::Tokenizer clip_tok = clip::Tokenizer::load(vocab_path, merges_path);
    clip::TextEncoder clip_text(clip::TextEncoderConfig{});
    brodiffusion::clip_image::ImageEncoder clip_img(
        brodiffusion::clip_image::ImageEncoderConfig{});
    auto clip_file = st::File::open(clip_path);
    clip_text.load_weights(clip_file, "text_model.");
    clip_img.load_weights(clip_file);
    brodiffusion::clip_score::CLIPScorer clip_scr(clip_tok, clip_text, clip_img);
    clip_scr.load_projections(clip_file);
    sampler.set_scorer([&](const std::vector<float>& img, int H, int W) {
        return clip_scr.score(img, H, W);
    });

    const int total_runs = static_cast<int>(prompts.size()) * n_seeds;
    std::printf("sd-mcts-collect: %zu prompts × %d seeds = %d runs, "
                "B=%d, decision-interval=%d, bias-mag=%.2f, steps=%d, %dx%d\n",
                prompts.size(), n_seeds, total_runs,
                mcfg.branching_factor, mcfg.decision_interval,
                static_cast<double>(mcfg.bias_magnitude),
                opts_base.num_inference_steps,
                opts_base.width, opts_base.height);

    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();
    int run_idx = 0;
    for (std::size_t pi = 0; pi < prompts.size(); ++pi) {
        const std::string& prompt = prompts[pi];
        clip_scr.set_prompt(prompt);
        for (int si = 0; si < n_seeds; ++si) {
            const std::uint64_t seed = start_seed + static_cast<std::uint64_t>(si);
            pl::GenerateOptions opts = opts_base;
            opts.seed = seed;
            mcfg.seed = seed;
            // Re-bind the sampler's config: simplest way is a fresh sampler,
            // but we can keep the one we have if sd-mcts seed mutation is
            // limited to enumerate_actions's bias-pattern generation, which
            // reads cfg_.seed. Build a fresh sampler per run to be safe.
            sm::Sampler this_sampler(pipeline, mcfg);
            this_sampler.set_scorer([&](const std::vector<float>& img, int H, int W) {
                return clip_scr.score(img, H, W);
            });

            const auto t_run0 = clk::now();
            auto rollouts = this_sampler.enumerate_actions(prompt, opts, /*capture_latents=*/true);
            const auto t_run1 = clk::now();
            const double ms = std::chrono::duration<double, std::milli>(t_run1 - t_run0).count();

            const int C_lat = 4;
            const int H_lat = opts.height / 8;
            const int W_lat = opts.width  / 8;

            char tag[64];
            std::snprintf(tag, sizeof(tag), "p%04zu_s%llu",
                          pi, static_cast<unsigned long long>(seed));

            double sum_score = 0.0;
            float min_score = rollouts.empty() ? 0.0f : rollouts.front().score;
            float max_score = min_score;
            for (const auto& r : rollouts) {
                const int D = static_cast<int>(r.decision_latents.size());
                const std::string tfn = std::string(out_dir) + "/trace_" + tag +
                                        "_a" + std::to_string(r.action) + ".bin";
                if (!write_trace_file(tfn, r.action, D, C_lat, H_lat, W_lat,
                                      r.score, seed,
                                      r.decision_step_indices,
                                      r.decision_latents)) {
                    return 1;
                }
                sum_score += r.score;
                if (r.score < min_score) min_score = r.score;
                if (r.score > max_score) max_score = r.score;
            }
            const double mean_score = rollouts.empty() ? 0.0
                : sum_score / static_cast<double>(rollouts.size());

            ++run_idx;
            const double elapsed = std::chrono::duration<double>(clk::now() - t0).count();
            const double eta = elapsed * (total_runs - run_idx) / std::max(1, run_idx);
            std::printf("[%d/%d] p%zu s%llu (%.1fs) score mean=%.4f spread=%.4f  ETA %.0fs  prompt=%.60s\n",
                        run_idx, total_runs, pi,
                        static_cast<unsigned long long>(seed),
                        ms / 1000.0, mean_score, max_score - min_score,
                        eta, prompt.c_str());
            std::fflush(stdout);
        }
    }
    const double total_s = std::chrono::duration<double>(clk::now() - t0).count();
    std::printf("Done in %.1fs (%.1fs/run avg). Trace files in %s\n",
                total_s, total_s / std::max(1, total_runs), out_dir);
    return 0;
}

int run_value_head(int argc, char** argv) {
    namespace vh = brodiffusion::value_head;
    namespace fs = std::filesystem;

    const char* traces_dir = arg_after(argc, argv, "--traces");
    const char* out_path   = arg_after(argc, argv, "--out");
    const char* epochs_s   = arg_after(argc, argv, "--epochs");
    const char* batch_s    = arg_after(argc, argv, "--batch");
    const char* lr_s       = arg_after(argc, argv, "--lr");
    const char* hidden_s   = arg_after(argc, argv, "--hidden");
    const char* seed_s     = arg_after(argc, argv, "--seed");
    const char* branch_s   = arg_after(argc, argv, "--branching");
    bool eval_only = false;
    const char* load_path  = arg_after(argc, argv, "--load");
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--eval") == 0) eval_only = true;
    }

    if (!traces_dir || (!out_path && !eval_only)) {
        std::fprintf(stderr,
            "value-head: --traces <dir> required; --out <weights> required unless --eval\n"
            "            [--epochs N=20] [--batch N=64] [--lr F=1e-3]\n"
            "            [--hidden N=128] [--branching B=3] [--seed N=1]\n"
            "            [--load <weights>] [--eval]\n"
            "\n"
            "Reads trace_a<N>.bin produced by `sd-mcts --enumerate-trace-out`,\n"
            "trains a small MLP value head that predicts the recorded terminal\n"
            "score from (latent_at_decision_t, action_one_hot). With --eval,\n"
            "loads --load weights and just reports per-example MSE / Pearson r.\n");
        return 2;
    }

    int epochs   = epochs_s  ? std::atoi(epochs_s)  : 20;
    int batch    = batch_s   ? std::atoi(batch_s)   : 64;
    float lr     = lr_s      ? static_cast<float>(std::atof(lr_s)) : 1e-3f;
    int hidden   = hidden_s  ? std::atoi(hidden_s)  : 128;
    int branchB  = branch_s  ? std::atoi(branch_s)  : 3;
    std::uint64_t seed = seed_s
        ? static_cast<std::uint64_t>(std::strtoull(seed_s, nullptr, 10)) : 1;

    // ── Walk the traces dir, load every trace_a*.bin, build example matrix. ─
    // Examples: one per (trace_file, decision_index). Input vector is
    //   [latent_flat | action_one_hot(B)] of size (latent_dim + B).
    // Target: the terminal score recorded in the trace header.
    int latent_dim = -1;
    int C_lat = -1, H_lat = -1, W_lat = -1;
    std::vector<float> X_flat;   // concatenated rows
    std::vector<float> Y_flat;
    int n_examples = 0;

    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(traces_dir)) {
        const auto& p = e.path();
        const auto stem = p.stem().string();
        if (stem.rfind("trace_", 0) == 0 && p.extension() == ".bin") {
            files.push_back(p);
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        std::fprintf(stderr, "value-head: no trace_a*.bin files in %s\n", traces_dir);
        return 1;
    }

    for (const auto& p : files) {
        std::ifstream f(p, std::ios::binary);
        if (!f) { std::fprintf(stderr, "  skip (open): %s\n", p.string().c_str()); continue; }
        std::int32_t magic=0, ver=0, action=0, D=0, C=0, H=0, W=0;
        float score=0.0f; std::uint64_t fseed=0;
        f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        f.read(reinterpret_cast<char*>(&ver),   sizeof(ver));
        f.read(reinterpret_cast<char*>(&action),sizeof(action));
        f.read(reinterpret_cast<char*>(&D),     sizeof(D));
        f.read(reinterpret_cast<char*>(&C),     sizeof(C));
        f.read(reinterpret_cast<char*>(&H),     sizeof(H));
        f.read(reinterpret_cast<char*>(&W),     sizeof(W));
        f.read(reinterpret_cast<char*>(&score), sizeof(score));
        f.read(reinterpret_cast<char*>(&fseed), sizeof(fseed));
        if (magic != 0x42445354 || ver != 1) {
            std::fprintf(stderr, "  skip (magic): %s\n", p.string().c_str()); continue;
        }
        if (action >= branchB) {
            std::fprintf(stderr,
                "  skip (action %d >= --branching %d): %s\n",
                action, branchB, p.string().c_str());
            continue;
        }
        const int ld = C * H * W;
        if (latent_dim < 0) {
            latent_dim = ld; C_lat = C; H_lat = H; W_lat = W;
        } else if (latent_dim != ld) {
            std::fprintf(stderr,
                "  skip (latent dim %d != %d): %s\n",
                ld, latent_dim, p.string().c_str());
            continue;
        }
        std::vector<float> latent(static_cast<std::size_t>(ld));
        for (int d = 0; d < D; ++d) {
            std::int32_t step_idx = 0;
            f.read(reinterpret_cast<char*>(&step_idx), sizeof(step_idx));
            f.read(reinterpret_cast<char*>(latent.data()),
                   static_cast<std::streamsize>(latent.size() * sizeof(float)));
            if (!f) { std::fprintf(stderr, "  short read: %s\n", p.string().c_str()); break; }

            // Append [latent | one-hot(action, B)] to X_flat, score to Y_flat.
            X_flat.insert(X_flat.end(), latent.begin(), latent.end());
            for (int b = 0; b < branchB; ++b) X_flat.push_back(b == action ? 1.0f : 0.0f);
            Y_flat.push_back(score);
            ++n_examples;
        }
    }

    if (n_examples == 0) {
        std::fprintf(stderr, "value-head: 0 examples extracted\n");
        return 1;
    }

    const int in_dim = latent_dim + branchB;
    std::printf("value-head: loaded %d examples from %zu trace files\n",
                n_examples, files.size());
    std::printf("  latent: %d (C=%d,H=%d,W=%d), in_dim=%d, B=%d\n",
                latent_dim, C_lat, H_lat, W_lat, in_dim, branchB);

    // Target stats — useful sanity check (if all targets are equal there's
    // no signal to learn).
    double y_mean = 0.0, y_min = Y_flat[0], y_max = Y_flat[0];
    for (float v : Y_flat) {
        y_mean += v;
        if (v < y_min) y_min = v;
        if (v > y_max) y_max = v;
    }
    y_mean /= Y_flat.size();
    double y_var = 0.0;
    for (float v : Y_flat) { double d = v - y_mean; y_var += d * d; }
    y_var /= Y_flat.size();
    std::printf("  target: mean=%.6f var=%.6e (min=%.4f max=%.4f spread=%.4f)\n",
                y_mean, y_var, y_min, y_max, y_max - y_min);

    brotensor::cuda_init();

    vh::Config cfg;
    cfg.latent_dim = latent_dim;
    cfg.branching  = branchB;
    cfg.hidden_dim = hidden;
    cfg.lr         = lr;
    cfg.seed       = seed;
    vh::ValueHead head(cfg);
    if (load_path) {
        std::printf("  loading initial weights: %s\n", load_path);
        head.load(load_path);
    }

    // Permutation indices for SGD.
    std::vector<int> perm(static_cast<std::size_t>(n_examples));
    for (int i = 0; i < n_examples; ++i) perm[static_cast<std::size_t>(i)] = i;
    std::mt19937_64 rng(seed ^ 0xBAD5EEDULL);

    brotensor::GpuTensor X_BD, Y_pred_B1, dY_B1;
    std::vector<float> X_batch_host(static_cast<std::size_t>(batch) * in_dim);
    std::vector<float> Y_target_host(static_cast<std::size_t>(batch));
    std::vector<float> Y_pred_host(static_cast<std::size_t>(batch));
    std::vector<float> dY_host(static_cast<std::size_t>(batch));

    if (eval_only) {
        // Single forward pass over the whole set in `batch`-sized chunks.
        double mse = 0.0, sxy = 0.0, sx2 = 0.0, sy2 = 0.0, sx = 0.0, sy = 0.0;
        int n_done = 0;
        for (int start = 0; start < n_examples; start += batch) {
            const int B_actual = std::min(batch, n_examples - start);
            X_batch_host.resize(static_cast<std::size_t>(B_actual) * in_dim);
            Y_target_host.resize(static_cast<std::size_t>(B_actual));
            for (int b = 0; b < B_actual; ++b) {
                const int idx = start + b;
                const float* src = &X_flat[static_cast<std::size_t>(idx) * in_dim];
                std::copy(src, src + in_dim,
                          X_batch_host.begin() + static_cast<std::ptrdiff_t>(b) * in_dim);
                Y_target_host[static_cast<std::size_t>(b)] =
                    Y_flat[static_cast<std::size_t>(idx)];
            }
            brotensor::upload(X_batch_host.data(), B_actual, in_dim, X_BD);
            head.forward(X_BD, Y_pred_B1);
            Y_pred_host.resize(static_cast<std::size_t>(B_actual));
            brotensor::download(Y_pred_B1, Y_pred_host.data());
            for (int b = 0; b < B_actual; ++b) {
                const double yp = Y_pred_host[static_cast<std::size_t>(b)];
                const double yt = Y_target_host[static_cast<std::size_t>(b)];
                const double d  = yp - yt;
                mse += d * d;
                sxy += yp * yt; sx2 += yp * yp; sy2 += yt * yt;
                sx  += yp;      sy  += yt;
                ++n_done;
            }
        }
        mse /= n_done;
        const double mx = sx / n_done, my = sy / n_done;
        const double cov = sxy / n_done - mx * my;
        const double vx  = sx2 / n_done - mx * mx;
        const double vy  = sy2 / n_done - my * my;
        const double r   = cov / std::sqrt(std::max(1e-30, vx * vy));
        std::printf("eval: MSE=%.6e Pearson r=%.4f over %d examples\n", mse, r, n_done);
        return 0;
    }

    // Training loop.
    const int steps_per_epoch = (n_examples + batch - 1) / batch;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(perm.begin(), perm.end(), rng);
        double epoch_loss = 0.0;
        int seen = 0;
        for (int s = 0; s < steps_per_epoch; ++s) {
            const int start = s * batch;
            const int B_actual = std::min(batch, n_examples - start);
            X_batch_host.resize(static_cast<std::size_t>(B_actual) * in_dim);
            Y_target_host.resize(static_cast<std::size_t>(B_actual));
            for (int b = 0; b < B_actual; ++b) {
                const int idx = perm[static_cast<std::size_t>(start + b)];
                const float* src = &X_flat[static_cast<std::size_t>(idx) * in_dim];
                std::copy(src, src + in_dim,
                          X_batch_host.begin() + static_cast<std::ptrdiff_t>(b) * in_dim);
                Y_target_host[static_cast<std::size_t>(b)] =
                    Y_flat[static_cast<std::size_t>(idx)];
            }
            brotensor::upload(X_batch_host.data(), B_actual, in_dim, X_BD);
            head.forward(X_BD, Y_pred_B1);
            Y_pred_host.resize(static_cast<std::size_t>(B_actual));
            brotensor::download(Y_pred_B1, Y_pred_host.data());

            // MSE: loss = 0.5 * (pred - target)^2 / B_actual ; dY = (pred - target) / B_actual.
            dY_host.resize(static_cast<std::size_t>(B_actual));
            double batch_loss = 0.0;
            const float inv_B = 1.0f / static_cast<float>(B_actual);
            for (int b = 0; b < B_actual; ++b) {
                const float d = Y_pred_host[static_cast<std::size_t>(b)] -
                                Y_target_host[static_cast<std::size_t>(b)];
                dY_host[static_cast<std::size_t>(b)] = d * inv_B;
                batch_loss += 0.5 * d * d;
            }
            brotensor::upload(dY_host.data(), B_actual, 1, dY_B1);
            head.backward_and_step(dY_B1);
            epoch_loss += batch_loss;
            seen += B_actual;
        }
        std::printf("  epoch %d: avg loss = %.6e (over %d examples)\n",
                    epoch, epoch_loss / seen, seen);
    }

    head.save(out_path);
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
    if (std::strcmp(argv[1], "sd-mcts-collect") == 0) {
        try {
            return run_sd_mcts_collect(argc, argv);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "sd-mcts-collect: %s\n", e.what());
            return 1;
        }
    }
    if (std::strcmp(argv[1], "value-head") == 0) {
        try {
            return run_value_head(argc, argv);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "value-head: %s\n", e.what());
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
