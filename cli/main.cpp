// brodiffusion CLI — usage text and the subcommand dispatcher.
//
// Every subcommand body lives in a sibling translation unit (cmd_txt2img.cpp,
// cmd_model_fwd.cpp, cmd_ardy.cpp, cmd_terrain.cpp) declared in commands.h;
// the shared argv / PNG / raw-f32 helpers live in cli_common.cpp. main() only
// routes argv[1] and turns an escaping exception into "<subcommand>: <what>".

#include "commands.h"

#include "brodiffusion/detail/jit_fusion.h"
#include "brodiffusion/pipeline.h"
#include "brodiffusion/version.h"

#include "brolm/tokenizer.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include "broimage/encode.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace cli  = brodiffusion::cli;
namespace pl   = brodiffusion::pipeline;
namespace st   = brotensor::safetensors;
namespace clip = brolm::clip;

int usage() {
    std::printf(
        "brodiffusion %s\n"
        "\n"
        "Usage:\n"
        "  brodiffusion --version\n"
        "  brodiffusion txt2img --model <dir> --prompt <text> --out <png>\n"
        "                       [--negative <text>] [--steps N] [--cfg F]\n"
        "                       [--width N] [--height N] [--seed N]\n"
        "  brodiffusion txt2img --text <st> --unet <st> --vae <st>\n"
        "                       --vocab <vocab.json> --merges <merges.txt>\n"
        "                       --prompt <text> --out <png>\n"
        "                       [--negative <text>] [--steps N] [--cfg F]\n"
        "                       [--width N] [--height N] [--seed N]\n"
        "                       [--scheduler ddim|lcm]\n"
        "                       [--noise internal|torch]\n"
        "                       [--latent-in <f32>] [--latent-out <f32>]\n"
        "                       [--lora <path>[:<scale>]]... [--lcm-lora <path>]\n"
        "                       [--quantize-unet]\n"
        "  brodiffusion img2img --init <png> [--strength F] [--vae-sample]\n"
        "                       (all txt2img flags also accepted; SD1.5 only)\n"
        "  brodiffusion inpaint --init <png> --mask <png>\n"
        "                       [--strength F] [--vae-sample]\n"
        "                       (all txt2img flags also accepted; SD1.5 only;\n"
        "                        white mask pixels = inpaint, black = keep)\n"
        "\n"
        "  --control <weights> --control-image <png>\n"
        "                       [--control-scale F] [--control-window S:E]\n"
        "                       register a ControlNet (SD1.5 only). Repeat\n"
        "                       the group for multi-ControlNet stacking — the\n"
        "                       residuals are summed position-wise (each\n"
        "                       weighted by its --control-scale, default 1.0)\n"
        "                       and fed into the UNet skips. --control-window\n"
        "                       S:E (each in [0,1]) restricts the net to a\n"
        "                       half-open fraction of the schedule (default\n"
        "                       0:1 = full). --control-image / --control-scale\n"
        "                       / --control-window attach to the most recent\n"
        "                       --control entry. LCM scheduler is supported;\n"
        "                       trace mode is supported (cond pass only).\n"
        "  brodiffusion make-mask --out <png> [--width N] [--height N]\n"
        "                       writes a center-square binary inpaint mask\n"
        "                       (white quarter-area box on black, default 512x512).\n"
        "  brodiffusion bench   --text <st> --unet <st> --vae <st>\n"
        "                       --vocab <vocab.json> --merges <merges.txt>\n"
        "                       [--steps N] [--iters N] [--warmup N]\n"
        "                       [--scheduler ddim|lcm] [--lora <path>[:<scale>]]...\n"
        "\n"
        "  --model <dir>    load a diffusers model directory (model_index.json +\n"
        "                   component subdirs). Detects SD1.5 vs Flux automatically\n"
        "                   and loads all weights + tokenizers; the explicit\n"
        "                   --text/--unet/--vae/--vocab/--merges flags are then\n"
        "                   unused. For a Flux model --steps defaults to 4.\n"
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
        "  --noise torch    generate the initial latent with a torch.randn-\n"
        "                   compatible RNG, so --seed N reproduces a PyTorch\n"
        "                   reference run seeded the same way (default: internal).\n"
        "  --latent-in <f32>   load the initial latent noise from a raw\n"
        "                   little-endian float32 file (NCHW flat, 4*H/8*W/8\n"
        "                   values) instead of any RNG.\n"
        "  --latent-out <f32>  dump the final denoised latent (pre-VAE) to a\n"
        "                   raw float32 file, for cross-implementation diffing.\n"
        "\n"
        "  brodiffusion t5  --weights <st> --tokenizer <json> --prompt <text>\n"
        "                   [--max-length N] [--quantize]\n"
        "                   load the T5-XXL text encoder, encode <text>, run a\n"
        "                   forward pass, and print output stats. --quantize\n"
        "                   loads it as INT8 (W8A16); --max-length defaults 128.\n"
        "\n"
        "  --no-jit         disable every trace-JIT fusion site and run the\n"
        "                   eager op sequences instead (same as setting\n"
        "                   BRODIFFUSION_JIT=0). Accepted by every subcommand.\n"
        "\n"
        "Writes an RGB PNG via broimage.\n",
        brodiffusion::version_string());
    return 0;
}

// One (name -> entry point) row of the dispatch table. Every subcommand is
// wrapped identically: run it, and on an escaping exception print
// "<name>: <what>" and exit 1.
struct Command {
    const char* name;
    int (*fn)(int, char**);
};

const Command kCommands[] = {
    {"txt2img",            cli::run_txt2img},
    {"t5",                 cli::run_t5},
    {"pixart-fwd",         cli::run_pixart_fwd},
    {"krea2-vae-fwd",      cli::run_krea2_vae_fwd},
    {"qi21-vae-fwd",       cli::run_qi21_vae_fwd},
    {"krea2-text-fwd",     cli::run_krea2_text_fwd},
    {"krea2-fwd",          cli::run_krea2_fwd},
    {"qi21-fwd",           cli::run_qi21_fwd},
    {"qi21-text-fwd",      cli::run_qi21_text_fwd},
    {"terrain-unet-fwd",   cli::run_terrain_unet_fwd},
    {"terrain-laplacian",  cli::run_terrain_laplacian},
    {"terrain-coarse",     cli::run_terrain_coarse},
    {"terrain-sample",     cli::run_terrain_sample},
    {"terrain-rng",        cli::run_terrain_rng},
    {"terrain-synth",      cli::run_terrain_synth},
    {"terrain-itensor",    cli::run_terrain_itensor},
    {"ardy-motionrep-fwd", cli::run_ardy_motionrep_fwd},
    {"ardy-fsq-detok",     cli::run_ardy_fsq_detok},
    {"ardy-backbone-fwd",  cli::run_ardy_backbone_fwd},
    {"ardy-denoiser-fwd",  cli::run_ardy_denoiser_fwd},
    {"ardy-sample",        cli::run_ardy_sample},
    {"ardy-generate",      cli::run_ardy_generate},
    {"ardy-detok-motion",  cli::run_ardy_detok_motion},
    {"ardy-text-feat",     cli::run_ardy_text_feat},
};

// make-mask: write a center-square binary inpaint mask (white quarter-area box
// on black). Small enough to live here rather than in a command file.
int run_make_mask(int argc, char** argv) {
    const char* out_path = cli::arg_after(argc, argv, "--out");
    const char* w_s = cli::arg_after(argc, argv, "--width");
    const char* h_s = cli::arg_after(argc, argv, "--height");
    if (!out_path) {
        std::fprintf(stderr, "make-mask: --out <png> required\n");
        return 2;
    }
    const int W = w_s ? std::atoi(w_s) : 512;
    const int H = h_s ? std::atoi(h_s) : 512;
    if (W <= 0 || H <= 0) {
        std::fprintf(stderr, "make-mask: width/height must be positive\n");
        return 2;
    }
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(3) * H * W, 0);
    const int cx = W / 4, cy = H / 4, cw = W / 2, ch = H / 2;
    for (int y = cy; y < cy + ch; ++y) {
        for (int x = cx; x < cx + cw; ++x) {
            auto i = static_cast<std::size_t>(y * W + x) * 3;
            rgb[i + 0] = rgb[i + 1] = rgb[i + 2] = 255;
        }
    }
    if (!broimage::encode_png_file(out_path, rgb.data(), W, H, 3)) {
        std::fprintf(stderr, "make-mask: cannot write PNG %s\n", out_path);
        return 1;
    }
    std::printf("Wrote %s (%dx%d center-square mask)\n", out_path, W, H);
    return 0;
}

// bench: SD1.5 end-to-end throughput harness over the explicit component files.
int run_bench(int argc, char** argv) {
    const char* text_path   = cli::arg_after(argc, argv, "--text");
    const char* unet_path   = cli::arg_after(argc, argv, "--unet");
    const char* vae_path    = cli::arg_after(argc, argv, "--vae");
    const char* vocab_path  = cli::arg_after(argc, argv, "--vocab");
    const char* merges_path = cli::arg_after(argc, argv, "--merges");
    const char* steps_s     = cli::arg_after(argc, argv, "--steps");
    const char* iters_s     = cli::arg_after(argc, argv, "--iters");
    const char* warmup_s    = cli::arg_after(argc, argv, "--warmup");
    const char* sched_s     = cli::arg_after(argc, argv, "--scheduler");
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

    brotensor::init();
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
    for (const auto& spec : cli::collect_loras(argc, argv)) {
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
}

int dispatch(const char* name, int (*fn)(int, char**), int argc, char** argv) {
    try {
        return fn(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", name, e.what());
        return 1;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();

    // --no-jit is global rather than per-subcommand: it turns off every trace-
    // JIT fusion site at once, so a before/after comparison is one flag on an
    // otherwise identical command line. Equivalent to BRODIFFUSION_JIT=0.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-jit") == 0) {
            brodiffusion::detail::set_jit_enabled(false);
        }
    }

    if (std::strcmp(argv[1], "--version") == 0 ||
        std::strcmp(argv[1], "-v") == 0) {
        std::printf("brodiffusion %s\n", brodiffusion::version_string());
        return 0;
    }

    for (const Command& c : kCommands) {
        if (std::strcmp(argv[1], c.name) == 0) {
            return dispatch(c.name, c.fn, argc, argv);
        }
    }

    // img2img / inpaint are sugar for txt2img plus their own required flags.
    // Both share run_txt2img — the active branch is selected by whether --init
    // (and --mask) were passed.
    if (std::strcmp(argv[1], "img2img") == 0) {
        if (!cli::arg_after(argc, argv, "--init")) {
            std::fprintf(stderr, "img2img: --init <png> is required\n");
            return 2;
        }
        return dispatch("img2img", cli::run_txt2img, argc, argv);
    }
    if (std::strcmp(argv[1], "inpaint") == 0) {
        if (!cli::arg_after(argc, argv, "--init") ||
            !cli::arg_after(argc, argv, "--mask")) {
            std::fprintf(stderr,
                "inpaint: --init <png> and --mask <png> are required\n");
            return 2;
        }
        return dispatch("inpaint", cli::run_txt2img, argc, argv);
    }

    if (std::strcmp(argv[1], "make-mask") == 0) {
        return dispatch("make-mask", run_make_mask, argc, argv);
    }
    if (std::strcmp(argv[1], "bench") == 0) {
        return dispatch("bench", run_bench, argc, argv);
    }

    return usage();
}
