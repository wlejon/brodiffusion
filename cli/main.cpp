#include "brodiffusion/pipeline.h"
#include "brodiffusion/vae_qwenimage.h"
#include "brodiffusion/krea2_text.h"
#include "brodiffusion/dit/krea2.h"
#include "brodiffusion/detail/json.h"
#include "brolm/qwen3vl_config.h"
#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"
#include "brotensor/safetensors.h"
#include "brolm/tokenizer.h"
#include "brolm/t5.h"
#include "brolm/tokenizer_t5.h"
#include "brodiffusion/version.h"
#include "brodiffusion/ardy/motion_rep.h"
#include "brodiffusion/ardy/fsq_decoder.h"
#include "brodiffusion/ardy/denoiser_backbone.h"
#include "brodiffusion/ardy/denoiser.h"
#include "brodiffusion/ardy/sampler.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "broimage/encode.h"
#include "broimage/preproc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pl   = brodiffusion::pipeline;
namespace st   = brotensor::safetensors;
namespace clip = brolm::clip;

static int usage() {
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
        "Writes an RGB PNG via broimage.\n",
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

struct ControlSpec {
    std::string weights_path;
    std::string image_path;
    float       scale      = 1.0f;
    float       start_step = 0.0f;
    float       end_step   = 1.0f;
};

// Collect repeated --control / --control-image / --control-scale /
// --control-window arguments in positional order. Each --control adds a
// new entry; --control-image / --control-scale / --control-window fill
// in subsequent fields of the *most recent* entry. The CLI form mirrors
// the API: one ControlNet, one input. Throws by returning empty on a
// usage error after printing a message.
std::vector<ControlSpec> collect_controls(int argc, char** argv,
                                          bool& usage_error) {
    std::vector<ControlSpec> out;
    usage_error = false;
    for (int i = 1; i < argc - 1; ++i) {
        const char* a = argv[i];
        const char* v = argv[i + 1];
        if (std::strcmp(a, "--control") == 0) {
            ControlSpec s;
            s.weights_path = v;
            out.push_back(std::move(s));
        } else if (std::strcmp(a, "--control-image") == 0) {
            if (out.empty()) {
                std::fprintf(stderr,
                    "controlnet: --control-image must follow --control\n");
                usage_error = true;
                return {};
            }
            out.back().image_path = v;
        } else if (std::strcmp(a, "--control-scale") == 0) {
            if (out.empty()) {
                std::fprintf(stderr,
                    "controlnet: --control-scale must follow --control\n");
                usage_error = true;
                return {};
            }
            out.back().scale = static_cast<float>(std::atof(v));
        } else if (std::strcmp(a, "--control-window") == 0) {
            if (out.empty()) {
                std::fprintf(stderr,
                    "controlnet: --control-window must follow --control\n");
                usage_error = true;
                return {};
            }
            // Format: "<start>:<end>" (both fractions in [0,1]).
            std::string raw = v;
            auto pos = raw.find(':');
            if (pos == std::string::npos) {
                std::fprintf(stderr,
                    "controlnet: --control-window expects <start>:<end>\n");
                usage_error = true;
                return {};
            }
            out.back().start_step =
                static_cast<float>(std::atof(raw.substr(0, pos).c_str()));
            out.back().end_step =
                static_cast<float>(std::atof(raw.substr(pos + 1).c_str()));
        }
    }
    for (const auto& s : out) {
        if (s.image_path.empty()) {
            std::fprintf(stderr,
                "controlnet: every --control needs a matching "
                "--control-image\n");
            usage_error = true;
            return {};
        }
    }
    return out;
}

// Convert a planar [-1,1] FP32 image (3*H*W NCHW) to PNG via broimage.
// f32_nchw_to_u8_nhwc with scale=127.5, bias=127.5 maps [-1,1] -> [0,255]
// with the same round+clamp the old hand-rolled PPM writer used.
int write_png(const char* out_path, const std::vector<float>& img,
              int W, int H) {
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(3) * H * W);
    broimage::f32_nchw_to_u8_nhwc(img.data(), 1, 3, H, W,
                                  127.5f, 127.5f, rgb.data());
    if (!broimage::encode_png_file(out_path, rgb.data(), W, H, 3)) {
        std::fprintf(stderr, "txt2img: cannot write PNG %s\n", out_path);
        return 1;
    }
    std::printf("Wrote %s\n", out_path);
    return 0;
}

// Read a raw little-endian float32 file into a vector, checking the element
// count. Used by --latent-in to feed an externally-generated initial latent.
std::vector<float> load_latent_f32(const char* path, int expected_count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error(std::string("cannot open --latent-in: ") + path);
    }
    const std::streamsize bytes = f.tellg();
    const std::streamsize want  =
        static_cast<std::streamsize>(expected_count) * 4;
    if (bytes != want) {
        throw std::runtime_error(
            "--latent-in size mismatch: " + std::to_string(bytes) +
            " bytes, expected " + std::to_string(want) + " (" +
            std::to_string(expected_count) + " float32)");
    }
    std::vector<float> v(static_cast<std::size_t>(expected_count));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), want);
    return v;
}

// Write a tensor to a raw little-endian float32 file. Handles the FP16
// compute dtype (GPU backend) by upconverting on the way out. Used by
// --latent-out to dump the final denoised latent for cross-impl comparison.
void dump_latent_f32(const char* path, const brotensor::Tensor& t) {
    const std::size_t n = static_cast<std::size_t>(t.rows) * t.cols;
    std::vector<float> vals;
    if (t.dtype == brotensor::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        t.copy_to_host_fp16(bits.data());
        brotensor::sync_all();
        vals.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            vals[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        }
    } else {
        vals = t.to_host_vector();
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        throw std::runtime_error(std::string("cannot open --latent-out: ") + path);
    }
    f.write(reinterpret_cast<const char*>(vals.data()),
            static_cast<std::streamsize>(vals.size() * sizeof(float)));
}

// txt2img against a diffusers model directory (--model). Auto-detects SD1.5
// vs Flux from the loaded config.
int run_txt2img_model_dir(int argc, char** argv, const char* model_dir) {
    const char* prompt   = arg_after(argc, argv, "--prompt");
    const char* out_path = arg_after(argc, argv, "--out");
    const char* neg      = arg_after(argc, argv, "--negative");
    const char* steps_s  = arg_after(argc, argv, "--steps");
    const char* cfg_s    = arg_after(argc, argv, "--cfg");
    const char* width_s  = arg_after(argc, argv, "--width");
    const char* height_s = arg_after(argc, argv, "--height");
    const char* seed_s   = arg_after(argc, argv, "--seed");
    const char* latent_out = arg_after(argc, argv, "--latent-out");
    const char* latent_in  = arg_after(argc, argv, "--latent-in");
    const char* noise_s    = arg_after(argc, argv, "--noise");
    const char* init_path = arg_after(argc, argv, "--init");
    const char* mask_path = arg_after(argc, argv, "--mask");
    const char* strength_s = arg_after(argc, argv, "--strength");
    bool vae_sample = false;
    bool quantize   = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--vae-sample") == 0) vae_sample = true;
        if (std::strcmp(argv[i], "--quantize-unet") == 0) quantize = true;
    }

    if (!prompt || !out_path) {
        std::fprintf(stderr,
            "txt2img: --model mode requires --prompt and --out\n");
        return 2;
    }

    brotensor::init();

    std::printf("Loading model directory: %s%s\n", model_dir,
                quantize ? " (INT8 W8A16 denoiser + text encoder)" : "");
    pl::Pipeline::ModelDirOptions dir_opts;
    dir_opts.quantize = quantize;
    // Device-memory report (BRODIFFUSION_TIME): used/total for the active
    // GPU. "used" is device-wide, so it includes other processes on the card.
    auto print_vram = [](const char* when) {
        if (!std::getenv("BRODIFFUSION_TIME")) return;
        std::size_t fb = 0, tb = 0;
        if (brotensor::device_mem_info(brotensor::default_device(), fb, tb)) {
            std::fprintf(stderr, "[mem] %s: %.2f / %.2f GiB used\n", when,
                         static_cast<double>(tb - fb) / (1024.0 * 1024 * 1024),
                         static_cast<double>(tb) / (1024.0 * 1024 * 1024));
        }
    };

    const auto t_load0 = std::chrono::steady_clock::now();
    pl::Pipeline pipeline = pl::Pipeline::from_model_dir(model_dir, dir_opts);
    if (std::getenv("BRODIFFUSION_TIME")) {
        std::fprintf(stderr, "[time] model load: %.2f s\n",
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t_load0).count());
    }
    print_vram("after model load");
    const bool is_flux =
        pipeline.config().model_class == brodiffusion::ModelClass::Flux;
    const bool is_sana =
        pipeline.config().model_class == brodiffusion::ModelClass::Sana;
    const bool is_pixart =
        pipeline.config().model_class == brodiffusion::ModelClass::PixArt;
    const bool is_krea2 =
        pipeline.config().model_class == brodiffusion::ModelClass::Krea2;
    // Krea 2 ships two checkpoints under one _class_name: Raw (real CFG) and
    // the distilled Turbo (no CFG, few steps).
    const bool is_krea2_turbo = is_krea2 && pipeline.config().krea2.is_distilled;
    std::printf("Model class: %s\n",
                is_flux ? "Flux" : (is_sana ? "Sana"
                       : (is_pixart ? "PixArt-Sigma"
                       : (is_krea2 ? (is_krea2_turbo ? "Krea 2 Turbo" : "Krea 2 Raw")
                       : "StableDiffusion"))));

    // Sana-Sprint is the guidance-distilled (guidance_embeds), SCM/TrigFlow
    // few-step variant — it defaults to 2 steps, vs base Sana's 20.
    const bool is_sana_sprint = is_sana && pipeline.config().sana.guidance_embeds;

    pl::GenerateOptions opts;
    // Sana reference defaults: 1024px, guidance 4.5 (native 1024 model; DC-AE
    // downsamples 32x so 1024 -> a 32x32 latent). Base Sana takes 20 steps;
    // Sana-Sprint just 2.
    if (is_sana) { opts.width = 1024; opts.height = 1024;
                   opts.num_inference_steps = is_sana_sprint ? 2 : 20;
                   opts.guidance_scale = 4.5f; }
    // PixArt-Sigma-XL-2-1024-MS native resolution + reference sampling defaults
    // (DPM-Solver++ 20 steps, CFG 4.5).
    if (is_pixart) { opts.width = 1024; opts.height = 1024;
                     opts.num_inference_steps = 20;
                     opts.guidance_scale = 4.5f; }
    // Krea 2 native 1024px. The model card quotes guidance g under the
    // convention standard_scale = 1 + g: Raw g=4.5 -> --guidance-scale 5.5
    // (28 steps); Turbo g=0.0 -> --guidance-scale 1.0 (no CFG, 8 steps).
    if (is_krea2) { opts.width = 1024; opts.height = 1024;
                    opts.num_inference_steps = is_krea2_turbo ? 8 : 28;
                    opts.guidance_scale = is_krea2_turbo ? 1.0f : 5.5f; }
    if (neg)      opts.negative_prompt = neg;
    if (steps_s)  opts.num_inference_steps = std::atoi(steps_s);
    else if (is_flux) opts.num_inference_steps = 4;  // flux-schnell default
    if (cfg_s)    opts.guidance_scale = static_cast<float>(std::atof(cfg_s));
    if (width_s)  opts.width  = std::atoi(width_s);
    if (height_s) opts.height = std::atoi(height_s);
    if (seed_s)   opts.seed =
        static_cast<std::uint64_t>(std::strtoull(seed_s, nullptr, 10));

    // --noise selects the initial-latent RNG ('torch' reproduces a PyTorch
    // reference run's starting latent; default internal).
    if (noise_s) {
        if (std::strcmp(noise_s, "torch") == 0) {
            opts.noise_source = pl::NoiseSource::Torch;
        } else if (std::strcmp(noise_s, "internal") == 0) {
            opts.noise_source = pl::NoiseSource::Internal;
        } else {
            std::fprintf(stderr, "txt2img: --noise must be 'internal' or 'torch'\n");
            return 2;
        }
    }
    // --latent-in overrides the RNG entirely with raw N(0,1) noise from a file
    // (NCHW flat float32) — the strongest form of cross-impl parity. The latent
    // element count is model-class-specific: Sana DC-AE is 32 channels at 32x
    // downsample, Flux 16ch / 8x, SD1.5 4ch / 8x.
    if (latent_in) {
        const int ds = is_sana ? 32 : 8;
        const int ch = is_sana ? 32 : (is_flux ? 16 : 4);
        const int n_lat = ch * (opts.height / ds) * (opts.width / ds);
        opts.init_noise = load_latent_f32(latent_in, n_lat);
        std::printf("Initial latent noise loaded from %s (%d float32)\n",
                    latent_in, n_lat);
    }

    if (init_path) {
        if (is_flux) {
            std::fprintf(stderr,
                "img2img: --init is not supported for Flux model dirs "
                "(SD1.5 only for now)\n");
            return 2;
        }
        opts.init_image_path   = init_path;
        opts.vae_encode_sample = vae_sample;
        if (strength_s) {
            opts.strength = static_cast<float>(std::atof(strength_s));
        }
    }
    if (mask_path) {
        if (!init_path) {
            std::fprintf(stderr,
                "inpaint: --mask requires --init (SD1.5 only)\n");
            return 2;
        }
        if (is_flux) {
            std::fprintf(stderr,
                "inpaint: --mask is not supported for Flux model dirs "
                "(SD1.5 only)\n");
            return 2;
        }
        opts.mask_image_path = mask_path;
    }

    // ControlNet: each --control adds a registered net; --control-image /
    // --control-scale / --control-window fill the most recent entry.
    // Repeatable for multi-ControlNet stacking. SD1.5 only.
    {
        bool cn_usage_err = false;
        auto controls = collect_controls(argc, argv, cn_usage_err);
        if (cn_usage_err) return 2;
        if (!controls.empty() && is_flux) {
            std::fprintf(stderr,
                "controlnet: not supported for Flux model dirs "
                "(SD1.5 only)\n");
            return 2;
        }
        for (const auto& cs : controls) {
            std::printf("Loading ControlNet: %s\n", cs.weights_path.c_str());
            auto cn_file = st::File::open(cs.weights_path);
            pipeline.add_controlnet(cn_file);
            opts.controls.push_back(pl::ControlNetInput{
                cs.image_path, cs.scale, cs.start_step, cs.end_step});
        }
    }

    std::printf("Generating %dx%d, %d steps, CFG=%.1f, seed=%llu\n",
                opts.width, opts.height, opts.num_inference_steps,
                static_cast<double>(opts.guidance_scale),
                static_cast<unsigned long long>(opts.seed));

    const auto t_gen0 = std::chrono::steady_clock::now();
    std::vector<float> img;
    if (latent_out) {
        // Step-wise API (bit-equivalent to generate()) so the final denoised
        // latent can be dumped before the VAE decode for cross-impl comparison.
        auto state = pipeline.prime(prompt, opts);
        while (state.step_index < state.n_steps) pipeline.step_once(state, opts);
        dump_latent_f32(latent_out, state.latent);
        std::printf("Final latent written to %s\n", latent_out);
        img = pipeline.decode(state);
    } else {
        img = pipeline.generate(prompt, opts);
    }
    if (std::getenv("BRODIFFUSION_TIME")) {
        std::fprintf(stderr, "[time] generate (encode+sample+decode): %.2f s\n",
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t_gen0).count());
    }
    print_vram("after generate");
    return write_png(out_path, img, opts.width, opts.height);
}

int run_txt2img(int argc, char** argv) {
    // --model <dir>: load a whole diffusers model directory and skip the
    // explicit per-component file flags.
    if (const char* model_dir = arg_after(argc, argv, "--model")) {
        return run_txt2img_model_dir(argc, argv, model_dir);
    }

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
    const char* noise_s     = arg_after(argc, argv, "--noise");
    const char* latent_in   = arg_after(argc, argv, "--latent-in");
    const char* latent_out  = arg_after(argc, argv, "--latent-out");
    // img2img flags: --init <png> [--strength F] [--vae-sample]. When --init
    // is present, the same code path constructs the pipeline and just sets
    // the img2img opts; the `img2img` subcommand routes here too. Keeping
    // them merged avoids duplicating the full --text/--unet/--vae setup; a
    // future PR can split if either branch grows.
    const char* init_path   = arg_after(argc, argv, "--init");
    const char* mask_path   = arg_after(argc, argv, "--mask");
    const char* strength_s  = arg_after(argc, argv, "--strength");
    bool quantize_unet = false;
    bool vae_sample    = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quantize-unet") == 0) quantize_unet = true;
        if (std::strcmp(argv[i], "--vae-sample") == 0)    vae_sample    = true;
    }

    if (!text_path || !unet_path || !vae_path ||
        !vocab_path || !merges_path || !prompt || !out_path) {
        std::fprintf(stderr,
            "txt2img: --text, --unet, --vae, --vocab, --merges, --prompt, --out are required\n"
            "         (or use --model <dir> --prompt <text> --out <png>)\n");
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

    // --noise selects the initial-latent RNG. 'torch' makes --seed reproduce a
    // PyTorch reference run's starting latent (torch.randn under a CPU
    // Generator), so the two pipelines can be compared with the RNG removed.
    if (noise_s) {
        if (std::strcmp(noise_s, "torch") == 0) {
            opts.noise_source = pl::NoiseSource::Torch;
        } else if (std::strcmp(noise_s, "internal") == 0) {
            opts.noise_source = pl::NoiseSource::Internal;
        } else {
            std::fprintf(stderr,
                "txt2img: --noise must be 'internal' or 'torch'\n");
            return 2;
        }
    }

    // img2img wiring — empty init_image_path => txt2img (existing behavior).
    if (init_path) {
        opts.init_image_path   = init_path;
        opts.vae_encode_sample = vae_sample;
        if (strength_s) {
            opts.strength = static_cast<float>(std::atof(strength_s));
        }
        if (latent_in) {
            std::fprintf(stderr,
                "img2img: --latent-in is incompatible with --init "
                "(use one or the other)\n");
            return 2;
        }
    }
    // Inpaint: --mask requires --init (validated below); empty path =
    // img2img / txt2img per init_image_path.
    if (mask_path) {
        if (!init_path) {
            std::fprintf(stderr,
                "inpaint: --mask requires --init (SD1.5 only)\n");
            return 2;
        }
        opts.mask_image_path = mask_path;
    }

    // --latent-in overrides the RNG entirely with raw N(0,1) noise from a
    // file (NCHW flat float32) — the strongest form of cross-impl parity.
    if (latent_in) {
        const int n_lat = 4 * (opts.height / 8) * (opts.width / 8);
        opts.init_noise = load_latent_f32(latent_in, n_lat);
        std::printf("Initial latent noise loaded from %s (%d float32)\n",
                    latent_in, n_lat);
    }

    brotensor::init();

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

    // ControlNet: each --control adds a registered net; --control-image /
    // --control-scale / --control-window fill the most recent entry.
    // Repeatable for multi-ControlNet stacking. Loaded AFTER the base
    // weights, BEFORE generate.
    {
        bool cn_usage_err = false;
        auto controls = collect_controls(argc, argv, cn_usage_err);
        if (cn_usage_err) return 2;
        for (const auto& cs : controls) {
            std::printf("Loading ControlNet: %s\n", cs.weights_path.c_str());
            auto cn_file = st::File::open(cs.weights_path);
            pipeline.add_controlnet(cn_file);
            opts.controls.push_back(pl::ControlNetInput{
                cs.image_path, cs.scale, cs.start_step, cs.end_step});
        }
    }

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

    // (3*H*W) NCHW, FP32 in [-1, 1]. When --latent-out is set, run the
    // step-wise API (bit-equivalent to generate()) so the final denoised
    // latent can be dumped before the VAE decode.
    std::vector<float> img;
    if (latent_out) {
        auto state = pipeline.prime(prompt, opts);
        for (int s = 0; s < state.n_steps; ++s) pipeline.step_once(state, opts);
        dump_latent_f32(latent_out, state.latent);
        std::printf("Final latent written to %s\n", latent_out);
        img = pipeline.decode(state);
    } else {
        img = pipeline.generate(prompt, opts);
    }
    return write_png(out_path, img, opts.width, opts.height);
}

// Load the T5-XXL encoder, encode a prompt, run one forward, print stats.
// Doubles as the load/works check for the t5-xxl weight download and the
// manual FP16-vs-INT8 comparison driver (run once with and once without
// --quantize and diff the printed mean / L2 norm).
int run_t5(int argc, char** argv) {
    const char* weights_path = arg_after(argc, argv, "--weights");
    const char* tok_path     = arg_after(argc, argv, "--tokenizer");
    const char* prompt       = arg_after(argc, argv, "--prompt");
    const char* maxlen_s     = arg_after(argc, argv, "--max-length");
    const char* dump_path    = arg_after(argc, argv, "--out");
    bool quantize = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quantize") == 0) quantize = true;
    }
    if (!weights_path || !tok_path || !prompt) {
        std::fprintf(stderr,
            "t5: --weights, --tokenizer, --prompt are required\n");
        return 2;
    }
    const int max_length = maxlen_s ? std::atoi(maxlen_s) : 128;

    brotensor::init();

    auto tok = brolm::t5::Tokenizer::load(tok_path);
    std::vector<std::int32_t> ids = tok.encode(prompt, max_length);

    brolm::t5::T5Config cfg;
    cfg.quantize_weights = quantize;
    brolm::t5::TextEncoder enc(cfg);

    std::printf("Loading T5-XXL weights: %s%s\n", weights_path,
                quantize ? "  (INT8 W8A16)" : "");
    auto file = st::File::open(weights_path);

    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    enc.load_weights(file, "");
    brotensor::sync_all();
    auto t1 = clk::now();
    std::printf("Loaded (%s) in %.2f s\n",
                quantize ? "quantized to INT8" : "FP16",
                std::chrono::duration<double>(t1 - t0).count());

    brotensor::Tensor out;
    auto t2 = clk::now();
    enc.forward(ids.data(), static_cast<int>(ids.size()), out);
    brotensor::sync_all();
    auto t3 = clk::now();

    const int L = out.rows;
    const int D = out.cols;
    std::vector<float> vals;
    if (out.dtype == brotensor::Dtype::FP16) {
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(L) * D);
        out.copy_to_host_fp16(bits.data());
        brotensor::sync_all();
        vals.resize(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            vals[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        }
    } else if (out.dtype == brotensor::Dtype::BF16) {
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(L) * D);
        out.copy_to_host_bf16(bits.data());
        brotensor::sync_all();
        vals.resize(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            vals[i] = brotensor::bf16_bits_to_fp32(bits[i]);
        }
    } else {
        vals = out.to_host_vector();
    }

    int nonfinite = 0;
    double sum = 0.0, sumsq = 0.0;
    for (float v : vals) {
        if (!std::isfinite(v)) { ++nonfinite; continue; }
        sum += v;
        sumsq += static_cast<double>(v) * static_cast<double>(v);
    }
    const double mean = vals.empty() ? 0.0 : sum / static_cast<double>(vals.size());
    const double l2   = std::sqrt(sumsq);
    std::printf("Output: (%d, %d)  tokens=%d  forward=%.1f ms\n",
                L, D, static_cast<int>(ids.size()),
                std::chrono::duration<double, std::milli>(t3 - t2).count());
    std::printf("  non-finite : %d\n", nonfinite);
    std::printf("  mean       : %.6f\n", mean);
    std::printf("  L2 norm    : %.4f\n", l2);
    if (vals.size() >= 5) {
        std::printf("  first 5    : %.5f %.5f %.5f %.5f %.5f\n",
                    vals[0], vals[1], vals[2], vals[3], vals[4]);
    }
    if (dump_path) {
        std::ofstream of(dump_path, std::ios::binary | std::ios::trunc);
        of.write(reinterpret_cast<const char*>(vals.data()),
                 static_cast<std::streamsize>(vals.size() * sizeof(float)));
        std::printf("  dumped (%d,%d) to %s\n", L, D, dump_path);
    }
    return nonfinite == 0 ? 0 : 1;
}

// Hidden debug subcommand: run ONE PixArtDenoiser forward on fixed inputs read
// from raw float32 files, dump the epsilon. Used to diff against a diffusers
// reference (scripts/pixart_ref.py). Not in usage().
int run_pixart_fwd(int argc, char** argv) {
    const char* w  = arg_after(argc, argv, "--weights");
    const char* lp = arg_after(argc, argv, "--latent");
    const char* cp = arg_after(argc, argv, "--ctx");
    const char* op = arg_after(argc, argv, "--out");
    const char* ts = arg_after(argc, argv, "--t");
    const char* Hs = arg_after(argc, argv, "--H");
    const char* Ws = arg_after(argc, argv, "--W");
    const char* Ls = arg_after(argc, argv, "--L");
    if (!w || !lp || !cp || !op || !ts || !Hs || !Ws || !Ls) {
        std::fprintf(stderr, "pixart-fwd: need --weights --latent --ctx --out "
                             "--t --H --W --L\n");
        return 2;
    }
    const int H = std::atoi(Hs), W = std::atoi(Ws), L = std::atoi(Ls);
    const float t = static_cast<float>(std::atof(ts));
    brotensor::init();
    brodiffusion::dit::PixArtConfig cfg;
    brodiffusion::dit::PixArtDenoiser den(cfg);
    auto f = st::File::open(w);
    den.load_weights(f, "");
    den.finalize_weights();

    auto lat_h = load_latent_f32(lp, cfg.in_channels * H * W);
    auto ctx_h = load_latent_f32(cp, L * cfg.caption_channels);
    brotensor::Tensor lat =
        brotensor::Tensor::from_host(lat_h.data(), 1, cfg.in_channels * H * W)
            .to(brotensor::default_device());
    brotensor::Tensor ctx =
        brotensor::Tensor::from_host(ctx_h.data(), L, cfg.caption_channels)
            .to(brotensor::default_device());
    // Cast latent to the denoiser compute dtype (forward also handles this).
    brodiffusion::Conditioning cond;
    cond.text_embeddings = ctx;
    cond.has_uncond = false;
    auto prep = den.prepare(cond);
    brotensor::Tensor out;
    den.forward(lat, H, W, t, prep, brodiffusion::Branch::Cond, out);
    brotensor::sync_all();
    dump_latent_f32(op, out);
    std::printf("pixart-fwd: wrote epsilon (%d,%d) to %s\n", out.rows, out.cols, op);
    return 0;
}

// Hidden debug subcommand: run ONE Qwen-Image VAE (Krea 2) decode on a fixed
// latent read from a raw float32 file, dump the RGB output. Used to diff
// against a diffusers reference (scripts/krea2_vae_ref.py). Not in usage().
int run_krea2_vae_fwd(int argc, char** argv) {
    const char* w  = arg_after(argc, argv, "--weights");
    const char* lp = arg_after(argc, argv, "--latent");
    const char* op = arg_after(argc, argv, "--out");
    const char* Hs = arg_after(argc, argv, "--H");
    const char* Ws = arg_after(argc, argv, "--W");
    if (!w || !lp || !op || !Hs || !Ws) {
        std::fprintf(stderr, "krea2-vae-fwd: need --weights --latent --out --H --W\n");
        return 2;
    }
    const int H = std::atoi(Hs), W = std::atoi(Ws);
    brotensor::init();

    namespace vq = brodiffusion::vae_qwenimage;
    vq::Config cfg;
    vq::Decoder dec(cfg);
    auto f = st::File::open(w);
    dec.load_weights(f, "");

    auto lat_h = load_latent_f32(lp, cfg.z_dim * H * W);
    brotensor::Tensor lat =
        brotensor::Tensor::from_host(lat_h.data(), 1, cfg.z_dim * H * W)
            .to(brotensor::default_device());

    brotensor::Tensor out;
    dec.decode(lat, H, W, out);
    brotensor::sync_all();

    // dump_latent_f32 only special-cases FP16; force_upcast may put `out` at
    // BF16 on a CUDA backend, so normalize to FP32 before dumping.
    if (out.dtype != brotensor::Dtype::FP32) {
        brotensor::Tensor out_f32;
        brotensor::cast(out, out_f32, brotensor::Dtype::FP32);
        brotensor::sync_all();
        out = std::move(out_f32);
    }
    dump_latent_f32(op, out);
    std::printf("krea2-vae-fwd: wrote image (%d,%d) to %s\n", out.rows, out.cols, op);
    return 0;
}

// Hidden debug subcommand: run the Krea 2 text-conditioning pathway (Qwen3-VL
// tapped hidden states → fused-fusion input tensors) for one prompt and dump
// prompt_embeds (and optionally the validity mask) as raw float32. Diffed
// against scripts/krea2_text_ref.py. Not in usage().
int run_krea2_text_fwd(int argc, char** argv) {
    const char* wdir = arg_after(argc, argv, "--weights-dir");
    const char* tdir = arg_after(argc, argv, "--tokenizer-dir");
    const char* prompt = arg_after(argc, argv, "--prompt");
    const char* op = arg_after(argc, argv, "--out");
    const char* mp = arg_after(argc, argv, "--mask-out");
    if (!wdir || !tdir || !prompt || !op) {
        std::fprintf(stderr,
            "krea2-text-fwd: need --weights-dir --tokenizer-dir --prompt --out "
            "[--mask-out]\n");
        return 2;
    }
    brotensor::init();

    const std::string wd = wdir, td = tdir;
    auto tok = brolm::qwen3vl::Tokenizer::load(td + "/vocab.json",
                                               td + "/merges.txt");
    auto cfg = brolm::qwen3vl::Qwen3VLConfig::load(wd + "/text_encoder/config.json");
    brolm::qwen3vl::TextModel model(cfg.text);
    auto f = st::File::open(wd + "/text_encoder/model.safetensors");
    model.load_weights(f, "language_model.");

    auto cond = brodiffusion::krea2::encode_prompt(tok, model, prompt);
    brotensor::sync_all();

    dump_latent_f32(op, cond.prompt_embeds);
    std::printf("krea2-text-fwd: wrote prompt_embeds (%d,%d) to %s\n",
                cond.prompt_embeds.rows, cond.prompt_embeds.cols, op);
    if (mp) {
        dump_latent_f32(mp, cond.prompt_embeds_mask);
        std::printf("krea2-text-fwd: wrote mask (%d,%d) to %s\n",
                    cond.prompt_embeds_mask.rows, cond.prompt_embeds_mask.cols, mp);
    }
    return 0;
}

// Parse a Krea2Transformer2DModel config.json into a Krea2Config.
brodiffusion::dit::Krea2Config load_krea2_config(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open config: " + path);
    std::string text((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    namespace j = brodiffusion::detail::json;
    j::Value v = j::parse(text);
    brodiffusion::dit::Krea2Config c;
    c.in_channels = v.get_int("in_channels", c.in_channels);
    c.num_layers = v.get_int("num_layers", c.num_layers);
    c.attention_head_dim = v.get_int("attention_head_dim", c.attention_head_dim);
    c.num_attention_heads = v.get_int("num_attention_heads", c.num_attention_heads);
    c.num_key_value_heads = v.get_int("num_key_value_heads", c.num_key_value_heads);
    c.intermediate_size = v.get_int("intermediate_size", c.intermediate_size);
    c.timestep_embed_dim = v.get_int("timestep_embed_dim", c.timestep_embed_dim);
    c.text_hidden_dim = v.get_int("text_hidden_dim", c.text_hidden_dim);
    c.num_text_layers = v.get_int("num_text_layers", c.num_text_layers);
    c.text_num_attention_heads =
        v.get_int("text_num_attention_heads", c.text_num_attention_heads);
    c.text_num_key_value_heads =
        v.get_int("text_num_key_value_heads", c.text_num_key_value_heads);
    c.text_intermediate_size =
        v.get_int("text_intermediate_size", c.text_intermediate_size);
    c.num_layerwise_text_blocks =
        v.get_int("num_layerwise_text_blocks", c.num_layerwise_text_blocks);
    c.num_refiner_text_blocks =
        v.get_int("num_refiner_text_blocks", c.num_refiner_text_blocks);
    c.axes_dims_rope = v.get_int_array("axes_dims_rope", c.axes_dims_rope);
    c.rope_theta = v.get_float("rope_theta", c.rope_theta);
    c.norm_eps = v.get_float("norm_eps", c.norm_eps);
    return c;
}

// Hidden debug subcommand: run ONE Krea2 DiT forward on fixed inputs (packed
// latent, stage-1 conditioning, timestep, packed grid) read from raw float32
// files, dump the velocity. Diffed against scripts/krea2_dit_ref.py. Sharded
// checkpoint (--weights-dir holds config.json + the diffusion_pytorch_model
// shards). Not in usage().
int run_krea2_fwd(int argc, char** argv) {
    const char* wdir = arg_after(argc, argv, "--weights-dir");
    const char* lp = arg_after(argc, argv, "--latent");
    const char* ep = arg_after(argc, argv, "--embeds");
    const char* mp = arg_after(argc, argv, "--mask");
    const char* op = arg_after(argc, argv, "--out");
    const char* ts = arg_after(argc, argv, "--t");
    const char* hps = arg_after(argc, argv, "--hp");
    const char* wps = arg_after(argc, argv, "--wp");
    const char* seqs = arg_after(argc, argv, "--seq");
    bool quantize = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quantize") == 0) quantize = true;
    }
    if (!wdir || !lp || !ep || !mp || !op || !ts || !hps || !wps) {
        std::fprintf(stderr, "krea2-fwd: need --weights-dir --latent --embeds "
                             "--mask --out --t --hp --wp [--seq] [--quantize]\n");
        return 2;
    }
    const int hp = std::atoi(hps), wp = std::atoi(wps);
    const int text_seq = seqs ? std::atoi(seqs) : 512;
    const float t = static_cast<float>(std::atof(ts));
    brotensor::init();

    const std::string wd = wdir;
    auto cfg = load_krea2_config(wd + "/config.json");
    cfg.quantize_weights = quantize;
    brodiffusion::dit::Krea2Transformer2DModel model(cfg);

    // Open every shard listed in the index (or the single-file checkpoint).
    std::vector<st::File> files;
    std::vector<const st::File*> shards;
    const std::string index = wd + "/diffusion_pytorch_model.safetensors.index.json";
    std::ifstream idxf(index, std::ios::binary);
    if (idxf) {
        std::string text((std::istreambuf_iterator<char>(idxf)),
                         std::istreambuf_iterator<char>());
        namespace j = brodiffusion::detail::json;
        j::Value v = j::parse(text);
        const auto& wm = v.at("weight_map");
        std::vector<std::string> names;
        for (const auto& m : wm.as_object()) {
            const std::string s = m.second.as_string();
            if (std::find(names.begin(), names.end(), s) == names.end())
                names.push_back(s);
        }
        files.reserve(names.size());
        for (const std::string& n : names) files.push_back(st::File::open(wd + "/" + n));
    } else {
        files.push_back(st::File::open(wd + "/diffusion_pytorch_model.safetensors"));
    }
    for (const st::File& f : files) shards.push_back(&f);
    model.load_weights(shards, "");

    const int img_len = hp * wp;
    auto lat_h = load_latent_f32(lp, img_len * cfg.in_channels);
    auto emb_h = load_latent_f32(ep, text_seq * cfg.num_text_layers * cfg.text_hidden_dim);
    auto msk_h = load_latent_f32(mp, text_seq);
    brotensor::Tensor lat =
        brotensor::Tensor::from_host(lat_h.data(), img_len, cfg.in_channels)
            .to(brotensor::default_device());
    brotensor::Tensor emb =
        brotensor::Tensor::from_host(emb_h.data(),
                                     text_seq * cfg.num_text_layers,
                                     cfg.text_hidden_dim)
            .to(brotensor::default_device());
    brotensor::Tensor msk =
        brotensor::Tensor::from_host(msk_h.data(), text_seq, 1)
            .to(brotensor::default_device());

    brotensor::Tensor out;
    model.forward(lat, hp, wp, emb, msk, t, out);
    brotensor::sync_all();
    // dump_latent_f32 only special-cases FP16; the compute dtype is BF16 on a
    // CUDA backend, so normalize to FP32 before dumping.
    if (out.dtype != brotensor::Dtype::FP32) {
        brotensor::Tensor out_f32;
        brotensor::cast(out, out_f32, brotensor::Dtype::FP32);
        brotensor::sync_all();
        out = std::move(out_f32);
    }
    dump_latent_f32(op, out);
    std::printf("krea2-fwd: wrote velocity (%d,%d) to %s\n", out.rows, out.cols, op);
    return 0;
}

// ── ARDY motion-rep parity (scripts/ardy_motionrep_ref.py) ──────────────────
// Reads float64 local rotation matrices + root positions, runs the ARDY motion
// codec forward (features) and inverse (posed joints + recovered local rots),
// and dumps all three as raw float64 to diff against the PyTorch reference.
namespace {
std::vector<double> load_f64(const char* path, size_t n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<double> v(n);
    in.read(reinterpret_cast<char*>(v.data()),
            static_cast<std::streamsize>(n * sizeof(double)));
    if (!in) throw std::runtime_error(std::string("short read from ") + path);
    return v;
}
void dump_f64(const char* path, const std::vector<double>& v) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error(std::string("cannot write ") + path);
    out.write(reinterpret_cast<const char*>(v.data()),
              static_cast<std::streamsize>(v.size() * sizeof(double)));
}
}  // namespace

// Read a 1-D float32 numpy .npy array (n elements) into host floats.
namespace {
std::vector<float> load_npy_f32(const char* path, int n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    char magic[8];
    in.read(magic, 8);  // \x93NUMPY + version(2)
    unsigned char hl[2];
    in.read(reinterpret_cast<char*>(hl), 2);
    const int hlen = hl[0] | (hl[1] << 8);
    in.seekg(10 + hlen, std::ios::beg);  // data starts after the header
    std::vector<float> v(n);
    in.read(reinterpret_cast<char*>(v.data()),
            static_cast<std::streamsize>(n * sizeof(float)));
    if (!in) throw std::runtime_error(std::string("short read from ") + path);
    return v;
}
// Read a 1-D float64 numpy .npy array (n elements) into host floats.
std::vector<float> load_npy_f64_as_f32(const char* path, int n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    char magic[8];
    in.read(magic, 8);
    unsigned char hl[2];
    in.read(reinterpret_cast<char*>(hl), 2);
    const int hlen = hl[0] | (hl[1] << 8);
    in.seekg(10 + hlen, std::ios::beg);
    std::vector<double> d(n);
    in.read(reinterpret_cast<char*>(d.data()),
            static_cast<std::streamsize>(n * sizeof(double)));
    if (!in) throw std::runtime_error(std::string("short read from ") + path);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = static_cast<float>(d[i]);
    return v;
}
std::vector<float> load_raw_f32(const char* path, size_t n) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<float> v(n);
    in.read(reinterpret_cast<char*>(v.data()),
            static_cast<std::streamsize>(n * sizeof(float)));
    if (!in) throw std::runtime_error(std::string("short read from ") + path);
    return v;
}
}  // namespace

int run_ardy_fsq_detok(int argc, char** argv) {
    const char* w  = arg_after(argc, argv, "--weights");
    const char* mp = arg_after(argc, argv, "--pq-mean");
    const char* sp = arg_after(argc, argv, "--pq-std");
    const char* tp = arg_after(argc, argv, "--tokens");
    const char* rp = arg_after(argc, argv, "--local-root");
    const char* Ts = arg_after(argc, argv, "--T-tok");
    const char* op = arg_after(argc, argv, "--out");
    if (!w || !mp || !sp || !tp || !rp || !Ts || !op) {
        std::fprintf(stderr, "ardy-fsq-detok: need --weights --pq-mean --pq-std "
                             "--tokens --local-root --T-tok --out\n");
        return 2;
    }
    const int T_tok = std::atoi(Ts);
    brotensor::init();
    brodiffusion::ardy::FsqMotionDecoder dec;
    const auto& cfg = dec.config();

    auto f = st::File::open(w);
    dec.load_weights(f);
    auto mean = load_npy_f32(mp, cfg.token_dim);
    auto std_ = load_npy_f32(sp, cfg.token_dim);
    dec.set_post_quant_stats(mean.data(), std_.data(), cfg.token_dim);

    auto tokens = load_raw_f32(tp, static_cast<size_t>(T_tok) * cfg.token_dim);
    auto lroot  = load_raw_f32(rp, static_cast<size_t>(T_tok) *
                                       cfg.num_frames_per_token * cfg.external_cond_dim);

    brotensor::Tensor out;
    dec.detokenize(tokens.data(), lroot.data(), T_tok, out);
    brotensor::sync_all();
    dump_latent_f32(op, out);
    std::printf("ardy-fsq-detok: T_tok=%d out(%d,%d) -> %d frames x %d\n",
                T_tok, out.rows, out.cols, T_tok * cfg.num_frames_per_token,
                cfg.output_dim);
    return 0;
}

int run_ardy_denoiser_fwd(int argc, char** argv) {
    const char* w  = arg_after(argc, argv, "--weights");
    const char* mp = arg_after(argc, argv, "--mean");     // motion mean.npy (414)
    const char* sp = arg_after(argc, argv, "--std");      // motion std.npy (414)
    const char* xp = arg_after(argc, argv, "--hybrid");   // (T_tok, 148) raw f32
    const char* tp = arg_after(argc, argv, "--text");     // (1, 4096) raw f32
    const char* Ts = arg_after(argc, argv, "--T-tok");
    const char* ts = arg_after(argc, argv, "--timestep");
    const char* hs = arg_after(argc, argv, "--heading");
    const char* op = arg_after(argc, argv, "--out");
    if (!w || !mp || !sp || !xp || !tp || !Ts || !ts || !hs || !op) {
        std::fprintf(stderr, "ardy-denoiser-fwd: need --weights --mean --std "
                             "--hybrid --text --T-tok --timestep --heading --out\n");
        return 2;
    }
    const int T_tok = std::atoi(Ts);
    const int timestep = std::atoi(ts);
    const float heading = static_cast<float>(std::atof(hs));

    brotensor::init();
    brodiffusion::ardy::ArdyDenoiser dn;
    const auto& cfg = dn.config();

    auto f = st::File::open(w);
    dn.load_weights(f);
    const int sdim = dn.stats_dim();  // 418 (float64 stats)
    auto mean = load_npy_f64_as_f32(mp, sdim);
    auto std_ = load_npy_f64_as_f32(sp, sdim);
    dn.set_motion_stats(mean.data(), std_.data(), sdim);

    auto hyb  = load_raw_f32(xp, static_cast<size_t>(T_tok) * dn.hybrid_dim());
    auto text = load_raw_f32(tp, 4096);

    brotensor::Tensor out;
    dn.forward(hyb.data(), text.data(), timestep, heading, T_tok, out);
    brotensor::sync_all();
    dump_latent_f32(op, out);
    std::printf("ardy-denoiser-fwd: T_tok=%d out(%d,%d)\n", T_tok, out.rows, out.cols);
    return 0;
}

// Full spaced-DDIM text-to-motion window: denoise a fresh generation window from
// initial noise with text-only CFG. Matches the reference generation loop.
int run_ardy_sample(int argc, char** argv) {
    const char* w  = arg_after(argc, argv, "--weights");
    const char* mp = arg_after(argc, argv, "--mean");     // motion mean.npy (418, f64)
    const char* sp = arg_after(argc, argv, "--std");      // motion std.npy  (418, f64)
    const char* xp = arg_after(argc, argv, "--x-init");   // (T_tok,148) raw f32 noise
    const char* tp = arg_after(argc, argv, "--text");     // (1,4096) raw f32
    const char* Ts = arg_after(argc, argv, "--T-tok");
    const char* Ss = arg_after(argc, argv, "--steps");
    const char* cs = arg_after(argc, argv, "--cfg");
    const char* hs = arg_after(argc, argv, "--heading");
    const char* op = arg_after(argc, argv, "--out");
    if (!w || !mp || !sp || !xp || !tp || !Ts || !Ss || !cs || !hs || !op) {
        std::fprintf(stderr, "ardy-sample: need --weights --mean --std --x-init "
                             "--text --T-tok --steps --cfg --heading --out\n");
        return 2;
    }
    const int T_tok = std::atoi(Ts);
    const int steps = std::atoi(Ss);
    const float cfg_w = static_cast<float>(std::atof(cs));
    const float heading = static_cast<float>(std::atof(hs));

    brotensor::init();
    brodiffusion::ardy::ArdyDenoiser dn;
    auto f = st::File::open(w);
    dn.load_weights(f);
    const int sdim = dn.stats_dim();  // 418 (float64 stats)
    auto mean = load_npy_f64_as_f32(mp, sdim);
    auto std_ = load_npy_f64_as_f32(sp, sdim);
    dn.set_motion_stats(mean.data(), std_.data(), sdim);

    auto x_init = load_raw_f32(xp, static_cast<size_t>(T_tok) * dn.hybrid_dim());
    auto text   = load_raw_f32(tp, 4096);

    brodiffusion::ardy::ArdyWindowSampler sampler(dn);
    std::vector<float> out;
    sampler.sample(x_init.data(), text.data(), T_tok, heading, steps, cfg_w, out);

    std::ofstream of(op, std::ios::binary | std::ios::trunc);
    if (!of) { std::fprintf(stderr, "ardy-sample: cannot write %s\n", op); return 1; }
    of.write(reinterpret_cast<const char*>(out.data()),
             static_cast<std::streamsize>(out.size() * sizeof(float)));
    std::printf("ardy-sample: T_tok=%d steps=%d cfg=%.3f out(%d,%d)\n",
                T_tok, steps, cfg_w, T_tok, dn.hybrid_dim());
    return 0;
}

int run_ardy_backbone_fwd(int argc, char** argv) {
    const char* w  = arg_after(argc, argv, "--weights");
    const char* st_ = arg_after(argc, argv, "--stage");     // root | body
    const char* xp = arg_after(argc, argv, "--x");          // (T, latent) raw f32
    const char* tp = arg_after(argc, argv, "--text");       // (num_text_tokens, llm) raw f32
    const char* Ts = arg_after(argc, argv, "--T");
    const char* ts = arg_after(argc, argv, "--timestep");
    const char* hs = arg_after(argc, argv, "--heading");
    const char* op = arg_after(argc, argv, "--out");
    if (!w || !st_ || !xp || !tp || !Ts || !ts || !hs || !op) {
        std::fprintf(stderr, "ardy-backbone-fwd: need --weights --stage --x "
                             "--text --T --timestep --heading --out\n");
        return 2;
    }
    const int T = std::atoi(Ts);
    const int timestep = std::atoi(ts);
    const float heading = static_cast<float>(std::atof(hs));
    const bool is_body = std::strcmp(st_, "body") == 0;

    brotensor::init();
    brodiffusion::ardy::ArdyDenoiserBackbone::Config cfg;
    cfg.output_dim = is_body ? 128 : 20;
    brodiffusion::ardy::ArdyDenoiserBackbone bb(cfg);

    auto f = st::File::open(w);
    bb.load_weights(f, std::string("denoiser.backbone.") +
                       (is_body ? "body_model." : "root_model."));

    auto xh = load_raw_f32(xp, static_cast<size_t>(T) * cfg.latent_dim);
    auto th = load_raw_f32(tp, static_cast<size_t>(cfg.num_text_tokens) * cfg.llm_dim);
    brotensor::Tensor x = brodiffusion::detail::upload_host(xh.data(), T, cfg.latent_dim);

    // First-generation window: token_index = arange(T) (origin at frame 0).
    std::vector<int> tok_idx(T);
    for (int i = 0; i < T; ++i) tok_idx[i] = i;

    brotensor::Tensor out;
    bb.forward(x, th.data(), timestep, heading, tok_idx.data(), T,
               /*key_mask=*/nullptr, out);
    brotensor::sync_all();
    dump_latent_f32(op, out);
    std::printf("ardy-backbone-fwd: stage=%s T=%d out(%d,%d)\n",
                is_body ? "body" : "root", T, out.rows, out.cols);
    return 0;
}

int run_ardy_motionrep_fwd(int argc, char** argv) {
    const char* lr = arg_after(argc, argv, "--local-rots");
    const char* rp = arg_after(argc, argv, "--root-pos");
    const char* Ts = arg_after(argc, argv, "--T");
    const char* of = arg_after(argc, argv, "--out-features");
    const char* op = arg_after(argc, argv, "--out-posed");
    const char* ol = arg_after(argc, argv, "--out-local");
    if (!lr || !rp || !Ts || !of || !op || !ol) {
        std::fprintf(stderr, "ardy-motionrep-fwd: need --local-rots --root-pos "
                             "--T --out-features --out-posed --out-local\n");
        return 2;
    }
    const int T = std::atoi(Ts);
    constexpr int Jn = brodiffusion::ardy::ArdyMotionRep::kNumJoints;   // 34
    constexpr int Fd = brodiffusion::ardy::ArdyMotionRep::kFeatureDim;  // 414

    auto local = load_f64(lr, static_cast<size_t>(T) * Jn * 9);
    auto root  = load_f64(rp, static_cast<size_t>(T) * 3);

    brodiffusion::ardy::ArdyMotionRep rep(/*fps=*/25.0);
    std::vector<double> feats;
    rep.forward(local.data(), root.data(), T, feats);

    auto dec = rep.inverse(feats.data(), T, /*posed_from_rot=*/true);

    dump_f64(of, feats);
    dump_f64(op, dec.posed_joints);
    dump_f64(ol, dec.local_rot_mats);
    std::printf("ardy-motionrep-fwd: T=%d features(%d,%d) posed(%d,%d,3) "
                "local(%d,%d,3,3)\n", T, T, Fd, T, Jn, T, Jn);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    if (std::strcmp(argv[1], "pixart-fwd") == 0) {
        try { return run_pixart_fwd(argc, argv); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "pixart-fwd: %s\n", e.what()); return 1;
        }
    }
    if (std::strcmp(argv[1], "krea2-vae-fwd") == 0) {
        try { return run_krea2_vae_fwd(argc, argv); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "krea2-vae-fwd: %s\n", e.what()); return 1;
        }
    }
    if (std::strcmp(argv[1], "krea2-text-fwd") == 0) {
        try { return run_krea2_text_fwd(argc, argv); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "krea2-text-fwd: %s\n", e.what()); return 1;
        }
    }
    if (std::strcmp(argv[1], "krea2-fwd") == 0) {
        try { return run_krea2_fwd(argc, argv); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "krea2-fwd: %s\n", e.what()); return 1;
        }
    }
    if (std::strcmp(argv[1], "ardy-motionrep-fwd") == 0) {
        try { return run_ardy_motionrep_fwd(argc, argv); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "ardy-motionrep-fwd: %s\n", e.what()); return 1;
        }
    }
    if (std::strcmp(argv[1], "ardy-fsq-detok") == 0) {
        try { return run_ardy_fsq_detok(argc, argv); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "ardy-fsq-detok: %s\n", e.what()); return 1;
        }
    }
    if (std::strcmp(argv[1], "ardy-backbone-fwd") == 0) {
        try { return run_ardy_backbone_fwd(argc, argv); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "ardy-backbone-fwd: %s\n", e.what()); return 1;
        }
    }
    if (std::strcmp(argv[1], "ardy-denoiser-fwd") == 0) {
        try { return run_ardy_denoiser_fwd(argc, argv); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "ardy-denoiser-fwd: %s\n", e.what()); return 1;
        }
    }
    if (std::strcmp(argv[1], "ardy-sample") == 0) {
        try { return run_ardy_sample(argc, argv); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "ardy-sample: %s\n", e.what()); return 1;
        }
    }
    if (std::strcmp(argv[1], "--version") == 0 || std::strcmp(argv[1], "-v") == 0) {
        std::printf("brodiffusion %s\n", brodiffusion::version_string());
        return 0;
    }
    if (std::strcmp(argv[1], "txt2img") == 0) {
        try {
            return run_txt2img(argc, argv);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "txt2img: %s\n", e.what());
            return 1;
        }
    }
    // img2img is sugar for txt2img + img2img-specific flags. Both subcommands
    // share run_txt2img — the active branch is selected by whether --init was
    // passed. (img2img without --init falls back to txt2img with a friendly
    // error from the option-parsing layer.)
    if (std::strcmp(argv[1], "img2img") == 0) {
        try {
            if (!arg_after(argc, argv, "--init")) {
                std::fprintf(stderr,
                    "img2img: --init <png> is required\n");
                return 2;
            }
            return run_txt2img(argc, argv);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "img2img: %s\n", e.what());
            return 1;
        }
    }
    // inpaint is sugar for txt2img + --init + --mask. Same run_txt2img code
    // path; the active branch is selected by whether mask_image_path is set.
    if (std::strcmp(argv[1], "inpaint") == 0) {
        try {
            if (!arg_after(argc, argv, "--init") ||
                !arg_after(argc, argv, "--mask")) {
                std::fprintf(stderr,
                    "inpaint: --init <png> and --mask <png> are required\n");
                return 2;
            }
            return run_txt2img(argc, argv);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "inpaint: %s\n", e.what());
            return 1;
        }
    }
    if (std::strcmp(argv[1], "t5") == 0) {
        try {
            return run_t5(argc, argv);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "t5: %s\n", e.what());
            return 1;
        }
    }
    if (std::strcmp(argv[1], "make-mask") == 0) {
        try {
            const char* out_path = arg_after(argc, argv, "--out");
            const char* w_s = arg_after(argc, argv, "--width");
            const char* h_s = arg_after(argc, argv, "--height");
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
            // Center-square binary mask: white quarter-area box on black.
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
        } catch (const std::exception& e) {
            std::fprintf(stderr, "make-mask: %s\n", e.what());
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
