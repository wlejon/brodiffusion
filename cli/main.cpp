#include "brodiffusion/pipeline.h"
#include "brotensor/safetensors.h"
#include "brolm/tokenizer.h"
#include "brolm/t5.h"
#include "brolm/tokenizer_t5.h"
#include "brodiffusion/version.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include "broimage/encode.h"
#include "broimage/preproc.h"

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
    const char* init_path = arg_after(argc, argv, "--init");
    const char* strength_s = arg_after(argc, argv, "--strength");
    bool vae_sample = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--vae-sample") == 0) vae_sample = true;
    }

    if (!prompt || !out_path) {
        std::fprintf(stderr,
            "txt2img: --model mode requires --prompt and --out\n");
        return 2;
    }

    brotensor::init();

    std::printf("Loading model directory: %s\n", model_dir);
    pl::Pipeline pipeline = pl::Pipeline::from_model_dir(model_dir);
    const bool is_flux =
        pipeline.config().model_class == brodiffusion::ModelClass::Flux;
    std::printf("Model class: %s\n", is_flux ? "Flux" : "StableDiffusion");

    pl::GenerateOptions opts;
    if (neg)      opts.negative_prompt = neg;
    if (steps_s)  opts.num_inference_steps = std::atoi(steps_s);
    else if (is_flux) opts.num_inference_steps = 4;  // flux-schnell default
    if (cfg_s)    opts.guidance_scale = static_cast<float>(std::atof(cfg_s));
    if (width_s)  opts.width  = std::atoi(width_s);
    if (height_s) opts.height = std::atoi(height_s);
    if (seed_s)   opts.seed =
        static_cast<std::uint64_t>(std::strtoull(seed_s, nullptr, 10));

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

    std::printf("Generating %dx%d, %d steps, CFG=%.1f, seed=%llu\n",
                opts.width, opts.height, opts.num_inference_steps,
                static_cast<double>(opts.guidance_scale),
                static_cast<unsigned long long>(opts.seed));

    auto img = pipeline.generate(prompt, opts);
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
    return nonfinite == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
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
    if (std::strcmp(argv[1], "t5") == 0) {
        try {
            return run_t5(argc, argv);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "t5: %s\n", e.what());
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
