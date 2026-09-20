// txt2img / img2img / inpaint subcommands.
//
// Two forms share this file: `--model <dir>` loads a whole diffusers model
// directory (run_txt2img_model_dir, which auto-detects the model class), and
// the explicit `--text/--unet/--vae/--vocab/--merges` form builds an SD1.5
// pipeline by hand (run_txt2img). `img2img` and `inpaint` are sugar for the
// latter plus --init / --mask, so they route through the same entry point.

#include "commands.h"

#include "brodiffusion/pipeline.h"
#include "brodiffusion/model_config.h"
#include "brotensor/safetensors.h"
#include "brolm/tokenizer.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace brodiffusion::cli {

namespace pl   = brodiffusion::pipeline;
namespace st   = brotensor::safetensors;
namespace clip = brolm::clip;

// txt2img against a diffusers model directory (--model). Auto-detects the
// model class from the loaded config.
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
    const bool is_qi21 =
        pipeline.config().model_class == brodiffusion::ModelClass::QwenImage21;
    // Krea 2 ships two checkpoints under one _class_name: Raw (real CFG) and
    // the distilled Turbo (no CFG, few steps).
    const bool is_krea2_turbo = is_krea2 && pipeline.config().krea2.is_distilled;
    std::printf("Model class: %s\n",
                is_flux ? "Flux" : (is_sana ? "Sana"
                       : (is_pixart ? "PixArt-Sigma"
                       : (is_krea2 ? (is_krea2_turbo ? "Krea 2 Turbo" : "Krea 2 Raw")
                       : (is_qi21 ? "Qwen-Image 2.1"
                       : "StableDiffusion")))));

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
    // Qwen-Image 2.1 reference defaults: 1024px, 40 steps, true_cfg_scale 1.0
    // (the model is meant to be sampled without guidance — 1.0 skips the
    // uncond branch entirely).
    if (is_qi21) { opts.width = 1024; opts.height = 1024;
                   opts.num_inference_steps = 40;
                   opts.guidance_scale = 1.0f; }
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
    // downsample, Qwen-Image 2.1 is 64 channels at 16x, Flux 16ch / 8x, SD1.5
    // 4ch / 8x.
    if (latent_in) {
        const int ds = pipeline.vae_scale_factor();
        const int ch = is_sana ? 32 : (is_qi21 ? 64 : (is_flux ? 16 : 4));
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

}  // namespace brodiffusion::cli
