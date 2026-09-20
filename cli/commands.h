#pragma once

// brodiffusion CLI — subcommand entry points and the argv / file helpers they
// share.
//
// `main.cpp` is only the dispatcher plus `usage()`; every subcommand body
// lives in a sibling translation unit grouped by model family:
//
//   cmd_txt2img.cpp     txt2img / img2img / inpaint (both the --model dir and
//                       the explicit --text/--unet/--vae forms)
//   cmd_model_fwd.cpp   single-component forward drivers behind the parity
//                       scripts: t5, pixart-fwd, krea2-*, qi21-*
//   cmd_ardy.cpp        ardy-* (motion tokenizer / denoiser / sampler)
//   cmd_terrain.cpp     terrain-* (map UNet, RNG, infinite tensor, sampler)
//
// Every function here returns a process exit code (0 ok, 1 error, 2 usage)
// and may throw std::runtime_error — main() wraps each call in a try/catch
// that prints the subcommand name and the message.

#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brodiffusion::cli {

// ── argv helpers ────────────────────────────────────────────────────────────

// Value following `flag` in argv, or nullptr when the flag is absent (or is
// the last argument, with no value after it).
const char* arg_after(int argc, char** argv, const char* flag);

struct LoraSpec {
    std::string path;
    float       scale = 1.0f;
};

// Collect every "--lora <path>[:<scale>]" from argv. Repeatable.
std::vector<LoraSpec> collect_loras(int argc, char** argv);

struct ControlSpec {
    std::string weights_path;
    std::string image_path;
    float       scale      = 1.0f;
    float       start_step = 0.0f;
    float       end_step   = 1.0f;
};

// Collect repeated --control / --control-image / --control-scale /
// --control-window arguments in positional order. Each --control adds a new
// entry; the others fill in fields of the *most recent* entry. On a usage
// error the message is printed, `usage_error` is set and the result is empty.
std::vector<ControlSpec> collect_controls(int argc, char** argv,
                                          bool& usage_error);

// ── image / latent IO ───────────────────────────────────────────────────────

// Convert a planar [-1,1] FP32 image (3*H*W NCHW) to PNG via broimage.
int write_png(const char* out_path, const std::vector<float>& img,
              int W, int H);

// Read a raw little-endian float32 file into a vector, checking the element
// count. Used by --latent-in.
std::vector<float> load_latent_f32(const char* path, int expected_count);

// Write a tensor to a raw little-endian float32 file, upconverting a 16-bit
// compute dtype on the way out. Used by --latent-out.
void dump_latent_f32(const char* path, const brotensor::Tensor& t);

// ── txt2img family (cmd_txt2img.cpp) ────────────────────────────────────────

int run_txt2img(int argc, char** argv);
int run_txt2img_model_dir(int argc, char** argv, const char* model_dir);

// ── single-component forward drivers (cmd_model_fwd.cpp) ────────────────────

int run_t5(int argc, char** argv);
int run_pixart_fwd(int argc, char** argv);
int run_krea2_vae_fwd(int argc, char** argv);
int run_qi21_vae_fwd(int argc, char** argv);
int run_krea2_text_fwd(int argc, char** argv);
int run_krea2_fwd(int argc, char** argv);
int run_qi21_fwd(int argc, char** argv);
int run_qi21_text_fwd(int argc, char** argv);

// ── ardy (cmd_ardy.cpp) ─────────────────────────────────────────────────────

int run_ardy_fsq_detok(int argc, char** argv);
int run_ardy_denoiser_fwd(int argc, char** argv);
int run_ardy_sample(int argc, char** argv);
int run_ardy_generate(int argc, char** argv);
int run_ardy_detok_motion(int argc, char** argv);
int run_ardy_text_feat(int argc, char** argv);
int run_ardy_backbone_fwd(int argc, char** argv);
int run_ardy_motionrep_fwd(int argc, char** argv);

// ── terrain (cmd_terrain.cpp) ───────────────────────────────────────────────

int run_terrain_unet_fwd(int argc, char** argv);
int run_terrain_rng(int argc, char** argv);
int run_terrain_itensor(int argc, char** argv);
int run_terrain_synth(int argc, char** argv);
int run_terrain_laplacian(int argc, char** argv);
int run_terrain_coarse(int argc, char** argv);
int run_terrain_sample(int argc, char** argv);

}  // namespace brodiffusion::cli
