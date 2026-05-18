#pragma once

// Stable Diffusion 1.5 text-to-image pipeline.
//
// Owns the tokenizer, CLIP text encoder, UNet, VAE decoder, and DDIM
// scheduler — wires them into a single `generate(prompt, ...)` call that
// produces a raw FP32 RGB image in [-1, 1] (the VAE output range). Callers
// clamp + rescale to uint8 / encode to PNG outside this library.
//
// Inference-only, FP16 throughout the GPU path, batch size N = 1. Classifier-
// free guidance runs the U-Net twice per step (conditional + unconditional)
// rather than batching, since the rest of the inference stack is N=1.

#include "brodiffusion/clip.h"
#include "brodiffusion/scheduler.h"
#include "brodiffusion/tokenizer.h"
#include "brodiffusion/unet.h"
#include "brodiffusion/vae.h"

#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brodiffusion::safetensors { class File; }

namespace brodiffusion::pipeline {

struct PipelineConfig {
    unet::UNetConfig         unet;
    vae::DecoderConfig       vae;
    clip::TextEncoderConfig  text_encoder;
    scheduler::DDIMConfig    scheduler;
};

struct GenerateOptions {
    // Image dimensions in pixels. Latent dims are H/8, W/8. Both must be
    // multiples of 8 and at least 8 (so latent dims are >= 1).
    int height = 512;
    int width  = 512;

    int   num_inference_steps = 30;
    float guidance_scale      = 7.5f;   // CFG; 1.0 disables (uncond is skipped).
    std::string negative_prompt;

    // RNG seed for the initial latent noise.
    std::uint64_t seed = 0;
};

class Pipeline {
public:
    Pipeline(const PipelineConfig& cfg, clip::Tokenizer tokenizer);

    // Load every sub-module's weights from a single safetensors file using
    // the prefixes used by an SD1.5 full checkpoint:
    //   "cond_stage_model.transformer.text_model." for CLIP
    //   "model.diffusion_model."                   for the U-Net
    //   "first_stage_model.decoder."               for the VAE decoder
    void load_weights(const safetensors::File& f);

    // Load with explicit prefixes (e.g. when sub-modules ship in separate
    // diffusers exports).
    void load_weights(const safetensors::File& f,
                      const std::string& text_prefix,
                      const std::string& unet_prefix,
                      const std::string& vae_prefix);

    // Load each module from its own diffusers-format safetensors file. Uses
    // the diffusers default prefixes ("text_model.", "", "decoder.").
    void load_weights(const safetensors::File& text_file,
                      const safetensors::File& unet_file,
                      const safetensors::File& vae_file);

    // Generate an image. Returns a freshly-allocated host buffer of
    // 3 * height * width FP32 values in NCHW (C=3, [-1, 1]).
    std::vector<float> generate(std::string_view prompt,
                                const GenerateOptions& opts);

    const PipelineConfig& config() const { return cfg_; }

    // Install a capture hook on the inner UNet (see unet::InletCaptureHook).
    // Used by the `capture-inlet` CLI subcommand. Pass nullptr to disable.
    void set_inlet_capture_hook(unet::InletCaptureHook* h) {
        unet_.set_capture_hook(h);
    }

    // Direct access to inner UNet for advanced wiring (e.g. loading trained
    // inlet weights post-construction).
    unet::UNet&       unet()       { return unet_; }
    const unet::UNet& unet() const { return unet_; }

private:
    void encode_prompt_(std::string_view prompt, brotensor::GpuTensor& out);

    PipelineConfig     cfg_;
    clip::Tokenizer    tokenizer_;
    clip::TextEncoder  text_encoder_;
    unet::UNet         unet_;
    vae::Decoder       vae_;
    scheduler::DDIM    scheduler_;

    // Scratch tensors kept alive across generate() calls.
    brotensor::GpuTensor ctx_cond_, ctx_uncond_;
    brotensor::GpuTensor latent_, noise_pred_cond_, noise_pred_uncond_;
    brotensor::GpuTensor decoded_, scratch_;

    // Cross-attention K/V projected from the text context once per generate(),
    // reused for all denoising steps. CFG (cond + uncond) keeps two caches.
    unet::UNet::CrossAttnKVCache xattn_cache_cond_;
    unet::UNet::CrossAttnKVCache xattn_cache_uncond_;
};

}  // namespace brodiffusion::pipeline
