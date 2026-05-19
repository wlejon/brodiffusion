#pragma once

// Unconditional pixel-space DDPM pipeline for small images (CIFAR-10 et al.).
//
// Wraps `ddpm_unet::UNet` + `scheduler::DDPM` into a `generate(opts)` call
// that produces a raw FP32 RGB image in [-1, 1] (the DDPM's training range
// with `clip_sample=true`). Callers clamp + rescale to uint8 outside.
//
// Inference-only, FP16 throughout, batch size N = 1. No text encoder, no
// VAE — the "latent" *is* the image. The step-wise API mirrors the SD1.5
// pipeline so MCTS / tree-search tooling can `prime() → fork → step_once →
// score → decode` without caring which backbone it's wrapping.

#include "brodiffusion/ddpm_scheduler.h"
#include "brodiffusion/ddpm_unet.h"

#include "brotensor/tensor.h"

#include <cstdint>
#include <random>
#include <vector>

namespace brodiffusion::safetensors { class File; }

namespace brodiffusion::cifar_pipeline {

struct PipelineConfig {
    ddpm_unet::UNetConfig  unet;
    scheduler::DDPMConfig  scheduler;
};

// Snapshot of mid-generation state. Cheap to fork — only the FP16 sample
// carries device memory; everything else is host scalars / RNG state. The
// UNet has no caches dependent on prior steps (no text context), so a fork
// is a single device clone.
struct PipelineState {
    brotensor::GpuTensor sample;     // (1, C * H * W) FP16
    std::mt19937_64 rng;             // initial noise + per-step ancestral noise
    int step_index = 0;
    int n_steps    = 0;
    int H          = 0;
    int W          = 0;

    PipelineState clone() const;
};

struct GenerateOptions {
    // Image dimensions in pixels. Must match what the loaded checkpoint was
    // trained for — CIFAR-10 is 32x32. The UNet config's block count fixes
    // the minimum (H, W must be divisible by 2^(num_blocks - 1)).
    int height = 32;
    int width  = 32;

    // Number of denoising steps. Defaults to the full DDPM chain (1000); a
    // lower value uniformly subsamples the schedule the same way DDIM does.
    int num_inference_steps = 1000;

    // RNG seed for both the initial sample and the per-step ancestral noise.
    std::uint64_t seed = 0;
};

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& cfg = {});

    // Load all UNet weights from a diffusers `UNet2DModel` safetensors export
    // (the format google/ddpm-cifar10-32 ships in).
    void load_weights(const safetensors::File& unet_file);

    // One-shot generation. Internally: prime() → loop(step_once) → decode().
    std::vector<float> generate(const GenerateOptions& opts);

    // ── Step-wise API (for MCTS / tree-search tooling) ────────────────────
    PipelineState prime(const GenerateOptions& opts);
    void          step_once(PipelineState& state);
    // VAE-equivalent: just clamp the sample to [-1, 1] and copy to host as a
    // (3 * H * W) FP32 NCHW buffer. (For DDPM `clip_sample=true`, the sample
    // is already in range after the final step; the clamp is defensive.)
    std::vector<float> decode(const PipelineState& state);

    const ddpm_unet::UNet& unet() const { return unet_; }
    const PipelineConfig&  config() const { return cfg_; }

private:
    PipelineConfig    cfg_;
    ddpm_unet::UNet   unet_;
    scheduler::DDPM   scheduler_;

    // Working buffers reused across step_once() calls. The current sample
    // lives on PipelineState, not here.
    brotensor::GpuTensor eps_pred_;
    brotensor::GpuTensor scratch_;
    brotensor::GpuTensor noise_step_;
    brotensor::GpuTensor decoded_;
};

}  // namespace brodiffusion::cifar_pipeline
