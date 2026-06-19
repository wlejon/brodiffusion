#pragma once

// SanaTransformer2DModel — NVIDIA Sana's linear-attention DiT denoiser.
//
// A Denoiser implementation for the Sana rectified-flow transformer (a sibling
// of the Flux DiT behind brodiffusion's model-agnostic Denoiser base).
// Forward-only, batch size N = 1. Runs on whichever backend brotensor resolves
// at runtime — CPU by default (FP32, the test path), CUDA when available
// (FP16, with the BF16 internal stream the DiT compute dtype provides).
//
// Architecture (HF diffusers SanaTransformer2DModel):
//   - patch_embed: patch_size x patch_size conv over the 32-channel latent →
//     inner_dim tokens (patch_size = 1, so a 1x1 conv / per-pixel projection)
//   - caption_projection: Gemma-2 caption sequence (caption_channels = 2304)
//     → inner_dim, plus a learned global caption token
//   - time_embed: timestep → 6*inner_dim AdaLN modulation (shared across
//     blocks — Sana uses a single global AdaLN-single table)
//   - num_layers SanaTransformerBlocks: linear self-attention (ReLU-kernel
//     softmax-free attention, no RoPE), standard softmax cross-attention to the
//     caption tokens (num_cross_attention_heads), and a GLU-style MLP
//     (mlp_ratio = 2.5)
//   - norm_out (AdaLayerNormSingle) → proj_out → out_channels → unpatchify
//
// Unlike Flux, Sana 0.6B is NOT guidance-distilled: it runs classifier-free
// guidance (uses_cfg() == true). Pairs with the Flow-DPM-Solver scheduler;
// prediction_type() is Velocity.

#include "brodiffusion/denoiser.h"

#include "brotensor/tensor.h"

#include <string>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::dit {

struct SanaConfig {
    int in_channels              = 32;     // latent channel count (DC-AE f32c32)
    int out_channels             = 32;
    int num_layers               = 28;
    int attention_head_dim       = 32;
    int num_attention_heads      = 36;     // inner_dim = heads * head_dim = 1152
    int num_cross_attention_heads = 16;
    int cross_attention_head_dim = 72;
    int cross_attention_dim      = 1152;
    int caption_channels         = 2304;   // Gemma-2 caption width
    float mlp_ratio              = 2.5f;
    int patch_size               = 1;
    int sample_size              = 32;
    bool attention_bias          = false;
    bool norm_elementwise_affine = false;
    float norm_eps               = 1e-6f;

    int inner_dim() const { return num_attention_heads * attention_head_dim; }
    int latent_channels() const { return in_channels; }
};

class SanaDenoiser final : public Denoiser {
public:
    explicit SanaDenoiser(const SanaConfig& cfg);
    ~SanaDenoiser();

    SanaDenoiser(const SanaDenoiser&) = delete;
    SanaDenoiser& operator=(const SanaDenoiser&) = delete;
    SanaDenoiser(SanaDenoiser&&) noexcept = default;
    SanaDenoiser& operator=(SanaDenoiser&&) noexcept = default;

    // ── Denoiser interface ────────────────────────────────────────────────
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "") override;
    void finalize_weights() override;
    PreparedConditioning prepare(const Conditioning& cond) override;
    void forward(const brotensor::Tensor& latent, int H_lat, int W_lat,
                 float timestep, const PreparedConditioning& prepared,
                 Branch branch, brotensor::Tensor& out) override;
    int latent_channels() const override { return cfg_.in_channels; }
    PredictionType prediction_type() const override {
        return PredictionType::Velocity;
    }
    // Sana 0.6B is not guidance-distilled — it runs true classifier-free
    // guidance (a separate uncond branch + CFG combine), unlike Flux.
    bool uses_cfg() const override { return true; }
    brotensor::Dtype compute_dtype() const override;

    const SanaConfig& config() const { return cfg_; }

private:
    SanaConfig cfg_;
    bool finalized_ = false;
};

}  // namespace brodiffusion::dit
