#pragma once

// FluxTransformer2DModel — the Flux.1 DiT denoiser.
//
// A Denoiser implementation for the Flux.1 rectified-flow transformer (the
// SD1.5 UNet's sibling behind brodiffusion's model-agnostic Denoiser base).
// Forward-only, batch size N = 1. Runs on whichever backend brotensor
// resolves at runtime — CPU by default (FP32, the test path), CUDA when
// available (FP16).
//
// Architecture (HF diffusers FluxTransformer2DModel):
//   - x_embedder: 2x2-patch-packed latent (64-dim) → inner_dim
//   - context_embedder: T5 sequence (4096-dim) → inner_dim
//   - time_text_embed: timestep + (optional guidance) + pooled-CLIP → temb
//   - num_layers double-stream blocks (separate image / text streams,
//     joint attention)
//   - num_single_layers single-stream blocks (concatenated [text; image])
//   - norm_out (AdaLayerNormContinuous) → proj_out → 64-dim → unpack
//
// The joint sequence is [text tokens ; image tokens]; 2D axial RoPE supplies
// position information (no learned position embedding). AdaLN modulation
// (no affine LayerNorm) is driven by `temb`.
//
// Pairs with the FlowMatch (rectified-flow Euler) scheduler;
// prediction_type() is Velocity and uses_cfg() is false (Flux uses a
// distilled single-branch guidance embedding, not classifier-free guidance).

#include "brodiffusion/denoiser.h"

#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::dit {

struct FluxConfig {
    int in_channels         = 64;    // PACKED latent dim (= latent_channels * 4)
    int num_layers          = 19;    // double-stream blocks
    int num_single_layers   = 38;    // single-stream blocks
    int attention_head_dim  = 128;
    int num_attention_heads = 24;    // inner_dim = heads * head_dim = 3072
    int joint_attention_dim = 4096;  // T5 context width
    int pooled_projection_dim = 768; // CLIP pooled width
    bool guidance_embeds    = false; // false = flux-schnell, true = flux-dev
    std::vector<int> axes_dims_rope = {16, 56, 56};  // sums to attention_head_dim

    int inner_dim() const { return num_attention_heads * attention_head_dim; }
    int latent_channels() const { return in_channels / 4; }
};

class FluxDenoiser final : public Denoiser {
public:
    explicit FluxDenoiser(const FluxConfig& cfg);
    ~FluxDenoiser();

    FluxDenoiser(const FluxDenoiser&) = delete;
    FluxDenoiser& operator=(const FluxDenoiser&) = delete;
    FluxDenoiser(FluxDenoiser&&) noexcept = default;
    FluxDenoiser& operator=(FluxDenoiser&&) noexcept = default;

    // ── Denoiser interface ────────────────────────────────────────────────
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "") override;

    // Load from a *sharded* safetensors set: every tensor is searched across
    // all `shards` (the first match wins; a name missing in every shard
    // throws). The single-File overload above is a one-element wrapper over
    // this path. The Flux transformer ships sharded in diffusers format.
    void load_weights(
        const std::vector<const brotensor::safetensors::File*>& shards,
        const std::string& prefix = "");
    void finalize_weights() override;
    PreparedConditioning prepare(const Conditioning& cond) override;
    void forward(const brotensor::Tensor& latent, int H_lat, int W_lat,
                 float timestep, const PreparedConditioning& prepared,
                 Branch branch, brotensor::Tensor& out) override;
    int latent_channels() const override { return cfg_.latent_channels(); }
    PredictionType prediction_type() const override {
        return PredictionType::Velocity;
    }
    bool uses_cfg() const override { return false; }
    brotensor::Dtype compute_dtype() const override;

    const FluxConfig& config() const { return cfg_; }

private:
    // A biased linear layer (weight (out,in), bias (out,1)).
    struct Linear {
        brotensor::Tensor W;
        brotensor::Tensor b;
    };

    // Double-stream block (FluxTransformerBlock).
    struct DoubleBlock {
        Linear norm1;          // inner_dim → 6*inner_dim
        Linear norm1_context;  // inner_dim → 6*inner_dim
        Linear to_q, to_k, to_v;
        Linear add_q, add_k, add_v;
        Linear to_out;         // attn.to_out.0
        Linear to_add_out;     // attn.to_add_out
        brotensor::Tensor norm_q, norm_k;              // (head_dim,1) RMSNorm gain
        brotensor::Tensor norm_added_q, norm_added_k;  // (head_dim,1)
        Linear ff0, ff2;        // ff.net.0.proj (D→4D), ff.net.2 (4D→D)
        Linear ffc0, ffc2;      // ff_context.net.0.proj, .net.2
    };

    // Single-stream block (FluxSingleTransformerBlock).
    struct SingleBlock {
        Linear norm;            // inner_dim → 3*inner_dim
        Linear to_q, to_k, to_v;
        brotensor::Tensor norm_q, norm_k;  // (head_dim,1)
        Linear proj_mlp;        // D → 4D
        Linear proj_out;        // 5D → D
    };

    void run_double_block_(const DoubleBlock& blk,
                           const brotensor::Tensor& temb,
                           const brotensor::Tensor& cos,
                           const brotensor::Tensor& sin,
                           int txt_len, int img_len);
    void run_single_block_(const SingleBlock& blk,
                           const brotensor::Tensor& temb,
                           const brotensor::Tensor& cos,
                           const brotensor::Tensor& sin,
                           int L);

    FluxConfig cfg_;
    bool finalized_ = false;

    // Top-level weights.
    Linear x_embedder_;
    Linear context_embedder_;
    Linear te_time_l1_, te_time_l2_;     // timestep_embedder
    Linear te_text_l1_, te_text_l2_;     // text_embedder
    Linear te_guidance_l1_, te_guidance_l2_;  // guidance_embedder (dev only)
    Linear norm_out_;                    // AdaLayerNormContinuous (D→2D)
    Linear proj_out_;                    // D → in_channels

    std::vector<DoubleBlock> double_blocks_;
    std::vector<SingleBlock> single_blocks_;

    // AdaLN affine-free LayerNorm params: all-ones gamma, all-zeros beta,
    // both (inner_dim,1) at the compute dtype. Allocated once.
    brotensor::Tensor ada_gamma_, ada_beta_;

    // ── per-forward scratch (kept alive across calls to avoid realloc) ─────
    brotensor::Tensor img_, txt_, x_;          // residual streams
    brotensor::Tensor ln_, mod_;               // layernorm / modulate output
    brotensor::Tensor silu_;                   // silu(temb)
    brotensor::Tensor chunk_row_;              // modulation MLP output row
    brotensor::Tensor q_, k_, v_;              // projections
    brotensor::Tensor qn_, kn_;                // per-head RMSNorm output
    brotensor::Tensor Q_, K_, V_;              // concatenated attention inputs
    brotensor::Tensor Qr_, Kr_;                // RoPE-rotated Q / K
    brotensor::Tensor attn_;                   // attention output
    brotensor::Tensor proj_;                   // attn output projection
    brotensor::Tensor ff_mid_, ff_out_;        // FFN intermediates
    brotensor::Tensor gated_;                  // broadcast_mul output
    brotensor::Tensor mlp_;                    // single-block MLP branch
    brotensor::Tensor cat5_;                   // single-block [attn ; mlp]
    brotensor::Tensor ts_, freq_;              // timestep embedding scratch
    brotensor::Tensor temb_, temb_time_, temb_guid_;
};

}  // namespace brodiffusion::dit
