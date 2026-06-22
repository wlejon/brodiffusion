#pragma once

// PixArtTransformer2DModel — PixArt-Sigma's diffusion transformer denoiser.
//
// A Denoiser for the PixArt-Sigma-XL-2 DiT (HF diffusers Transformer2DModel
// with norm_type="ada_norm_single"). Forward-only, batch size N = 1. Runs on
// whichever backend brotensor resolves at runtime — CPU (FP32) by default,
// CUDA (FP16) when available — at the pipeline compute dtype.
//
// Architecture (the released XL/2 1024-MS checkpoint):
//   - pos_embed.proj: a patch_size x patch_size (2x2), stride-2 Conv2d over the
//     4-channel latent -> inner_dim tokens, plus an added 2D sin-cos positional
//     embedding (a non-persistent buffer, recomputed here per grid size).
//   - caption_projection (PixArt text MLP): T5-XXL caption (caption_channels =
//     4096) -> inner_dim, Linear -> GELU(tanh) -> Linear. NO caption_norm.
//   - adaln_single: timestep -> embedded_timestep (1, D) and a single global
//     6*D AdaLN-single modulation row (shared across blocks; each block adds its
//     own learned scale_shift_table offset).
//   - num_layers PixArt blocks, each: AdaLN-single self-attention (standard
//     softmax MHA, biased q/k/v), then cross-attention to the caption tokens
//     (NO norm / modulation), then a GELU(tanh) feed-forward (mlp ratio 4).
//   - norm_out (AdaLN with the top-level 2-row scale_shift_table offset by
//     embedded_timestep) -> proj_out -> unpatchify. proj_out emits
//     patch^2 * out_channels with out_channels = 2 * latent_channels (epsilon +
//     learned variance, IDDPM-style); the variance half is discarded.
//
// PixArt-Sigma is epsilon-prediction (prediction_type() == Epsilon) and runs
// true classifier-free guidance (uses_cfg() == true). Pairs with the
// DPMSolverMultistep (DPM-Solver++) scheduler. The 1024-MS checkpoint uses
// neither KV token compression (that is the 2K/4K models) nor any additional
// resolution/aspect micro-conditioning (use_additional_conditions = false).

#include "brodiffusion/denoiser.h"

#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::dit {

struct PixArtConfig {
    int   in_channels         = 4;      // latent channels (SDXL KL-VAE)
    int   out_channels        = 8;      // 2 * in_channels (epsilon + variance)
    int   num_layers          = 28;
    int   attention_head_dim  = 72;
    int   num_attention_heads = 16;     // inner_dim = heads * head_dim = 1152
    int   cross_attention_dim = 1152;
    int   caption_channels    = 4096;   // T5-XXL caption width
    int   patch_size          = 2;
    int   sample_size         = 128;    // base latent grid (1024px / 8)
    int   interpolation_scale = 2;
    float norm_eps            = 1e-6f;

    int inner_dim() const { return num_attention_heads * attention_head_dim; }
    int latent_channels() const { return in_channels; }
};

class PixArtDenoiser final : public Denoiser {
public:
    explicit PixArtDenoiser(const PixArtConfig& cfg);
    ~PixArtDenoiser();

    PixArtDenoiser(const PixArtDenoiser&) = delete;
    PixArtDenoiser& operator=(const PixArtDenoiser&) = delete;
    PixArtDenoiser(PixArtDenoiser&&) noexcept = default;
    PixArtDenoiser& operator=(PixArtDenoiser&&) noexcept = default;

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
        return PredictionType::Epsilon;
    }
    bool uses_cfg() const override { return true; }
    brotensor::Dtype compute_dtype() const override;

    // ── CUDA-graph step-capture seam ──────────────────────────────────────
    // The per-step time-embedding chain (host-dependent) is prepare_step; the
    // remaining device-only block stack + GPU unpatchify is forward_body, which
    // after warm-up is allocation-stable and host-free (no proj download — the
    // unpatchify runs on the GPU via brotensor::patch_unpack_forward). forward()
    // is exactly prepare_step + forward_body, so the eager and captured paths
    // share one body.
    bool supports_step_capture() const override { return true; }
    void prepare_step(float timestep,
                      const PreparedConditioning& prepared) override;
    void forward_body(const brotensor::Tensor& latent, int H_lat, int W_lat,
                      const PreparedConditioning& prepared, Branch branch,
                      brotensor::Tensor& out) override;

    const PixArtConfig& config() const { return cfg_; }

private:
    // A linear layer (weight (out,in)); bias optional (empty => bias-free).
    struct Linear {
        brotensor::Tensor W;
        brotensor::Tensor b;
        bool has_bias() const { return b.size() > 0; }
    };

    // One PixArt transformer block's weights.
    struct Block {
        brotensor::Tensor scale_shift;   // per-block (1, 6*inner) AdaLN offset
        Linear q1, k1, v1, out1;          // self-attention (attn1), all biased
        Linear q2, k2, v2, out2;          // cross-attention (attn2), all biased
        Linear ff1, ff2;                  // ff.net.0.proj (D->4D), ff.net.2 (4D->D)
    };

    void lin_(const Linear& l, const brotensor::Tensor& X,
              brotensor::Tensor& Y);
    // Build / cache the 2D sin-cos positional embedding for a (hp, wp) token
    // grid at the compute dtype.
    const brotensor::Tensor& pos_embed_for_(int hp, int wp);

    // Matmul / attention storage dtype. FP16 on CUDA (tensor-core WMMA + flash;
    // accumulation stays FP32 inside those kernels), FP32 on CPU. The residual
    // stream, layernorms, modulation and gating all stay at compute_dtype()
    // (FP32) — only the heavy per-sublayer GEMMs and attention drop to FP16.
    brotensor::Dtype mm_dtype() const;

    // Manual FP32 multi-head attention core from already-projected (FP32) Q/K/V
    // (per-head matmul + softmax + matmul); writes ca_ocat_ (FP32), no out-proj.
    // Used by cross-attention, whose query is the unnormed wide-range residual:
    // FP16 score storage overflows and BF16 is too coarse, so the core stays
    // FP32 (cheap — the caption context is short) while its projections ride the
    // FP16 tensor-core path.
    void manual_core_(int nh, const brotensor::Tensor& Q,
                      const brotensor::Tensor& K, const brotensor::Tensor& V);
    // FP32 manual attention + out-proj in one (CPU path).
    void manual_attention_(int nh, const brotensor::Tensor& Q,
                           const brotensor::Tensor& K, const brotensor::Tensor& V,
                           const Linear& out_proj, brotensor::Tensor& out);

    // Shared device-side body: patch-embed -> blocks -> norm_out -> proj_out ->
    // GPU unpatchify, against an already-selected (and projected) caption
    // context. Reads the persistent emb_/temb6_ written by prepare_step. After
    // one warm-up call it allocates no named buffer and reads/writes no host
    // memory, so the Pipeline can record it with CudaGraphCapture.
    void body_(const brotensor::Tensor& latent, int H_lat, int W_lat,
               const brotensor::Tensor& ctx, brotensor::Tensor& out);

    // Per-block sub-layers; each writes its (N, inner) output into `out`.
    void self_attention_(const Block& blk, const brotensor::Tensor& x_mod,
                         brotensor::Tensor& out);
    void cross_attention_(const Block& blk, const brotensor::Tensor& ctx,
                          const brotensor::Tensor& hidden,
                          brotensor::Tensor& out);
    void feed_forward_(const Block& blk, const brotensor::Tensor& x_mod,
                       brotensor::Tensor& out);

    PixArtConfig cfg_;
    bool finalized_ = false;

    // ── weights ───────────────────────────────────────────────────────────
    Linear patch_proj_;         // pos_embed.proj (conv 2x2 as (D, IC*4))
    Linear te_l1_, te_l2_;      // adaln_single.emb.timestep_embedder.linear_1/2
    Linear adaln_proj_;         // adaln_single.linear (D -> 6*D)
    Linear cap_l1_, cap_l2_;    // caption_projection.linear_1/2
    brotensor::Tensor norm_out_sst_;  // top-level scale_shift_table (1, 2*inner)
    Linear proj_out_;           // (patch^2 * out_channels, inner)
    std::vector<Block> blocks_;

    // AdaLN affine-free LayerNorm params: ones gamma, zeros beta (inner,1).
    brotensor::Tensor ada_gamma_, ada_beta_;

    // Cached positional embedding (compute dtype) keyed on the token grid.
    brotensor::Tensor pos_embed_;
    int pos_hp_ = -1, pos_wp_ = -1;

    // ── per-step scratch (kept alive to avoid realloc) ─────────────────────
    // Time-embedding chain (written by prepare_step into STABLE buffers — the
    // captured body reads emb_ and temb6_ by pointer, so prepare_step must never
    // reassign/clone them, only overwrite via ops). ts_ is consumed inside
    // prepare_step itself, so its per-step reallocation is harmless.
    brotensor::Tensor ts_, freq_, freq_cd_;
    brotensor::Tensor te_h_, emb_, emb_silu_, temb6_;
    // Body scratch.
    brotensor::Tensor lat_cd_;                 // FP32 latent (cast of the input)
    brotensor::Tensor mod_row_, ln_, mod_;
    std::vector<brotensor::Tensor> ch_, no_;   // sliced modulation chunks
    brotensor::Tensor shift_, scale_;          // norm_out shift/scale (no clone)
    brotensor::Tensor hidden_, patch_nchw_;
    brotensor::Tensor gated_, sub_out_;
    brotensor::Tensor out_unpack_;             // GPU unpatchify result (pre-cast)
    // Mixed-precision scratch: FP16 (mm_dtype) views of the FP32 modulated input
    // / residual that feed the tensor-core GEMMs, and the FP16 sublayer output
    // before it is cast back into the FP32 residual.
    brotensor::Tensor mod16_, hid16_, sub16_;
    // Cross-attn FP32 upcasts of the FP16 Q/K/V projections (for the exact core),
    // and the FP16 view of its output ahead of the FP16 out-projection.
    brotensor::Tensor qf32_, kf32_, vf32_, o16_;
    // attention scratch: FP16 projections + flash output on CUDA, FP32 on CPU.
    brotensor::Tensor q_, k_, v_, attn_o_;
    brotensor::Tensor mha_qh_, mha_kh_, mha_vh_, mha_attn_, mha_yc_;  // mha caches (CPU)
    // manual_attention_ per-head scratch (FP32).
    brotensor::Tensor ca_qh_, ca_kh_, ca_vh_, ca_kt_, ca_sc_, ca_oh_, ca_ocat_;
    // feed-forward scratch
    brotensor::Tensor ff_h_;
    brotensor::Tensor proj_;                // proj_out output (N, p^2*OC)
};

}  // namespace brodiffusion::dit
