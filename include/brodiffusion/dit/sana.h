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
#include <vector>

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
    // Sana-Sprint extras. guidance_embeds: the DiT is guidance-distilled and
    // takes an embedded guidance scalar (no CFG) via
    // SanaCombinedTimestepGuidanceEmbeddings; guidance_embeds_scale multiplies
    // the raw guidance scale before embedding. qk_norm: apply RMSNorm across the
    // full inner_dim ("rms_norm_across_heads") to the self- and cross-attention
    // queries/keys. Both are false for the base Sana 0.6B / 1.6B models.
    bool  guidance_embeds        = false;
    float guidance_embeds_scale  = 0.1f;
    bool  qk_norm                = false;

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
    // Base Sana is not guidance-distilled — it runs true classifier-free
    // guidance (a separate uncond branch + CFG combine), unlike Flux.
    // Sana-Sprint IS guidance-distilled (guidance_embeds): the guidance scale
    // is fed in as an embedding, so no uncond branch / CFG combine runs.
    bool uses_cfg() const override { return !cfg_.guidance_embeds; }
    brotensor::Dtype compute_dtype() const override;

    const SanaConfig& config() const { return cfg_; }

private:
    // A linear layer (weight (out,in)); bias optional (empty => bias-free).
    struct Linear {
        brotensor::Tensor W;
        brotensor::Tensor b;
        bool has_bias() const { return b.size() > 0; }
    };

    // One SanaTransformerBlock's weights.
    struct Block {
        // Per-block AdaLN-single offset table, flattened (1, 6*inner).
        brotensor::Tensor scale_shift;
        // Self-attention (attn1, ReLU linear attn): q/k/v are bias-free,
        // to_out.0 carries a bias.
        Linear q1, k1, v1, out1;
        // Cross-attention (attn2, softmax MHA to caption): all four projections
        // have a bias.
        Linear q2, k2, v2, out2;
        // qk_norm gains (Sana-Sprint only): RMSNorm-across-heads weights over
        // the full inner_dim for the self- (nq1/nk1) and cross- (nq2/nk2)
        // attention queries/keys. Empty (size 0) when cfg_.qk_norm is false.
        brotensor::Tensor nq1, nk1, nq2, nk2;
        // GLU-MBConv feed-forward (Mix-FFN). conv_inverted 1x1 (D -> 2*hidden,
        // biased) -> depthwise 3x3 (2*hidden, biased) -> GLU split -> conv_point
        // 1x1 (hidden -> D, bias-free). No internal norm / residual.
        brotensor::Tensor ff_inv_W, ff_inv_b;   // (2*hidden, D), (2*hidden,1)
        brotensor::Tensor ff_depth_W, ff_depth_b; // (2*hidden, 9), (2*hidden,1)
        brotensor::Tensor ff_point_W;            // (D, hidden)
    };

    static void lin_(const Linear& l, const brotensor::Tensor& X,
                     brotensor::Tensor& Y);
    void ensure_ones_(int n);

    // Per-block sub-layers. x_mod is the modulated, normalized hidden state
    // (N, inner); each writes its sub-layer output (N, inner) into `out`.
    void self_attention_(const Block& blk, int N, int H, int W,
                         const brotensor::Tensor& x_mod, brotensor::Tensor& out);
    void cross_attention_(const Block& blk, const brotensor::Tensor& ctx,
                          const brotensor::Tensor& hidden,
                          brotensor::Tensor& out);
    void mix_ffn_(const Block& blk, int H, int W,
                  const brotensor::Tensor& x_mod, brotensor::Tensor& out);

    SanaConfig cfg_;
    bool finalized_ = false;

    // ── weights ───────────────────────────────────────────────────────────
    Linear patch_embed_;        // (inner, in_channels) 1x1-conv-as-linear
    Linear te_l1_, te_l2_;      // timestep_embedder (256->inner, inner->inner)
    Linear ge_l1_, ge_l2_;      // guidance_embedder (256->inner, inner->inner),
                                //   Sana-Sprint only (cfg_.guidance_embeds)
    Linear te_proj_;            // time_embed.linear (inner -> 6*inner)
    Linear cap_l1_, cap_l2_;    // caption_projection (PixArt text MLP)
    brotensor::Tensor caption_norm_g_;  // (inner,1) RMSNorm gain (caption_norm)
    brotensor::Tensor norm_out_sst_;    // (1, 2*inner) flat scale_shift_table
    Linear proj_out_;           // (patch^2*out_channels, inner)
    std::vector<Block> blocks_;

    // AdaLN affine-free LayerNorm params: ones gamma, zeros beta (inner,1).
    brotensor::Tensor ada_gamma_, ada_beta_;

    // ── per-forward scratch (kept alive across calls to avoid realloc) ─────
    brotensor::Tensor hidden_;                 // (N, inner) residual stream
    brotensor::Tensor ts_, freq_, freq_cd_;    // timestep embed scratch
    brotensor::Tensor gv_, gfreq_, gfreq_cd_;  // guidance embed scratch (Sprint)
    brotensor::Tensor gemb_, cond_;            // guidance embedding + sum (Sprint)
    brotensor::Tensor qn_, kn_;                // qk-norm transpose scratch (Sprint)
    brotensor::Tensor emb_, emb_silu_, temb6_; // embedded timestep + AdaLN row
    brotensor::Tensor mod_row_, ln_, mod_;     // modulation / layernorm
    brotensor::Tensor gated_, sub_out_;        // gate * sublayer, sub-layer out
    // self-attention
    brotensor::Tensor xcm_, q_, k_, v_, qkv_, qkv_f_, attn_f_, attn_c_;
    brotensor::Tensor vp_, kt_, scores_, hid_, recip_, ones_;
    // Batched ReLU-linear-attention core (CUDA/BF16 path).
    brotensor::Tensor ones_bf_;                  // (1,N) ones at compute dtype
    brotensor::Tensor sa_S_, sa_z_, sa_qt_;      // KᵀV (nh,hd,hd), colsum (D,1), Qrᵀ (nh,N,hd)
    brotensor::Tensor sa_num_, sa_den_;          // (D,N) numerator, (nh,N) denominator
    brotensor::Tensor sa_denf_, sa_recip_;       // FP32 1/den scratch, (nh,N) recip
    // cross-attention (ca_*f_: flash-dtype casts; ca_of_: FP32 output cast)
    brotensor::Tensor ca_q_, ca_k_, ca_v_, ca_o_;
    brotensor::Tensor ca_qf_, ca_kf_, ca_vf_, ca_of_;
    // mix-ffn
    brotensor::Tensor ff_spatial_, ff_t1_, ff_t2_, ff_out_;
    brotensor::Tensor proj_;
    brotensor::Tensor out_cd_;   // (1, OC*N) compute-dtype velocity before FP32
};

}  // namespace brodiffusion::dit
