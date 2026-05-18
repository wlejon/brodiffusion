#pragma once

// UNet2DConditionModel for SD1.5 (the noise-prediction backbone).
//
// Inference-only, FP16, batch size N = 1 (the forward path hard-codes N=1
// throughout; the underlying brotensor ops support N > 1, so generalizing
// is mostly a matter of plumbing the batch dim through). Architecture mirrors Hugging
// Face's diffusers `UNet2DConditionModel` defaults for SD1.5:
//
//   block_out_channels = [320, 640, 1280, 1280]
//   layers_per_block   = 2
//   norm_num_groups    = 32
//   cross_attention_dim = 768            (CLIP ViT-L/14 hidden dim)
//   attention_head_dim  = 8              (actually means num_heads — diffusers
//                                          backwards-compat quirk; head_dim is
//                                          block_out_channels[i] / 8)
//   time_embed_dim      = 4 * block_out_channels[0] = 1280
//
// Top-level structure:
//   conv_in (in_channels=4 -> 320, 3x3)
//   time embedding: sinusoidal(t, 320) -> Linear(320,1280) -> SiLU -> Linear(1280,1280)
//   4 down_blocks:
//     [0..nb-2]: CrossAttnDownBlock — layers_per_block × (ResBlock + Transformer2D),
//                then a stride-2 3x3 conv downsampler.
//     [nb-1]:    DownBlock          — layers_per_block × ResBlock, no downsampler.
//   mid_block: ResBlock -> Transformer2D -> ResBlock
//   4 up_blocks (mirrors down, reversed channel order):
//     [0]:       UpBlock            — (layers_per_block+1) × ResBlock, upsample
//     [1..nb-1]: CrossAttnUpBlock   — (layers_per_block+1) × (ResBlock + Transformer2D),
//                                     upsample, EXCEPT the last block has no upsampler.
//     Each up-block layer first concats the latent with a popped skip activation
//     along the channel axis, then runs ResBlock(+Transformer).
//   conv_norm_out (GroupNorm) -> SiLU -> conv_out (320 -> out_channels=4, 3x3)
//
// Transformer2D (BasicTransformerBlock × 1, SD1.5 uses_linear_projection=False):
//   GroupNorm -> 1x1 proj_in -> seq layout
//     LayerNorm -> self-attn (no Q/K/V bias, biased Wo, no causal mask)
//     LayerNorm -> cross-attn (no Q/K/V bias, biased Wo, ctx = text encoder out)
//     LayerNorm -> Linear(D, 8D) -> GEGLU -> Linear(4D, D)   (FF, all biased)
//   1x1 proj_out -> NCHW layout -> residual add
//
// ResBlock (ResnetBlock2D):
//   GroupNorm(C_in) -> SiLU -> Conv3x3 -> + time_emb_proj(SiLU(temb))
//   GroupNorm(C_out) -> SiLU -> Conv3x3
//   + (1x1 conv shortcut if C_in != C_out)
//
// Caller is responsible for cuda_sync() before reading the output. All
// weights and activations are FP16 — convert host-side if your checkpoint
// ships FP32 weights.

#include "brotensor/device_buffer.h"
#include "brotensor/tensor.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace brodiffusion::safetensors { class File; struct TensorView; }

namespace brodiffusion::unet {

struct UNetConfig {
    int in_channels   = 4;
    int out_channels  = 4;
    std::vector<int> block_out_channels = {320, 640, 1280, 1280};
    int layers_per_block = 2;
    int norm_num_groups  = 32;
    float eps            = 1e-5f;
    int cross_attention_dim = 768;
    // Per-level num_heads is computed as block_out_channels[i] / attention_head_dim.
    int attention_head_dim  = 8;
    // time_embed_dim = block_out_channels[0] * time_embed_dim_mult.
    int time_embed_dim_mult = 4;

    // Optional guidance-scale-embedding projection dimension. When > 0 the
    // UNet expects the checkpoint to contain `time_embedding.cond_proj.weight`
    // of shape (time_embed_dim, time_cond_proj_dim), no bias, and the caller
    // must use the forward() overload that takes a `guidance_scale_embedding`
    // float. When 0 (default — vanilla SD1.5) no cond_proj weight is loaded
    // and the existing forward() overloads are used.
    //
    // LCM-distilled checkpoints (e.g. SimianLuo/LCM_Dreamshaper_v7) ship with
    // time_cond_proj_dim = 256; the projected guidance-scale embedding is
    // added to the time embedding *after* linear_1 and *before* the SiLU,
    // matching diffusers' TimestepEmbedding.forward.
    int time_cond_proj_dim = 0;
};

class UNet {
public:
    explicit UNet(const UNetConfig& cfg);
    ~UNet();

    UNet(const UNet&) = delete;
    UNet& operator=(const UNet&) = delete;
    UNet(UNet&&) noexcept = default;
    UNet& operator=(UNet&&) noexcept = default;

    // Load all weights from a safetensors file. Names follow Hugging Face's
    // `UNet2DConditionModel` convention. SD1.5 diffusers exports use an empty
    // prefix; SD1.5 full checkpoints typically use "model.diffusion_model.".
    //
    // Every tensor must be FP16. Throws std::runtime_error on missing names,
    // shape mismatches, or dtype mismatch.
    void load_weights(const brodiffusion::safetensors::File& f,
                      const std::string& prefix = "");

    // Forward pass.
    //   sample:                (1, in_channels * H * W) FP16 — noisy latent
    //   H, W:                  spatial dims of `sample`. H and W must each be
    //                          divisible by 2^(num_blocks-1) (typically 8).
    //   timestep:              continuous timestep value (typically in [0, 1000)).
    //   encoder_hidden_states: (L_text, cross_attention_dim) FP16, e.g. CLIP output.
    //   out:                   (1, out_channels * H * W) FP16, resized as needed.
    void forward(const brotensor::GpuTensor& sample,
                 int H, int W,
                 float timestep,
                 const brotensor::GpuTensor& encoder_hidden_states,
                 brotensor::GpuTensor& out);

    // Cached cross-attention K/V for a single context tensor — one (K, V) pair
    // per Transformer2D block, FP16, layout matching
    // brotensor::flash_attention_forward_gpu's K/V args (Lk, C). The text
    // context is fixed across all denoising steps so projecting K/V once per
    // generate() per CFG branch eliminates 16 × steps × 2 redundant matmuls.
    struct CrossAttnKVCacheEntry {
        brotensor::GpuTensor K;  // (Lk, C)
        brotensor::GpuTensor V;  // (Lk, C)
    };
    using CrossAttnKVCache = std::vector<CrossAttnKVCacheEntry>;

    // Populate `cache` with one (K, V) pair per Transformer2D block (in the
    // same traversal order the forward pass visits them: down blocks,
    // mid block, up blocks). `cache` is resized as needed.
    void prime_xattn_cache(const brotensor::GpuTensor& ctx,
                           CrossAttnKVCache& cache);

    // Variant of forward that uses a pre-primed K/V cache built from the
    // same `encoder_hidden_states` (or, more precisely, from a ctx with the
    // exact tokens the cache was primed against). Cross-attention layers
    // skip the K/V projection step; self-attention is unchanged.
    void forward(const brotensor::GpuTensor& sample,
                 int H, int W,
                 float timestep,
                 const brotensor::GpuTensor& encoder_hidden_states,
                 const CrossAttnKVCache& xattn_cache,
                 brotensor::GpuTensor& out);

    // LCM guidance-scale-embedding forward.
    //
    // Only valid when the UNet was built with `time_cond_proj_dim > 0`. The
    // raw guidance scale `w` (e.g. 7.5) is passed in; this overload internally
    // computes the sinusoidal `w_emb = sinusoidal(w * 1000, time_cond_proj_dim)`
    // matching diffusers' `get_guidance_scale_embedding`, then projects it
    // through the loaded `time_embedding.cond_proj.weight` and adds it to the
    // time embedding before the SiLU between linear_1 and linear_2.
    //
    // The non-LCM overloads above will throw if the UNet was built with
    // `time_cond_proj_dim > 0`; this overload will throw if it was 0.
    void forward(const brotensor::GpuTensor& sample,
                 int H, int W,
                 float timestep,
                 float guidance_scale_embedding,
                 const brotensor::GpuTensor& encoder_hidden_states,
                 const CrossAttnKVCache& xattn_cache,
                 brotensor::GpuTensor& out);

    // Number of Transformer2D (cross-attn) blocks in the model — matches the
    // size of any cache returned by prime_xattn_cache.
    int num_xattn_blocks() const;

    // Fold a LoRA delta into the base FP16 weight identified by `target_path`.
    // `target_path` is a diffusers path *within* the UNet (no "unet." prefix),
    // e.g. "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q" or
    // "ff.net.0.proj". Throws if the path is unknown or shapes don't match.
    //
    // The merge is in-place:
    //     W += scale_total * (lora_up @ lora_down)
    // where `scale_total = (alpha / rank) * user_scale` is computed by the
    // caller. `lora_down` is (rank, in_dim) and `lora_up` is (out_dim, rank);
    // F16 or F32 source are both accepted (F32 converted host-side, matching
    // the rest of the UNet loader).
    //
    // Must be called *after* `load_weights()`. May be called more than once
    // (e.g. to stack multiple LoRAs).
    void apply_lora_delta(const std::string& target_path,
                          const brodiffusion::safetensors::TensorView& lora_down,
                          const brodiffusion::safetensors::TensorView& lora_up,
                          float scale_total);

    const UNetConfig& config() const { return cfg_; }

private:
    struct Resnet {
        brotensor::GpuTensor n1g, n1b, W1, b1;
        brotensor::GpuTensor temb_W, temb_b;
        brotensor::GpuTensor n2g, n2b, W2, b2;
        brotensor::GpuTensor Ws, bs;
        bool has_shortcut = false;
        int  C_in = 0, C_out = 0;
    };
    struct AttnFFN {
        brotensor::GpuTensor n1g, n1b;
        brotensor::GpuTensor Wq1, Wk1, Wv1, Wo1, bo1;
        brotensor::GpuTensor n2g, n2b;
        brotensor::GpuTensor Wq2, Wk2, Wv2, Wo2, bo2;
        brotensor::GpuTensor n3g, n3b;
        brotensor::GpuTensor ff1_W, ff1_b;
        brotensor::GpuTensor ff2_W, ff2_b;
    };
    struct Transformer2D {
        brotensor::GpuTensor gn_g, gn_b;
        brotensor::GpuTensor pi_W, pi_b;
        brotensor::GpuTensor po_W, po_b;
        std::vector<AttnFFN> blocks;
        int  C = 0;
        int  num_heads = 0;
    };
    struct SampleConv {
        brotensor::GpuTensor W, b;
    };
    struct DownBlock {
        std::vector<Resnet>        resnets;
        std::vector<Transformer2D> transformers;
        SampleConv                 downsampler;
        bool has_attention   = false;
        bool has_downsampler = false;
        int  C_out = 0;
    };
    struct MidBlock {
        Resnet         r0, r1;
        Transformer2D  t;
    };
    struct UpBlock {
        std::vector<Resnet>        resnets;
        std::vector<Transformer2D> transformers;
        SampleConv                 upsampler;
        bool has_attention = false;
        bool has_upsampler = false;
        int  C_out = 0;
    };

    void load_resnet_(const brodiffusion::safetensors::File& f,
                      const std::string& prefix,
                      int C_in, int C_out, Resnet& r);
    void load_transformer_(const brodiffusion::safetensors::File& f,
                           const std::string& prefix,
                           int C, int num_heads, Transformer2D& t);

    void apply_resnet_(const Resnet& r, int H, int W,
                       brotensor::GpuTensor& x, brotensor::GpuTensor& tmp);
    // If `cache_entry` is non-null, its (K, V) replace the cross-attn K/V
    // projections (must have been primed against the same `ctx`).
    void apply_transformer_(const Transformer2D& t,
                            const brotensor::GpuTensor& ctx,
                            const CrossAttnKVCacheEntry* cache_entry,
                            int H, int W,
                            brotensor::GpuTensor& x);
    // Shared forward worker; xattn_cache may be null (legacy path) or point
    // at a cache with exactly num_xattn_blocks() entries. If `gs_emb` is
    // non-null the LCM cond_proj path is used (requires time_cond_proj_dim>0);
    // otherwise the vanilla SD1.5 time-embedding path is used.
    void forward_impl_(const brotensor::GpuTensor& sample,
                       int H, int W,
                       float timestep,
                       const float* gs_emb,
                       const brotensor::GpuTensor& encoder_hidden_states,
                       const CrossAttnKVCache* xattn_cache,
                       brotensor::GpuTensor& out);
    // Returns a pointer to the FP16 base weight identified by `target_path`
    // (a diffusers tail within the UNet, e.g. "down_blocks.0.attentions.0.
    // transformer_blocks.0.attn1.to_q"). Returns nullptr if the path doesn't
    // match a recognized LoRA-patchable layer.
    brotensor::GpuTensor* lora_target_(const std::string& target_path);
    // Sub-dispatchers used by lora_target_. `sub` is the diffusers path tail
    // remaining after the block-level prefix has been consumed.
    static brotensor::GpuTensor* resolve_transformer_target_(Transformer2D& tr,
                                                              const std::string& sub);
    static brotensor::GpuTensor* resolve_resnet_target_(Resnet& r,
                                                        const std::string& tail);

    void apply_conv3x3_(const brotensor::GpuTensor& W,
                        const brotensor::GpuTensor& b,
                        int C_in, int C_out, int H, int W_,
                        int stride, int pad,
                        const brotensor::GpuTensor& in,
                        brotensor::GpuTensor& out);

    UNetConfig cfg_;
    int time_embed_dim_ = 0;
    int freq_dim_       = 0;

    brotensor::GpuTensor conv_in_W_,  conv_in_b_;
    brotensor::GpuTensor te_l1_W_, te_l1_b_, te_l2_W_, te_l2_b_;
    // cond_proj weight (LCM only; empty when time_cond_proj_dim == 0).
    brotensor::GpuTensor te_cond_W_;
    std::vector<DownBlock> down_blocks_;
    MidBlock               mid_;
    std::vector<UpBlock>   up_blocks_;
    brotensor::GpuTensor norm_out_g_, norm_out_b_;
    brotensor::GpuTensor conv_out_W_, conv_out_b_;

    brotensor::GpuTensor x_, y_;
    brotensor::GpuTensor freq_emb_, temb_a_, temb_b_, temb_silu_, temb_proj_;
    // LCM scratch: w_emb_ holds the sinusoidal guidance-scale embedding,
    // temb_cond_ holds cond_proj(w_emb_); both unused when time_cond_proj_dim==0.
    brotensor::GpuTensor w_emb_, temb_cond_;
    brotensor::GpuTensor cat_buf_;
    brotensor::GpuTensor gn_, seq_, proj_in_seq_, tseq_, ln_;
    brotensor::GpuTensor attn_proj_;
    brotensor::GpuTensor ff_mid_, ff_act_, ff_out_;
    brotensor::GpuTensor proj_out_seq_, proj_out_nchw_;
};

}  // namespace brodiffusion::unet
