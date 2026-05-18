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

#include "brodiffusion/student_selfattn.h"

#include "brotensor/device_buffer.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brodiffusion::safetensors { class File; }

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

    // Phase-1 distillation: replace the self-attention sub-layer of the
    // four L = 4096 (top-resolution, C = 320) Transformer2D blocks with a
    // cheap residual depthwise-separable conv stack (SelfAttnStudent).
    //
    // Order of the 4 student slots (matches forward-pass visit order):
    //   [0] down_blocks[0].transformers[0]   (encoder, top-res, layer 0)
    //   [1] down_blocks[0].transformers[1]   (encoder, top-res, layer 1)
    //   [2] up_blocks[nb-1].transformers[1]  (decoder, top-res, second-to-last)
    //   [3] up_blocks[nb-1].transformers[2]  (decoder, top-res, last)
    //
    // Cross-attention + FF sub-layers of those blocks are *unchanged*. Other
    // resolutions (L = 1024, 256, 64) always use the original self-attention.
    //
    // When false (default) the student tensors are not even allocated.
    // When true and the student weights aren't present in the safetensors
    // file, the student is left zero-initialised (identity behaviour — it
    // skips the self-attention contribution entirely, which is the expected
    // pre-training swap-in state).
    bool enable_selfattn_student_L4096 = false;
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
    //
    // student_prefix: when the self-attn student is enabled, the per-block
    //   weights are looked up at
    //     <student_prefix>block<i>.dw.<j>.weight  etc.  (i = 0..3, j = 0..2)
    //   If the prefix is empty or the weights are absent, the students stay
    //   at their zero-init (identity) state. Ignored if the flag is off.
    void load_weights(const brodiffusion::safetensors::File& f,
                      const std::string& prefix = "",
                      const std::string& student_prefix = "");

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

    // Number of Transformer2D (cross-attn) blocks in the model — matches the
    // size of any cache returned by prime_xattn_cache.
    int num_xattn_blocks() const;

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
    // If `student` is non-null, its forward() replaces the self-attention
    // sub-layer of the transformer block; the surrounding GroupNorm /
    // proj_in / proj_out path is unchanged (we still go sequence-shape for
    // cross-attn + FF, but the self-attn step consumes NCHW directly via
    // the student's pre-seq-transpose hook). See apply_transformer_ for
    // the exact swap point.
    void apply_transformer_(const Transformer2D& t,
                            const brotensor::GpuTensor& ctx,
                            const CrossAttnKVCacheEntry* cache_entry,
                            const student::SelfAttnStudent* student,
                            int H, int W,
                            brotensor::GpuTensor& x);
    // Shared forward worker; xattn_cache may be null (legacy path) or point
    // at a cache with exactly num_xattn_blocks() entries.
    void forward_impl_(const brotensor::GpuTensor& sample,
                       int H, int W,
                       float timestep,
                       const brotensor::GpuTensor& encoder_hidden_states,
                       const CrossAttnKVCache* xattn_cache,
                       brotensor::GpuTensor& out);
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
    std::vector<DownBlock> down_blocks_;
    MidBlock               mid_;
    std::vector<UpBlock>   up_blocks_;
    brotensor::GpuTensor norm_out_g_, norm_out_b_;
    brotensor::GpuTensor conv_out_W_, conv_out_b_;

    brotensor::GpuTensor x_, y_;
    brotensor::GpuTensor freq_emb_, temb_a_, temb_b_, temb_silu_, temb_proj_;
    brotensor::GpuTensor cat_buf_;
    brotensor::GpuTensor gn_, seq_, proj_in_seq_, tseq_, ln_;
    brotensor::GpuTensor attn_proj_;
    brotensor::GpuTensor ff_mid_, ff_act_, ff_out_;
    brotensor::GpuTensor proj_out_seq_, proj_out_nchw_;

    // Self-attn student modules (Phase-1 distillation). Empty unless
    // cfg_.enable_selfattn_student_L4096 == true; otherwise sized to 4
    // and ordered to match the forward-pass visit order described on
    // UNetConfig::enable_selfattn_student_L4096.
    std::vector<student::SelfAttnStudent> students_;
    // Scratch buffer for the student's intermediate (post-dwconv) NCHW
    // activation. Reused across blocks and across timesteps.
    brotensor::GpuTensor student_scratch_;
    // Holds the student forward output (NCHW); the result is swapped back
    // into `x` after the student call.
    brotensor::GpuTensor student_out_;
};

}  // namespace brodiffusion::unet
