#pragma once

// UNet2DModel for pixel-space DDPMs (Ho et al. 2020 lineage). Tracks Hugging
// Face diffusers' `UNet2DModel` (the unconditional / image-space variant),
// distinct from the `UNet2DConditionModel` used by Stable Diffusion in
// `brodiffusion/unet.h`. Differences vs. SD1.5's UNet that drove the split:
//
//   - No text conditioning. No cross-attention. No Transformer2D wrapper —
//     attention layers are bare `AttnBlock`s (GroupNorm → Q/K/V/proj_attn →
//     residual add) interleaved with the resnets inside a block, with biased
//     Q/K/V/O linears.
//   - Time embedding uses diffusers' positional convention with
//     `flip_sin_to_cos=False` and `freq_shift=1` (sin first, then cos;
//     exponent divided by `half-1`), matching the original DDPM formulation.
//   - Downsampling can use asymmetric pre-pad `(0,1,0,1)` followed by a
//     stride-2 padding-0 conv (diffusers' `downsample_padding=0` path),
//     selectable per-config.
//   - Attention placement is per-block-type (`DownBlock2D` vs
//     `AttnDownBlock2D`), not a fixed "all but the last" rule.
//   - block_out_channels and layers_per_block are otherwise the same idea
//     as SD1.5's UNet config.
//
// Inference-only, FP16 throughout, batch size N = 1. brotensor owns every
// kernel involved (conv2d, group_norm, silu, flash_attention_forward, the
// fused resblock op, nearest-neighbour upsample). The asymmetric-pad helper
// for `downsample_padding=0` ships as a small CUDA kernel in this lib (see
// `src/ddpm_pad.cu`).

#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brodiffusion::safetensors { class File; }

namespace brodiffusion::ddpm_unet {

// One per down-side block. The mirror up_block_types vector inverts the order
// (last entry is closest to the output side).
enum class DownBlockType {
    Down,         // DownBlock2D    — `layers_per_block` resnets, no attention.
    AttnDown,     // AttnDownBlock2D — `layers_per_block` resnets, each followed
                  //                   by a single AttnBlock at the block's
                  //                   spatial resolution.
};
enum class UpBlockType {
    Up,           // UpBlock2D
    AttnUp,       // AttnUpBlock2D — `layers_per_block + 1` resnets, each
                  //                  followed by a single AttnBlock.
};

struct UNetConfig {
    int in_channels  = 3;
    int out_channels = 3;
    std::vector<int> block_out_channels = {128, 256, 256, 256};
    std::vector<DownBlockType> down_block_types =
        {DownBlockType::Down, DownBlockType::AttnDown,
         DownBlockType::Down, DownBlockType::Down};
    std::vector<UpBlockType> up_block_types =
        {UpBlockType::Up, UpBlockType::Up,
         UpBlockType::AttnUp, UpBlockType::Up};
    int   layers_per_block = 2;
    int   norm_num_groups  = 32;
    float eps              = 1.0e-6f;

    // time_embed_dim = block_out_channels[0] * time_embed_dim_mult.
    int time_embed_dim_mult = 4;

    // Diffusers' downsample_padding=0 ⇒ asymmetric prepad (0,1,0,1), conv
    // with padding=0. =1 ⇒ standard symmetric padding=1, no prepad.
    int downsample_padding = 0;

    // Number of attention heads inside each AttnBlock. diffusers' default
    // (attention_head_dim=null in the config JSON) maps to single-head
    // attention with head_dim equal to the block's channel count; that's
    // the setting google/ddpm-cifar10-32 was trained with. Override only
    // if loading a checkpoint that was trained multi-head at this layer.
    int attention_num_heads = 1;
};

class UNet {
public:
    explicit UNet(const UNetConfig& cfg);
    ~UNet();

    UNet(const UNet&) = delete;
    UNet& operator=(const UNet&) = delete;
    UNet(UNet&&) noexcept = default;
    UNet& operator=(UNet&&) noexcept = default;

    // Load all weights from a safetensors file. Names follow HF's
    // `UNet2DModel` convention (the format google/ddpm-cifar10-32 ships in).
    // Accepts F16 (used as-is) or F32 (converted host-side); google's CIFAR
    // checkpoint ships as F32.
    void load_weights(const brodiffusion::safetensors::File& f,
                      const std::string& prefix = "");

    // Forward pass.
    //   sample:   (1, in_channels * H * W) FP16 — noisy image.
    //   H, W:     spatial dims of sample (must each be divisible by
    //             2^(num_blocks - 1)).
    //   timestep: continuous timestep value (typically in [0, 1000)).
    //   out:      (1, out_channels * H * W) FP16 — predicted noise; resized
    //             as needed.
    void forward(const brotensor::GpuTensor& sample,
                 int H, int W, float timestep,
                 brotensor::GpuTensor& out);

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
    struct AttnBlock {
        brotensor::GpuTensor gn_g, gn_b;
        brotensor::GpuTensor Wq, bq;
        brotensor::GpuTensor Wk, bk;
        brotensor::GpuTensor Wv, bv;
        brotensor::GpuTensor Wo, bo;
        int C = 0;
    };
    struct SampleConv {
        brotensor::GpuTensor W, b;
    };
    struct DownBlock {
        std::vector<Resnet>    resnets;
        std::vector<AttnBlock> attns;   // empty unless type == AttnDown
        SampleConv             downsampler;
        bool has_attention   = false;
        bool has_downsampler = false;
        int  C_out = 0;
    };
    struct MidBlock {
        Resnet     r0, r1;
        AttnBlock  attn;
    };
    struct UpBlock {
        std::vector<Resnet>    resnets;
        std::vector<AttnBlock> attns;   // empty unless type == AttnUp
        SampleConv             upsampler;
        bool has_attention = false;
        bool has_upsampler = false;
        int  C_out = 0;
    };

    void load_resnet_(const brodiffusion::safetensors::File& f,
                      const std::string& prefix,
                      int C_in, int C_out, Resnet& r);
    void load_attn_(const brodiffusion::safetensors::File& f,
                    const std::string& prefix, int C, AttnBlock& a);

    void apply_resnet_(const Resnet& r, int H, int W,
                       brotensor::GpuTensor& x, brotensor::GpuTensor& tmp);
    void apply_attn_(const AttnBlock& a, int H, int W,
                     brotensor::GpuTensor& x);
    void apply_conv3x3_(const brotensor::GpuTensor& W,
                        const brotensor::GpuTensor& b,
                        int C_in, int C_out, int H, int W_,
                        int stride, int pad,
                        const brotensor::GpuTensor& in,
                        brotensor::GpuTensor& out);
    void apply_downsample_(const SampleConv& d, int C, int H, int W,
                           brotensor::GpuTensor& in,
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

    // Scratch buffers reused across calls.
    brotensor::GpuTensor x_, y_;
    brotensor::GpuTensor freq_emb_, temb_a_, temb_b_, temb_silu_, temb_proj_;
    brotensor::GpuTensor cat_buf_;
    brotensor::GpuTensor pad_buf_;
    brotensor::GpuTensor gn_, seq_, q_, k_, v_, attn_out_, proj_, proj_nchw_;
};

}  // namespace brodiffusion::ddpm_unet

namespace brodiffusion {
// Asymmetric zero-pad helper used by ddpm_unet's downsamplers. Copies an
// FP16 NCHW activation `in` of shape (N, C, H, W) into `out` of shape
// (N, C, H+1, W+1), filling the right column and bottom row with zero.
// Both tensors are FP16; `out` is resized as needed. Defined in
// src/ddpm_pad.cu.
void pad_right_bottom_zero_fp16(const brotensor::GpuTensor& in,
                                int N, int C, int H, int W,
                                brotensor::GpuTensor& out);
}  // namespace brodiffusion
