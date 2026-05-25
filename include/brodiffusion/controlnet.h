#pragma once

// ControlNet for SD1.5 — a standalone "encoder + mid" UNet half plus a
// control-image conditioning CNN and 13 zero-convolutions whose outputs are
// added as residuals onto the UNet's skip-stack pushes (12 down + 1 mid).
//
// ControlNet runs once per UNet denoising step, sharing the timestep and
// (typically) the cond-branch text context. It does NOT do CFG itself; the
// caller (Pipeline) chooses whether to run it for cond, uncond, or both
// branches. Vanilla SD1.5 only — no LCM cond_proj, no LoRA, no INT8.
//
// Block-level code (ResnetBlock2D, Transformer2D, conv helpers) is reused
// verbatim from brodiffusion::unet::detail (Phase D1's extracted helpers).

#include "brodiffusion/detail/unet_blocks.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::controlnet {

struct ControlNetConfig {
    // Latent input (matches UNet's in_channels).
    int in_channels      = 4;
    // Control image input channels (RGB).
    int control_channels = 3;
    // Same shape ladder as UNet's SD1.5 default.
    std::vector<int> block_out_channels = {320, 640, 1280, 1280};
    int layers_per_block = 2;
    int norm_num_groups  = 32;
    float eps            = 1e-5f;
    // Sinusoidal freq dim (matches block_out_channels[0]).
    int freq_dim         = 320;
    int time_embed_dim   = 1280;
    // CLIP-L cross-attention dim.
    int cross_attention_dim   = 768;
    // diffusers backward-compat quirk: this is actually num_heads.
    int transformer_num_heads = 8;
    // Conditioning embedding channel ladder. The HF SD1.5 ControlNet uses
    // {16, 32, 96, 256} -> block_out_channels[0]=320. Pairs are
    // (same-channel, stride-2-channel-increase).
    std::vector<int> conditioning_embedding_channels = {16, 32, 96, 256};
};

class ControlNet {
public:
    explicit ControlNet(const ControlNetConfig& cfg);
    ~ControlNet();

    ControlNet(const ControlNet&) = delete;
    ControlNet& operator=(const ControlNet&) = delete;
    ControlNet(ControlNet&&) noexcept = default;
    ControlNet& operator=(ControlNet&&) noexcept = default;

    // Load weights from a diffusers ControlNet safetensors export. Empty
    // prefix matches the lllyasviel/sd-controlnet-* layout. Throws on missing
    // tensors, shape mismatches, or unsupported dtype.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");

    // Forward pass. See module-level comment for the high-level contract.
    //   sample:        (1, in_channels * H_lat * W_lat) at compute dtype —
    //                  the same latent the paired UNet step receives.
    //   H_lat, W_lat:  latent spatial dims.
    //   timestep:      continuous timestep (same value the UNet sees).
    //   ctx:           (Lk, cross_attention_dim) CLIP text context.
    //   control_image: (1, control_channels * (H_lat*8) * (W_lat*8)) in
    //                  [-1, 1] at compute dtype.
    //   conditioning_scale: scalar multiplier applied to every output residual
    //                  (matches diffusers' `conditioning_scale`).
    //   down_residuals_out: resized to num_down_residuals() entries; each
    //                  tensor's channel/spatial shape matches the
    //                  corresponding UNet skip activation.
    //   mid_residual_out:   matches the UNet mid-block output shape.
    void forward(const brotensor::Tensor& sample,
                 int H_lat, int W_lat,
                 float timestep,
                 const brotensor::Tensor& ctx,
                 const brotensor::Tensor& control_image,
                 float conditioning_scale,
                 std::vector<brotensor::Tensor>& down_residuals_out,
                 brotensor::Tensor& mid_residual_out);

    int num_down_residuals() const { return num_down_residuals_; }
    const ControlNetConfig& config() const { return cfg_; }

private:
    using Resnet        = brodiffusion::unet::detail::Resnet;
    using Transformer2D = brodiffusion::unet::detail::Transformer2D;
    using AttnFFN       = brodiffusion::unet::detail::AttnFFN;
    using SampleConv    = brodiffusion::unet::detail::SampleConv;
    using DownBlock     = brodiffusion::unet::detail::DownBlock;
    using MidBlock      = brodiffusion::unet::detail::MidBlock;
    using BlockScratch  = brodiffusion::unet::detail::BlockScratch;

    // Conditioning-embedding conv. 3x3 with stride 1 or 2, pad 1, biased.
    struct CondConv {
        brotensor::Tensor W, b;
        int  C_in = 0, C_out = 0;
        int  stride = 1;
    };
    // 1x1 zero-conv. Same-channel: C_out == C_in == C.
    struct ZeroConv {
        brotensor::Tensor W, b;
        int  C = 0;
    };

    ControlNetConfig cfg_;
    int              num_down_residuals_ = 0;

    // Time embedding (mirrors UNet).
    brotensor::Tensor te_l1_W_, te_l1_b_, te_l2_W_, te_l2_b_;

    // Latent input.
    brotensor::Tensor conv_in_W_, conv_in_b_;

    // Conditioning-embedding CNN.
    CondConv             cond_conv_in_;
    std::vector<CondConv> cond_blocks_;
    CondConv             cond_conv_out_;

    // Encoder + mid (same shape as UNet's).
    std::vector<DownBlock> down_blocks_;
    MidBlock               mid_;

    // Output 1x1 zero-convs: one per skip push + one for mid.
    std::vector<ZeroConv> down_zero_convs_;
    ZeroConv              mid_zero_conv_;

    // Scratch.
    BlockScratch     scratch_;
    brotensor::Tensor freq_emb_, temb_a_, temb_;
    brotensor::Tensor x_, y_;
    brotensor::Tensor cond_x_, cond_y_;
};

}  // namespace brodiffusion::controlnet
