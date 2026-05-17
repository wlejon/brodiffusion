#pragma once

// VAE decoder for SD1.5 (AutoencoderKL, decoder branch only).
//
// Inference-only, FP16 throughout. Takes a latent (1, 4, H/8, W/8) and
// produces a raw image tensor (1, 3, H, W) — the caller is responsible for
// any post-processing (clamp to [-1, 1], rescale to uint8, etc.).
//
// Architecture follows Hugging Face's diffusers AutoencoderKL decoder:
//   conv_in: in_channels (4) -> block_out_channels.back() (512)
//   mid_block:
//     resnet -> attention -> resnet  (all at mid_channels)
//   up_blocks[i] (in reverse channel order, so block_out_channels.back() first):
//     layers_per_block + 1 resnets, where the first resnet handles the
//     C_prev -> C_i channel transition via a 1x1 conv shortcut.
//     If i < num_blocks - 1: upsample-2x-nearest -> 3x3 conv (same channels)
//   conv_norm_out (GroupNorm) -> SiLU -> conv_out: block_out_channels.front()
//   (128) -> out_channels (3).

#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brodiffusion::safetensors { class File; }

namespace brodiffusion::vae {

struct DecoderConfig {
    int in_channels   = 4;
    int out_channels  = 3;
    std::vector<int> block_out_channels = {128, 256, 512, 512};
    int layers_per_block = 2;
    int norm_num_groups  = 32;
    // SD applies `latent = latent / scaling_factor` before decode. Set to
    // 1.0f to disable (e.g. for unit tests with synthetic data).
    float scaling_factor = 0.18215f;
    float eps            = 1e-6f;
    int num_attention_heads = 1;   // SD VAE mid-block is single-head.
};

class Decoder {
public:
    explicit Decoder(const DecoderConfig& cfg);
    ~Decoder();

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    Decoder(Decoder&&) noexcept = default;
    Decoder& operator=(Decoder&&) noexcept = default;

    // Load weights from a safetensors file. Default prefix matches a
    // standalone diffusers VAE export; SD1.5 full checkpoints typically use
    // "first_stage_model.decoder." — pass that explicitly.
    //
    // Required tensors (all FP16):
    //   {prefix}conv_in.{weight, bias}            Conv2d 3x3 (C_mid, in_ch, 3, 3)
    //   {prefix}mid_block.resnets.{0,1}.{...}     see ResnetWeights
    //   {prefix}mid_block.attentions.0.group_norm.{weight, bias}    (C_mid,)
    //   {prefix}mid_block.attentions.0.query.{weight, bias}         (C_mid, C_mid)
    //   {prefix}mid_block.attentions.0.key.{weight, bias}
    //   {prefix}mid_block.attentions.0.value.{weight, bias}
    //   {prefix}mid_block.attentions.0.proj_attn.{weight, bias}
    //   {prefix}up_blocks.{i}.resnets.{j}.{...}   for each (i, j)
    //   {prefix}up_blocks.{i}.upsamplers.0.conv.{weight, bias}
    //                                             for i in [0, num_blocks-1)
    //   {prefix}conv_norm_out.{weight, bias}      (C_first,)
    //   {prefix}conv_out.{weight, bias}           Conv2d 3x3 (out_ch, C_first, 3, 3)
    //
    // Resnet weights at each (...).resnets.{j}:
    //   norm1.{weight,bias}  (C_in,)
    //   conv1.{weight,bias}  Conv2d 3x3 (C_out, C_in, 3, 3)
    //   norm2.{weight,bias}  (C_out,)
    //   conv2.{weight,bias}  Conv2d 3x3 (C_out, C_out, 3, 3)
    //   conv_shortcut.{weight,bias}  Conv2d 1x1 (C_out, C_in, 1, 1) — only when C_in != C_out
    //
    // Throws std::runtime_error on missing tensor, shape mismatch, or wrong
    // dtype (everything must be FP16).
    void load_weights(const brodiffusion::safetensors::File& f,
                      const std::string& prefix = "decoder.");

    // Decode latent → image.
    //   latent: (1, in_channels * H_lat * W_lat) FP16, NCHW with N=1.
    //   out:    (1, out_channels * 8*H_lat * 8*W_lat) FP16, resized as needed.
    // Caller is responsible for cuda_sync() before reading.
    void decode(const brotensor::GpuTensor& latent,
                int H_lat, int W_lat,
                brotensor::GpuTensor& out);

    const DecoderConfig& config() const { return cfg_; }

private:
    struct Resnet {
        brotensor::GpuTensor norm1_g, norm1_b;
        brotensor::GpuTensor conv1_W, conv1_b;
        brotensor::GpuTensor norm2_g, norm2_b;
        brotensor::GpuTensor conv2_W, conv2_b;
        brotensor::GpuTensor short_W, short_b;
        bool has_shortcut = false;
        int  C_in = 0, C_out = 0;
    };
    struct Attention {
        brotensor::GpuTensor gn_g, gn_b;
        brotensor::GpuTensor Wq, bq, Wk, bk, Wv, bv, Wo, bo;
        int C = 0;
    };
    struct UpsampleConv {
        brotensor::GpuTensor W, b;  // 3x3 same-channel
    };
    struct UpBlock {
        std::vector<Resnet> resnets;
        UpsampleConv        upsampler;
        bool has_upsampler = false;
        int  C_out = 0;
    };

    void load_resnet_(const brodiffusion::safetensors::File& f,
                      const std::string& prefix,
                      int C_in, int C_out, Resnet& r);
    void apply_resnet_(const Resnet& r, int H, int W,
                       brotensor::GpuTensor& x, brotensor::GpuTensor& tmp);
    void apply_attention_(const Attention& a, int H, int W,
                          brotensor::GpuTensor& x);
    void apply_upsample_(const UpsampleConv& u, int C, int H, int W,
                         brotensor::GpuTensor& x, brotensor::GpuTensor& tmp);
    void apply_conv3x3_(const brotensor::GpuTensor& W,
                        const brotensor::GpuTensor& b,
                        int C_in, int C_out, int H, int W_,
                        brotensor::GpuTensor& x_in, brotensor::GpuTensor& x_out);

    DecoderConfig cfg_;

    // Weights.
    brotensor::GpuTensor conv_in_W_,  conv_in_b_;
    Resnet               mid_res0_,   mid_res1_;
    Attention            mid_attn_;
    std::vector<UpBlock> up_blocks_;
    brotensor::GpuTensor norm_out_g_, norm_out_b_;
    brotensor::GpuTensor conv_out_W_, conv_out_b_;

    // Scratch (re-used across decode calls).
    brotensor::GpuTensor x_, y_;      // ping-pong residual stream
    brotensor::GpuTensor seq_, Q_, K_, V_, attn_seq_, proj_seq_, attn_nchw_;
    brotensor::GpuTensor ln_nchw_;
};

}  // namespace brodiffusion::vae
