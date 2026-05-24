#pragma once

// VAE decoder for SD1.5 (AutoencoderKL, decoder branch only).
//
// Inference-only. Runs on whichever backend brotensor resolves at runtime —
// CPU by default, CUDA when available — at that backend's compute dtype (FP32
// on CPU, FP16 on a GPU). Takes a latent (1, 4, H/8, W/8) and produces a raw
// image tensor (1, 3, H, W) — the caller is responsible for any
// post-processing (clamp to [-1, 1], rescale to uint8, etc.).
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

namespace brotensor::safetensors { class File; }

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
    // Some VAEs (e.g. Flux) shift the latent after the scaling division:
    //   latent = latent / scaling_factor + shift_factor
    // SD1.5's VAE uses 0.0 (no shift).
    float shift_factor = 0.0f;
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
    // Required tensors (F16 or F32 source; loaded at the compute dtype):
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
    // Throws std::runtime_error on missing tensor, shape mismatch, or a
    // source dtype that is neither F16 nor F32.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "decoder.");

    // Decode latent → image. Tensors carry the compute dtype (FP32 on CPU,
    // FP16 on a GPU backend).
    //   latent: (1, in_channels * H_lat * W_lat), NCHW with N=1.
    //   out:    (1, out_channels * 8*H_lat * 8*W_lat), resized as needed.
    // Caller is responsible for sync_all() before reading.
    void decode(const brotensor::Tensor& latent,
                int H_lat, int W_lat,
                brotensor::Tensor& out);

    const DecoderConfig& config() const { return cfg_; }

private:
    struct Resnet {
        brotensor::Tensor norm1_g, norm1_b;
        brotensor::Tensor conv1_W, conv1_b;
        brotensor::Tensor norm2_g, norm2_b;
        brotensor::Tensor conv2_W, conv2_b;
        brotensor::Tensor short_W, short_b;
        bool has_shortcut = false;
        int  C_in = 0, C_out = 0;
    };
    struct Attention {
        brotensor::Tensor gn_g, gn_b;
        brotensor::Tensor Wq, bq, Wk, bk, Wv, bv, Wo, bo;
        int C = 0;
    };
    struct UpsampleConv {
        brotensor::Tensor W, b;  // 3x3 same-channel
    };
    struct UpBlock {
        std::vector<Resnet> resnets;
        UpsampleConv        upsampler;
        bool has_upsampler = false;
        int  C_out = 0;
    };

    void load_resnet_(const brotensor::safetensors::File& f,
                      const std::string& prefix,
                      int C_in, int C_out, Resnet& r);
    void apply_resnet_(const Resnet& r, int H, int W,
                       brotensor::Tensor& x, brotensor::Tensor& tmp);
    void apply_attention_(const Attention& a, int H, int W,
                          brotensor::Tensor& x);
    void apply_upsample_(const UpsampleConv& u, int C, int H, int W,
                         brotensor::Tensor& x, brotensor::Tensor& tmp);
    void apply_conv3x3_(const brotensor::Tensor& W,
                        const brotensor::Tensor& b,
                        int C_in, int C_out, int H, int W_,
                        brotensor::Tensor& x_in, brotensor::Tensor& x_out);

    DecoderConfig cfg_;

    // Weights.
    // post_quant_conv lives ABOVE the "decoder." prefix in diffusers' VAE; it's
    // a 1x1 conv (in_channels -> in_channels) applied between the latent scale
    // and the decoder proper. Optional: legacy/encoder-less checkpoints may
    // omit it (has_post_quant_conv_ == false).
    brotensor::Tensor post_quant_W_, post_quant_b_;
    bool                 has_post_quant_conv_ = false;
    brotensor::Tensor conv_in_W_,  conv_in_b_;
    Resnet               mid_res0_,   mid_res1_;
    Attention            mid_attn_;
    std::vector<UpBlock> up_blocks_;
    brotensor::Tensor norm_out_g_, norm_out_b_;
    brotensor::Tensor conv_out_W_, conv_out_b_;

    // Scratch (re-used across decode calls).
    brotensor::Tensor x_, y_;      // ping-pong residual stream
    brotensor::Tensor seq_, proj_seq_, attn_nchw_;
    brotensor::Tensor ln_nchw_;
};

// Encoder mirrors Decoder. Phase A: deterministic-or-sampled forward only.
//
// Architecture (diffusers AutoencoderKL encoder):
//   conv_in: out_channels (3) -> block_out_channels.front() (128)
//   down_blocks[i] for i in [0, num_blocks):
//     layers_per_block resnets at block_out_channels[i] (first resnet handles
//     the C_prev -> C_i channel transition via a 1x1 conv shortcut)
//     if i < num_blocks - 1: stride-2 3x3 conv (same channels)
//   mid_block: resnet -> attention -> resnet at block_out_channels.back()
//   conv_norm_out -> SiLU
//   conv_out: block_out_channels.back() -> 2 * in_channels (mean, logvar)
//   quant_conv (above the "encoder." subtree): 1x1 conv 2*latent -> 2*latent
//
// Field names match DecoderConfig for code-share clarity:
//   in_channels       = latent channel count   (= 4 for SD1.5)
//   out_channels      = RGB channel count       (= 3)
struct EncoderConfig {
    int in_channels   = 4;
    int out_channels  = 3;
    std::vector<int> block_out_channels = {128, 256, 512, 512};
    int layers_per_block = 2;
    int norm_num_groups  = 32;
    // Pipeline applies   latent_out = (sample - shift_factor) * scaling_factor
    // — the inverse of Decoder's pre-scaling. Set to 1.0f / 0.0f for tests on
    // synthetic data.
    float scaling_factor = 0.18215f;
    float shift_factor   = 0.0f;
    float eps            = 1e-6f;
    int num_attention_heads = 1;
};

class Encoder {
public:
    explicit Encoder(const EncoderConfig& cfg);
    ~Encoder();

    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;
    Encoder(Encoder&&) noexcept = default;
    Encoder& operator=(Encoder&&) noexcept = default;

    // Load encoder weights. Default prefix matches a standalone diffusers VAE
    // export ("encoder."); SD1.5 full checkpoints use
    // "first_stage_model.encoder." — pass that explicitly. `quant_conv` lives
    // above the "encoder." subtree (e.g. plain "quant_conv.{weight,bias}"
    // for a diffusers VAE, or "first_stage_model.quant_conv.{weight,bias}").
    // Optional: legacy/decoder-only checkpoints may omit it.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "encoder.");

    // Encode image -> latent.
    //   image: (1, out_channels * H * W) in NCHW at the pipeline compute dtype,
    //          values in [-1, 1]. H and W must be positive multiples of 8.
    //   eps:   if non-null, sample = mean + exp(0.5*logvar) * eps  (eps in
    //          standard-Gaussian units, shape (1, in_channels * H/8 * W/8),
    //          same dtype as image). If null, deterministic mode: sample =
    //          mean.
    //   out:   (1, in_channels * H/8 * W/8) — the final latent
    //          = (sample - shift_factor) * scaling_factor.
    void encode(const brotensor::Tensor& image,
                int H, int W,
                const brotensor::Tensor* eps,
                brotensor::Tensor& out);

    const EncoderConfig& config() const { return cfg_; }

private:
    // Internal block structs mirror Decoder. Kept private to Encoder for
    // Phase A — a future refactor can lift them into a shared header once a
    // third user appears.
    struct Resnet {
        brotensor::Tensor norm1_g, norm1_b;
        brotensor::Tensor conv1_W, conv1_b;
        brotensor::Tensor norm2_g, norm2_b;
        brotensor::Tensor conv2_W, conv2_b;
        brotensor::Tensor short_W, short_b;
        bool has_shortcut = false;
        int  C_in = 0, C_out = 0;
    };
    struct Attention {
        brotensor::Tensor gn_g, gn_b;
        brotensor::Tensor Wq, bq, Wk, bk, Wv, bv, Wo, bo;
        int C = 0;
    };
    struct DownsampleConv {
        brotensor::Tensor W, b;  // 3x3 stride-2 same-channel
    };
    struct DownBlock {
        std::vector<Resnet> resnets;
        DownsampleConv      downsampler;
        bool has_downsampler = false;
        int  C_out = 0;
    };

    void load_resnet_(const brotensor::safetensors::File& f,
                      const std::string& prefix,
                      int C_in, int C_out, Resnet& r);
    void apply_resnet_(const Resnet& r, int H, int W,
                       brotensor::Tensor& x, brotensor::Tensor& tmp);
    void apply_attention_(const Attention& a, int H, int W,
                          brotensor::Tensor& x);
    // Diffusers Downsample2D: F.pad(x, (0, 1, 0, 1)) followed by stride-2 3x3
    // conv with pad=0. We mirror that asymmetric pad via pad2d_forward.
    void apply_downsample_(const DownsampleConv& d, int C, int H, int W,
                           brotensor::Tensor& x, brotensor::Tensor& tmp);
    void apply_conv3x3_(const brotensor::Tensor& W,
                        const brotensor::Tensor& b,
                        int C_in, int C_out, int H, int W_,
                        brotensor::Tensor& x_in, brotensor::Tensor& x_out);

    EncoderConfig cfg_;

    // Weights.
    brotensor::Tensor conv_in_W_, conv_in_b_;
    std::vector<DownBlock> down_blocks_;
    Resnet               mid_res0_, mid_res1_;
    Attention            mid_attn_;
    brotensor::Tensor norm_out_g_, norm_out_b_;
    brotensor::Tensor conv_out_W_, conv_out_b_;
    // quant_conv lives ABOVE "encoder." (sibling of post_quant_conv); 1x1 conv
    // (2*in_channels -> 2*in_channels) applied after the encoder body.
    brotensor::Tensor quant_W_, quant_b_;
    bool                 has_quant_conv_ = false;

    // Scratch.
    brotensor::Tensor x_, y_, pad_;
    brotensor::Tensor seq_, proj_seq_, attn_nchw_;
    brotensor::Tensor ln_nchw_;
    brotensor::Tensor moments_;     // (1, 2*C_lat*H_lat*W_lat) post-quant_conv
    brotensor::Tensor logvar_;      // (1, C_lat*H_lat*W_lat) — for exp(0.5*lv)
};

}  // namespace brodiffusion::vae
