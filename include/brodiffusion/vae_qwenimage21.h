#pragma once

// Qwen-Image 2.1 VAE (diffusers AutoencoderKLQwenImage21): the Wan 2.2-style
// *residual* image VAE with a 16x spatial stride and a 64-channel latent.
// Structurally it is vae_qwenimage.h's graph with three changes, all of
// which this module specialises to the single-image (num_frames=1) case
// exactly as that header does:
//
//   1. The convolutions are already 2D. QwenImage21CausalConv3d subclasses
//      nn.Conv2d and squeezes the (always length-1) frame axis away, so the
//      checkpoint stores plain [Cout,Cin,kh,kw] filters — no last-tap slicing.
//   2. Every down/up stage carries a parameter-free residual shortcut
//      (is_residual=true): the encoder adds AvgDown3D(x) to its main path, the
//      decoder adds DupUp3D(x). With one frame these reduce to channel
//      gathers around a 2x2 average pool / nearest 2x upsample:
//        AvgDown3D  ft=1,fs=2 (in==out)  : plain 2x2 avg-pool
//        AvgDown3D  ft=2,fs=2 (out=2*in) : the zero frame padded at the FRONT
//                                          of T lands on the even output
//                                          channels; odd channel o is the
//                                          2x2 avg-pool of input channel o/2
//        AvgDown3D  ft=1,fs=1 (in==out)  : identity (last encoder stage)
//        DupUp3D    ft=2,fs=2 (in==out)  : nearest 2x (first_chunk keeps the
//                                          odd frame, whose 4 sub-slots all
//                                          come from the same input channel)
//        DupUp3D    ft=2,fs=2 (in=2*out) : nearest 2x of the ODD input channels
//        DupUp3D    ft=1,fs=2 (in=2*out) : repeat_interleave(2) + pixel_shuffle
//                                          == brotensor pixel_shuffle_upsample
//      The decoder's up-stage channel count is NOT halved (is_residual), and
//      its Resample conv keeps out_dim -> out_dim.
//   3. Resample "upsample3d"/"downsample3d" own a `time_conv` that the
//      single-frame feat_cache protocol never executes (first call stashes a
//      sentinel and skips it). Those tensors are accepted and ignored.
//
// Norms are QwenImage21RMS_norm (F.normalize over C * sqrt(C) * gamma ==
// rms_norm_forward with eps 1e-12 on the (H*W, C) sequence layout), the
// mid-block attention is the same single-head fused-QKV block as v1, and the
// decoder clamps its output to [-1, 1] (diffusers does this inside
// AutoencoderKLQwenImage21._decode, so decode() here clamps too).
//
// The VAE is 4-channel (RGBA) on both ends: encode() takes (1, 4*H*W) and
// decode() returns (1, 4*H*W). The pipeline drops alpha for RGB output and
// pads RGB inputs with an opaque alpha channel.
//
// Latent (de)normalisation follows vae_qwenimage.h: decode() applies
// latent*std+mean per channel before post_quant_conv; encode() returns
// (z - mean)/std.

#include "brodiffusion/detail/jit_fusion.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::vae_qwenimage21 {

struct Config {
    int base_dim = 96;          // encoder channel base
    int decoder_base_dim = 144; // decoder channel base
    int z_dim = 64;
    std::vector<int> dim_mult = {1, 2, 4, 8, 8};
    int num_res_blocks = 2;
    std::vector<float> attn_scales = {};   // must stay empty (as v1)
    // Per-transition temporal flag (encoder order). Selects which shortcut
    // reduction applies (see header comment); the time_conv weights it
    // implies are never executed for one frame.
    std::vector<bool> temperal_downsample = {false, true, true, true};
    float dropout = 0.0f;
    int in_channels = 4;
    int out_channels = 4;
    std::vector<float> latents_mean = {
         0.5126f,  0.7721f, -0.0631f,  1.3506f, -0.7855f, -2.1025f, -0.3458f,  1.3722f,
         1.8873f, -1.7177f, -0.6510f,  0.2732f,  0.7562f, -0.6163f, -1.0277f,  3.8363f,
         2.0210f,  0.0472f,  0.9320f,  2.0087f,  2.4954f, -0.1391f, -1.4249f,  1.8464f,
        -0.5236f,  1.2826f,  3.7046f, -1.3035f,  2.7286f, -1.4518f, -1.9036f, -1.9955f,
        -0.0342f, -1.0265f, -0.7636f,  3.0555f,  0.0746f, -3.0751f, -0.1076f,  1.7376f,
        -1.0914f, -1.9435f, -0.2784f, -1.3680f,  0.4809f, -0.4433f,  0.3764f,  0.5729f,
        -2.0595f,  1.0960f, -1.3260f, -2.0211f, -5.0179f,  0.5275f,  4.0162f,  1.8505f,
         0.3026f,  1.9373f,  1.4937f,  0.2632f,  0.5547f, -1.7121f, -0.1562f,  0.0304f};
    std::vector<float> latents_std = {
        3.2001f, 3.2936f, 3.4321f, 3.0091f, 3.1061f, 4.0379f, 4.0705f, 3.7910f,
        3.0785f, 3.6500f, 3.9308f, 3.0904f, 2.8778f, 3.7675f, 3.7320f, 5.0756f,
        3.2864f, 4.0397f, 3.1317f, 4.0443f, 2.9249f, 3.9454f, 3.0988f, 4.2489f,
        3.4896f, 3.8513f, 3.9323f, 3.4719f, 3.7498f, 4.2830f, 3.5694f, 4.2467f,
        3.9037f, 3.2947f, 5.0770f, 3.5075f, 3.2700f, 3.4767f, 2.8063f, 5.1125f,
        3.5327f, 4.7833f, 3.1286f, 4.1819f, 3.8527f, 3.8312f, 3.5605f, 4.3875f,
        3.9624f, 4.0168f, 3.5643f, 4.0550f, 5.5614f, 4.2963f, 4.4080f, 3.4959f,
        3.8747f, 3.7608f, 3.5735f, 3.1490f, 3.7662f, 3.6746f, 3.4563f, 3.8161f};
    // 64-channel latent: same FP16-fragile class as the 16-channel VAEs.
    // BF16 arithmetic on CUDA when set.
    bool force_upcast = true;

    // Spatial stride between image and latent (16 for the default dim_mult).
    int spatial_scale() const { return 1 << (static_cast<int>(dim_mult.size()) - 1); }
};

class Decoder {
public:
    explicit Decoder(const Config& cfg);
    ~Decoder();

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    Decoder(Decoder&&) noexcept = default;
    Decoder& operator=(Decoder&&) noexcept = default;

    // Expected tensors under `{p}` (default ""), all plain 2D convs:
    //   {p}post_quant_conv.{weight,bias}         (z_dim,z_dim,1,1)
    //   {p}decoder.conv_in.{weight,bias}         (dims[0],z_dim,3,3)
    //   {p}decoder.mid_block.resnets.{0,1}.*, mid_block.attentions.0.*
    //   {p}decoder.up_blocks.{i}.resnets.{j}.*   j in [0, num_res_blocks]
    //   {p}decoder.up_blocks.{i}.upsampler.resample.1.{weight,bias}
    //                                            (out,out,3,3) for i < last
    //   {p}decoder.up_blocks.{i}.upsampler.time_conv.*   ignored
    //   {p}decoder.norm_out.gamma, decoder.conv_out.{weight,bias}
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");

    // latent: (1, z_dim*H_lat*W_lat) NCHW, raw pipeline-scale (the per-channel
    //         denormalisation is applied internally).
    // out:    (1, out_channels * S*H_lat * S*W_lat), S = spatial_scale(),
    //         clamped to [-1, 1]. Caller syncs before reading.
    void decode(const brotensor::Tensor& latent,
                int H_lat, int W_lat,
                brotensor::Tensor& out);

    const Config& config() const { return cfg_; }

private:
    struct Resnet {
        brotensor::Tensor norm1_g;
        brotensor::Tensor conv1_W, conv1_b;
        brotensor::Tensor norm2_g;
        brotensor::Tensor conv2_W, conv2_b;
        brotensor::Tensor short_W, short_b;
        bool has_shortcut = false;
        int  C_in = 0, C_out = 0;
    };
    struct Attention {
        brotensor::Tensor norm_g;
        brotensor::Tensor qkv_W, qkv_b;
        brotensor::Tensor proj_W, proj_b;
        int C = 0;
    };
    struct UpBlock {
        std::vector<Resnet> resnets;
        brotensor::Tensor up_W, up_b;   // (C_out, C_out*9), (C_out,1)
        bool has_upsampler = false;     // == has residual shortcut
        bool temporal = false;          // DupUp3D factor_t == 2
        int  C_in = 0, C_out = 0;
    };

    void load_resnet_(const brotensor::safetensors::File& f,
                      const std::string& prefix,
                      int C_in, int C_out, Resnet& r);
    void apply_rmsnorm_(const brotensor::Tensor& gamma, int C, int H, int W,
                       const brotensor::Tensor& x, brotensor::Tensor& out);
    // RMSNorm immediately followed by SiLU — every norm in the graph except
    // the attention block's. Traced as one kernel when the JIT is on, which
    // removes a whole read and write of the feature map; `site` names the
    // seam so each keeps its own compiled binding per (gamma, shape).
    void apply_rmsnorm_silu_(detail::JitSite& site, const brotensor::Tensor& gamma,
                             int C, int H, int W, const brotensor::Tensor& x,
                             brotensor::Tensor& out);
    void apply_resnet_(const Resnet& r, int H, int W, brotensor::Tensor& x);
    void apply_attention_(const Attention& a, int H, int W, brotensor::Tensor& x);
    void apply_upsample_(const UpBlock& u, int H, int W, brotensor::Tensor& x);
    // DupUp3D(x_copy) at (H,W) -> (2H,2W) into `out`.
    void apply_dup_shortcut_(const UpBlock& u, int H, int W,
                             const brotensor::Tensor& x_copy, brotensor::Tensor& out);

    Config cfg_;
    brotensor::Dtype arith_dtype_ = brotensor::Dtype::FP32;

    brotensor::Tensor post_quant_W_, post_quant_b_;
    brotensor::Tensor conv_in_W_, conv_in_b_;
    Resnet    mid_res0_, mid_res1_;
    Attention mid_attn_;
    std::vector<UpBlock> up_blocks_;
    brotensor::Tensor norm_out_g_;
    brotensor::Tensor conv_out_W_, conv_out_b_;

    brotensor::Tensor x_, y_, h_, n1_, n2_;
    brotensor::Tensor seq_, seq2_, up_t_, x_copy_, short_, gather_;

    // One site per seam rather than one for the whole decoder: a site
    // remembers a bounded number of bindings, and each of these sees one per
    // resnet (a distinct gamma) per feature-map size.
    detail::JitSite jit_norm1_{"qi21vae.dec.resnet.norm1_silu"};
    detail::JitSite jit_norm2_{"qi21vae.dec.resnet.norm2_silu"};
    detail::JitSite jit_norm_out_{"qi21vae.dec.norm_out_silu"};
};

class Encoder {
public:
    explicit Encoder(const Config& cfg);
    ~Encoder();

    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;
    Encoder(Encoder&&) noexcept = default;
    Encoder& operator=(Encoder&&) noexcept = default;

    // Expected tensors under `{p}`:
    //   {p}encoder.conv_in.{weight,bias}                  (dims[0],in_ch,3,3)
    //   {p}encoder.down_blocks.{i}.resnets.{j}.*          j in [0,num_res_blocks)
    //   {p}encoder.down_blocks.{i}.downsampler.resample.1.{weight,bias}
    //                                                     (out,out,3,3), i < last
    //   {p}encoder.down_blocks.{i}.downsampler.time_conv.*   ignored
    //   {p}encoder.mid_block.*, encoder.norm_out.gamma, encoder.conv_out.*
    //   {p}quant_conv.{weight,bias}                       (2z,2z,1,1)
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");

    // image: (1, in_channels*H*W) NCHW in [-1,1] (RGBA); H, W multiples of
    //        spatial_scale().
    // eps:   if non-null, sample = mean + exp(0.5*logvar)*eps, shape
    //        (1, z_dim*H/S*W/S); null -> sample = mean (pipeline "argmax").
    // out:   (1, z_dim*H/S*W/S) = (sample - latents_mean)/latents_std.
    void encode(const brotensor::Tensor& image,
                int H, int W,
                const brotensor::Tensor* eps,
                brotensor::Tensor& out);

    const Config& config() const { return cfg_; }

private:
    struct Resnet {
        brotensor::Tensor norm1_g;
        brotensor::Tensor conv1_W, conv1_b;
        brotensor::Tensor norm2_g;
        brotensor::Tensor conv2_W, conv2_b;
        brotensor::Tensor short_W, short_b;
        bool has_shortcut = false;
        int  C_in = 0, C_out = 0;
    };
    struct Attention {
        brotensor::Tensor norm_g;
        brotensor::Tensor qkv_W, qkv_b;
        brotensor::Tensor proj_W, proj_b;
        int C = 0;
    };
    struct DownBlock {
        std::vector<Resnet> resnets;
        brotensor::Tensor down_W, down_b;   // (C_out,C_out*9), (C_out,1) stride-2
        bool has_downsampler = false;
        bool temporal = false;              // AvgDown3D factor_t == 2
        int  C_in = 0, C_out = 0;
    };

    void load_resnet_(const brotensor::safetensors::File& f,
                      const std::string& prefix,
                      int C_in, int C_out, Resnet& r);
    void apply_rmsnorm_(const brotensor::Tensor& gamma, int C, int H, int W,
                       const brotensor::Tensor& x, brotensor::Tensor& out);
    // See the decoder's overload: RMSNorm + SiLU as one traced kernel.
    void apply_rmsnorm_silu_(detail::JitSite& site, const brotensor::Tensor& gamma,
                             int C, int H, int W, const brotensor::Tensor& x,
                             brotensor::Tensor& out);
    void apply_resnet_(const Resnet& r, int H, int W, brotensor::Tensor& x);
    void apply_attention_(const Attention& a, int H, int W, brotensor::Tensor& x);
    void apply_downsample_(const DownBlock& d, int H, int W, brotensor::Tensor& x);
    // AvgDown3D(x_copy) at (H,W) into `out` (H/2,W/2 when the stage
    // downsamples; identity otherwise).
    void apply_avg_shortcut_(const DownBlock& d, int H, int W,
                             const brotensor::Tensor& x_copy, brotensor::Tensor& out);

    Config cfg_;
    brotensor::Dtype arith_dtype_ = brotensor::Dtype::FP32;

    brotensor::Tensor conv_in_W_, conv_in_b_;
    std::vector<DownBlock> down_blocks_;
    Resnet    mid_res0_, mid_res1_;
    Attention mid_attn_;
    brotensor::Tensor norm_out_g_;
    brotensor::Tensor conv_out_W_, conv_out_b_;
    brotensor::Tensor quant_W_, quant_b_;

    brotensor::Tensor x_, y_, h_, n1_, n2_, pad_;
    brotensor::Tensor seq_, seq2_, x_copy_, short_, pool_;
    brotensor::Tensor moments_, logvar_;

    detail::JitSite jit_norm1_{"qi21vae.enc.resnet.norm1_silu"};
    detail::JitSite jit_norm2_{"qi21vae.enc.resnet.norm2_silu"};
    detail::JitSite jit_norm_out_{"qi21vae.enc.norm_out_silu"};
};

}  // namespace brodiffusion::vae_qwenimage21
