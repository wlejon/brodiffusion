#pragma once

// Qwen-Image VAE (diffusers AutoencoderKLQwenImage) — the VAE Krea 2 decodes
// latents with. Architecturally a Wan-video-style causal-Conv3d VAE (RMSNorm
// resblocks, causal temporal padding, optional per-stage temporal down/up-
// sampling) — but Krea 2 only ever runs it at num_frames=1 (a single image,
// never a video clip), so this implementation specializes the whole graph to
// that case instead of carrying the general video feature-caching machinery
// diffusers needs for streaming multi-frame decode.
//
// ─── The num_frames=1 reduction (read this before touching the .cpp) ──────
//
// diffusers' QwenImageCausalConv3d pads the time axis causally: a kernel-3
// conv gets 2 zero frames prepended (front) and 0 appended (back), then runs
// as an ordinary (now-unpadded) Conv3d. diffusers additionally carries a
// `feat_cache` so a *second* call on the *next* chunk of frames can reuse the
// tail of the previous chunk instead of re-deriving it. Krea 2's decode/encode
// loops only ever run ONE chunk (num_frame == 1), so feat_cache is always
// still at its just-cleared initial state (every slot None) for the one and
// only call that happens — meaning the cache-reuse branches never fire and
// every causal conv sees a plain (uncached) input.
//
// With a single real frame and 2 zero frames prepended, a kernel-3 causal
// conv's temporal contraction collapses to ONE multiply-add: of its 3 taps,
// taps 0 and 1 land on the zero-padding and vanish; only tap 2 (the LAST
// temporal slice of the 3x3x3 filter) ever multiplies real data. So every
// kernel-3 causal Conv3d in this graph (conv_in, conv_out, and every resnet's
// conv1/conv2) reduces EXACTLY to an ordinary 2D 3x3 convolution using only
// the filter's last temporal slice, W[:, :, 2, :, :] — no temporal
// convolution is ever actually computed. Kernel-1 causal convs (conv_shortcut,
// quant_conv, post_quant_conv) have only one temporal tap to begin with, so
// they were already equivalent to a plain 2D 1x1 conv.
//
// QwenImageResample's "upsample3d"/"downsample3d" modes go further: on the
// very first (feat_cache-slot-still-None) invocation they skip their
// `time_conv` entirely (diffusers stashes a "Rep" sentinel / clones the
// current frame for a *future* call that never comes) and fall straight
// through to the plain 2D nearest-upsample+conv / pad+stride-2-conv path.
// Since Krea 2's decode/encode loops never issue that future call, time_conv
// is dead weight for this use case — this loader does not even read its
// tensors, and "upsample3d"/"downsample3d" behave identically to
// "upsample2d"/"downsample2d" (`temperal_downsample` in Config is carried only
// for config-shape fidelity with the JSON; it does not affect the forward
// pass).
//
// Net effect: this module needs no conv3d op at all. Every 3x3x3 checkpoint
// weight is sliced host-side at load time down to its last-tap 2D (Cout,
// Cin*9) filter and run through the ordinary conv2d path.
//
// ─── Normalization: QwenImageRMS_norm, not GroupNorm ───────────────────────
//
// Every norm in this VAE (resnets' norm1/norm2, the attention block's norm,
// norm_out) is QwenImageRMS_norm: F.normalize(x, dim=channel) * sqrt(C) *
// gamma. F.normalize(dim=C) divides by the channel vector's L2 norm; the
// sqrt(C) factor turns that into a divide-by-RMS, so the whole thing is
// exactly brotensor's rms_norm_forward (y = x*gamma/rms(x)) applied per
// spatial position over the channel axis (NCHW -> (H*W, C) sequence ->
// rms_norm_forward -> back). None of these norms carry a bias tensor (the
// model's `bias=False` default, and no `.bias` key ever appears in the
// checkpoint) — only a `.gamma` weight.
//
// ─── Attention: always present in the mid-block, despite empty attn_scales
//
// attn_scales is [] in every known Qwen-Image / Krea 2 config, which means
// the down/up-sampling stages never get an attention layer (their insertion
// is gated on `scale in attn_scales`, impossible for an empty list) — but the
// mid-block's one attention layer is unconditional (QwenImageMidBlock always
// builds `num_layers=1` attention regardless of attn_scales). This loader
// therefore always loads exactly one mid-block attention (encoder and
// decoder each) and never loads a down/up-block attention; a non-empty
// `attn_scales` is rejected at load time as unsupported.
//
// The mid-block attention is a single fused-QKV single-head self-attention
// (to_qkv: 1x1 conv C -> 3C, split by output-channel thirds into Q/K/V; proj:
// 1x1 conv C -> C), non-causal despite its docstring — no mask is applied.
//
// ─── Latent (de)normalization ──────────────────────────────────────────────
//
// Unlike SD's scalar `scaling_factor`, Qwen-Image normalizes per-channel:
// diffusers' pipeline computes, before calling decode(),
//   latent = latent * latents_std + latents_mean          (elementwise per z_dim channel)
// and, before calling encode() on a sampled latent,
//   latent = (latent - latents_mean) / latents_std
// This module bakes the decode-side formula into Decoder::decode() (applied
// before post_quant_conv) and the encode-side formula into Encoder::encode()
// (applied after quant_conv), so callers pass/receive raw pixel-space-scale
// latents exactly like brodiffusion::vae::Decoder/Encoder do for SD1.5.
//
// ─── Architecture summary ──────────────────────────────────────────────────
//
//   Decoder: conv_in (z_dim -> dims[0]) -> mid_block (resnet, attention,
//     resnet, all at dims[0]) -> up_blocks (deep -> shallow; each stage runs
//     num_res_blocks+1 resnets, the first handling the channel transition via
//     a 1x1 conv_shortcut when in!=out, then — except the last stage — a 2x
//     nearest-upsample + 3x3 conv that HALVES the channel count) -> norm_out
//     (RMSNorm) -> SiLU -> conv_out (dims.back() -> input_channels).
//   Encoder mirrors it: conv_in (input_channels -> dims[0]) -> down_blocks
//     (shallow -> deep; each stage runs num_res_blocks resnets then — except
//     the last stage — a stride-2 3x3 conv, SAME channel count) -> mid_block
//     -> norm_out -> SiLU -> conv_out (dims.back() -> 2*z_dim, mean+logvar).
//   quant_conv / post_quant_conv are 1x1 convs living OUTSIDE the encoder./
//   decoder. subtree (siblings, like SD1.5), applied after conv_out / before
//   conv_in respectively.
//
// This is a 16-channel latent VAE (z_dim=16), the same class of numerically
// fragile case as Flux's 16-channel AutoencoderKL — force_upcast defaults on.

#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::vae_qwenimage {

struct Config {
    int base_dim = 96;
    int z_dim = 16;
    std::vector<int> dim_mult = {1, 2, 4, 4};
    int num_res_blocks = 2;
    // Always empty for every known Qwen-Image/Krea-2 checkpoint (see the
    // header comment above): down/up-block attention insertion is gated on
    // `scale in attn_scales`, so a non-empty list would need per-stage
    // attention support this loader does not implement. load_weights()
    // throws if this is non-empty.
    std::vector<float> attn_scales = {};
    // Per-transition temporal down/upsample flag (encoder order). Carried for
    // 1:1 config fidelity; does not affect the forward pass — see the
    // num_frames=1 reduction note above.
    std::vector<bool> temperal_downsample = {false, true, true};
    float dropout = 0.0f;   // eval-mode dropout is a no-op; unused.
    int input_channels = 3;
    std::vector<float> latents_mean = {
        -0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
         0.4134f, -0.0715f,  0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
    std::vector<float> latents_std = {
        2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
        3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};
    // See vae.h's DecoderConfig::force_upcast: 16-channel VAEs are the known
    // FP16-fragile class in this codebase (the Flux VAE NaNs every pixel at
    // FP16). Default on; the arithmetic dtype becomes BF16 on a CUDA backend
    // (FP32 stays the CPU backend's only dtype either way).
    bool force_upcast = true;
};

class Decoder {
public:
    explicit Decoder(const Config& cfg);
    ~Decoder();

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    Decoder(Decoder&&) noexcept = default;
    Decoder& operator=(Decoder&&) noexcept = default;

    // Load weights from a safetensors file. Default prefix matches a
    // standalone diffusers AutoencoderKLQwenImage export.
    //
    // Expected tensors (F16/F32/BF16 source; loaded at the module's
    // arithmetic dtype). `{p}` = prefix, default "" (post_quant_conv and
    // conv_in etc. live directly at top level plus "decoder." for the body):
    //   {p}post_quant_conv.{weight,bias}      Conv3d 1x1x1 (z_dim,z_dim,1,1,1)
    //   {p}decoder.conv_in.{weight,bias}      Conv3d 3x3x3 (dims[0],z_dim,3,3,3)
    //   {p}decoder.mid_block.resnets.{0,1}.*  see Resnet below (dims[0])
    //   {p}decoder.mid_block.attentions.0.{norm.gamma,to_qkv.{weight,bias},
    //                                        proj.{weight,bias}}
    //   {p}decoder.up_blocks.{i}.resnets.{j}.*      j in [0, num_res_blocks]
    //   {p}decoder.up_blocks.{i}.upsamplers.0.resample.1.{weight,bias}
    //                                          Conv2d 3x3 (C_out/2, C_out, 3,3)
    //                                          for i in [0, num_stages-1)
    //   {p}decoder.norm_out.gamma             (dims.back(),)
    //   {p}decoder.conv_out.{weight,bias}     Conv3d 3x3x3 (input_channels,
    //                                          dims.back(),3,3,3)
    //
    // Resnet weights at each "...resnets.{j}.":
    //   norm1.gamma  (C_in,)                 no bias tensor (bias=False)
    //   conv1.{weight,bias}  Conv3d 3x3x3 (C_out,C_in,3,3,3)
    //   norm2.gamma  (C_out,)
    //   conv2.{weight,bias}  Conv3d 3x3x3 (C_out,C_out,3,3,3)
    //   conv_shortcut.{weight,bias}  Conv3d 1x1x1 (C_out,C_in,1,1,1) — only
    //                                when C_in != C_out
    //
    // Throws std::runtime_error on a missing tensor, a shape mismatch, a
    // source dtype that isn't F16/F32/BF16, or a non-empty
    // Config::attn_scales.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");

    // Decode latent -> image.
    //   latent: (1, z_dim*H_lat*W_lat), NCHW with N=1. Raw (pipeline-scale)
    //           latent — the per-channel latents_mean/latents_std
    //           denormalization is applied internally.
    //   out:    (1, input_channels * 8*H_lat * 8*W_lat), resized as needed.
    // Caller is responsible for sync_all() before reading.
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
        brotensor::Tensor qkv_W, qkv_b;   // (3C,C), (3C,1) — split by row-third
        brotensor::Tensor proj_W, proj_b;
        int C = 0;
    };
    struct UpBlock {
        std::vector<Resnet> resnets;
        brotensor::Tensor up_W, up_b;   // (C_out/2, C_out*9), (C_out/2,1)
        bool has_upsampler = false;
        int  C_out = 0;
    };

    void load_resnet_(const brotensor::safetensors::File& f,
                      const std::string& prefix,
                      int C_in, int C_out, Resnet& r);
    void apply_rmsnorm_(const brotensor::Tensor& gamma, int C, int H, int W,
                       const brotensor::Tensor& x, brotensor::Tensor& out);
    void apply_resnet_(const Resnet& r, int H, int W, brotensor::Tensor& x);
    void apply_attention_(const Attention& a, int H, int W, brotensor::Tensor& x);
    void apply_upsample_(const UpBlock& u, int H, int W, brotensor::Tensor& x);

    Config cfg_;
    brotensor::Dtype arith_dtype_ = brotensor::Dtype::FP32;

    brotensor::Tensor post_quant_W_, post_quant_b_;
    brotensor::Tensor conv_in_W_, conv_in_b_;
    Resnet    mid_res0_, mid_res1_;
    Attention mid_attn_;
    std::vector<UpBlock> up_blocks_;
    brotensor::Tensor norm_out_g_;
    brotensor::Tensor conv_out_W_, conv_out_b_;

    // Scratch (re-used across decode calls).
    brotensor::Tensor x_, y_, h_, n1_, n2_;
    brotensor::Tensor seq_, seq2_, up_t_;
};

// Encoder mirrors Decoder. See the header comment above for the flat
// down_blocks checkpoint layout (unlike the decoder's per-stage up_blocks.{i}
// grouping, diffusers' QwenImageEncoder3d stores every resnet AND every
// downsampler as one running flat index — this loader reconstructs the
// per-stage grouping from that flat index at load time).
class Encoder {
public:
    explicit Encoder(const Config& cfg);
    ~Encoder();

    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;
    Encoder(Encoder&&) noexcept = default;
    Encoder& operator=(Encoder&&) noexcept = default;

    // Load weights. `{p}` = prefix, default "".
    //   {p}encoder.conv_in.{weight,bias}      Conv3d 3x3x3 (dims[0],in_ch,3,3,3)
    //   {p}encoder.down_blocks.{k}...          flat index k — see .cpp
    //   {p}encoder.mid_block.*                 as Decoder
    //   {p}encoder.norm_out.gamma / conv_out.{weight,bias}
    //   {p}quant_conv.{weight,bias}            Conv3d 1x1x1 (2*z_dim,2*z_dim,...)
    //
    // Throws std::runtime_error as Decoder::load_weights.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "");

    // Encode image -> latent.
    //   image: (1, input_channels*H*W) in NCHW at the pipeline compute dtype,
    //          values in [-1,1]. H, W must be positive multiples of the
    //          total spatial downsample (8 for the default dim_mult).
    //   eps:   if non-null, sample = mean + exp(0.5*logvar) * eps (shape
    //          (1, z_dim*H/8*W/8), same dtype as image); if null, deterministic
    //          (sample = mean).
    //   out:   (1, z_dim*H/8*W/8) = (sample - latents_mean) / latents_std.
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
        brotensor::Tensor down_W, down_b;   // (C,C*9), (C,1) stride-2
        bool has_downsampler = false;
        int  C_out = 0;
    };

    void load_resnet_(const brotensor::safetensors::File& f,
                      const std::string& prefix,
                      int C_in, int C_out, Resnet& r);
    void apply_rmsnorm_(const brotensor::Tensor& gamma, int C, int H, int W,
                       const brotensor::Tensor& x, brotensor::Tensor& out);
    void apply_resnet_(const Resnet& r, int H, int W, brotensor::Tensor& x);
    void apply_attention_(const Attention& a, int H, int W, brotensor::Tensor& x);
    void apply_downsample_(const DownBlock& d, int H, int W, brotensor::Tensor& x);

    Config cfg_;
    brotensor::Dtype arith_dtype_ = brotensor::Dtype::FP32;

    brotensor::Tensor conv_in_W_, conv_in_b_;
    std::vector<DownBlock> down_blocks_;
    Resnet    mid_res0_, mid_res1_;
    Attention mid_attn_;
    brotensor::Tensor norm_out_g_;
    brotensor::Tensor conv_out_W_, conv_out_b_;
    brotensor::Tensor quant_W_, quant_b_;

    // Scratch.
    brotensor::Tensor x_, y_, h_, n1_, n2_, pad_;
    brotensor::Tensor seq_, seq2_;
    brotensor::Tensor moments_, logvar_;
};

}  // namespace brodiffusion::vae_qwenimage
