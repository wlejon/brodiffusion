#pragma once

// Deep-Compression Autoencoder (AutoencoderDC) decoder — the f32c32 VAE that
// pairs with NVIDIA's Sana DiT.
//
// Inference-only, decoder branch. Runs on whichever backend brotensor resolves
// at runtime — CPU by default (FP32), CUDA when available (FP16). Takes a
// 32-channel latent (1, 32, H/32, W/32) — the DC-AE downsamples by 32x, not
// the 8x of SD/Flux AutoencoderKL — and produces a raw image tensor
// (1, 3, H, W). The caller handles post-processing (clamp to [-1, 1], rescale
// to uint8, etc.).
//
// Architecture (HF diffusers AutoencoderDC, dc-ae-f32c32-sana-1.0):
//   conv_in: latent_channels (32) -> decoder_block_out_channels.back() (1024)
//   up_blocks (in reverse channel order, so block_out_channels.back() first):
//     decoder_layers_per_block[i] blocks, each either a ResBlock or an
//     EfficientViTBlock (linear-attention transformer with multi-scale
//     depthwise-conv QKV aggregation; the qkv_multiscales list gives the extra
//     kernel sizes per stage). Stage transitions use "interpolate" upsampling
//     (pixel-shuffle / nearest) — there is no GroupNorm; norms are RMSNorm and
//     the activation is SiLU.
//   norm_out (RMSNorm) -> SiLU -> conv_out: block_out_channels.front()
//     (128) -> image_channels (3).
//
// The decoder applies `latent = latent / scaling_factor` before conv_in (DC-AE
// has no shift term).

#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::dcae {

struct DecoderConfig {
    int latent_channels = 32;   // decoder input channels (= AutoencoderDC latent_channels)
    int image_channels  = 3;    // decoder output channels (= AutoencoderDC in_channels)

    // Per-stage channel widths, in encoder order (conv_in consumes the LAST,
    // conv_out produces the FIRST). dc-ae-f32c32: {128,256,512,512,1024,1024}.
    std::vector<int> block_out_channels = {128, 256, 512, 512, 1024, 1024};

    // Per-stage block count. dc-ae-f32c32 decoder: {3,3,3,3,3,3}.
    std::vector<int> layers_per_block = {3, 3, 3, 3, 3, 3};

    // Per-stage block kind: false = ResBlock, true = EfficientViTBlock
    // (the linear-attention transformer stage). dc-ae-f32c32: the last three
    // stages are attention.
    std::vector<bool> is_attention = {false, false, false, true, true, true};

    // Per-stage EfficientViTBlock multi-scale aggregation kernel sizes
    // (decoder_qkv_multiscales). Empty for the ResBlock stages; {5} for each
    // attention stage in dc-ae-f32c32.
    std::vector<std::vector<int>> qkv_multiscales = {
        {}, {}, {}, {5}, {5}, {5}};

    int attention_head_dim = 32;     // EfficientViTBlock head width
    // DC-AE applies `latent = latent / scaling_factor` before decode. Set to
    // 1.0f to disable (e.g. for unit tests with synthetic data).
    float scaling_factor = 0.41407f;
    float eps = 1e-6f;               // RMSNorm epsilon
};

class Decoder {
public:
    explicit Decoder(const DecoderConfig& cfg);
    ~Decoder();

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    Decoder(Decoder&&) noexcept = default;
    Decoder& operator=(Decoder&&) noexcept = default;

    // Load decoder weights from a safetensors file. Default prefix matches a
    // standalone diffusers AutoencoderDC export.
    //
    // Throws std::runtime_error on missing tensor, shape mismatch, or a source
    // dtype that is neither F16 nor F32.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "decoder.");

    // Decode latent -> image. Tensors carry the compute dtype (FP32 on CPU,
    // FP16 on a GPU backend).
    //   latent: (1, latent_channels * H_lat * W_lat), NCHW with N=1.
    //   out:    (1, image_channels * 32*H_lat * 32*W_lat), resized as needed.
    // Caller is responsible for sync_all() before reading.
    void decode(const brotensor::Tensor& latent,
                int H_lat, int W_lat,
                brotensor::Tensor& out);

    const DecoderConfig& config() const { return cfg_; }

private:
    // ── per-block weight bundles (mirror the diffusers AutoencoderDC modules) ──

    // ResBlock: conv1(3x3) → SiLU → conv2(3x3, no bias) → RMSNorm(channels)
    //           → + residual. C_in == C_out for every decoder ResBlock.
    struct ResBlock {
        brotensor::Tensor conv1_W, conv1_b;   // (C, C*9), (C,1)
        brotensor::Tensor conv2_W;            // (C, C*9) — bias=False
        brotensor::Tensor norm_g, norm_b;     // (C,1) RMSNorm weight+bias
        int C = 0;
    };

    // GLUMBConv: 1x1 inverted (C→8C) → SiLU → 3x3 depthwise (8C) → GLU split
    //   (h, gate) → h*SiLU(gate) → 1x1 point (4C→C, no bias) → RMSNorm → +res.
    struct GLUMB {
        brotensor::Tensor inv_W, inv_b;       // (8C, C), (8C,1)
        brotensor::Tensor depth_W, depth_b;   // (8C, 9), (8C,1) depthwise
        brotensor::Tensor point_W;            // (C, 4C) — bias=False
        brotensor::Tensor norm_g, norm_b;     // (C,1)
        int C = 0;                            // hidden2 = 8C, gated = 4C
    };

    // SanaMultiscaleLinearAttention (ReLU linear attention, +1 multiscale scale).
    struct Attn {
        brotensor::Tensor Wq, Wk, Wv;         // (inner, C) — bias=False
        brotensor::Tensor ms_in_W;            // (3*inner, 25) depthwise 5x5
        brotensor::Tensor ms_out_W;           // (3*inner, head_dim) grouped 1x1
        brotensor::Tensor Wo;                 // (C, (1+S)*inner) — bias=False
        brotensor::Tensor norm_g, norm_b;     // (C,1) RMSNorm
        int C = 0, inner = 0, num_heads = 0;
    };

    struct EViT {
        Attn  attn;
        GLUMB glu;
    };

    struct UpBlock {
        bool is_attention = false;
        bool has_upsampler = false;
        // DCUpBlock2d (interpolate): nearest-2x → conv(3x3) + pixel-shuffle skip.
        brotensor::Tensor up_W, up_b;         // (C_out, C_in*9), (C_out,1)
        int up_Cin = 0, up_Cout = 0;
        std::vector<ResBlock> resblocks;      // populated when !is_attention
        std::vector<EViT>     evitblocks;     // populated when is_attention
        int C = 0;                            // stage output channels
    };

    // ── forward helpers (channel-major NCHW, N=1, compute dtype) ──
    void apply_rmsnorm_(const brotensor::Tensor& g, const brotensor::Tensor& b,
                        int C, int H, int W,
                        brotensor::Tensor& x, brotensor::Tensor& out);
    void apply_resblock_(const ResBlock& r, int H, int W, brotensor::Tensor& x);
    void apply_glumb_(const GLUMB& m, int H, int W, brotensor::Tensor& x);
    void apply_attn_(const Attn& a, int H, int W, brotensor::Tensor& x);
    void apply_upsample_(const UpBlock& u, int H, int W, brotensor::Tensor& x);
    void ensure_ones_(int n);

    DecoderConfig cfg_;

    // Weights.
    brotensor::Tensor conv_in_W_, conv_in_b_;
    std::vector<UpBlock> up_blocks_;          // indexed by checkpoint order 0..5
    brotensor::Tensor norm_out_g_, norm_out_b_;
    brotensor::Tensor conv_out_W_, conv_out_b_;
    int in_shortcut_repeats_ = 0;             // 0 ⇒ no input shortcut

    // Scratch (re-used across decode calls; ping-pong + per-op temporaries).
    brotensor::Tensor x_, y_, t1_, t2_, t3_;
    brotensor::Tensor seq_, seq2_, repbuf_, up_t_, short_t_;
    brotensor::Tensor qkv_, ms_, hs_, hsf_, attn_f_, attn_c_, toout_;
    brotensor::Tensor vp_, kt_, scores_, hidden_, recip_, ones_;
};

}  // namespace brodiffusion::dcae
