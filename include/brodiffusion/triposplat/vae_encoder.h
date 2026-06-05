#pragma once
//
// TripoSplat — Flux.2 VAE image encoder (conditioning feature2).
//
// TripoSplat (VAST-AI/TripoSplat) conditions its flow-matching DiT on two image
// features: `feature1` from the DINOv3 ViT-H backbone (brovisionml) and
// `feature2` from this Flux.2 VAE encoder. Only the encoder half of the VAE is
// used — the image is encoded to a 32-channel latent, then pixel-shuffled to
// 128 channels at half spatial resolution, batch-norm-normalized with the
// checkpoint's running statistics, and flattened to a token sequence.
//
// The heavy conv/groupnorm/attention encoder body is the same diffusers
// AutoencoderKL encoder that `vae::Encoder` already implements; this class wraps
// one configured for Flux.2 (32-channel latent) and adds the TripoSplat-specific
// tail (pixel-shuffle + bn-normalize + tokenize).
//
// Reference: model.py `Flux2VAEEncoder` in the upstream TripoSplat repo. Its
// `encode(images, deterministic)` takes images in [-1, 1]; the pipeline feeds
// `img * 2 - 1` (img in [0, 1]). With `deterministic=True` the latent is the
// posterior mean (no reparameterization noise) — the form used for parity.

#include "brodiffusion/vae.h"
#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::triposplat {

class Flux2VaeEncoder {
public:
    Flux2VaeEncoder();
    ~Flux2VaeEncoder();

    Flux2VaeEncoder(const Flux2VaeEncoder&) = delete;
    Flux2VaeEncoder& operator=(const Flux2VaeEncoder&) = delete;
    Flux2VaeEncoder(Flux2VaeEncoder&&) noexcept = default;
    Flux2VaeEncoder& operator=(Flux2VaeEncoder&&) noexcept = default;

    // Load the encoder body (via vae::Encoder), the quant_conv, and the
    // latent-normalizer batch-norm running statistics from a Flux.2 VAE
    // safetensors file. The decoder/post_quant_conv tensors in the file are
    // ignored. Throws on a missing tensor or shape mismatch.
    void load_weights(const brotensor::safetensors::File& f);

    // Encode an image into conditioning tokens.
    //   image: (1, 3 * H * W) NCHW at the pipeline compute dtype, values in
    //          [-1, 1]. H and W must be positive multiples of 16 (the encoder
    //          downsamples 8x and the pixel-shuffle a further 2x).
    //   out:   (T, 128) at the compute dtype, where T = (H/16) * (W/16). Row t
    //          is the 128-d feature for spatial cell t (row-major over the
    //          H/16 x W/16 grid).
    // Deterministic (posterior mean); the stochastic reparameterization noise
    // term is intentionally omitted (it is a single additive op and is not part
    // of this primitive). Caller must sync_all() before reading `out`.
    void encode(const brotensor::Tensor& image, int H, int W,
                brotensor::Tensor& out);

    // Number of feature channels per token (128).
    static constexpr int kTokenDim = 128;
    // Total downsample factor from image to token grid (8x VAE + 2x shuffle).
    static constexpr int kTokenStride = 16;

private:
    vae::Encoder enc_;
    // BatchNorm1d(128, affine=False) latent normalizer: y = (x - mean)/sqrt(var+eps).
    std::vector<float> bn_mean_;   // (128,)
    std::vector<float> bn_inv_std_;// (128,) = 1/sqrt(var + eps), precomputed
    static constexpr float kBnEps = 1e-5f;

    brotensor::Tensor latent_;     // scratch: raw 32-ch latent (mean)
};

}  // namespace brodiffusion::triposplat
