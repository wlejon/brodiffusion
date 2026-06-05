#include "brodiffusion/triposplat/vae_encoder.h"

#include "brodiffusion/detail/compute.h"
#include "brotensor/safetensors.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::triposplat {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("triposplat::Flux2VaeEncoder: " + msg);
}

// Flux.2 VAE encoder body == diffusers AutoencoderKL encoder with a 32-channel
// latent. The Flux.2-specific tail (pixel-shuffle + bn + tokenize) lives here,
// not in vae::Encoder, so the generic SD/FLUX VAE stays untouched.
vae::EncoderConfig flux2_config() {
    vae::EncoderConfig cfg;
    cfg.in_channels        = 32;                       // latent channels
    cfg.out_channels       = 3;                        // RGB
    cfg.block_out_channels = {128, 256, 512, 512};
    cfg.layers_per_block   = 2;
    cfg.norm_num_groups    = 32;
    cfg.scaling_factor     = 1.0f;                     // bn replaces scale/shift
    cfg.shift_factor       = 0.0f;
    cfg.eps                = 1e-6f;
    cfg.num_attention_heads = 1;
    return cfg;
}

// Read an F32 1-D running statistic of length n from the checkpoint into host
// floats (these are normalization constants, not device tensors).
std::vector<float> read_f32_vec(const st::File& f, const std::string& key, int n) {
    const auto* v = f.find(key);
    if (!v) fail("missing tensor '" + key + "'");
    if (v->dtype != st::Dtype::F32) {
        fail(key + ": expected F32, got " + st::dtype_name(v->dtype));
    }
    if (v->numel() != n) {
        fail(key + ": expected " + std::to_string(n) + " elements, got " +
             std::to_string(v->numel()));
    }
    const float* p = reinterpret_cast<const float*>(v->data);
    return std::vector<float>(p, p + n);
}

}  // namespace

Flux2VaeEncoder::Flux2VaeEncoder() : enc_(flux2_config()) {}
Flux2VaeEncoder::~Flux2VaeEncoder() = default;

void Flux2VaeEncoder::load_weights(const st::File& f) {
    // Encoder body + quant_conv (quant_conv lives above the "encoder." subtree).
    enc_.load_weights(f, "encoder.");

    // BatchNorm1d(128, affine=False), frozen running stats. Fold the inference
    // normalization y = (x - mean)/sqrt(var+eps) into modulate's affine
    // Y = X*(1+scale)+shift: scale = inv_std - 1, shift = -mean*inv_std.
    std::vector<float> mean = read_f32_vec(f, "bn.running_mean", kTokenDim);
    std::vector<float> var  = read_f32_vec(f, "bn.running_var", kTokenDim);
    std::vector<float> scale(kTokenDim), shift(kTokenDim);
    for (int i = 0; i < kTokenDim; ++i) {
        const float inv_std = 1.0f / std::sqrt(var[i] + kBnEps);
        scale[i] = inv_std - 1.0f;
        shift[i] = -mean[i] * inv_std;
    }
    bn_mod_scale_ = detail::upload_host(scale.data(), 1, kTokenDim);
    bn_mod_shift_ = detail::upload_host(shift.data(), 1, kTokenDim);
}

void Flux2VaeEncoder::encode(const bt::Tensor& image, int H, int W,
                             bt::Tensor& out) {
    if (bn_mod_scale_.size() == 0) fail("encode: weights not loaded");
    if (H <= 0 || W <= 0) fail("encode: H and W must be positive");
    if (H % kTokenStride != 0 || W % kTokenStride != 0) {
        fail("encode: H and W must be multiples of 16");
    }

    const int Cl = 32;            // latent channels
    const int hl = H / 8;         // latent spatial
    const int wl = W / 8;
    const int ho = hl / 2;        // token grid (after pixel-shuffle 2x)
    const int wo = wl / 2;

    // The whole tail runs on-device at the compute dtype — no host round-trip.
    // Mirrors model.py:
    //   latents.view(B,C,H/2,2,W/2,2).permute(0,1,3,5,2,4).reshape(B,C*4,H/2,W/2)
    //   (latents - bn_mean) / sqrt(bn_var + eps)  ->  flatten(2).transpose(1,2)
    //
    // 1. VAE encoder body -> raw 32-channel latent (posterior mean). With
    //    scaling=1/shift=0 the encoder returns `mean` directly.
    enc_.encode(image, H, W, /*eps=*/nullptr, latent_);

    // 2. pixel-shuffle 2x: (32, hl, wl) -> (128, ho, wo), channel-major layout
    //    co = c*4 + sh*2 + sw (torch F.pixel_unshuffle ordering).
    bt::spatial_merge_2x2_forward(latent_, /*N=*/1, Cl, hl, wl,
                                  /*channel_major=*/true, shuffled_);

    // 3. tokenize: NCHW (1,128,ho,wo) -> (T=ho*wo, 128) row-major over the grid.
    bt::nchw_to_sequence(shuffled_, /*N=*/1, kTokenDim, ho, wo, tokens_);

    // 4. bn-normalize folded into a per-channel affine over the 128 columns.
    //    Applying it post-tokenize is identical to pre-tokenize (per-channel).
    bt::modulate(tokens_, bn_mod_scale_, bn_mod_shift_, out);
}

}  // namespace brodiffusion::triposplat
