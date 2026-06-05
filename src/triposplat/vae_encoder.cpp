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

    // BatchNorm1d(128, affine=False) latent normalizer: precompute 1/sqrt(var+eps).
    bn_mean_ = read_f32_vec(f, "bn.running_mean", kTokenDim);
    std::vector<float> var = read_f32_vec(f, "bn.running_var", kTokenDim);
    bn_inv_std_.resize(kTokenDim);
    for (int i = 0; i < kTokenDim; ++i) {
        bn_inv_std_[i] = 1.0f / std::sqrt(var[i] + kBnEps);
    }
}

void Flux2VaeEncoder::encode(const bt::Tensor& image, int H, int W,
                             bt::Tensor& out) {
    if (bn_mean_.empty()) fail("encode: weights not loaded");
    if (H <= 0 || W <= 0) fail("encode: H and W must be positive");
    if (H % kTokenStride != 0 || W % kTokenStride != 0) {
        fail("encode: H and W must be multiples of 16");
    }

    // 1. VAE encoder body -> raw 32-channel latent (posterior mean). With
    //    scaling=1/shift=0 the encoder returns `mean` directly.
    enc_.encode(image, H, W, /*eps=*/nullptr, latent_);

    const int Cl = 32;            // latent channels
    const int hl = H / 8;         // latent spatial
    const int wl = W / 8;
    const int ho = hl / 2;        // token grid (after pixel-shuffle 2x)
    const int wo = wl / 2;
    const int T  = ho * wo;

    // 2. Download the latent to host. The pixel-shuffle below is a pure
    //    reshape/permute; the bn-normalize a per-channel affine; the tokenize a
    //    transpose. Doing this marshalling host-side (once per image — the DiT
    //    sampler and decoder dominate runtime) keeps it simple and exact.
    bt::sync_all();
    std::vector<float> lat;
    if (latent_.dtype == bt::Dtype::FP16) {
        std::vector<uint16_t> bits = latent_.to_host_vector_fp16();
        lat.resize(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            lat[i] = bt::fp16_bits_to_fp32(bits[i]);
        }
    } else {
        lat = latent_.to_host_vector();
    }

    // 3. pixel-shuffle (C, hl, wl) -> (C*4, ho, wo) with channel layout
    //    co = c*4 + sh*2 + sw, sampling input (y=2*yo+sh, x=2*xo+sw); then the
    //    per-channel bn-normalize; then tokenize to (T, 128) row-major over the
    //    token grid. Mirrors model.py:
    //      latents.view(B,C,H/2,2,W/2,2).permute(0,1,3,5,2,4).reshape(B,C*4,H/2,W/2)
    //      (latents - bn_mean) / sqrt(bn_var + eps)  ->  flatten(2).transpose(1,2)
    std::vector<float> tok(static_cast<std::size_t>(T) * kTokenDim);
    for (int c = 0; c < Cl; ++c) {
        for (int sh = 0; sh < 2; ++sh) {
            for (int sw = 0; sw < 2; ++sw) {
                const int co = c * 4 + sh * 2 + sw;
                const float m = bn_mean_[co];
                const float inv = bn_inv_std_[co];
                for (int yo = 0; yo < ho; ++yo) {
                    const int y = yo * 2 + sh;
                    for (int xo = 0; xo < wo; ++xo) {
                        const int x = xo * 2 + sw;
                        const float v =
                            lat[(static_cast<std::size_t>(c) * hl + y) * wl + x];
                        const int t = yo * wo + xo;
                        tok[static_cast<std::size_t>(t) * kTokenDim + co] =
                            (v - m) * inv;
                    }
                }
            }
        }
    }

    // 4. Upload tokens at the pipeline compute dtype.
    out = detail::upload_host(tok.data(), T, kTokenDim);
}

}  // namespace brodiffusion::triposplat
