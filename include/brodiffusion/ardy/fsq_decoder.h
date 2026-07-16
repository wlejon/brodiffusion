#pragma once
//
// ARDY FSQ motion-tokenizer decoder (detokenize path).
//
// ARDY's motion "tokenizer" is an FSQ (finite-scalar-quantization) transformer
// autoencoder. The diffusion denoiser works in the hybrid latent space —
// explicit global root + FSQ-quantized body tokens — and this decoder turns the
// body tokens back into explicit body motion features, conditioned on the local
// root. It is the `FSQVAETransformer.detokenize` half (the encoder half is only
// needed for training / history conditioning, deferred for v1 text-to-motion).
//
// Reference: ardy/model/autoencoder/{fsq.py (FSQVAETransformer.detokenize),
// transformer.py (DoubleCondDecoderTransformer)}. g152 config: token dim 128
// (128 FSQ channels x 64 levels), latent 512, 8 post-norm causal Transformer
// layers, 4 heads, ff 1024, gelu; external-root conditioning injected once
// before the stack (Linear(latent + fpt*root_dim -> latent) + ReLU). Output is
// the "pose" feature block (local_root[4] + body[409] = 413) per frame.
//
// detokenize(tokens): unnormalize with the post-quantization stats, re-round to
// the FSQ grid (round(clamp(x,-1,1)*half)/half, half = level/2 = 32), then run
// the conditioned decoder.

#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brotensor::safetensors { class File; }

namespace brodiffusion::ardy {

class FsqMotionDecoder {
public:
    struct Config {
        int token_dim            = 128;   // FSQ channels (== latent_embedding_dim)
        int output_dim           = 413;   // pose = local_root(4) + body(409)
        int local_root_dim       = 4;     // leading slice of output_dim
        int num_frames_per_token = 4;
        int latent_dim           = 512;
        int num_heads            = 4;
        int ff_size              = 1024;
        int num_layers           = 8;
        int external_cond_dim    = 4;     // local root, per frame
        int fsq_level            = 64;    // half-width = fsq_level / 2 = 32
        bool causal              = true;
    };

    explicit FsqMotionDecoder(const Config& cfg = {});
    ~FsqMotionDecoder();

    FsqMotionDecoder(const FsqMotionDecoder&) = delete;
    FsqMotionDecoder& operator=(const FsqMotionDecoder&) = delete;

    const Config& config() const { return cfg_; }

    // Load decoder weights from the ARDY tokenizer.safetensors. Keys live under
    // `prefix` (default "pose_net.decoder."). Throws on any missing tensor.
    void load_weights(const brotensor::safetensors::File& f,
                      const std::string& prefix = "pose_net.decoder.");

    // Post-quantization normalization stats (length token_dim = 128). These are
    // the mean/std the denoiser's token outputs are normalized by; detokenize
    // unnormalizes with them (x * sqrt(std^2 + eps) + mean, matching ardy Stats)
    // before re-rounding. eps defaults to the ardy Stats default (1e-5).
    void set_post_quant_stats(const float* mean, const float* std, int n,
                              float eps = 1e-5f);

    // Decode body tokens into pose features.
    //   tokens_norm: (T_tok, token_dim) normalized FSQ token embeddings (host).
    //   local_root:  (T_tok * num_frames_per_token, local_root_dim) local-root
    //                external condition (host), row-major == (T_tok, fpt*rdim).
    //   out:         (T_tok, num_frames_per_token * output_dim) at compute dtype
    //                — flat-identical to (num_frames, output_dim). Caller must
    //                sync_all() before reading.
    void detokenize(const float* tokens_norm, const float* local_root, int T_tok,
                    brotensor::Tensor& out);

private:
    struct Linear { brotensor::Tensor W, b; };
    struct Layer {
        Linear q, k, v, out_proj;   // self-attention (packed in_proj split 3x)
        Linear linear1, linear2;    // FFN
        brotensor::Tensor n1g, n1b, n2g, n2b;  // norm1/norm2 affine
    };

    Config cfg_;
    Linear input_proj_, external_cond_, output_proj_;
    std::vector<Layer> layers_;
    std::vector<float> pq_mean_, pq_std_;
};

}  // namespace brodiffusion::ardy
