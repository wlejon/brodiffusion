#include "brodiffusion/ardy/fsq_decoder.h"

#include "brodiffusion/detail/compute.h"
#include "brotensor/safetensors.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"

#include <cmath>
#include <stdexcept>

namespace brodiffusion::ardy {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& m) {
    throw std::runtime_error("ardy::FsqMotionDecoder: " + m);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    if (const auto* v = f.find(key)) return *v;
    fail("missing tensor '" + key + "'");
}

// Load a torch Linear (weight [out,in], bias [out]) to device at compute dtype.
void load_linear(const st::File& f, const std::string& key, int out, int in,
                 bt::Tensor& W, bt::Tensor& b) {
    st::upload_compute_checked(need(f, key + ".weight"), out, in, W, key);
    st::upload_compute_checked(need(f, key + ".bias"), out, 1, b, key);
}

bt::Tensor load_vec(const st::File& f, const std::string& key, int n) {
    bt::Tensor t;
    st::upload_compute_checked(need(f, key), n, 1, t, key);
    return t;
}

}  // namespace

FsqMotionDecoder::FsqMotionDecoder(const Config& cfg) : cfg_(cfg) {}
FsqMotionDecoder::~FsqMotionDecoder() = default;

void FsqMotionDecoder::load_weights(const st::File& f, const std::string& p) {
    const int D = cfg_.latent_dim;          // 512
    const int FF = cfg_.ff_size;             // 1024
    const int fpt = cfg_.num_frames_per_token;

    load_linear(f, p + "input_proj", D, cfg_.token_dim, input_proj_.W, input_proj_.b);
    // external_cond_blocks.0: Linear(latent + fpt*ext_dim -> latent)
    load_linear(f, p + "external_cond_blocks.0",
                D, D + fpt * cfg_.external_cond_dim, external_cond_.W, external_cond_.b);
    // output_proj: Linear(latent -> fpt*output_dim)
    load_linear(f, p + "output_proj", fpt * cfg_.output_dim, D, output_proj_.W, output_proj_.b);

    layers_.resize(cfg_.num_layers);
    for (int i = 0; i < cfg_.num_layers; ++i) {
        const std::string lp =
            p + "seqTransEncoder.layers." + std::to_string(i) + ".";
        Layer& L = layers_[i];

        // self_attn: packed in_proj_weight [3D, D] / in_proj_bias [3D] -> q,k,v.
        bt::Tensor inW, inB;
        st::upload_compute_checked(need(f, lp + "self_attn.in_proj_weight"),
                                   3 * D, D, inW, lp);
        st::upload_compute_checked(need(f, lp + "self_attn.in_proj_bias"),
                                   3 * D, 1, inB, lp);
        auto split = [&](int idx, Linear& lin) {
            lin.W = bt::Tensor::zeros_on(inW.device, D, D, inW.dtype);
            lin.b = bt::Tensor::zeros_on(inB.device, D, 1, inB.dtype);
            bt::copy_d2d(inW, static_cast<int>(static_cast<std::size_t>(idx) * D * D),
                         lin.W, 0, D * D);
            bt::copy_d2d(inB, idx * D, lin.b, 0, D);
        };
        split(0, L.q);
        split(1, L.k);
        split(2, L.v);
        load_linear(f, lp + "self_attn.out_proj", D, D, L.out_proj.W, L.out_proj.b);

        load_linear(f, lp + "linear1", FF, D, L.linear1.W, L.linear1.b);
        load_linear(f, lp + "linear2", D, FF, L.linear2.W, L.linear2.b);
        L.n1g = load_vec(f, lp + "norm1.weight", D);
        L.n1b = load_vec(f, lp + "norm1.bias", D);
        L.n2g = load_vec(f, lp + "norm2.weight", D);
        L.n2b = load_vec(f, lp + "norm2.bias", D);
    }
}

void FsqMotionDecoder::set_post_quant_stats(const float* mean, const float* std,
                                            int n, float eps) {
    if (n != cfg_.token_dim) fail("post-quant stats length != token_dim");
    pq_mean_.assign(mean, mean + n);
    // ardy Stats.unnormalize uses std_eps = sqrt(std^2 + eps), not raw std.
    pq_std_.resize(n);
    for (int i = 0; i < n; ++i) pq_std_[i] = std::sqrt(std[i] * std[i] + eps);
}

void FsqMotionDecoder::requantize(float* tokens_norm, int T_tok) const {
    if (pq_mean_.empty()) fail("requantize: post-quant stats not set");
    const int Cd = cfg_.token_dim;  // 128
    const float half = static_cast<float>(cfg_.fsq_level) / 2.0f;  // 32
    for (int t = 0; t < T_tok; ++t) {
        for (int c = 0; c < Cd; ++c) {
            const size_t i = static_cast<size_t>(t) * Cd + c;
            float x = tokens_norm[i] * pq_std_[c] + pq_mean_[c];  // unnormalize
            if (x > 1.0f) x = 1.0f;
            if (x < -1.0f) x = -1.0f;
            const float q = std::nearbyint(x * half) / half;      // snap to grid
            tokens_norm[i] = (q - pq_mean_[c]) / pq_std_[c];       // renormalize
        }
    }
}

void FsqMotionDecoder::detokenize(const float* tokens_norm,
                                  const float* local_root, int T_tok,
                                  bt::Tensor& out) {
    if (layers_.empty()) fail("detokenize: weights not loaded");
    if (pq_mean_.empty()) fail("detokenize: post-quant stats not set");

    const int D   = cfg_.latent_dim;
    const int Cd  = cfg_.token_dim;          // 128
    const int fpt = cfg_.num_frames_per_token;
    const int Ext = fpt * cfg_.external_cond_dim;  // 16
    const int H   = cfg_.num_heads;
    const float half = static_cast<float>(cfg_.fsq_level) / 2.0f;  // 32

    // ── requantize tokens on host: unnormalize (x*std_eps+mean, std_eps folded
    //    into pq_std_) then re-round to the FSQ grid round(clamp(x,-1,1)*half)/
    //    half. nearbyint == torch.round (round-half-to-even under the default
    //    FE rounding mode). ──
    std::vector<float> tq(static_cast<size_t>(T_tok) * Cd);
    for (int t = 0; t < T_tok; ++t) {
        for (int c = 0; c < Cd; ++c) {
            const size_t i = static_cast<size_t>(t) * Cd + c;
            float x = tokens_norm[i] * pq_std_[c] + pq_mean_[c];
            if (x > 1.0f) x = 1.0f;
            if (x < -1.0f) x = -1.0f;
            tq[i] = std::nearbyint(x * half) / half;
        }
    }
    bt::Tensor tok = detail::upload_host(tq.data(), T_tok, Cd);
    bt::Tensor ext = detail::upload_host(local_root, T_tok, Ext);

    // h = input_proj(tokens)
    bt::Tensor h;
    detail::linear_batched(input_proj_.W, &input_proj_.b, tok, h);

    // external-root conditioning: cat([h, ext], -1) -> Linear(D+Ext, D) -> ReLU
    bt::Tensor cat = bt::Tensor::zeros_on(h.device, T_tok, D + Ext, h.dtype);
    for (int t = 0; t < T_tok; ++t) {
        bt::copy_d2d(h,   t * D,   cat, t * (D + Ext),       D);
        bt::copy_d2d(ext, t * Ext, cat, t * (D + Ext) + D,   Ext);
    }
    detail::linear_batched(external_cond_.W, &external_cond_.b, cat, h);
    bt::relu_forward(h, h);

    // positional encoding (non-learned sinusoidal, added in-place)
    {
        std::vector<float> pe(static_cast<size_t>(T_tok) * D);
        for (int pos = 0; pos < T_tok; ++pos) {
            for (int i2 = 0; i2 < D / 2; ++i2) {
                const float div = std::pow(10000.0f,
                    -static_cast<float>(2 * i2) / static_cast<float>(D));
                const float a = static_cast<float>(pos) * div;
                pe[static_cast<size_t>(pos) * D + 2 * i2]     = std::sin(a);
                pe[static_cast<size_t>(pos) * D + 2 * i2 + 1] = std::cos(a);
            }
        }
        bt::Tensor pet = detail::upload_host(pe.data(), T_tok, D);
        bt::add_inplace(h, pet);
    }

    // 8 post-norm causal Transformer encoder layers.
    bt::Tensor q, k, v, attn, ao, res, ff;
    for (Layer& L : layers_) {
        // self-attention block
        detail::linear_batched(L.q.W, &L.q.b, h, q);
        detail::linear_batched(L.k.W, &L.k.b, h, k);
        detail::linear_batched(L.v.W, &L.v.b, h, v);
        bt::flash_attention_gqa_forward(q, k, v, /*d_mask=*/nullptr, H, H,
                                        cfg_.causal, attn);
        detail::linear_batched(L.out_proj.W, &L.out_proj.b, attn, ao);
        bt::add_inplace(ao, h);                       // h + sa(h)
        detail::layernorm_batched(ao, L.n1g, L.n1b, h, 1e-5f);  // norm1

        // feed-forward block
        detail::linear_batched(L.linear1.W, &L.linear1.b, h, ff);
        bt::gelu_exact_forward(ff, ff);  // torch activation="gelu" == erf GELU
        detail::linear_batched(L.linear2.W, &L.linear2.b, ff, res);
        bt::add_inplace(res, h);                      // h + ff(h)
        detail::layernorm_batched(res, L.n2g, L.n2b, h, 1e-5f);  // norm2
    }

    // output projection: (T_tok, D) -> (T_tok, fpt*output_dim). Row-major flat
    // == (num_frames, output_dim) (rearrange "b t (f d) -> b (t f) d").
    detail::linear_batched(output_proj_.W, &output_proj_.b, h, out);
}

}  // namespace brodiffusion::ardy
