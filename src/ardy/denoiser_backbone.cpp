#include "brodiffusion/ardy/denoiser_backbone.h"

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
    throw std::runtime_error("ardy::ArdyDenoiserBackbone: " + m);
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

// Sinusoidal PE row for an arbitrary (possibly negative) index into a d-dim
// vector: [2k] = sin(i*div_k), [2k+1] = cos(i*div_k), div_k = 10000^(-2k/d).
// (PositionalEncoding + PositionalEncodingNegativeIndex both reduce to this.)
void pe_row(int idx, int d, float* dst) {
    for (int k = 0; k < d / 2; ++k) {
        const float div = std::pow(10000.0f,
            -static_cast<float>(2 * k) / static_cast<float>(d));
        const float a = static_cast<float>(idx) * div;
        dst[2 * k]     = std::sin(a);
        dst[2 * k + 1] = std::cos(a);
    }
}

}  // namespace

ArdyDenoiserBackbone::ArdyDenoiserBackbone(const Config& cfg) : cfg_(cfg) {}
ArdyDenoiserBackbone::~ArdyDenoiserBackbone() = default;

void ArdyDenoiserBackbone::load_weights(const st::File& f, const std::string& p) {
    const int D  = cfg_.latent_dim;   // 1024
    const int FF = cfg_.ff_size;      // 2048

    load_linear(f, p + "embed_text", D, cfg_.llm_dim, embed_text_.W, embed_text_.b);
    load_linear(f, p + "embed_timestep.time_embed.0", D, D, time0_.W, time0_.b);
    load_linear(f, p + "embed_timestep.time_embed.2", D, D, time2_.W, time2_.b);
    if (cfg_.input_first_heading_angle) {
        load_linear(f, p + "linear_first_heading_angle", D, 2,
                    first_heading_.W, first_heading_.b);
    }
    load_linear(f, p + "output_linear", cfg_.output_dim, D,
                output_linear_.W, output_linear_.b);

    // Learned prefix PE table: (num_prefix, latent_dim).
    st::upload_compute_checked(need(f, p + "learned_prefix_embedding.embedding.weight"),
                               num_prefix(), D, learned_prefix_, p);

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
        L.n1g = bt::Tensor{}; L.n1b = bt::Tensor{};
        L.n2g = bt::Tensor{}; L.n2b = bt::Tensor{};
        st::upload_compute_checked(need(f, lp + "norm1.weight"), D, 1, L.n1g, lp);
        st::upload_compute_checked(need(f, lp + "norm1.bias"),   D, 1, L.n1b, lp);
        st::upload_compute_checked(need(f, lp + "norm2.weight"), D, 1, L.n2g, lp);
        st::upload_compute_checked(need(f, lp + "norm2.bias"),   D, 1, L.n2b, lp);
    }
}

void ArdyDenoiserBackbone::forward(const bt::Tensor& x,
                                   const float* text_feat,
                                   int timestep,
                                   float first_heading_angle,
                                   const int* token_index,
                                   int T,
                                   const float* key_mask,
                                   bt::Tensor& out) {
    if (layers_.empty()) fail("forward: weights not loaded");
    const int D  = cfg_.latent_dim;
    const int H  = cfg_.num_heads;
    const int NP = num_prefix();
    const int L  = NP + T;

    // ── prefix conditioning tokens ──
    // text: (num_text_tokens, llm_dim) -> embed_text -> (num_text_tokens, D)
    bt::Tensor txt = detail::upload_host(text_feat, cfg_.num_text_tokens, cfg_.llm_dim);
    bt::Tensor emb_text;
    detail::linear_batched(embed_text_.W, &embed_text_.b, txt, emb_text);

    // timestep: time_embed(sinusoidalPE[timestep]) = Linear -> SiLU -> Linear.
    std::vector<float> pe_t(D);
    pe_row(timestep, D, pe_t.data());
    bt::Tensor tin = detail::upload_host(pe_t.data(), 1, D);
    bt::Tensor th, emb_time;
    detail::linear_batched(time0_.W, &time0_.b, tin, th);
    bt::silu_forward(th, th);
    detail::linear_batched(time2_.W, &time2_.b, th, emb_time);

    // first heading angle: [cos, sin] -> Linear(2 -> D).
    bt::Tensor emb_fha;
    if (cfg_.input_first_heading_angle) {
        const float fha[2] = {std::cos(first_heading_angle),
                              std::sin(first_heading_angle)};
        bt::Tensor fin = detail::upload_host(fha, 1, 2);
        detail::linear_batched(first_heading_.W, &first_heading_.b, fin, emb_fha);
    }

    // ── assemble xseq = [prefix (+ learned PE); motion (+ sinusoidal PE)] ──
    bt::Tensor xseq = bt::Tensor::zeros_on(x.device, L, D, x.dtype);
    int row = 0;
    bt::copy_d2d(emb_text, 0, xseq, row * D, cfg_.num_text_tokens * D);
    row += cfg_.num_text_tokens;
    bt::copy_d2d(emb_time, 0, xseq, row * D, D);
    row += 1;
    if (cfg_.input_first_heading_angle) {
        bt::copy_d2d(emb_fha, 0, xseq, row * D, D);
        row += 1;
    }
    // learned positional embedding over the NP prefix rows.
    {
        bt::Tensor pref = bt::Tensor::zeros_on(x.device, NP, D, x.dtype);
        bt::copy_d2d(xseq, 0, pref, 0, NP * D);
        bt::add_inplace(pref, learned_prefix_);
        bt::copy_d2d(pref, 0, xseq, 0, NP * D);
    }
    // motion tokens + sinusoidal PE (indexed by token_index, may be negative).
    {
        std::vector<float> pe(static_cast<size_t>(T) * D);
        for (int t = 0; t < T; ++t) pe_row(token_index[t], D, &pe[static_cast<size_t>(t) * D]);
        bt::Tensor pet = detail::upload_host(pe.data(), T, D);
        bt::Tensor xm = bt::Tensor::zeros_on(x.device, T, D, x.dtype);
        bt::copy_d2d(x, 0, xm, 0, T * D);
        bt::add_inplace(xm, pet);
        bt::copy_d2d(xm, 0, xseq, NP * D, T * D);
    }

    // ── post-norm BIDIRECTIONAL Transformer encoder layers ──
    bt::Tensor h = xseq;  // (L, D)
    bt::Tensor q, k, v, attn, ao, res, ff;
    for (Layer& lyr : layers_) {
        detail::linear_batched(lyr.q.W, &lyr.q.b, h, q);
        detail::linear_batched(lyr.k.W, &lyr.k.b, h, k);
        detail::linear_batched(lyr.v.W, &lyr.v.b, h, v);
        bt::flash_attention_gqa_forward(q, k, v, key_mask, H, H,
                                        /*causal=*/false, attn);
        detail::linear_batched(lyr.out_proj.W, &lyr.out_proj.b, attn, ao);
        bt::add_inplace(ao, h);                        // h + sa(h)
        detail::layernorm_batched(ao, lyr.n1g, lyr.n1b, h, 1e-5f);  // norm1

        detail::linear_batched(lyr.linear1.W, &lyr.linear1.b, h, ff);
        bt::gelu_exact_forward(ff, ff);                // torch activation="gelu"
        detail::linear_batched(lyr.linear2.W, &lyr.linear2.b, ff, res);
        bt::add_inplace(res, h);                       // h + ff(h)
        detail::layernorm_batched(res, lyr.n2g, lyr.n2b, h, 1e-5f);  // norm2
    }

    // strip the prefix rows, then output projection.
    bt::Tensor motion = bt::Tensor::zeros_on(h.device, T, D, h.dtype);
    bt::copy_d2d(h, NP * D, motion, 0, T * D);
    detail::linear_batched(output_linear_.W, &output_linear_.b, motion, out);
}

}  // namespace brodiffusion::ardy
