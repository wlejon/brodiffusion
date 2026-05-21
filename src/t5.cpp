#include "brodiffusion/t5.h"

#include "brotensor/safetensors.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::t5 {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

using st::upload_compute_checked;

// ─── helpers ───────────────────────────────────────────────────────────────

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("t5::TextEncoder: " + msg);
}

// Find a tensor by name across one or more shards; first match wins.
const st::TensorView& need(const std::vector<const st::File*>& shards,
                           const std::string& key) {
    for (const st::File* f : shards) {
        if (const auto* v = f->find(key)) return *v;
    }
    throw std::runtime_error("t5::TextEncoder: missing tensor '" + key + "'");
}

// Find a tensor by name across shards; nullptr if absent in every shard.
const st::TensorView* find_in(const std::vector<const st::File*>& shards,
                              const std::string& key) {
    for (const st::File* f : shards) {
        if (const auto* v = f->find(key)) return v;
    }
    return nullptr;
}

// Build a device-resident INT32 buffer holding `n` token ids. brotensor has
// no from_host path for INT32, so stage on the host then migrate to the
// default device.
bt::Tensor make_idx_device(const int32_t* host, int n) {
    bt::Tensor cpu = bt::Tensor::empty_on(bt::Device::CPU, n, 1, bt::Dtype::INT32);
    std::memcpy(cpu.host_raw_mut(), host,
                static_cast<std::size_t>(n) * sizeof(int32_t));
    return cpu.to(bt::default_device());
}

// Download a loaded weight tensor into a host FP32 vector, handling both the
// FP16 (GPU) and FP32 (CPU) compute-dtype cases — same pattern as
// pipeline.cpp's decode().
std::vector<float> download_fp32(const bt::Tensor& t) {
    const std::size_t n = static_cast<std::size_t>(t.size());
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        t.copy_to_host_fp16(bits.data());
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = bt::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return t.to_host_vector();
}

// T5 bidirectional relative-position bucket (encoder). Mirrors HF's
// _relative_position_bucket with bidirectional=True.
int relative_position_bucket(int relative_position, int num_buckets,
                             int max_distance) {
    int ret = 0;
    num_buckets /= 2;
    if (relative_position > 0) ret += num_buckets;
    int n = std::abs(relative_position);
    int max_exact = num_buckets / 2;
    if (n < max_exact) {
        ret += n;
    } else {
        int large = max_exact +
            static_cast<int>(std::log(static_cast<double>(n) / max_exact) /
                             std::log(static_cast<double>(max_distance) / max_exact) *
                             static_cast<double>(num_buckets - max_exact));
        if (large > num_buckets - 1) large = num_buckets - 1;
        ret += large;
    }
    return ret;
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

TextEncoder::TextEncoder(const T5Config& cfg) : cfg_(cfg) {
    if (cfg_.d_model <= 0 || cfg_.num_heads <= 0 ||
        cfg_.d_model % cfg_.num_heads != 0) {
        fail("d_model must be a positive multiple of num_heads");
    }
    if (cfg_.num_layers <= 0 || cfg_.vocab_size <= 0 || cfg_.d_ff <= 0 ||
        cfg_.relative_attention_num_buckets <= 0 ||
        cfg_.relative_attention_max_distance <= 0) {
        fail("config has non-positive dimension");
    }
    blocks_.resize(static_cast<std::size_t>(cfg_.num_layers));
}

TextEncoder::~TextEncoder() = default;

// ─── load_weights ──────────────────────────────────────────────────────────

void TextEncoder::load_weights(const st::File& f, const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights(shards, prefix);
}

void TextEncoder::load_weights(const std::vector<const st::File*>& shards,
                               const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");
    const int V  = cfg_.vocab_size;
    const int D  = cfg_.d_model;
    const int FF = cfg_.d_ff;
    const int NB = cfg_.relative_attention_num_buckets;
    const int H  = cfg_.num_heads;

    // Token embedding: standalone export uses "shared.weight"; an
    // encoder-only export may instead carry "encoder.embed_tokens.weight".
    {
        const std::string shared_key = prefix + "shared.weight";
        const std::string embed_key  = prefix + "encoder.embed_tokens.weight";
        const auto* tv = find_in(shards, shared_key);
        if (!tv) tv = find_in(shards, embed_key);
        if (!tv) {
            fail("missing token embedding ('" + shared_key + "' or '" +
                 embed_key + "')");
        }
        upload_compute_checked(*tv, V, D, token_embed_, "shared.weight");
    }

    for (int i = 0; i < cfg_.num_layers; ++i) {
        const std::string p =
            prefix + "encoder.block." + std::to_string(i) + ".";
        Block& B = blocks_[static_cast<std::size_t>(i)];

        upload_compute_checked(need(shards, p + "layer.0.layer_norm.weight"),
                               D, 1, B.ln0, "block.layer.0.layer_norm");

        const std::string sa = p + "layer.0.SelfAttention.";
        upload_compute_checked(need(shards, sa + "q.weight"), D, D, B.Wq, "SelfAttention.q");
        upload_compute_checked(need(shards, sa + "k.weight"), D, D, B.Wk, "SelfAttention.k");
        upload_compute_checked(need(shards, sa + "v.weight"), D, D, B.Wv, "SelfAttention.v");
        upload_compute_checked(need(shards, sa + "o.weight"), D, D, B.Wo, "SelfAttention.o");

        upload_compute_checked(need(shards, p + "layer.1.layer_norm.weight"),
                               D, 1, B.ln1, "block.layer.1.layer_norm");

        const std::string dr = p + "layer.1.DenseReluDense.";
        upload_compute_checked(need(shards, dr + "wi_0.weight"), FF, D, B.wi_0, "DenseReluDense.wi_0");
        upload_compute_checked(need(shards, dr + "wi_1.weight"), FF, D, B.wi_1, "DenseReluDense.wi_1");
        upload_compute_checked(need(shards, dr + "wo.weight"),   D, FF, B.wo,  "DenseReluDense.wo");
    }

    // Relative-position bias table — block 0 only, shared by every layer.
    {
        bt::Tensor rel;
        upload_compute_checked(
            need(shards, prefix + "encoder.block.0.layer.0.SelfAttention."
                             "relative_attention_bias.weight"),
            NB, H, rel, "relative_attention_bias");
        rel_attn_bias_ = download_fp32(rel);
    }

    upload_compute_checked(need(shards, prefix + "encoder.final_layer_norm.weight"),
                           D, 1, final_ln_, "final_layer_norm");

    // Invalidate any cached position bias — weights just changed.
    pos_bias_L_ = -1;

    // INT8 (W8A16) quantisation. GPU-only: the W8A16 ops have no CPU path,
    // so on the CPU backend the flag is ignored and every QWeight stays
    // inactive (the FP32 forward path is used). On a GPU backend each
    // per-block attention / FFN matrix is converted in place; the token
    // embedding, RMSNorm gains, and position-bias table stay FP16.
    if (cfg_.quantize_weights) {
        if (token_embed_.device == bt::Device::CPU) {
            std::fprintf(stderr,
                "brodiffusion: INT8 weight quantization is GPU-only; "
                "ignoring T5Config::quantize_weights on the CPU backend.\n");
        } else {
            for (Block& B : blocks_) {
                quantize_weight_inplace_(B.Wq,   B.Wq_q);
                quantize_weight_inplace_(B.Wk,   B.Wk_q);
                quantize_weight_inplace_(B.Wv,   B.Wv_q);
                quantize_weight_inplace_(B.Wo,   B.Wo_q);
                quantize_weight_inplace_(B.wi_0, B.wi_0_q);
                quantize_weight_inplace_(B.wi_1, B.wi_1_q);
                quantize_weight_inplace_(B.wo,   B.wo_q);
            }
        }
    }
}

// ─── INT8 (W8A16) quantisation ─────────────────────────────────────────────

void TextEncoder::quantize_weight_inplace_(bt::Tensor& W_fp16, QWeight& q) {
    if (W_fp16.size() == 0) return;             // nothing to quantise
    if (W_fp16.dtype != bt::Dtype::FP16) {
        fail("quantize_weight_inplace_: weight is not FP16");
    }
    const int out = W_fp16.rows;
    const int in  = W_fp16.cols;

    std::vector<std::uint16_t> host_fp16(
        static_cast<std::size_t>(out) * static_cast<std::size_t>(in));
    W_fp16.copy_to_host_fp16(host_fp16.data());

    std::vector<std::int8_t> host_int8(host_fp16.size());
    std::vector<float>       host_scales(static_cast<std::size_t>(out));
    bt::quantize_int8_per_row_host(host_fp16.data(), out, in,
                                   host_int8.data(), host_scales.data());

    // brotensor has no from_host path for INT8 — stage the quantised bytes
    // on the host, then migrate to W_fp16's device.
    bt::Tensor cpu_int8 =
        bt::Tensor::empty_on(bt::Device::CPU, out, in, bt::Dtype::INT8);
    std::memcpy(cpu_int8.host_raw_mut(), host_int8.data(),
                static_cast<std::size_t>(out) * static_cast<std::size_t>(in));
    q.W_int8 = cpu_int8.to(W_fp16.device);
    q.scales = bt::Tensor::from_host_on(W_fp16.device,
                                        host_scales.data(), out, 1);

    // Free the original FP16 storage.
    W_fp16 = bt::Tensor();
}

void TextEncoder::ffn_linear_(const bt::Tensor& W, const QWeight& q,
                              const bt::Tensor& X, bt::Tensor& Y) {
    if (q.active()) {
        bt::linear_forward_batched_int8w_fp16(q.W_int8, q.scales,
                                              /*bias=*/nullptr, X, Y);
    } else {
        detail::linear_batched(W, /*bias=*/nullptr, X, Y);
    }
}

// ─── relative-position bias ────────────────────────────────────────────────

void TextEncoder::rebuild_position_bias_(int L) {
    if (pos_bias_L_ == L && !pos_bias_.empty()) return;

    const int H  = cfg_.num_heads;
    const int NB = cfg_.relative_attention_num_buckets;
    const int MD = cfg_.relative_attention_max_distance;

    // bias[h*L + q][k] = rel_attn_bias[bucket(k - q)][h]
    std::vector<float> bias(static_cast<std::size_t>(H) * L * L);
    for (int q = 0; q < L; ++q) {
        for (int k = 0; k < L; ++k) {
            const int bucket = relative_position_bucket(k - q, NB, MD);
            for (int h = 0; h < H; ++h) {
                const std::size_t row =
                    static_cast<std::size_t>(h) * L + q;
                bias[row * L + k] =
                    rel_attn_bias_[static_cast<std::size_t>(bucket) * H + h];
            }
        }
    }

    // attn_bias is FP32 on every backend.
    bt::Tensor host = bt::Tensor::from_host(bias.data(), H * L, L);
    pos_bias_ = host.to(bt::default_device());
    pos_bias_L_ = L;
}

// ─── forward ───────────────────────────────────────────────────────────────

void TextEncoder::forward(const int32_t* ids, int L, bt::Tensor& out) {
    if (!ids) fail("forward: ids pointer is null");
    if (L <= 0) fail("forward: L must be positive");
    if (token_embed_.size() == 0) fail("forward: weights not loaded");
    if (rel_attn_bias_.empty()) fail("forward: position bias table not loaded");

    const int H = cfg_.num_heads;

    // T5-XXL's residual stream grows monotonically across its 24 pre-norm
    // blocks and overflows FP16 (±65504) after roughly a dozen layers — the
    // well-known "T5 fp16" problem. Clamp the residual back into range after
    // each sub-layer (HuggingFace does exactly this). It is sufficient
    // because every sub-layer input is RMS-normed, hence scale-invariant: a
    // clamped residual still feeds an O(1) normalised input to the next
    // block. The CPU/FP32 path has no overflow and needs no clamp.
    const bool clamp_residual =
        (brodiffusion::compute_dtype() == bt::Dtype::FP16);
    constexpr float kFp16Clamp = 64504.0f;   // finfo(fp16).max - 1000

    rebuild_position_bias_(L);

    // Embedding: x = embedding_lookup(shared, ids). No position embedding,
    // no embedding scaling.
    ids_dev_ = make_idx_device(ids, L);
    bt::embedding_lookup_forward(
        token_embed_, static_cast<const int32_t*>(ids_dev_.data), L, x_);
    // Own the residual stream — embedding output buffer is otherwise reused.
    x_ = x_.clone();

    for (auto& B : blocks_) {
        // ── self-attention sub-layer ──────────────────────────────────────
        bt::rms_norm_forward(x_, B.ln0, cfg_.layer_norm_eps, n_);
        if (B.Wq_q.active()) {
            bt::self_attention_bias_int8w_fp16(
                n_, B.Wq_q.W_int8, B.Wq_q.scales,
                B.Wk_q.W_int8, B.Wk_q.scales,
                B.Wv_q.W_int8, B.Wv_q.scales,
                B.Wo_q.W_int8, B.Wo_q.scales,
                /*d_mask=*/nullptr, &pos_bias_, H, /*scale=*/1.0f, attn_);
        } else {
            bt::self_attention_bias_forward(
                n_, B.Wq, B.Wk, B.Wv, B.Wo,
                /*d_mask=*/nullptr, &pos_bias_, H, /*scale=*/1.0f, attn_);
        }
        bt::add_inplace(x_, attn_);
        if (clamp_residual) bt::clamp(x_, -kFp16Clamp, kFp16Clamp);

        // ── FFN sub-layer (gated-gelu) ────────────────────────────────────
        bt::rms_norm_forward(x_, B.ln1, cfg_.layer_norm_eps, n_);
        ffn_linear_(B.wi_0, B.wi_0_q, n_, g_);
        bt::gelu_forward(g_, g_);                       // tanh-approx GELU
        ffn_linear_(B.wi_1, B.wi_1_q, n_, l_);
        bt::mul_inplace(g_, l_);                        // h = gelu(wi_0 n) * wi_1 n
        ffn_linear_(B.wo, B.wo_q, g_, ffn_out_);
        bt::add_inplace(x_, ffn_out_);
        if (clamp_residual) bt::clamp(x_, -kFp16Clamp, kFp16Clamp);
    }

    bt::rms_norm_forward(x_, final_ln_, cfg_.layer_norm_eps, out);
}

}  // namespace brodiffusion::t5
