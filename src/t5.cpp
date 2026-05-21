#include "brodiffusion/t5.h"

#include "brotensor/safetensors.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdint>
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
        bt::self_attention_bias_forward(
            n_, B.Wq, B.Wk, B.Wv, B.Wo,
            /*d_mask=*/nullptr, &pos_bias_, H, /*scale=*/1.0f, attn_);
        bt::add_inplace(x_, attn_);

        // ── FFN sub-layer (gated-gelu) ────────────────────────────────────
        bt::rms_norm_forward(x_, B.ln1, cfg_.layer_norm_eps, n_);
        detail::linear_batched(B.wi_0, /*bias=*/nullptr, n_, g_);
        bt::gelu_forward(g_, g_);                       // tanh-approx GELU
        detail::linear_batched(B.wi_1, /*bias=*/nullptr, n_, l_);
        bt::mul_inplace(g_, l_);                        // h = gelu(wi_0 n) * wi_1 n
        detail::linear_batched(B.wo, /*bias=*/nullptr, g_, ffn_out_);
        bt::add_inplace(x_, ffn_out_);
    }

    bt::rms_norm_forward(x_, final_ln_, cfg_.layer_norm_eps, out);
}

}  // namespace brodiffusion::t5
