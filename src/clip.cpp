#include "brodiffusion/clip.h"
#include "brodiffusion/safetensors.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::clip {

namespace bt = ::brotensor;
namespace st = ::brodiffusion::safetensors;

// ─── helpers ───────────────────────────────────────────────────────────────

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("clip::TextEncoder: " + msg);
}

// Upload a FP16 safetensors view, asserting the expected (rows, cols) shape.
// Permits the source tensor to be 1-D (treated as (n, 1)) or 2-D.
void upload_fp16_checked(const st::TensorView& v, int rows, int cols,
                         bt::GpuTensor& dst, const char* name) {
    if (v.dtype != st::Dtype::F16) {
        fail(std::string(name) + ": expected FP16, got " +
             st::dtype_name(v.dtype));
    }
    int64_t expected = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    if (v.numel() != expected) {
        fail(std::string(name) + ": shape mismatch (expected " +
             std::to_string(rows) + "x" + std::to_string(cols) + ", got " +
             std::to_string(v.numel()) + " elements)");
    }
    st::upload(v, rows, cols, dst);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    const auto* v = f.find(key);
    if (!v) throw std::runtime_error("clip::TextEncoder: missing tensor '" + key + "'");
    return *v;
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

TextEncoder::TextEncoder(const TextEncoderConfig& cfg) : cfg_(cfg) {
    if (cfg_.hidden_dim <= 0 || cfg_.num_heads <= 0 ||
        cfg_.hidden_dim % cfg_.num_heads != 0) {
        fail("hidden_dim must be a positive multiple of num_heads");
    }
    if (cfg_.num_layers <= 0 || cfg_.max_position <= 0 ||
        cfg_.vocab_size <= 0 || cfg_.intermediate_dim <= 0) {
        fail("config has non-positive dimension");
    }
    layers_.resize(static_cast<std::size_t>(cfg_.num_layers));
}

TextEncoder::~TextEncoder() = default;

// ─── load_weights ──────────────────────────────────────────────────────────

void TextEncoder::load_weights(const st::File& f, const std::string& prefix) {
    const int V = cfg_.vocab_size;
    const int P = cfg_.max_position;
    const int D = cfg_.hidden_dim;
    const int F = cfg_.intermediate_dim;

    upload_fp16_checked(need(f, prefix + "embeddings.token_embedding.weight"),
                        V, D, token_embed_, "token_embedding");
    upload_fp16_checked(need(f, prefix + "embeddings.position_embedding.weight"),
                        P, D, position_embed_, "position_embedding");

    for (int i = 0; i < cfg_.num_layers; ++i) {
        const std::string p = prefix + "encoder.layers." + std::to_string(i) + ".";
        Layer& L = layers_[static_cast<std::size_t>(i)];

        upload_fp16_checked(need(f, p + "layer_norm1.weight"), D, 1, L.ln1_gamma, "ln1.weight");
        upload_fp16_checked(need(f, p + "layer_norm1.bias"),   D, 1, L.ln1_beta,  "ln1.bias");

        upload_fp16_checked(need(f, p + "self_attn.q_proj.weight"), D, D, L.Wq, "q_proj.W");
        upload_fp16_checked(need(f, p + "self_attn.q_proj.bias"),   D, 1, L.bq, "q_proj.b");
        upload_fp16_checked(need(f, p + "self_attn.k_proj.weight"), D, D, L.Wk, "k_proj.W");
        upload_fp16_checked(need(f, p + "self_attn.k_proj.bias"),   D, 1, L.bk, "k_proj.b");
        upload_fp16_checked(need(f, p + "self_attn.v_proj.weight"), D, D, L.Wv, "v_proj.W");
        upload_fp16_checked(need(f, p + "self_attn.v_proj.bias"),   D, 1, L.bv, "v_proj.b");
        upload_fp16_checked(need(f, p + "self_attn.out_proj.weight"), D, D, L.Wo, "out_proj.W");
        upload_fp16_checked(need(f, p + "self_attn.out_proj.bias"),   D, 1, L.bo, "out_proj.b");

        upload_fp16_checked(need(f, p + "layer_norm2.weight"), D, 1, L.ln2_gamma, "ln2.weight");
        upload_fp16_checked(need(f, p + "layer_norm2.bias"),   D, 1, L.ln2_beta,  "ln2.bias");

        upload_fp16_checked(need(f, p + "mlp.fc1.weight"), F, D, L.fc1_W, "fc1.W");
        upload_fp16_checked(need(f, p + "mlp.fc1.bias"),   F, 1, L.fc1_b, "fc1.b");
        upload_fp16_checked(need(f, p + "mlp.fc2.weight"), D, F, L.fc2_W, "fc2.W");
        upload_fp16_checked(need(f, p + "mlp.fc2.bias"),   D, 1, L.fc2_b, "fc2.b");
    }

    upload_fp16_checked(need(f, prefix + "final_layer_norm.weight"),
                        D, 1, final_gamma_, "final_ln.weight");
    upload_fp16_checked(need(f, prefix + "final_layer_norm.bias"),
                        D, 1, final_beta_,  "final_ln.bias");

    // Position-id buffer is fixed [0, 1, ..., P-1]. Upload once.
    std::vector<int32_t> positions(static_cast<std::size_t>(P));
    for (int i = 0; i < P; ++i) positions[static_cast<std::size_t>(i)] = i;
    positions_dev_.upload(positions.data(), positions.size());
}

// ─── forward ───────────────────────────────────────────────────────────────

void TextEncoder::forward(const int32_t* ids, bt::GpuTensor& out) {
    if (!ids) fail("forward: ids pointer is null");
    if (token_embed_.size() == 0) fail("forward: weights not loaded");
    if (positions_dev_.empty()) fail("forward: position buffer not initialised");

    const int L = cfg_.max_position;
    const int H = cfg_.num_heads;

    // Upload token IDs and run embedding lookups.
    ids_dev_.upload(ids, static_cast<std::size_t>(L));

    bt::embedding_lookup_forward_gpu(token_embed_,    ids_dev_.device_ptr(),       L, tok_emb_);
    bt::embedding_lookup_forward_gpu(position_embed_, positions_dev_.device_ptr(), L, pos_emb_);

    // x = tok_emb + pos_emb  (in tok_emb_, which we then alias as x_).
    bt::add_inplace_gpu(tok_emb_, pos_emb_);

    // Move the residual stream into x_ via a clone — we want a stable name
    // and we'll be feeding x_ into the per-layer code repeatedly.
    x_ = tok_emb_.clone();

    for (auto& layer : layers_) {
        // ── self-attention block ──────────────────────────────────────────
        bt::layernorm_forward_inference_batched_fp16_gpu(
            x_, layer.ln1_gamma, layer.ln1_beta, ln_out_, cfg_.layer_norm_eps);

        bt::linear_forward_batched_fp16_gpu(layer.Wq, &layer.bq, ln_out_, Q_);
        bt::linear_forward_batched_fp16_gpu(layer.Wk, &layer.bk, ln_out_, K_);
        bt::linear_forward_batched_fp16_gpu(layer.Wv, &layer.bv, ln_out_, V_);

        bt::flash_attention_forward_gpu(Q_, K_, V_, /*d_mask=*/nullptr,
                                        H, /*causal=*/true, attn_out_);

        bt::linear_forward_batched_fp16_gpu(layer.Wo, &layer.bo, attn_out_, proj_out_);
        bt::add_inplace_gpu(x_, proj_out_);

        // ── MLP block ─────────────────────────────────────────────────────
        bt::layernorm_forward_inference_batched_fp16_gpu(
            x_, layer.ln2_gamma, layer.ln2_beta, ln_out_, cfg_.layer_norm_eps);

        bt::linear_forward_batched_fp16_gpu(layer.fc1_W, &layer.fc1_b, ln_out_, ffn_mid_);
        bt::quick_gelu_forward_gpu(ffn_mid_, ffn_act_);
        bt::linear_forward_batched_fp16_gpu(layer.fc2_W, &layer.fc2_b, ffn_act_, ffn_out_);

        bt::add_inplace_gpu(x_, ffn_out_);
    }

    bt::layernorm_forward_inference_batched_fp16_gpu(
        x_, final_gamma_, final_beta_, out, cfg_.layer_norm_eps);
}

}  // namespace brodiffusion::clip
