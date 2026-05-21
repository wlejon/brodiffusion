#include "brodiffusion/clip.h"
#include "brotensor/safetensors.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::clip {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

// Validate-and-upload a safetensors weight view at the compute dtype —
// shared across the CLIP text / image encoders and the scorer.
using st::upload_compute_checked;

// ─── helpers ───────────────────────────────────────────────────────────────

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("clip::TextEncoder: " + msg);
}

// Build a device-resident INT32 buffer holding `n` token/position ids.
// brotensor has no from_host path for INT32, so stage on the host then
// migrate to the default device.
bt::Tensor make_idx_device(const int32_t* host, int n) {
    bt::Tensor cpu = bt::Tensor::empty_on(bt::Device::CPU, n, 1, bt::Dtype::INT32);
    std::memcpy(cpu.host_raw_mut(), host, static_cast<std::size_t>(n) * sizeof(int32_t));
    return cpu.to(bt::default_device());
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

    upload_compute_checked(need(f, prefix + "embeddings.token_embedding.weight"),
                        V, D, token_embed_, "token_embedding");
    upload_compute_checked(need(f, prefix + "embeddings.position_embedding.weight"),
                        P, D, position_embed_, "position_embedding");

    for (int i = 0; i < cfg_.num_layers; ++i) {
        const std::string p = prefix + "encoder.layers." + std::to_string(i) + ".";
        Layer& L = layers_[static_cast<std::size_t>(i)];

        upload_compute_checked(need(f, p + "layer_norm1.weight"), D, 1, L.ln1_gamma, "ln1.weight");
        upload_compute_checked(need(f, p + "layer_norm1.bias"),   D, 1, L.ln1_beta,  "ln1.bias");

        upload_compute_checked(need(f, p + "self_attn.q_proj.weight"), D, D, L.Wq, "q_proj.W");
        upload_compute_checked(need(f, p + "self_attn.q_proj.bias"),   D, 1, L.bq, "q_proj.b");
        upload_compute_checked(need(f, p + "self_attn.k_proj.weight"), D, D, L.Wk, "k_proj.W");
        upload_compute_checked(need(f, p + "self_attn.k_proj.bias"),   D, 1, L.bk, "k_proj.b");
        upload_compute_checked(need(f, p + "self_attn.v_proj.weight"), D, D, L.Wv, "v_proj.W");
        upload_compute_checked(need(f, p + "self_attn.v_proj.bias"),   D, 1, L.bv, "v_proj.b");
        upload_compute_checked(need(f, p + "self_attn.out_proj.weight"), D, D, L.Wo, "out_proj.W");
        upload_compute_checked(need(f, p + "self_attn.out_proj.bias"),   D, 1, L.bo, "out_proj.b");

        upload_compute_checked(need(f, p + "layer_norm2.weight"), D, 1, L.ln2_gamma, "ln2.weight");
        upload_compute_checked(need(f, p + "layer_norm2.bias"),   D, 1, L.ln2_beta,  "ln2.bias");

        upload_compute_checked(need(f, p + "mlp.fc1.weight"), F, D, L.fc1_W, "fc1.W");
        upload_compute_checked(need(f, p + "mlp.fc1.bias"),   F, 1, L.fc1_b, "fc1.b");
        upload_compute_checked(need(f, p + "mlp.fc2.weight"), D, F, L.fc2_W, "fc2.W");
        upload_compute_checked(need(f, p + "mlp.fc2.bias"),   D, 1, L.fc2_b, "fc2.b");
    }

    upload_compute_checked(need(f, prefix + "final_layer_norm.weight"),
                        D, 1, final_gamma_, "final_ln.weight");
    upload_compute_checked(need(f, prefix + "final_layer_norm.bias"),
                        D, 1, final_beta_,  "final_ln.bias");

    // Optional projection. CLIPTextModelWithProjection ships a
    // {prefix}text_projection.weight of shape (D, D) (diffusers (out, in)
    // row-major). Plain CLIPTextModel — which is what Flux uses — omits it,
    // in which case the pooled output is the raw EOS hidden state.
    const auto* tp = f.find(prefix + "text_projection.weight");
    has_text_projection_ = (tp != nullptr);
    if (has_text_projection_) {
        upload_compute_checked(*tp, D, D, text_projection_, "text_projection.weight");
    }

    // Position-id buffer is fixed [0, 1, ..., P-1]. Upload once.
    std::vector<int32_t> positions(static_cast<std::size_t>(P));
    for (int i = 0; i < P; ++i) positions[static_cast<std::size_t>(i)] = i;
    positions_dev_ = make_idx_device(positions.data(), P);
}

// ─── forward ───────────────────────────────────────────────────────────────

void TextEncoder::forward(const int32_t* ids, bt::Tensor& out,
                          bt::Tensor* pooled) {
    if (!ids) fail("forward: ids pointer is null");
    if (token_embed_.size() == 0) fail("forward: weights not loaded");
    if (positions_dev_.empty()) fail("forward: position buffer not initialised");

    const int L = cfg_.max_position;
    const int H = cfg_.num_heads;

    // Upload token IDs and run embedding lookups.
    ids_dev_ = make_idx_device(ids, L);

    bt::embedding_lookup_forward(token_embed_,
                                 static_cast<const int32_t*>(ids_dev_.data),
                                 L, tok_emb_);
    bt::embedding_lookup_forward(position_embed_,
                                 static_cast<const int32_t*>(positions_dev_.data),
                                 L, pos_emb_);

    // x = tok_emb + pos_emb  (in tok_emb_, which we then alias as x_).
    bt::add_inplace(tok_emb_, pos_emb_);

    // Move the residual stream into x_ via a clone — we want a stable name
    // and we'll be feeding x_ into the per-layer code repeatedly.
    x_ = tok_emb_.clone();

    for (auto& layer : layers_) {
        // ── self-attention block ──────────────────────────────────────────
        detail::layernorm_batched(
            x_, layer.ln1_gamma, layer.ln1_beta, ln_out_, cfg_.layer_norm_eps);

        bt::flash_attention_qkvo_forward(
            ln_out_, /*Ctx=*/nullptr,
            layer.Wq, &layer.bq,
            layer.Wk, &layer.bk,
            layer.Wv, &layer.bv,
            layer.Wo, &layer.bo,
            /*d_mask=*/nullptr, H, /*causal=*/true,
            proj_out_);
        bt::add_inplace(x_, proj_out_);

        // ── MLP block ─────────────────────────────────────────────────────
        detail::layernorm_batched(
            x_, layer.ln2_gamma, layer.ln2_beta, ln_out_, cfg_.layer_norm_eps);

        detail::linear_batched(layer.fc1_W, &layer.fc1_b, ln_out_, ffn_mid_);
        bt::quick_gelu_forward(ffn_mid_, ffn_act_);
        detail::linear_batched(layer.fc2_W, &layer.fc2_b, ffn_act_, ffn_out_);

        bt::add_inplace(x_, ffn_out_);
    }

    detail::layernorm_batched(
        x_, final_gamma_, final_beta_, out, cfg_.layer_norm_eps);

    if (pooled != nullptr) {
        // First occurrence of the EOS token marks end-of-text.
        const int D = cfg_.hidden_dim;
        int eos_pos = L - 1;
        for (int i = 0; i < L; ++i) {
            if (ids[i] == cfg_.eos_token_id) { eos_pos = i; break; }
        }
        if (has_text_projection_) {
            // pooled = eos_row @ text_projection. Copy the EOS row out into a
            // (1, D) scratch, then run the bias-free batched linear.
            detail::resize_like(pooled_eos_, 1, D, compute_dtype(), out.device);
            bt::copy_d2d(out, /*src_off=*/eos_pos * D,
                         pooled_eos_, /*dst_off=*/0, /*count=*/D);
            detail::linear_batched(text_projection_, /*bias=*/nullptr,
                                   pooled_eos_, *pooled);
        } else {
            detail::resize_like(*pooled, 1, D, compute_dtype(), out.device);
            bt::copy_d2d(out, /*src_off=*/eos_pos * D,
                         *pooled, /*dst_off=*/0, /*count=*/D);
        }
    }
}

// ─── LoRA merge ────────────────────────────────────────────────────────────

namespace {

bool parse_clip_target_(const std::string& target_path,
                        int& layer_idx, std::string& proj) {
    // Expected: "text_model.encoder.layers.<i>.self_attn.<q|k|v|out>_proj".
    static const std::string head = "text_model.encoder.layers.";
    if (target_path.compare(0, head.size(), head) != 0) return false;
    std::size_t p = head.size();
    if (p >= target_path.size() ||
        !std::isdigit(static_cast<unsigned char>(target_path[p]))) {
        return false;
    }
    int v = 0;
    while (p < target_path.size() &&
           std::isdigit(static_cast<unsigned char>(target_path[p]))) {
        v = v * 10 + (target_path[p] - '0');
        ++p;
    }
    layer_idx = v;
    static const std::string mid = ".self_attn.";
    if (target_path.compare(p, mid.size(), mid) != 0) return false;
    proj = target_path.substr(p + mid.size());
    return (proj == "q_proj" || proj == "k_proj" ||
            proj == "v_proj" || proj == "out_proj");
}

}  // namespace

void TextEncoder::apply_lora_delta(const std::string& target_path,
                                   const st::TensorView& lora_down,
                                   const st::TensorView& lora_up,
                                   float scale_total) {
    int layer_idx = 0;
    std::string proj;
    if (!parse_clip_target_(target_path, layer_idx, proj)) {
        fail("apply_lora_delta: unknown target '" + target_path + "'");
    }
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layers_.size())) {
        fail("apply_lora_delta: layer index " + std::to_string(layer_idx) +
             " out of range");
    }
    Layer& L = layers_[static_cast<std::size_t>(layer_idx)];
    bt::Tensor* W = nullptr;
    if (proj == "q_proj")        W = &L.Wq;
    else if (proj == "k_proj")   W = &L.Wk;
    else if (proj == "v_proj")   W = &L.Wv;
    else if (proj == "out_proj") W = &L.Wo;
    if (!W || W->size() == 0) {
        fail("apply_lora_delta: target '" + target_path +
             "' has no weights loaded");
    }
    const int W_rows = W->rows;
    const int W_cols = W->cols;

    // lora_down: (rank, in_dim); lora_up: (out_dim, rank). CLIP targets are
    // pure linear (out=in=D), so the shape is always 2-D.
    if (lora_down.shape.size() != 2 || lora_up.shape.size() != 2) {
        fail("apply_lora_delta: CLIP LoRA tensors must be 2-D");
    }
    const int rank = static_cast<int>(lora_down.shape[0]);
    if (rank <= 0) fail("apply_lora_delta: rank <= 0");
    if (static_cast<int>(lora_up.shape[1]) != rank) {
        fail("apply_lora_delta: lora_up.cols (" +
             std::to_string(lora_up.shape[1]) + ") != rank (" +
             std::to_string(rank) + ")");
    }
    if (static_cast<int>(lora_up.shape[0]) != W_rows ||
        static_cast<int>(lora_down.shape[1]) != W_cols) {
        fail("apply_lora_delta: (out_dim, in_dim) shape mismatch for '" +
             target_path + "'");
    }

    bt::Tensor down_g, up_g;
    upload_compute_checked(lora_down, rank,   W_cols, down_g, target_path + ".lora_down");
    upload_compute_checked(lora_up,   W_rows, rank,   up_g,   target_path + ".lora_up");

    bt::Tensor delta;
    bt::matmul(up_g, down_g, delta);
    bt::scale_inplace(delta, scale_total);
    bt::add_inplace(*W, delta);
}

}  // namespace brodiffusion::clip
