#include "brodiffusion/clip_image.h"
#include "brodiffusion/safetensors.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::clip_image {

namespace bt = ::brotensor;
namespace st = ::brodiffusion::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("clip_image::ImageEncoder: " + msg);
}

void upload_fp16_checked(const st::TensorView& v, int rows, int cols,
                         bt::GpuTensor& dst, const char* name) {
    if (v.dtype != st::Dtype::F16 && v.dtype != st::Dtype::F32) {
        fail(std::string(name) + ": expected F16 or F32, got " +
             st::dtype_name(v.dtype));
    }
    int64_t expected = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    if (v.numel() != expected) {
        fail(std::string(name) + ": shape mismatch (expected " +
             std::to_string(rows) + "x" + std::to_string(cols) + ", got " +
             std::to_string(v.numel()) + " elements)");
    }
    st::upload_fp16(v, rows, cols, dst);
}

const st::TensorView* find_or(const st::File& f, const std::string& key) {
    return f.find(key);
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    const auto* v = f.find(key);
    if (!v) throw std::runtime_error("clip_image::ImageEncoder: missing tensor '" + key + "'");
    return *v;
}

}  // namespace

ImageEncoder::ImageEncoder(const ImageEncoderConfig& cfg) : cfg_(cfg) {
    if (cfg_.hidden_dim <= 0 || cfg_.num_heads <= 0 ||
        cfg_.hidden_dim % cfg_.num_heads != 0) {
        fail("hidden_dim must be a positive multiple of num_heads");
    }
    if (cfg_.num_layers <= 0 || cfg_.patch_size <= 0 ||
        cfg_.image_size <= 0 ||
        cfg_.image_size % cfg_.patch_size != 0) {
        fail("image_size must be a positive multiple of patch_size");
    }
    layers_.resize(static_cast<std::size_t>(cfg_.num_layers));
}

ImageEncoder::~ImageEncoder() = default;

// ─── load_weights ──────────────────────────────────────────────────────────

void ImageEncoder::load_weights(const st::File& f,
                                const std::string& prefix,
                                bool pre_layrnorm_alt) {
    const int D = cfg_.hidden_dim;
    const int F = cfg_.intermediate_dim;
    const int C = cfg_.in_channels;
    const int P = cfg_.patch_size;
    const int T = num_tokens(cfg_);

    // patch_embed conv: (D, C*P*P). HF stores (D, C, P, P) which is the
    // same C-contiguous layout brotensor's conv2d expects.
    upload_fp16_checked(need(f, prefix + "embeddings.patch_embedding.weight"),
                        D, C * P * P, patch_W_, "patch_embedding.weight");

    // class_embedding: (D,) stored 1-D in HF. Load as (1, D) row to ease
    // the copy into seq_ row 0.
    upload_fp16_checked(need(f, prefix + "embeddings.class_embedding"),
                        1, D, class_embed_, "class_embedding");

    upload_fp16_checked(need(f, prefix + "embeddings.position_embedding.weight"),
                        T, D, position_embed_, "position_embedding");

    // pre_layrnorm is the upstream HF typo. Accept either spelling.
    const std::string pre_key_typo = prefix + "pre_layrnorm.";
    const std::string pre_key_alt  = prefix + "pre_layernorm.";
    const auto* pre_g_v = find_or(f, pre_key_typo + "weight");
    const auto* pre_b_v = find_or(f, pre_key_typo + "bias");
    if (!pre_g_v && pre_layrnorm_alt) {
        pre_g_v = find_or(f, pre_key_alt + "weight");
        pre_b_v = find_or(f, pre_key_alt + "bias");
    }
    if (!pre_g_v || !pre_b_v) {
        fail("missing pre_layrnorm/pre_layernorm weight/bias");
    }
    upload_fp16_checked(*pre_g_v, D, 1, pre_g_, "pre_ln.weight");
    upload_fp16_checked(*pre_b_v, D, 1, pre_b_, "pre_ln.bias");

    for (int i = 0; i < cfg_.num_layers; ++i) {
        const std::string p = prefix + "encoder.layers." + std::to_string(i) + ".";
        Layer& L = layers_[static_cast<std::size_t>(i)];

        upload_fp16_checked(need(f, p + "layer_norm1.weight"), D, 1, L.ln1_g, "ln1.weight");
        upload_fp16_checked(need(f, p + "layer_norm1.bias"),   D, 1, L.ln1_b, "ln1.bias");

        upload_fp16_checked(need(f, p + "self_attn.q_proj.weight"), D, D, L.Wq, "q_proj.W");
        upload_fp16_checked(need(f, p + "self_attn.q_proj.bias"),   D, 1, L.bq, "q_proj.b");
        upload_fp16_checked(need(f, p + "self_attn.k_proj.weight"), D, D, L.Wk, "k_proj.W");
        upload_fp16_checked(need(f, p + "self_attn.k_proj.bias"),   D, 1, L.bk, "k_proj.b");
        upload_fp16_checked(need(f, p + "self_attn.v_proj.weight"), D, D, L.Wv, "v_proj.W");
        upload_fp16_checked(need(f, p + "self_attn.v_proj.bias"),   D, 1, L.bv, "v_proj.b");
        upload_fp16_checked(need(f, p + "self_attn.out_proj.weight"), D, D, L.Wo, "out_proj.W");
        upload_fp16_checked(need(f, p + "self_attn.out_proj.bias"),   D, 1, L.bo, "out_proj.b");

        upload_fp16_checked(need(f, p + "layer_norm2.weight"), D, 1, L.ln2_g, "ln2.weight");
        upload_fp16_checked(need(f, p + "layer_norm2.bias"),   D, 1, L.ln2_b, "ln2.bias");

        upload_fp16_checked(need(f, p + "mlp.fc1.weight"), F, D, L.fc1_W, "fc1.W");
        upload_fp16_checked(need(f, p + "mlp.fc1.bias"),   F, 1, L.fc1_b, "fc1.b");
        upload_fp16_checked(need(f, p + "mlp.fc2.weight"), D, F, L.fc2_W, "fc2.W");
        upload_fp16_checked(need(f, p + "mlp.fc2.bias"),   D, 1, L.fc2_b, "fc2.b");
    }

    upload_fp16_checked(need(f, prefix + "post_layernorm.weight"),
                        D, 1, post_g_, "post_ln.weight");
    upload_fp16_checked(need(f, prefix + "post_layernorm.bias"),
                        D, 1, post_b_, "post_ln.bias");
}

// ─── forward ───────────────────────────────────────────────────────────────

void ImageEncoder::forward(const bt::GpuTensor& pixels, bt::GpuTensor& cls_out) {
    if (patch_W_.size() == 0) fail("forward: weights not loaded");

    const int D = cfg_.hidden_dim;
    const int H = cfg_.num_heads;
    const int P = cfg_.patch_size;
    const int C = cfg_.in_channels;
    const int S = cfg_.image_size;
    const int G = S / P;                   // grid side (16 for 224/14)
    const int T = 1 + G * G;               // 257 tokens
    const float eps = cfg_.layer_norm_eps;

    // ── patch embed: (1, 3*S*S) -> (1, D*G*G) via stride-P conv2d ──────────
    // No bias on patch_embedding in HF CLIP.
    bt::conv2d_forward_gpu(pixels, patch_W_, /*bias=*/nullptr,
                           /*N=*/1, C, S, S, D, P, P,
                           /*stride_h=*/P, /*stride_w=*/P,
                           /*pad_h=*/0, /*pad_w=*/0,
                           /*dil_h=*/1, /*dil_w=*/1,
                           patch_nchw_);

    // ── reshape NCHW -> sequence: (1, D*G*G) -> (G*G, D) ───────────────────
    // Reuse ln_out_ as the (256, D) holding buffer for the patch sequence —
    // we'll copy it into seq_ rows [1..T-1] in the next step.
    bt::nchw_to_sequence_gpu(patch_nchw_, /*N=*/1, /*C=*/D, /*H=*/G, /*W=*/G, ln_out_);

    // ── build (T, D) sequence: [CLS; patches] + position_embed ─────────────
    seq_.resize(T, D, bt::Dtype::FP16);
    // Row 0 = class_embedding.
    bt::copy_d2d_gpu(class_embed_, /*src_off=*/0,
                     seq_,         /*dst_off=*/0,
                     /*count=*/D);
    // Rows 1..T-1 = patch sequence.
    bt::copy_d2d_gpu(ln_out_, /*src_off=*/0,
                     seq_,    /*dst_off=*/D,
                     /*count=*/(T - 1) * D);
    // x += position_embed.
    bt::add_inplace_gpu(seq_, position_embed_);

    // ── pre-LN over the full sequence ──────────────────────────────────────
    bt::layernorm_forward_inference_batched_fp16_gpu(
        seq_, pre_g_, pre_b_, ln_out_, eps);
    seq_ = ln_out_.clone();

    // ── 24 transformer layers (non-causal, biased Q/K/V/O, QuickGELU MLP) ──
    for (auto& L : layers_) {
        bt::layernorm_forward_inference_batched_fp16_gpu(
            seq_, L.ln1_g, L.ln1_b, ln_out_, eps);

        bt::flash_attention_qkvo_forward_gpu(
            ln_out_, /*Ctx=*/nullptr,
            L.Wq, &L.bq, L.Wk, &L.bk, L.Wv, &L.bv, L.Wo, &L.bo,
            /*d_mask=*/nullptr, H, /*causal=*/false,
            proj_out_);
        bt::add_inplace_gpu(seq_, proj_out_);

        bt::layernorm_forward_inference_batched_fp16_gpu(
            seq_, L.ln2_g, L.ln2_b, ln_out_, eps);
        bt::linear_forward_batched_fp16_gpu(L.fc1_W, &L.fc1_b, ln_out_, ffn_mid_);
        bt::quick_gelu_forward_gpu(ffn_mid_, ffn_act_);
        bt::linear_forward_batched_fp16_gpu(L.fc2_W, &L.fc2_b, ffn_act_, ffn_out_);
        bt::add_inplace_gpu(seq_, ffn_out_);
    }

    // ── post-LN, then take CLS row 0 ──────────────────────────────────────
    // HF CLIPVisionTransformer applies post_layernorm to the *pooled* CLS
    // output only. Copy row 0 first, LN that 1×D row, emit.
    cls_out.resize(1, D, bt::Dtype::FP16);
    bt::copy_d2d_gpu(seq_, /*src_off=*/0, cls_out, /*dst_off=*/0, /*count=*/D);
    // In-place LN on cls_out by writing through ln_out_.
    bt::layernorm_forward_inference_batched_fp16_gpu(
        cls_out, post_g_, post_b_, ln_out_, eps);
    cls_out = ln_out_.clone();
}

}  // namespace brodiffusion::clip_image
