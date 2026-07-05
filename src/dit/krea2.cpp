#include "brodiffusion/dit/krea2.h"

#include "brodiffusion/dit/common.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"
#include "brotensor/detail/cpu/thread_pool.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::dit {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("dit::Krea2Transformer2DModel: " + msg);
}

const st::TensorView& need(const std::vector<const st::File*>& shards,
                           const std::string& key) {
    for (const st::File* f : shards) {
        if (const auto* v = f->find(key)) return *v;
    }
    fail("missing tensor '" + key + "'");
}

// Download a checked view to host FP32 regardless of storage dtype.
std::vector<float> view_to_fp32(const st::TensorView& v, int rows, int cols,
                                const std::string& name) {
    const std::int64_t expected =
        static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(cols);
    if (v.numel() != expected) {
        fail(name + " ('" + v.name + "'): shape mismatch (expected " +
             std::to_string(rows) + "x" + std::to_string(cols) + ")");
    }
    const std::size_t n = static_cast<std::size_t>(expected);
    std::vector<float> out(n);
    if (v.dtype == st::Dtype::F32) {
        std::memcpy(out.data(), v.data, n * 4);
    } else if (v.dtype == st::Dtype::F16) {
        const auto* b = reinterpret_cast<const std::uint16_t*>(v.data);
        for (std::size_t i = 0; i < n; ++i) out[i] = bt::fp16_bits_to_fp32(b[i]);
    } else if (v.dtype == st::Dtype::BF16) {
        const auto* b = reinterpret_cast<const std::uint16_t*>(v.data);
        for (std::size_t i = 0; i < n; ++i) out[i] = bt::bf16_bits_to_fp32(b[i]);
    } else {
        fail(name + " ('" + v.name + "'): expected F16/F32/BF16");
    }
    return out;
}

// Upload host FP32 values at the given dtype.
bt::Tensor upload_as(const std::vector<float>& h, int rows, int cols,
                     bt::Dtype dt) {
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    if (dt == bt::Dtype::BF16) {
        std::vector<std::uint16_t> bits(n);
        for (std::size_t i = 0; i < n; ++i) bits[i] = bt::fp32_to_bf16_bits(h[i]);
        return bt::Tensor::from_host_bf16(bits.data(), rows, cols);
    }
    if (dt == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        for (std::size_t i = 0; i < n; ++i) bits[i] = bt::fp32_to_fp16_bits(h[i]);
        return bt::Tensor::from_host_fp16(bits.data(), rows, cols);
    }
    return bt::Tensor::from_host(h.data(), rows, cols);
}

// ── activation / norm helpers (see krea2.h: RMSNorm gain is 1+weight, FP32) ──

// gelu with the tanh approximation (reference approximate="tanh").
void gelu_tanh_inplace(bt::Tensor& t) { bt::gelu_forward(t, t); }

// RMSNorm with a preloaded (1+weight) FP32 gain. rms_norm_forward accepts
// the FP32 gain against FP16/BF16 activations directly (its kernels
// accumulate and multiply in FP32 internally, rounding once at the store),
// so this is a single kernel call with no cast round-trip and no gain
// precision loss — matching the reference's _keep_in_fp32_modules.
bt::Tensor rmsnorm(const bt::Tensor& X, const bt::Tensor& gain, float eps) {
    bt::Tensor out;
    bt::rms_norm_forward(X, gain, eps, out);
    return out;
}

// Per-head RMSNorm: view (L, nh*hd) as (L*nh, hd), norm, relabel (L, nh*hd).
bt::Tensor headnorm(const bt::Tensor& X, const bt::Tensor& gain, float eps,
                    int nh, int hd) {
    const int L = X.rows;
    bt::Tensor xv = bt::Tensor::view(X.device, X.data, L * nh, hd, X.dtype);
    bt::Tensor normed = rmsnorm(xv, gain, eps);        // owns (L*nh, hd)
    normed.rows = L;
    normed.cols = nh * hd;
    return normed;
}

// Build a device INT32 buffer from host values (varlen cu_seqlens).
bt::Tensor make_idx_device(const std::vector<int32_t>& host) {
    const int n = static_cast<int>(host.size());
    bt::Tensor cpu = bt::Tensor::empty_on(bt::Device::CPU, n, 1, bt::Dtype::INT32);
    std::memcpy(cpu.host_raw_mut(), host.data(),
                static_cast<std::size_t>(n) * sizeof(int32_t));
    return cpu.to(bt::default_device());
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

Krea2Transformer2DModel::Krea2Transformer2DModel(const Krea2Config& cfg)
    : cfg_(cfg) {
    if (cfg_.num_attention_heads <= 0 || cfg_.attention_head_dim <= 0) {
        fail("num_attention_heads / attention_head_dim must be positive");
    }
    if (cfg_.num_attention_heads % cfg_.num_key_value_heads != 0) {
        fail("num_attention_heads must be a multiple of num_key_value_heads");
    }
    int sum = 0;
    for (int d : cfg_.axes_dims_rope) sum += d;
    if (sum != cfg_.attention_head_dim) {
        fail("axes_dims_rope must sum to attention_head_dim");
    }
    if (cfg_.text_hidden_dim % cfg_.text_num_attention_heads != 0) {
        fail("text_hidden_dim must be a multiple of text_num_attention_heads");
    }
    if (cfg_.text_num_attention_heads != cfg_.text_num_key_value_heads) {
        fail("text fusion attention must be plain MHA (nq == nkv)");
    }
    layerwise_blocks_.resize(static_cast<std::size_t>(cfg_.num_layerwise_text_blocks));
    refiner_blocks_.resize(static_cast<std::size_t>(cfg_.num_refiner_text_blocks));
    blocks_.resize(static_cast<std::size_t>(cfg_.num_layers));
}

Krea2Transformer2DModel::~Krea2Transformer2DModel() = default;

bt::Dtype Krea2Transformer2DModel::compute_dtype() const {
    return flux_compute_dtype();
}

// ─── load_weights ──────────────────────────────────────────────────────────

void Krea2Transformer2DModel::load_weights(const st::File& f,
                                           const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights(shards, prefix);
}

void Krea2Transformer2DModel::load_weights(
    const std::vector<const st::File*>& shards, const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no shards");
    load_impl_(shards, prefix);
    loaded_ = true;
}

void Krea2Transformer2DModel::load_impl_(
    const std::vector<const st::File*>& shards, const std::string& prefix) {
    const bt::Dtype dt = flux_compute_dtype();
    const int H = cfg_.hidden_size();
    const int IC = cfg_.in_channels;
    const int TE = cfg_.timestep_embed_dim;
    const int TH = cfg_.text_hidden_dim;
    const int hd_img = cfg_.attention_head_dim;
    const int nq_img = cfg_.num_attention_heads;
    const int nkv_img = cfg_.num_key_value_heads;
    const int hd_txt = TH / cfg_.text_num_attention_heads;
    const int nq_txt = cfg_.text_num_attention_heads;

    auto lin = [&](const std::string& key, int out, int in, bool bias,
                   Linear& l) {
        l.W = upload_as(view_to_fp32(need(shards, prefix + key + ".weight"),
                                     out, in, key),
                        out, in, dt);
        if (bias) {
            l.b = upload_as(view_to_fp32(need(shards, prefix + key + ".bias"),
                                         out, 1, key),
                            out, 1, dt);
        }
    };
    // RMSNorm gain: store (1 + weight) in FP32 (see krea2.h). rms_norm_forward
    // takes the FP32 gain directly against 16-bit activations — full gain
    // precision (the reference's _keep_in_fp32_modules) without upcasting
    // the activation tensor.
    auto norm = [&](const std::string& key, int dim, bt::Tensor& g) {
        std::vector<float> h = view_to_fp32(need(shards, prefix + key + ".weight"),
                                            dim, 1, key);
        for (float& x : h) x += 1.0f;
        g = bt::Tensor::from_host(h.data(), dim, 1).to(bt::default_device());
    };
    auto raw = [&](const std::string& key, int rows, int cols, bt::Tensor& t) {
        t = upload_as(view_to_fp32(need(shards, prefix + key), rows, cols, key),
                      rows, cols, dt);
    };

    bool quant = cfg_.quantize_weights;
    if (quant && bt::default_device() != bt::Device::CUDA) {
        std::fprintf(stderr,
            "Krea2Transformer2DModel: quantize_weights requested but the "
            "default device is not CUDA — loading dense weights instead (the "
            "fused INT8 dequant matmuls are GPU-only)\n");
        quant = false;
    }

    // Quantizing loader for the big per-block linears: converts the on-disk
    // weight (F16/F32/BF16) to FP16 bits host-side, quantizes to INT8 with
    // per-output-row symmetric FP32 scales, and uploads only the INT8 copy —
    // the dense weight never lands on the device, so peak VRAM during load is
    // the INT8 footprint (~12 GB), not the BF16 one (~24.6 GB). The bias (if
    // any) uploads dense at the compute dtype, exactly as the dense path does.
    auto lin_q = [&](const std::string& key, int out, int in, bool bias,
                     Linear& l) {
        if (!quant) { lin(key, out, in, bias, l); return; }
        const st::TensorView& wv = need(shards, prefix + key + ".weight");
        const std::int64_t expected =
            static_cast<std::int64_t>(out) * static_cast<std::int64_t>(in);
        if (wv.numel() != expected) {
            fail(key + ".weight: shape mismatch (expected " +
                 std::to_string(out) + "x" + std::to_string(in) + ")");
        }
        const std::size_t n = static_cast<std::size_t>(expected);
        std::vector<std::uint16_t> w16(n);
        if (wv.dtype == st::Dtype::F16) {
            std::memcpy(w16.data(), wv.data, n * 2);
        } else if (wv.dtype == st::Dtype::BF16) {
            // Row-parallel conversion: the 25 GB BF16 checkpoint makes this
            // loop a large share of quantized-load time when run serially.
            const auto* src = reinterpret_cast<const std::uint16_t*>(wv.data);
            bt::detail::cpu::parallel_for(
                static_cast<std::size_t>(out), [&](std::size_t r) {
                    const std::size_t base = r * static_cast<std::size_t>(in);
                    for (std::size_t i = base;
                         i < base + static_cast<std::size_t>(in); ++i) {
                        w16[i] = bt::fp32_to_fp16_bits(
                            bt::bf16_bits_to_fp32(src[i]));
                    }
                });
        } else if (wv.dtype == st::Dtype::F32) {
            const auto* src = reinterpret_cast<const float*>(wv.data);
            for (std::size_t i = 0; i < n; ++i) {
                w16[i] = bt::fp32_to_fp16_bits(src[i]);
            }
        } else {
            fail(key + ".weight: expected F16/F32/BF16");
        }
        std::vector<std::int8_t> q(n);
        std::vector<float> sc(static_cast<std::size_t>(out));
        bt::quantize_int8_per_row_host(w16.data(), out, in, q.data(), sc.data());
        l.W_int8 = bt::Tensor::from_host_int8(q.data(), out, in);
        l.scales = bt::Tensor::from_host(sc.data(), out, 1);
        if (bias) {
            l.b = upload_as(view_to_fp32(need(shards, prefix + key + ".bias"),
                                         out, 1, key),
                            out, 1, dt);
        }
    };

    auto load_attn = [&](const std::string& p, int dim, int hd, int nq, int nkv,
                         Attention& a) {
        lin_q(p + "to_q", hd * nq, dim, false, a.to_q);
        lin_q(p + "to_k", hd * nkv, dim, false, a.to_k);
        lin_q(p + "to_v", hd * nkv, dim, false, a.to_v);
        lin_q(p + "to_gate", dim, dim, false, a.to_gate);
        lin_q(p + "to_out.0", dim, dim, false, a.to_out);
        norm(p + "norm_q", hd, a.norm_q);
        norm(p + "norm_k", hd, a.norm_k);
    };
    auto load_ff = [&](const std::string& p, int dim, int inter, SwiGLU& f) {
        lin_q(p + "gate", inter, dim, false, f.gate);
        lin_q(p + "up", inter, dim, false, f.up);
        lin_q(p + "down", dim, inter, false, f.down);
    };
    auto load_fusion = [&](const std::string& p, FusionBlock& b) {
        norm(p + "norm1", TH, b.norm1);
        norm(p + "norm2", TH, b.norm2);
        load_attn(p + "attn.", TH, hd_txt, nq_txt, nq_txt, b.attn);
        load_ff(p + "ff.", TH, cfg_.text_intermediate_size, b.ff);
    };

    lin("img_in", H, IC, true, img_in_);
    lin("time_embed.linear_1", H, TE, true, time_l1_);
    lin("time_embed.linear_2", H, H, true, time_l2_);
    lin("time_mod_proj", 6 * H, H, true, time_mod_proj_);

    for (int i = 0; i < cfg_.num_layerwise_text_blocks; ++i) {
        load_fusion("text_fusion.layerwise_blocks." + std::to_string(i) + ".",
                    layerwise_blocks_[static_cast<std::size_t>(i)]);
    }
    raw("text_fusion.projector.weight", 1, cfg_.num_text_layers, projector_);
    for (int i = 0; i < cfg_.num_refiner_text_blocks; ++i) {
        load_fusion("text_fusion.refiner_blocks." + std::to_string(i) + ".",
                    refiner_blocks_[static_cast<std::size_t>(i)]);
    }

    norm("txt_in.norm", TH, txt_in_norm_);
    lin("txt_in.linear_1", H, TH, true, txt_in_l1_);
    lin("txt_in.linear_2", H, H, true, txt_in_l2_);

    for (int i = 0; i < cfg_.num_layers; ++i) {
        const std::string p = "transformer_blocks." + std::to_string(i) + ".";
        TransformerBlock& b = blocks_[static_cast<std::size_t>(i)];
        raw(p + "scale_shift_table", 6, H, b.scale_shift_table);
        norm(p + "norm1", H, b.norm1);
        norm(p + "norm2", H, b.norm2);
        load_attn(p + "attn.", H, hd_img, nq_img, nkv_img, b.attn);
        load_ff(p + "ff.", H, cfg_.intermediate_size, b.ff);
    }

    raw("final_layer.scale_shift_table", 2, H, final_scale_shift_table_);
    norm("final_layer.norm", H, final_norm_);
    lin("final_layer.linear", IC, H, true, final_linear_);
}

// ─── forward ───────────────────────────────────────────────────────────────

bt::Tensor Krea2Transformer2DModel::linb_(const Linear& l, const bt::Tensor& X) {
    bt::Tensor Y;
    if (l.quantized()) {
        bt::linear_forward_batched_int8w_fp16(
            l.W_int8, l.scales, l.has_bias() ? &l.b : nullptr, X, Y);
    } else {
        detail::linear_batched(l.W, l.has_bias() ? &l.b : nullptr, X, Y);
    }
    return Y;
}

void Krea2Transformer2DModel::encode_text(const bt::Tensor& prompt_embeds,
                                          const bt::Tensor& prompt_embeds_mask,
                                          bt::Tensor& txt_out) {
    if (!loaded_) fail("encode_text: weights not loaded");
    const bt::Dtype dt = flux_compute_dtype();
    const int NL = cfg_.num_text_layers;
    const int TH = cfg_.text_hidden_dim;
    const int hd_txt = TH / cfg_.text_num_attention_heads;
    const int nq_txt = cfg_.text_num_attention_heads;
    if (prompt_embeds.cols != TH || prompt_embeds.rows % NL != 0) {
        fail("encode_text: prompt_embeds shape mismatch");
    }
    const int text_seq = prompt_embeds.rows / NL;
    if (prompt_embeds_mask.rows != text_seq) {
        fail("encode_text: prompt_embeds_mask length mismatch");
    }

    auto linb = [&](const Linear& l, const bt::Tensor& X) { return linb_(l, X); };

    bt::Tensor hs;   // (text_seq*NL, TH)
    if (prompt_embeds.dtype != dt) bt::cast(prompt_embeds, hs, dt);
    else hs = prompt_embeds.to(bt::default_device());

    // ── compact to the valid tokens ──────────────────────────────────────
    // Every text row carries the identity RoPE position and masked rows are
    // excluded from every attention (fusion and body alike), so dropping
    // them is exact — and shrinks both the fusion stack below and the joint
    // sequence every body block runs on (512-token block → ~a few dozen).
    bt::Tensor mask_f32 = prompt_embeds_mask;
    if (mask_f32.dtype != bt::Dtype::FP32) {
        bt::Tensor t; bt::cast(prompt_embeds_mask, t, bt::Dtype::FP32); mask_f32 = t;
    }
    std::vector<float> mask_h = mask_f32.to(bt::Device::CPU).to_host_vector();
    std::vector<int> valid;
    valid.reserve(static_cast<std::size_t>(text_seq));
    for (int i = 0; i < text_seq; ++i) {
        if (mask_h[static_cast<std::size_t>(i)] > 0.5f) valid.push_back(i);
    }
    if (valid.empty()) fail("encode_text: mask has no valid tokens");
    const int n_valid = static_cast<int>(valid.size());
    if (n_valid < text_seq) {
        bt::Tensor hs_c;
        detail::resize_like(hs_c, n_valid * NL, TH, dt, bt::default_device());
        for (int v = 0; v < n_valid; ++v) {
            bt::copy_d2d(hs, valid[static_cast<std::size_t>(v)] * NL * TH,
                         hs_c, v * NL * TH, NL * TH);
        }
        hs = hs_c;
    }

    // Attention scratch, reused across every attention call.
    bt::Tensor Qr, Kr, Krep, Vrep;

    // Refiner attention (nq == nkv, no RoPE; all compacted rows are valid so
    // no mask either).
    auto attn_apply = [&](const Attention& a, const bt::Tensor& x) -> bt::Tensor {
        bt::Tensor q = linb(a.to_q, x);
        bt::Tensor k = linb(a.to_k, x);
        bt::Tensor v = linb(a.to_v, x);
        bt::Tensor gate = linb(a.to_gate, x);
        q = headnorm(q, a.norm_q, cfg_.norm_eps, nq_txt, hd_txt);
        k = headnorm(k, a.norm_k, cfg_.norm_eps, nq_txt, hd_txt);
        bt::Tensor attn;
        gqa_attention_masked(q, k, v, nullptr, nullptr, hd_txt, nq_txt,
                             nq_txt, nullptr, attn, Qr, Kr, Krep, Vrep);
        bt::sigmoid_forward(gate, gate);
        bt::mul_inplace(attn, gate);
        return linb(a.to_out, attn);
    };

    // FF sublayer (SwiGLU): down(silu(gate(x)) * up(x)).
    auto ff_apply = [&](const SwiGLU& f, const bt::Tensor& x) -> bt::Tensor {
        bt::Tensor g = linb(f.gate, x);
        bt::silu_forward(g, g);
        bt::Tensor u = linb(f.up, x);
        bt::mul_inplace(g, u);
        return linb(f.down, g);
    };

    // Layerwise blocks: attention over the NL-tap axis, batched per token.
    std::vector<int32_t> cu(static_cast<std::size_t>(n_valid + 1));
    for (int i = 0; i <= n_valid; ++i) cu[static_cast<std::size_t>(i)] = i * NL;
    bt::Tensor cu_dev = make_idx_device(cu);
    const int32_t* cu_ptr = static_cast<const int32_t*>(cu_dev.data);

    auto layerwise_attn = [&](const Attention& a, const bt::Tensor& x) -> bt::Tensor {
        bt::Tensor q = linb(a.to_q, x);
        bt::Tensor k = linb(a.to_k, x);
        bt::Tensor v = linb(a.to_v, x);
        bt::Tensor gate = linb(a.to_gate, x);
        q = headnorm(q, a.norm_q, cfg_.norm_eps, nq_txt, hd_txt);
        k = headnorm(k, a.norm_k, cfg_.norm_eps, nq_txt, hd_txt);
        bt::Tensor attn;
        bt::flash_attention_varlen_forward(q, k, v, cu_ptr, cu_ptr, n_valid,
                                           NL, NL, nq_txt, hd_txt,
                                           /*causal=*/false, attn);
        bt::sigmoid_forward(gate, gate);
        bt::mul_inplace(attn, gate);
        return linb(a.to_out, attn);
    };

    for (const FusionBlock& b : layerwise_blocks_) {
        bt::Tensor n1 = rmsnorm(hs, b.norm1, cfg_.norm_eps);
        bt::Tensor ao = layerwise_attn(b.attn, n1);
        bt::add_inplace(hs, ao);
        bt::Tensor n2 = rmsnorm(hs, b.norm2, cfg_.norm_eps);
        bt::Tensor fo = ff_apply(b.ff, n2);
        bt::add_inplace(hs, fo);
    }

    // Projector: collapse the NL-tap axis (Linear NL->1) into (n_valid, TH).
    std::vector<float> pw;
    {
        bt::Tensor pf = projector_;
        if (pf.dtype != bt::Dtype::FP32) { bt::Tensor t; bt::cast(projector_, t, bt::Dtype::FP32); pf = t; }
        pw = pf.to(bt::Device::CPU).to_host_vector();
    }
    bt::Tensor fused = bt::Tensor::zeros_on(bt::default_device(), n_valid, TH, dt);
    bt::Tensor gath;
    detail::resize_like(gath, n_valid, TH, dt, bt::default_device());
    for (int l = 0; l < NL; ++l) {
        bt::copy_d2d_strided(hs, l * TH, NL * TH, gath, 0, TH, TH, n_valid);
        bt::axpby_inplace(fused, gath, 1.0f, pw[static_cast<std::size_t>(l)]);
    }

    // Refiner blocks: attention over the (compacted) token sequence, no RoPE.
    for (const FusionBlock& b : refiner_blocks_) {
        bt::Tensor n1 = rmsnorm(fused, b.norm1, cfg_.norm_eps);
        bt::Tensor ao = attn_apply(b.attn, n1);
        bt::add_inplace(fused, ao);
        bt::Tensor n2 = rmsnorm(fused, b.norm2, cfg_.norm_eps);
        bt::Tensor fo = ff_apply(b.ff, n2);
        bt::add_inplace(fused, fo);
    }

    // txt_in (Krea2TextProjection): norm -> linear_1 -> gelu(tanh) -> linear_2.
    bt::Tensor txt = rmsnorm(fused, txt_in_norm_, cfg_.norm_eps);
    txt = linb(txt_in_l1_, txt);
    gelu_tanh_inplace(txt);
    txt_out = linb(txt_in_l2_, txt);             // (n_valid, H)
}

void Krea2Transformer2DModel::time_embed_(float timestep, bt::Tensor& temb,
                                          bt::Tensor& temb_mod) {
    const bt::Dtype dt = flux_compute_dtype();
    const float ts_val = timestep * 1000.0f;   // Krea 2 scales flow-time by 1000
    bt::Tensor ts = bt::Tensor::from_host_on(bt::Device::CPU, &ts_val, 1, 1);
    bt::Tensor freq;
    bt::timestep_embedding(ts, cfg_.timestep_embed_dim, 10000.0f, freq);  // (1,TE) FP32
    bt::Tensor freq_dev = freq.to(bt::default_device());
    bt::Tensor freq_cd = freq_dev;
    if (dt != bt::Dtype::FP32) bt::cast(freq_dev, freq_cd, dt);
    temb = linb_(time_l1_, freq_cd);
    gelu_tanh_inplace(temb);
    temb = linb_(time_l2_, temb);                // (1, H)  raw time embedding
    bt::Tensor temb_gelu = temb.clone();
    gelu_tanh_inplace(temb_gelu);
    temb_mod = linb_(time_mod_proj_, temb_gelu); // (1, 6H)
}

void Krea2Transformer2DModel::set_mod_delta(const bt::Tensor& delta,
                                            int block_lo, int block_hi) {
    if (delta.size() == 0) {
        mod_delta_ = bt::Tensor();
        mod_delta_lo_ = mod_delta_hi_ = 0;
        return;
    }
    const int H = cfg_.hidden_size();
    if ((int64_t)delta.size() != (int64_t)6 * H)
        fail("set_mod_delta: delta must have 6*hidden_size elements");
    const bt::Dtype dt = flux_compute_dtype();
    bt::Tensor d = delta.to(bt::default_device());
    if (d.dtype != dt) { bt::Tensor t; bt::cast(d, t, dt); d = t; }
    mod_delta_ = d;
    mod_delta_lo_ = block_lo;
    mod_delta_hi_ = block_hi;
}

void Krea2Transformer2DModel::set_gate_scale(float txt_scale,
                                             float img_scale, int block_lo,
                                             int block_hi) {
    gate_txt_scale_ = txt_scale;
    gate_img_scale_ = img_scale;
    gate_lo_ = block_lo;
    gate_hi_ = block_hi;
}

void Krea2Transformer2DModel::set_gate_mask(const bt::Tensor& mask,
                                            int block_lo, int block_hi) {
    if (mask.size() == 0) {
        gate_mask_ = bt::Tensor();
        gate_mask_lo_ = gate_mask_hi_ = 0;
        return;
    }
    const bt::Dtype dt = flux_compute_dtype();
    bt::Tensor m = mask.to(bt::default_device());
    if (m.dtype != dt) { bt::Tensor t; bt::cast(m, t, dt); m = t; }
    gate_mask_ = m;
    gate_mask_lo_ = block_lo;
    gate_mask_hi_ = block_hi;
}

void Krea2Transformer2DModel::capture_gates(std::vector<float>* sink) {
    gate_sink_ = sink;
}

void Krea2Transformer2DModel::compute_time_mod(float timestep,
                                               bt::Tensor& temb_out,
                                               bt::Tensor& mod_out) {
    if (!loaded_) fail("compute_time_mod: weights not loaded");
    bt::Tensor temb, temb_mod;
    time_embed_(timestep, temb, temb_mod);
    if (temb.dtype != bt::Dtype::FP32) bt::cast(temb, temb_out, bt::Dtype::FP32);
    else temb_out = temb;
    if (temb_mod.dtype != bt::Dtype::FP32) bt::cast(temb_mod, mod_out, bt::Dtype::FP32);
    else mod_out = temb_mod;
    bt::sync_all();
}

void Krea2Transformer2DModel::forward_with_text(const bt::Tensor& packed_latent,
                                                int hp, int wp,
                                                const bt::Tensor& txt,
                                                float timestep, bt::Tensor& out) {
    if (!loaded_) fail("forward_with_text: weights not loaded");
    const bt::Dtype dt = flux_compute_dtype();
    const int H = cfg_.hidden_size();
    const int hd_img = cfg_.attention_head_dim;
    const int nq_img = cfg_.num_attention_heads;
    const int nkv_img = cfg_.num_key_value_heads;
    const int img_len = hp * wp;
    if (txt.cols != H || txt.dtype != dt) {
        fail("forward_with_text: txt must be (n_valid, hidden) at the "
             "compute dtype (see encode_text)");
    }
    const int text_seq = txt.rows;

    auto linb = [&](const Linear& l, const bt::Tensor& X) { return linb_(l, X); };

    // ── timestep embeddings ──────────────────────────────────────────────
    bt::Tensor temb, temb_mod;
    time_embed_(timestep, temb, temb_mod);

    // Attention scratch, reused across every attention call.
    bt::Tensor Qr, Kr, Krep, Vrep;

    // Body-block index, visible to attn_apply for the gate research hooks
    // (this lambda only ever runs for body blocks in this method).
    int block_idx = 0;

    // Attention sublayer (Krea2AttnProcessor): project q/k/v/gate, per-head
    // qk-norm, optional RoPE, GQA masked attention, sigmoid gate, out proj.
    auto attn_apply = [&](const Attention& a, const bt::Tensor& x,
                          int hd, int nq, int nkv, const bt::Tensor* cos,
                          const bt::Tensor* sin, const float* dmask) -> bt::Tensor {
        bt::Tensor q = linb(a.to_q, x);
        bt::Tensor k = linb(a.to_k, x);
        bt::Tensor v = linb(a.to_v, x);
        bt::Tensor gate = linb(a.to_gate, x);
        q = headnorm(q, a.norm_q, cfg_.norm_eps, nq, hd);
        k = headnorm(k, a.norm_k, cfg_.norm_eps, nkv, hd);
        bt::Tensor attn;
        gqa_attention_masked(q, k, v, cos, sin, hd, nq, nkv, dmask, attn,
                             Qr, Kr, Krep, Vrep);
        bt::sigmoid_forward(gate, gate);
        if (gate_sink_) {   // research: per-row mean gate for this block
            if ((int)gate_ones_.rows != H || gate_ones_.dtype != dt) {
                gate_ones_ = bt::Tensor::zeros_on(bt::default_device(), H, 1, dt);
                bt::add_scalar_inplace(gate_ones_, 1.0f);
            }
            bt::Tensor gm;
            bt::matmul(gate, gate_ones_, gm);           // (seq, 1)
            bt::Tensor gm32 = gm;
            if (gm.dtype != bt::Dtype::FP32) bt::cast(gm, gm32, bt::Dtype::FP32);
            bt::sync_all();
            bt::Tensor gh = gm32.to(bt::Device::CPU);
            const float* p = gh.host_f32();
            const float inv_h = 1.0f / (float)H;
            float* dst = gate_sink_->data() + (size_t)block_idx * gate.rows;
            for (int r = 0; r < gate.rows; ++r) dst[r] = p[r] * inv_h;
        }
        if (block_idx >= gate_lo_ && block_idx < gate_hi_) {  // research hook
            const int isz = bt::dtype_size_bytes(dt);
            if (gate_txt_scale_ != 1.0f && text_seq > 0) {
                bt::Tensor gt = bt::Tensor::view(
                    bt::default_device(), gate.data, text_seq, H, dt);
                bt::scale_inplace(gt, gate_txt_scale_);
            }
            if (gate_img_scale_ != 1.0f) {
                bt::Tensor gi = bt::Tensor::view(
                    bt::default_device(),
                    (char*)gate.data + (size_t)text_seq * H * isz,
                    gate.rows - text_seq, H, dt);
                bt::scale_inplace(gi, gate_img_scale_);
            }
        }
        if (gate_mask_.size() == (size_t)gate.rows &&
            block_idx >= gate_mask_lo_ && block_idx < gate_mask_hi_) {
            if ((int)gate_ones_.rows != H || gate_ones_.dtype != dt) {
                gate_ones_ = bt::Tensor::zeros_on(bt::default_device(), H, 1, dt);
                bt::add_scalar_inplace(gate_ones_, 1.0f);
            }
            bt::Tensor mcol = bt::Tensor::view(
                bt::default_device(), gate_mask_.data, gate.rows, 1, dt);
            bt::Tensor ones_row = bt::Tensor::view(
                bt::default_device(), gate_ones_.data, 1, H, dt);
            bt::Tensor mfull;
            bt::matmul(mcol, ones_row, mfull);   // (seq, hidden) rank-1
            bt::mul_inplace(gate, mfull);
        }
        bt::mul_inplace(attn, gate);
        return linb(a.to_out, attn);
    };

    // FF sublayer (SwiGLU): down(silu(gate(x)) * up(x)).
    auto ff_apply = [&](const SwiGLU& f, const bt::Tensor& x) -> bt::Tensor {
        bt::Tensor g = linb(f.gate, x);
        bt::silu_forward(g, g);
        bt::Tensor u = linb(f.up, x);
        bt::mul_inplace(g, u);
        return linb(f.down, g);
    };

    // img_in and joint sequence (txt precomputed by encode_text).
    bt::Tensor lat = packed_latent;
    if (lat.dtype != dt) { bt::Tensor t; bt::cast(packed_latent, t, dt); lat = t; }
    else lat = packed_latent.to(bt::default_device());
    bt::Tensor img = linb(img_in_, lat);         // (img_len, H)

    bt::Tensor x;
    detail::resize_like(x, text_seq + img_len, H, dt, bt::default_device());
    bt::copy_d2d(txt, 0, x, 0, text_seq * H);
    bt::copy_d2d(img, 0, x, text_seq * H, img_len * H);

    // RoPE tables for the joint [text ; image] sequence (theta = rope_theta).
    RopeTables rope = build_axial_rope_tables(text_seq, hp, wp, hd_img,
                                              cfg_.axes_dims_rope, cfg_.rope_theta);

    // ── transformer blocks ───────────────────────────────────────────────
    if (gate_sink_) {
        gate_sink_->assign((size_t)blocks_.size() * (text_seq + img_len),
                           0.0f);
    }

    bt::Tensor table_flat;   // scratch: (1, 6H) view of a block's table
    std::vector<bt::Tensor> chunks;
    for (const TransformerBlock& b : blocks_) {
        // modulation = temb_mod + scale_shift_table (flattened), sliced to 6.
        bt::Tensor mod;
        detail::resize_like(mod, 1, 6 * H, dt, bt::default_device());
        bt::copy_d2d(b.scale_shift_table, 0, mod, 0, 6 * H);
        bt::add_inplace(mod, temb_mod);
        if (mod_delta_.size() > 0 && block_idx >= mod_delta_lo_ &&
            block_idx < mod_delta_hi_) {
            bt::add_inplace(mod, mod_delta_);   // research hook (set_mod_delta)
        }
        slice_modulation_chunks(mod, H, 6, chunks);
        const bt::Tensor& prescale = chunks[0];
        const bt::Tensor& preshift = chunks[1];
        const bt::Tensor& pregate = chunks[2];
        const bt::Tensor& postscale = chunks[3];
        const bt::Tensor& postshift = chunks[4];
        const bt::Tensor& postgate = chunks[5];

        bt::Tensor n1 = rmsnorm(x, b.norm1, cfg_.norm_eps);
        bt::Tensor n1m;
        bt::modulate(n1, prescale, preshift, n1m);
        // No mask: the compacted joint sequence is fully valid.
        bt::Tensor ao = attn_apply(b.attn, n1m, hd_img, nq_img, nkv_img,
                                   &rope.cos, &rope.sin, nullptr);
        bt::Tensor gated;
        bt::broadcast_mul(ao, pregate, gated);
        bt::add_inplace(x, gated);

        bt::Tensor n2 = rmsnorm(x, b.norm2, cfg_.norm_eps);
        bt::Tensor n2m;
        bt::modulate(n2, postscale, postshift, n2m);
        bt::Tensor fo = ff_apply(b.ff, n2m);
        bt::broadcast_mul(fo, postgate, gated);
        bt::add_inplace(x, gated);
        ++block_idx;
    }

    // ── final layer over image rows (uses the RAW temb) ──────────────────
    bt::Tensor img_x;
    detail::resize_like(img_x, img_len, H, dt, bt::default_device());
    bt::copy_d2d(x, text_seq * H, img_x, 0, img_len * H);

    bt::Tensor scale, shift;
    detail::resize_like(scale, 1, H, dt, bt::default_device());
    detail::resize_like(shift, 1, H, dt, bt::default_device());
    bt::copy_d2d(final_scale_shift_table_, 0, scale, 0, H);
    bt::copy_d2d(final_scale_shift_table_, H, shift, 0, H);
    bt::add_inplace(scale, temb);
    bt::add_inplace(shift, temb);

    bt::Tensor fn = rmsnorm(img_x, final_norm_, cfg_.norm_eps);
    bt::Tensor fnm;
    bt::modulate(fn, scale, shift, fnm);
    out = linb(final_linear_, fnm);              // (img_len, in_channels)
    bt::sync_all();
}

void Krea2Transformer2DModel::forward(const bt::Tensor& packed_latent,
                                      int hp, int wp,
                                      const bt::Tensor& prompt_embeds,
                                      const bt::Tensor& prompt_embeds_mask,
                                      float timestep, bt::Tensor& out) {
    bt::Tensor txt;
    encode_text(prompt_embeds, prompt_embeds_mask, txt);
    forward_with_text(packed_latent, hp, wp, txt, timestep, out);
}

// ═══════════════════════════════════════════════════════════════════════
// Krea2Denoiser — Denoiser-interface wrapper
// ═══════════════════════════════════════════════════════════════════════

namespace {

[[noreturn]] void fail_den(const std::string& msg) {
    throw std::runtime_error("dit::Krea2Denoiser: " + msg);
}

// Per-model prepared payload: the text_fusion output rows per CFG branch.
// encode_text (compaction + the whole fusion stack + txt_in) is timestep-
// independent, so it runs ONCE here instead of inside every denoise step —
// the reference pipeline re-runs it per forward, which is pure waste.
struct Krea2Prepared : PreparedConditioning::Impl {
    bt::Tensor txt;          // (n_valid_pos, hidden)
    bt::Tensor uncond_txt;   // (n_valid_neg, hidden)
    bool has_uncond = false;
};

}  // namespace

Krea2Denoiser::Krea2Denoiser(const Krea2Config& cfg, int patch_size)
    : model_(cfg), patch_size_(patch_size) {
    // The 2x2 pack/unpack helpers (dit::pack_latents) hardcode a 2x2 patch;
    // Krea 2 is always patch_size=2. Reject anything else at the boundary
    // rather than silently mis-packing.
    if (patch_size_ != 2) {
        fail_den("only patch_size=2 is supported (got " +
                 std::to_string(patch_size_) + ")");
    }
}

Krea2Denoiser::~Krea2Denoiser() = default;

void Krea2Denoiser::load_weights(const st::File& f, const std::string& prefix) {
    model_.load_weights(f, prefix);
}

void Krea2Denoiser::load_weights(const std::vector<const st::File*>& shards,
                                 const std::string& prefix) {
    model_.load_weights(shards, prefix);
}

PreparedConditioning Krea2Denoiser::prepare(const Conditioning& cond) {
    auto prep = std::make_unique<Krea2Prepared>();
    if (cond.text_embeddings.size() == 0 ||
        cond.text_embeddings_mask.size() == 0) {
        fail_den("prepare: text_embeddings / text_embeddings_mask are empty");
    }
    model_.encode_text(cond.text_embeddings, cond.text_embeddings_mask,
                       prep->txt);
    prep->has_uncond = cond.has_uncond;
    if (cond.has_uncond) {
        if (cond.uncond_embeddings.size() == 0 ||
            cond.uncond_embeddings_mask.size() == 0) {
            fail_den("prepare: has_uncond but uncond embeddings/mask are empty");
        }
        model_.encode_text(cond.uncond_embeddings, cond.uncond_embeddings_mask,
                           prep->uncond_txt);
    }
    return PreparedConditioning(std::move(prep));
}

brotensor::Tensor& Krea2Denoiser::fused_text(PreparedConditioning& prepared,
                                             bool uncond) {
    auto* prep = dynamic_cast<Krea2Prepared*>(prepared.get());
    if (!prep) fail_den("fused_text: prepared conditioning has the wrong type");
    if (uncond) {
        if (!prep->has_uncond) {
            fail_den("fused_text: uncond requested but no uncond conditioning "
                     "was prepared");
        }
        return prep->uncond_txt;
    }
    return prep->txt;
}

void Krea2Denoiser::forward(const bt::Tensor& latent, int H_lat, int W_lat,
                            float timestep,
                            const PreparedConditioning& prepared,
                            Branch branch, bt::Tensor& out) {
    if (!prepared) fail_den("forward: prepared conditioning is empty");
    const auto* prep = dynamic_cast<const Krea2Prepared*>(prepared.get());
    if (!prep) fail_den("forward: prepared conditioning has the wrong type");
    if (H_lat % patch_size_ != 0 || W_lat % patch_size_ != 0) {
        fail_den("forward: H_lat and W_lat must be multiples of patch_size");
    }

    const bt::Tensor* txt = &prep->txt;
    if (branch == Branch::Uncond) {
        if (!prep->has_uncond) {
            fail_den("forward: Uncond branch requested but no uncond "
                     "conditioning was prepared");
        }
        txt = &prep->uncond_txt;
    }

    const int LC = model_.config().latent_channels();
    const int hp = H_lat / patch_size_;
    const int wp = W_lat / patch_size_;

    // The Denoiser contract passes the flat NCHW latent; the transformer wants
    // 2x2-packed tokens. Pack (pipeline dtype) → transformer (casts to BF16
    // internally) → unpack the packed velocity back to NCHW (pipeline dtype).
    bt::Tensor packed = pack_latents(latent, LC, H_lat, W_lat);

    // The scheduler passes the continuous timestep sigma*num_train_timesteps
    // (sigma*1000); Krea2Transformer2DModel::forward expects the flow-time
    // sigma in [0,1] and re-multiplies by 1000 for its own embedding (matching
    // Flux's embedded value). So divide by 1000 here to invert that.
    const float t_flow = timestep / 1000.0f;

    model_.forward_with_text(packed, hp, wp, *txt, t_flow, tf_out_);
    unpack_latents(tf_out_, LC, H_lat, W_lat, out);
}

}  // namespace brodiffusion::dit
