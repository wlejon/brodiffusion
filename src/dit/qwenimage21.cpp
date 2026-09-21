// QwenImage21Transformer2DModel — construction, weight loading, txt_in, and
// the shared timestep/modulation path. The joint-sequence forward (RoPE
// tables, block-causal attention, the prefix KV cache) lives in
// qwenimage21_forward.cpp; the Denoiser wrapper in qwenimage21_denoiser.cpp.

#include "brodiffusion/dit/qwenimage21.h"

#include "qwenimage21_detail.h"

#include "brodiffusion/dit/common.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"
#include "brotensor/detail/cpu/thread_pool.h"

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

namespace qi21 {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("dit::QwenImage21Transformer2DModel: " + msg);
}

const st::TensorView& need(const std::vector<const st::File*>& shards,
                           const std::string& key) {
    for (const st::File* f : shards) {
        if (const auto* v = f->find(key)) return *v;
    }
    fail("missing tensor '" + key + "'");
}

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

bt::Tensor row_view(const bt::Tensor& t, int start, int n) {
    const std::size_t esz = static_cast<std::size_t>(bt::dtype_size_bytes(t.dtype));
    return bt::Tensor::view(
        t.device,
        static_cast<char*>(t.data) +
            static_cast<std::size_t>(start) * static_cast<std::size_t>(t.cols) * esz,
        n, t.cols, t.dtype);
}

}  // namespace qi21

using qi21::fail;
using qi21::need;
using qi21::row_view;
using qi21::upload_as;
using qi21::view_to_fp32;

// ─── ctor / dtor ───────────────────────────────────────────────────────────

QwenImage21Transformer2DModel::QwenImage21Transformer2DModel(
    const QwenImage21Config& cfg)
    : cfg_(cfg) {
    if (cfg_.num_attention_heads <= 0 || cfg_.attention_head_dim <= 0) {
        fail("num_attention_heads / attention_head_dim must be positive");
    }
    if (cfg_.attention_head_dim % 2 != 0) {
        fail("attention_head_dim must be even (RoPE pairs)");
    }
    int sum = 0;
    for (int d : cfg_.axes_dims_rope) {
        if (d % 2 != 0) fail("every axes_dims_rope entry must be even");
        sum += d;
    }
    if (cfg_.axes_dims_rope.size() != 3) {
        fail("axes_dims_rope must have 3 entries (frame, height, width)");
    }
    if (sum != cfg_.attention_head_dim) {
        fail("axes_dims_rope must sum to attention_head_dim");
    }
    if (cfg_.patch_size != 1) {
        fail("only patch_size=1 is supported (2.1 consumes latents unpatched)");
    }
    if (cfg_.num_layers <= 0) fail("num_layers must be positive");
    blocks_.resize(static_cast<std::size_t>(cfg_.num_layers));
}

QwenImage21Transformer2DModel::~QwenImage21Transformer2DModel() = default;

bt::Dtype QwenImage21Transformer2DModel::compute_dtype() const {
    return flux_compute_dtype();
}

// ─── load_weights ──────────────────────────────────────────────────────────

void QwenImage21Transformer2DModel::load_weights(const st::File& f,
                                                 const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights(shards, prefix);
}

void QwenImage21Transformer2DModel::load_weights(
    const std::vector<const st::File*>& shards, const std::string& prefix,
    const std::function<bool()>& should_cancel) {
    if (shards.empty()) fail("load_weights: no shards");
    load_impl_(shards, prefix, should_cancel);
    loaded_ = true;
}

void QwenImage21Transformer2DModel::load_impl_(
    const std::vector<const st::File*>& shards, const std::string& prefix,
    const std::function<bool()>& should_cancel) {
    const bt::Dtype dt = flux_compute_dtype();
    const int H = cfg_.hidden_size();
    const int IC = cfg_.in_channels;
    const int OC = cfg_.out_channels;
    const int TE = cfg_.timestep_embed_dim;
    const int CD = cfg_.context_in_dim;
    const int MH = cfg_.mlp_hidden_size();
    const int hd = cfg_.attention_head_dim;

    // Bias-free dense linear at the compute dtype.
    auto lin = [&](const std::string& key, int out, int in, Linear& l) {
        l.W = upload_as(view_to_fp32(need(shards, prefix + key + ".weight"),
                                     out, in, key),
                        out, in, dt);
        l.W_int8 = bt::Tensor();
        l.scales = bt::Tensor();
    };
    // Plain RMSNorm gain (attn q/k norms): the checkpoint stores the effective
    // scale, unlike the zero-centered txt_in.text_norm below. FP32 —
    // rms_norm_forward takes an FP32 gain against 16-bit activations directly.
    auto norm = [&](const std::string& key, int dim, bt::Tensor& g) {
        std::vector<float> h = view_to_fp32(need(shards, prefix + key + ".weight"),
                                            dim, 1, key);
        g = bt::Tensor::from_host(h.data(), dim, 1).to(bt::default_device());
    };
    // Zero-centered RMSNorm gain (QwenImage21ZeroCenterRMSNorm): the
    // checkpoint stores `scale - 1`, so pre-add 1 here.
    auto zc_norm = [&](const std::string& key, int dim, bt::Tensor& g) {
        std::vector<float> h = view_to_fp32(need(shards, prefix + key + ".weight"),
                                            dim, 1, key);
        for (float& x : h) x += 1.0f;
        g = bt::Tensor::from_host(h.data(), dim, 1).to(bt::default_device());
    };

    bool quant = cfg_.quantize_weights;
    if (quant && bt::default_device() != bt::Device::CUDA) {
        std::fprintf(stderr,
            "QwenImage21Transformer2DModel: quantize_weights requested but the "
            "default device is not CUDA — loading dense weights instead (the "
            "fused INT8 dequant matmuls are GPU-only)\n");
        quant = false;
    }

    // Quantizing loader for the big per-block linears: converts the on-disk
    // BF16 weight to FP16 bits host-side, quantizes to INT8 with per-output-row
    // symmetric FP32 scales, and uploads only the INT8 copy — so peak VRAM
    // during load is the INT8 footprint (~7.2 GB), not the BF16 one (~14.2 GB).
    auto lin_q = [&](const std::string& key, int out, int in, Linear& l) {
        if (!quant) { lin(key, out, in, l); return; }
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
        l.W = bt::Tensor();
        l.W_int8 = bt::Tensor::from_host_int8(q.data(), out, in);
        l.scales = bt::Tensor::from_host(sc.data(), out, 1);
    };

    lin("img_in", H, IC, img_in_);
    lin("time_text_embed.timestep_embedder.linear_1", H, TE, time_l1_);
    lin("time_text_embed.timestep_embedder.linear_2", H, H, time_l2_);
    lin("modulation.1", 4 * H, H, modulation_);

    zc_norm("txt_in.text_norm", CD, txt_norm_);
    lin("txt_in.in_layer", H, CD, txt_in_);
    lin("txt_in.out_layer", H, H, txt_out_);

    for (int i = 0; i < cfg_.num_layers; ++i) {
        // Cooperative cancellation: this loop is the bulk of the 14.2 GB load.
        if (should_cancel && should_cancel()) throw LoadCancelled{};
        const std::string p = "transformer_blocks." + std::to_string(i) + ".";
        Block& b = blocks_[static_cast<std::size_t>(i)];
        lin_q(p + "attn.to_q", H, H, b.to_q);
        lin_q(p + "attn.to_k", H, H, b.to_k);
        lin_q(p + "attn.to_v", H, H, b.to_v);
        lin_q(p + "attn.to_out.0", H, H, b.to_out);
        norm(p + "attn.norm_q", hd, b.norm_q);
        norm(p + "attn.norm_k", hd, b.norm_k);
        lin_q(p + "img_mlp.gate_layer", MH, H, b.mlp_gate);
        lin_q(p + "img_mlp.proj", MH, H, b.mlp_proj);
        lin_q(p + "img_mlp.out", H, MH, b.mlp_out);
    }

    lin("norm_out.linear", H, H, norm_out_lin_);
    lin("proj_out", OC, H, proj_out_);

    // Non-affine LayerNorm operands (gamma = 1, beta = 0) at the compute
    // dtype, plus the zero shift row modulate() takes.
    {
        std::vector<float> ones(static_cast<std::size_t>(H), 1.0f);
        std::vector<float> zeros(static_cast<std::size_t>(H), 0.0f);
        ln_gamma_ = upload_as(ones, H, 1, dt);
        ln_beta_ = upload_as(zeros, H, 1, dt);
        zero_shift_ = upload_as(zeros, 1, H, dt);
    }
}

// ─── linear dispatch ───────────────────────────────────────────────────────

bt::Tensor QwenImage21Transformer2DModel::lin_(const Linear& l,
                                               const bt::Tensor& X) {
    bt::Tensor Y;
    if (l.quantized()) {
        bt::linear_forward_batched_int8w_fp16(l.W_int8, l.scales, nullptr, X, Y);
    } else {
        detail::linear_batched(l.W, nullptr, X, Y);
    }
    return Y;
}

void QwenImage21Transformer2DModel::lin_into_(const Linear& l,
                                              const bt::Tensor& X,
                                              bt::Tensor& Y) {
    if (l.quantized()) {
        bt::linear_forward_batched_int8w_fp16(l.W_int8, l.scales, nullptr, X, Y);
    } else {
        detail::linear_batched(l.W, nullptr, X, Y);
    }
}

void QwenImage21Transformer2DModel::layernorm_(const bt::Tensor& X,
                                               bt::Tensor& Y) {
    // Non-affine LayerNorm, eps = cfg_.eps. brotensor's batched inference
    // LayerNorm wants gamma/beta at the activation dtype; they are the
    // constant ones/zeros built at load.
    bt::layernorm_forward_inference_batched(X, ln_gamma_, ln_beta_, Y, cfg_.eps);
}

// ─── txt_in (QwenImage21TextProjection) ────────────────────────────────────

void QwenImage21Transformer2DModel::encode_text(
    const bt::Tensor& encoder_hidden_states, bt::Tensor& txt_out) {
    if (!loaded_) fail("encode_text: weights not loaded");
    const bt::Dtype dt = flux_compute_dtype();
    if (encoder_hidden_states.cols != cfg_.context_in_dim) {
        fail("encode_text: encoder_hidden_states must be (L, context_in_dim)");
    }
    if (encoder_hidden_states.rows <= 0) {
        fail("encode_text: encoder_hidden_states has no rows");
    }

    bt::Tensor hs;
    if (encoder_hidden_states.dtype != dt) {
        bt::cast(encoder_hidden_states.to(bt::default_device()), hs, dt);
    } else {
        hs = encoder_hidden_states.to(bt::default_device());
    }

    // QwenImage21ZeroCenterRMSNorm: the gain is pre-added to 1 at load and
    // kept FP32, so rms_norm_forward's FP32 accumulation reproduces the
    // reference's explicit float() upcast.
    bt::Tensor normed;
    bt::rms_norm_forward(hs, txt_norm_, cfg_.eps, normed);
    bt::Tensor x = lin_(txt_in_, normed);
    bt::gelu_forward(x, x);              // nn.GELU(approximate="tanh")
    txt_out = lin_(txt_out_, x);
}

// ─── timestep embedding + shared modulation ────────────────────────────────

void QwenImage21Transformer2DModel::build_modulation_(
    float timestep, bool build_variants, bt::Tensor* raw_temb,
    bt::Tensor* raw_mod) {
    const bt::Dtype dt = flux_compute_dtype();
    const bt::Device dev = bt::default_device();

    // Two timestep rows: the sampled t and t = 0. Under causal_condition the
    // text / condition-image tokens read the t = 0 row, so their activations
    // (and hence the prefix KV cache) are step-independent.
    const float ts_vals[2] = {timestep * 1000.0f, 0.0f};
    const int n_rows = cfg_.causal_condition ? 2 : 1;
    bt::Tensor ts = bt::Tensor::from_host_on(bt::Device::CPU, ts_vals, n_rows, 1);
    bt::Tensor freq;
    bt::timestep_embedding(ts, cfg_.timestep_embed_dim, 10000.0f, freq);
    bt::Tensor freq_dev = freq.to(dev);
    bt::Tensor freq_cd = freq_dev;
    if (dt != bt::Dtype::FP32) bt::cast(freq_dev, freq_cd, dt);

    bt::Tensor temb = lin_(time_l1_, freq_cd);
    bt::silu_forward(temb, temb);                 // TimestepEmbedding act_fn
    temb = lin_(time_l2_, temb);                  // (n_rows, H)

    bt::Tensor temb_act = temb.clone();
    bt::silu_forward(temb_act, temb_act);         // modulation.0 == nn.SiLU
    bt::Tensor mod = lin_(modulation_, temb_act); // (n_rows, 4H)

    // norm_out: LN(x) * (1 + linear(silu(temb))). Only the target rows go
    // through it, so only the sampled row's scale is needed.
    bt::Tensor final_all = lin_(norm_out_lin_, temb_act);   // (n_rows, H)

    if (raw_temb) *raw_temb = temb;
    if (raw_mod)  *raw_mod  = mod;
    if (!build_variants) return;

    // ── the per-block variants ───────────────────────────────────────────
    //
    // Group the blocks by which bindings cover them, then build ONE finished
    // modulation per group. A group whose mod-delta list is non-empty gets a
    // deltaed copy of the raw output (the delta lands before the gates' tanh,
    // so it passes through it like the base value does); every group then has
    // its composed gate scale and post-tanh gate delta folded into its gate
    // rows. With nothing armed there is exactly one group and this is the old
    // single-modulation path.
    resolve_coverage_();
    mods_.resize(coverages_.size());
    bt::Tensor mod2;
    for (std::size_t c = 0; c < coverages_.size(); ++c) {
        const BlockCoverage& cov = coverages_[c];
        const bt::Tensor* src = &mod;
        if (!cov.mod_deltas.empty()) {
            // A scratch copy, reused across groups so the pool hands back one
            // buffer rather than one per group.
            detail::resize_like(mod2, mod.rows, mod.cols, mod.dtype, dev);
            bt::copy_d2d(mod, 0, mod2, 0, static_cast<int>(mod.size()));
            for (int j : cov.mod_deltas) {
                const auto& b = mod_deltas_[static_cast<std::size_t>(j)];
                const bool do_target = b.target != QwenImage21ModTarget::Prefix;
                const bool do_prefix = b.target != QwenImage21ModTarget::Target;
                if (do_target) {
                    bt::Tensor r0 = qi21::row_view(mod2, 0, 1);
                    bt::add_inplace(r0, b.delta);
                }
                if (do_prefix && n_rows == 2) {
                    bt::Tensor r1 = qi21::row_view(mod2, 1, 1);
                    bt::add_inplace(r1, b.delta);
                } else if (do_prefix && n_rows == 1 && !do_target) {
                    // causal_condition disabled: there is only the sampled
                    // row and every token reads it, so a Prefix-only delta is
                    // that row's delta.
                    bt::Tensor r0 = qi21::row_view(mod2, 0, 1);
                    bt::add_inplace(r0, b.delta);
                }
            }
            src = &mod2;
        }
        chunk_modulation_(*src, final_all, mods_[c]);
        fold_gate_hooks_(cov, mods_[c]);
    }
}

void QwenImage21Transformer2DModel::chunk_modulation_(
    const bt::Tensor& mod, const bt::Tensor& final_all, Modulation& m) {
    const bt::Dtype dt = flux_compute_dtype();
    const int H = cfg_.hidden_size();
    const bt::Device dev = bt::default_device();
    const int n_rows = mod.rows;

    // Chunk order: [mod1.scale, mod1.gate, mod2.scale, mod2.gate]. The gates
    // enter the residual through tanh(), and the modulation is shared by
    // every block — so both the slice and the tanh happen once per forward.
    auto slice = [&](const bt::Tensor& src, int row, int chunk, bt::Tensor& dst) {
        detail::resize_like(dst, 1, H, dt, dev);
        bt::copy_d2d(src, row * src.cols + chunk * H, dst, 0, H);
    };
    slice(mod, 0, 0, m.scale1_t);
    slice(mod, 0, 1, m.gate1_t);
    slice(mod, 0, 2, m.scale2_t);
    slice(mod, 0, 3, m.gate2_t);
    bt::tanh_forward(m.gate1_t, m.gate1_t);
    bt::tanh_forward(m.gate2_t, m.gate2_t);
    if (n_rows == 2) {
        slice(mod, 1, 0, m.scale1_0);
        slice(mod, 1, 1, m.gate1_0);
        slice(mod, 1, 2, m.scale2_0);
        slice(mod, 1, 3, m.gate2_0);
        bt::tanh_forward(m.gate1_0, m.gate1_0);
        bt::tanh_forward(m.gate2_0, m.gate2_0);
    } else {
        // causal_condition disabled: every token reads the sampled row. Copy
        // rather than assign — the prefix and target rows carry their own
        // gate scale factors, so they must stay separate buffers, and a
        // reused buffer keeps the JIT sites' bindings alive.
        auto dup = [&](const bt::Tensor& s, bt::Tensor& d) {
            detail::resize_like(d, 1, H, dt, dev);
            bt::copy_d2d(s, 0, d, 0, H);
        };
        dup(m.scale1_t, m.scale1_0);
        dup(m.gate1_t,  m.gate1_0);
        dup(m.scale2_t, m.scale2_0);
        dup(m.gate2_t,  m.gate2_0);
    }

    detail::resize_like(m.final_scale, 1, H, dt, dev);
    bt::copy_d2d(final_all, 0, m.final_scale, 0, H);
    // Research hook (set_norm_out_scale_delta).
    if (norm_out_delta_.size() > 0) bt::add_inplace(m.final_scale, norm_out_delta_);
}

}  // namespace brodiffusion::dit
