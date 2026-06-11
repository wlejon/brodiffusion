#include "brodiffusion/dit/flux.h"

#include "brodiffusion/dit/common.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::dit {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("dit::FluxDenoiser: " + msg);
}

// Upload a checked view at the FLUX compute dtype (BF16 on CUDA — see
// dit::flux_compute_dtype), not the pipeline dtype.
void upload_flux_checked(const st::TensorView& v, int rows, int cols,
                            bt::Tensor& dst, const std::string& name) {
    if (v.dtype != st::Dtype::F16 && v.dtype != st::Dtype::F32 &&
        v.dtype != st::Dtype::BF16) {
        fail(name + " ('" + v.name + "'): expected F16/F32/BF16, got " +
             st::dtype_name(v.dtype));
    }
    const std::int64_t expected =
        static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(cols);
    if (v.numel() != expected) {
        fail(name + " ('" + v.name + "'): shape mismatch (expected " +
             std::to_string(rows) + "x" + std::to_string(cols) + ")");
    }
    st::upload_as(v, rows, cols, flux_compute_dtype(), dst);
}

// Upload host FP32 values at the flux compute dtype.
bt::Tensor upload_host_flux(const float* src, int rows, int cols) {
    const bt::Dtype dt = flux_compute_dtype();
    const std::size_t n =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (dt == bt::Dtype::BF16) {
        std::vector<std::uint16_t> bits(n);
        for (std::size_t i = 0; i < n; ++i) {
            bits[i] = bt::fp32_to_bf16_bits(src[i]);
        }
        return bt::Tensor::from_host_bf16(bits.data(), rows, cols);
    }
    if (dt == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        for (std::size_t i = 0; i < n; ++i) {
            bits[i] = bt::fp32_to_fp16_bits(src[i]);
        }
        return bt::Tensor::from_host_fp16(bits.data(), rows, cols);
    }
    return bt::Tensor::from_host(src, rows, cols);
}

// Cast `src` to the flux compute dtype into `dst` (no-op pass-through copy
// avoided when the dtype already matches — `dst` then aliases nothing and
// the caller uses `src` directly via the returned reference).
const bt::Tensor& to_flux_dtype(const bt::Tensor& src, bt::Tensor& dst) {
    const bt::Dtype dt = flux_compute_dtype();
    if (src.dtype == dt) return src;
    bt::cast(src, dst, dt);
    return dst;
}

// Find a tensor by name across one or more shards; first match wins.
const st::TensorView& need(const std::vector<const st::File*>& shards,
                           const std::string& key) {
    for (const st::File* f : shards) {
        if (const auto* v = f->find(key)) return *v;
    }
    throw std::runtime_error(
        "dit::FluxDenoiser: missing tensor '" + key + "'");
}

// Concatenate two row-major tensors a (La, D) and b (Lb, D) into
// dst ((La+Lb), D), text/first part first. brotensor has no concat op; the
// row-major layout makes this two copy_d2d runs.
void concat_rows(const bt::Tensor& a, const bt::Tensor& b, bt::Tensor& dst) {
    const int D = a.cols;
    detail::resize_like(dst, a.rows + b.rows, D, a.dtype, a.device);
    bt::copy_d2d(a, 0, dst, 0, a.rows * D);
    bt::copy_d2d(b, 0, dst, a.rows * D, b.rows * D);
}

// Copy a row range [r0, r0+n) of src (L, D) into dst (n, D).
void slice_rows(const bt::Tensor& src, int r0, int n, bt::Tensor& dst) {
    const int D = src.cols;
    detail::resize_like(dst, n, D, src.dtype, src.device);
    bt::copy_d2d(src, r0 * D, dst, 0, n * D);
}

// Slice the rectangular block rows [r0, r0+nr) × cols [c0, c0+nc) out of a
// row-major (L_rows, L_cols) tensor into dst (nr, nc). One copy_d2d run per
// row — the column slice is not contiguous in the source.
void slice_block(const bt::Tensor& src, int r0, int nr, int c0, int nc,
                 bt::Tensor& dst) {
    const int L_cols = src.cols;
    detail::resize_like(dst, nr, nc, src.dtype, src.device);
    for (int r = 0; r < nr; ++r) {
        bt::copy_d2d(src, (r0 + r) * L_cols + c0, dst, r * nc, nc);
    }
}

// Diagnostic probe (BRODIFFUSION_FLUX_NANCHECK=1): per-block max-|activation|
// and non-finite detection, downloaded to host. Slow — debug only.
bool flux_nancheck() {
    const char* e = std::getenv("BRODIFFUSION_FLUX_NANCHECK");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
}

float max_abs_host(const bt::Tensor& t, bool& nonfinite) {
    bt::sync_all();
    bt::Tensor h = t.to(bt::Device::CPU);
    float mx = 0.0f;
    nonfinite = false;
    if (h.dtype == bt::Dtype::FP16) {
        auto v = h.to_host_vector_fp16();
        for (auto b : v) {
            const float f = bt::fp16_bits_to_fp32(b);
            if (!std::isfinite(f)) { nonfinite = true; continue; }
            mx = std::max(mx, std::fabs(f));
        }
    } else if (h.dtype == bt::Dtype::BF16) {
        auto v = h.to_host_vector_bf16();
        for (auto b : v) {
            const float f = bt::bf16_bits_to_fp32(b);
            if (!std::isfinite(f)) { nonfinite = true; continue; }
            mx = std::max(mx, std::fabs(f));
        }
    } else {
        auto v = h.to_host_vector();
        for (float f : v) {
            if (!std::isfinite(f)) { nonfinite = true; continue; }
            mx = std::max(mx, std::fabs(f));
        }
    }
    return mx;
}

void probe(const char* tag, int idx, const bt::Tensor& a, const bt::Tensor* b) {
    bool nf_a = false, nf_b = false;
    const float ma = max_abs_host(a, nf_a);
    const float mb = b ? max_abs_host(*b, nf_b) : 0.0f;
    std::fprintf(stderr, "[flux-nan] %s %d: max|a|=%g%s max|b|=%g%s\n",
                 tag, idx, ma, nf_a ? " NONFINITE" : "",
                 mb, nf_b ? " NONFINITE" : "");
}

}  // namespace

// ─── ctor / dtor ───────────────────────────────────────────────────────────

FluxDenoiser::FluxDenoiser(const FluxConfig& cfg) : cfg_(cfg) {
    if (cfg_.in_channels <= 0 || cfg_.in_channels % 4 != 0) {
        fail("in_channels must be a positive multiple of 4");
    }
    if (cfg_.num_attention_heads <= 0 || cfg_.attention_head_dim <= 0) {
        fail("num_attention_heads / attention_head_dim must be positive");
    }
    if (cfg_.attention_head_dim % 2 != 0) {
        fail("attention_head_dim must be even");
    }
    int sum = 0;
    for (int d : cfg_.axes_dims_rope) sum += d;
    if (sum != cfg_.attention_head_dim) {
        fail("axes_dims_rope must sum to attention_head_dim");
    }
    if (cfg_.num_layers < 0 || cfg_.num_single_layers < 0) {
        fail("num_layers / num_single_layers must be non-negative");
    }
    double_blocks_.resize(static_cast<std::size_t>(cfg_.num_layers));
    single_blocks_.resize(static_cast<std::size_t>(cfg_.num_single_layers));
}

FluxDenoiser::~FluxDenoiser() = default;

bt::Dtype FluxDenoiser::compute_dtype() const {
    return brodiffusion::compute_dtype();
}

// ─── load_weights ──────────────────────────────────────────────────────────

void FluxDenoiser::load_weights(const st::File& f, const std::string& prefix) {
    const std::vector<const st::File*> shards = {&f};
    load_weights(shards, prefix);
}

void FluxDenoiser::load_weights(const std::vector<const st::File*>& shards,
                                const std::string& prefix) {
    if (shards.empty()) fail("load_weights: no safetensors shards");
    const int D    = cfg_.inner_dim();
    const int IC   = cfg_.in_channels;
    const int JD   = cfg_.joint_attention_dim;
    const int PD   = cfg_.pooled_projection_dim;
    const int HD   = cfg_.attention_head_dim;

    bool quant = cfg_.quantize_weights;
    if (quant && bt::default_device() != bt::Device::CUDA) {
        std::fprintf(stderr,
            "FluxDenoiser: quantize_weights requested but the default device "
            "is not CUDA — loading dense weights instead (the fused INT8 "
            "dequant matmuls are GPU-only)\n");
        quant = false;
    }

    auto load_lin = [&](const std::string& key, int out, int in, Linear& lin) {
        upload_flux_checked(need(shards, key + ".weight"), out, in, lin.W,
                               key);
        upload_flux_checked(need(shards, key + ".bias"), out, 1, lin.b, key);
    };

    // Quantizing loader for the big block linears: converts the on-disk
    // weight (F16/F32/BF16) to FP16 bits host-side, quantizes to INT8 with
    // per-output-row symmetric FP32 scales, and uploads only the INT8 copy —
    // the FP16 weight never lands on the device, so peak VRAM during load is
    // the INT8 footprint (~12 GB for Flux.1), not the FP16 one (~24 GB).
    auto load_lin_q = [&](const std::string& key, int out, int in,
                          Linear& lin) {
        if (!quant) { load_lin(key, out, in, lin); return; }
        const st::TensorView& wv = need(shards, key + ".weight");
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
            for (std::size_t i = 0; i < n; ++i) {
                w16[i] = bt::fp32_to_fp16_bits(bt::bf16_bits_to_fp32(src[i]));
            }
        } else if (wv.dtype == st::Dtype::F32) {
            const auto* src = reinterpret_cast<const float*>(wv.data);
            for (std::size_t i = 0; i < n; ++i) {
                w16[i] = bt::fp32_to_fp16_bits(src[i]);
            }
        } else {
            fail(key + ".weight: expected F16/F32/BF16, got " +
                 st::dtype_name(wv.dtype));
        }
        std::vector<std::int8_t> q(n);
        std::vector<float> sc(static_cast<std::size_t>(out));
        bt::quantize_int8_per_row_host(w16.data(), out, in, q.data(),
                                       sc.data());
        lin.W_int8 = bt::Tensor::from_host_int8(q.data(), out, in);
        lin.scales = bt::Tensor::from_host(sc.data(), out, 1);
        upload_flux_checked(need(shards, key + ".bias"), out, 1, lin.b, key);
    };

    load_lin(prefix + "x_embedder", D, IC, x_embedder_);
    load_lin(prefix + "context_embedder", D, JD, context_embedder_);

    const std::string tt = prefix + "time_text_embed.";
    load_lin(tt + "timestep_embedder.linear_1", D, 256, te_time_l1_);
    load_lin(tt + "timestep_embedder.linear_2", D, D, te_time_l2_);
    load_lin(tt + "text_embedder.linear_1", D, PD, te_text_l1_);
    load_lin(tt + "text_embedder.linear_2", D, D, te_text_l2_);
    if (cfg_.guidance_embeds) {
        load_lin(tt + "guidance_embedder.linear_1", D, 256, te_guidance_l1_);
        load_lin(tt + "guidance_embedder.linear_2", D, D, te_guidance_l2_);
    }

    for (int i = 0; i < cfg_.num_layers; ++i) {
        DoubleBlock& B = double_blocks_[static_cast<std::size_t>(i)];
        const std::string p =
            prefix + "transformer_blocks." + std::to_string(i) + ".";
        load_lin_q(p + "norm1.linear", 6 * D, D, B.norm1);
        load_lin_q(p + "norm1_context.linear", 6 * D, D, B.norm1_context);
        load_lin_q(p + "attn.to_q", D, D, B.to_q);
        load_lin_q(p + "attn.to_k", D, D, B.to_k);
        load_lin_q(p + "attn.to_v", D, D, B.to_v);
        load_lin_q(p + "attn.add_q_proj", D, D, B.add_q);
        load_lin_q(p + "attn.add_k_proj", D, D, B.add_k);
        load_lin_q(p + "attn.add_v_proj", D, D, B.add_v);
        load_lin_q(p + "attn.to_out.0", D, D, B.to_out);
        load_lin_q(p + "attn.to_add_out", D, D, B.to_add_out);
        upload_flux_checked(need(shards, p + "attn.norm_q.weight"), HD, 1,
                               B.norm_q, "attn.norm_q");
        upload_flux_checked(need(shards, p + "attn.norm_k.weight"), HD, 1,
                               B.norm_k, "attn.norm_k");
        upload_flux_checked(need(shards, p + "attn.norm_added_q.weight"), HD, 1,
                               B.norm_added_q, "attn.norm_added_q");
        upload_flux_checked(need(shards, p + "attn.norm_added_k.weight"), HD, 1,
                               B.norm_added_k, "attn.norm_added_k");
        load_lin_q(p + "ff.net.0.proj", 4 * D, D, B.ff0);
        load_lin_q(p + "ff.net.2", D, 4 * D, B.ff2);
        load_lin_q(p + "ff_context.net.0.proj", 4 * D, D, B.ffc0);
        load_lin_q(p + "ff_context.net.2", D, 4 * D, B.ffc2);
    }

    for (int i = 0; i < cfg_.num_single_layers; ++i) {
        SingleBlock& B = single_blocks_[static_cast<std::size_t>(i)];
        const std::string p =
            prefix + "single_transformer_blocks." + std::to_string(i) + ".";
        load_lin_q(p + "norm.linear", 3 * D, D, B.norm);
        load_lin_q(p + "attn.to_q", D, D, B.to_q);
        load_lin_q(p + "attn.to_k", D, D, B.to_k);
        load_lin_q(p + "attn.to_v", D, D, B.to_v);
        upload_flux_checked(need(shards, p + "attn.norm_q.weight"), HD, 1,
                               B.norm_q, "attn.norm_q");
        upload_flux_checked(need(shards, p + "attn.norm_k.weight"), HD, 1,
                               B.norm_k, "attn.norm_k");
        load_lin_q(p + "proj_mlp", 4 * D, D, B.proj_mlp);
        load_lin_q(p + "proj_out", D, 5 * D, B.proj_out);
    }

    load_lin(prefix + "norm_out.linear", 2 * D, D, norm_out_);
    load_lin(prefix + "proj_out", IC, D, proj_out_);

    // AdaLN affine-free LayerNorm params: ones gamma, zeros beta (inner_dim,1).
    {
        std::vector<float> ones(static_cast<std::size_t>(D), 1.0f);
        std::vector<float> zeros(static_cast<std::size_t>(D), 0.0f);
        ada_gamma_ = upload_host_flux(ones.data(), D, 1);
        ada_beta_  = upload_host_flux(zeros.data(), D, 1);
    }
}

void FluxDenoiser::finalize_weights() {
    // Quantization (when enabled) already happened streaming inside
    // load_weights — finalize is a no-op, idempotent.
    finalized_ = true;
}

void FluxDenoiser::lin_(const Linear& l, const bt::Tensor& X, bt::Tensor& Y) {
    if (l.quantized()) {
        bt::linear_forward_batched_int8w_fp16(l.W_int8, l.scales, &l.b, X, Y);
    } else {
        detail::linear_batched(l.W, &l.b, X, Y);
    }
}

// ─── prepared conditioning ─────────────────────────────────────────────────

namespace {

struct FluxPrepared : public PreparedConditioning::Impl {
    bt::Tensor context;   // (txt_len, inner_dim) — context_embedder(T5 seq)
    bt::Tensor text_emb;  // (1, inner_dim) — text_embedder(pooled CLIP)
    float guidance = 0.0f;
    int txt_len = 0;
};

}  // namespace

PreparedConditioning FluxDenoiser::prepare(const Conditioning& cond) {
    if (x_embedder_.W.size() == 0) fail("prepare: weights not loaded");
    const int D = cfg_.inner_dim();

    auto prep = std::make_unique<FluxPrepared>();

    // Project the T5 context sequence (txt_len, joint_attention_dim) → inner.
    // The raw conditioning arrives at the pipeline dtype; cast to the flux
    // compute dtype (BF16 on CUDA) before touching flux weights.
    if (cond.text_embeddings.cols != cfg_.joint_attention_dim) {
        fail("prepare: text_embeddings width != joint_attention_dim");
    }
    prep->txt_len = cond.text_embeddings.rows;
    bt::Tensor ctx_cast, pooled_cast;
    const bt::Tensor& ctx = to_flux_dtype(cond.text_embeddings, ctx_cast);
    detail::linear_batched(context_embedder_.W, &context_embedder_.b,
                           ctx, prep->context);
    prep->context = prep->context.clone();

    // text_embedder(pooled CLIP) — depends only on the fixed pooled vector.
    if (cond.pooled.cols != cfg_.pooled_projection_dim || cond.pooled.rows != 1) {
        fail("prepare: pooled must be (1, pooled_projection_dim)");
    }
    const bt::Tensor& pooled = to_flux_dtype(cond.pooled, pooled_cast);
    bt::Tensor h1, h2;
    detail::linear_batched(te_text_l1_.W, &te_text_l1_.b, pooled, h1);
    bt::silu_forward(h1, h1);
    detail::linear_batched(te_text_l2_.W, &te_text_l2_.b, h1, h2);
    prep->text_emb = h2.clone();
    (void)D;

    prep->guidance = cond.guidance;

    return PreparedConditioning(std::move(prep));
}

// ─── double-stream block ───────────────────────────────────────────────────

void FluxDenoiser::run_double_block_(const DoubleBlock& blk,
                                     const bt::Tensor& temb,
                                     const bt::Tensor& cos,
                                     const bt::Tensor& sin,
                                     int txt_len, int img_len,
                                     bt::Tensor* trace_out_entry,
                                     const bt::Tensor* attn_bias) {
    const int D  = cfg_.inner_dim();
    const int HD = cfg_.attention_head_dim;
    const int NH = cfg_.num_attention_heads;
    const bt::Dtype dt = flux_compute_dtype();
    const bt::Device dev = img_.device;

    // silu(temb) once; both AdaLN modulation MLPs consume it.
    bt::silu_forward(temb, silu_);

    // norm1 (image) and norm1_context (text): 6 chunks each.
    std::vector<bt::Tensor> img_mod, txt_mod;
    lin_(blk.norm1, silu_, chunk_row_);
    slice_modulation_chunks(chunk_row_, D, 6, img_mod);
    // copy out before reusing chunk_row_
    std::vector<bt::Tensor> img_c;
    img_c.reserve(6);
    for (auto& t : img_mod) img_c.push_back(t.clone());
    lin_(blk.norm1_context, silu_, chunk_row_);
    slice_modulation_chunks(chunk_row_, D, 6, txt_mod);
    std::vector<bt::Tensor> txt_c;
    txt_c.reserve(6);
    for (auto& t : txt_mod) txt_c.push_back(t.clone());

    // chunk order: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    const bt::Tensor& i_shift_msa = img_c[0];
    const bt::Tensor& i_scale_msa = img_c[1];
    const bt::Tensor& i_gate_msa  = img_c[2];
    const bt::Tensor& i_shift_mlp = img_c[3];
    const bt::Tensor& i_scale_mlp = img_c[4];
    const bt::Tensor& i_gate_mlp  = img_c[5];
    const bt::Tensor& t_shift_msa = txt_c[0];
    const bt::Tensor& t_scale_msa = txt_c[1];
    const bt::Tensor& t_gate_msa  = txt_c[2];
    const bt::Tensor& t_shift_mlp = txt_c[3];
    const bt::Tensor& t_scale_mlp = txt_c[4];
    const bt::Tensor& t_gate_mlp  = txt_c[5];

    // ── attention sub-layer ───────────────────────────────────────────────
    // img_mod = modulate(LN(img), scale_msa, shift_msa)
    bt::Tensor img_modulated, txt_modulated;
    detail::layernorm_batched(img_, ada_gamma_, ada_beta_, ln_, 1e-6f);
    bt::modulate(ln_, i_scale_msa, i_shift_msa, mod_);
    img_modulated = mod_.clone();
    detail::layernorm_batched(txt_, ada_gamma_, ada_beta_, ln_, 1e-6f);
    bt::modulate(ln_, t_scale_msa, t_shift_msa, mod_);
    txt_modulated = mod_.clone();

    // Per-head RMSNorm helper: view (L,D) as (L*NH, HD), rms_norm, return a
    // (L,D) view of the normalized buffer.
    auto headnorm = [&](bt::Tensor& src, const bt::Tensor& gain,
                        bt::Tensor& dst) -> bt::Tensor {
        const int L = src.rows;
        bt::Tensor src_v =
            bt::Tensor::view(src.device, src.data, L * NH, HD, src.dtype);
        bt::rms_norm_forward(src_v, gain, 1e-6f, dst);
        return bt::Tensor::view(dst.device, dst.data, L, D, dst.dtype);
    };

    // image q/k/v
    lin_(blk.to_q, img_modulated, q_);
    lin_(blk.to_k, img_modulated, k_);
    lin_(blk.to_v, img_modulated, v_);
    bt::Tensor img_q = headnorm(q_, blk.norm_q, qn_);
    bt::Tensor img_k = headnorm(k_, blk.norm_k, kn_);
    bt::Tensor img_q_c = img_q.clone();
    bt::Tensor img_k_c = img_k.clone();
    bt::Tensor img_v_c = v_.clone();

    // text q/k/v
    lin_(blk.add_q, txt_modulated, q_);
    lin_(blk.add_k, txt_modulated, k_);
    lin_(blk.add_v, txt_modulated, v_);
    bt::Tensor txt_q = headnorm(q_, blk.norm_added_q, qn_);
    bt::Tensor txt_k = headnorm(k_, blk.norm_added_k, kn_);
    bt::Tensor txt_q_c = txt_q.clone();
    bt::Tensor txt_k_c = txt_k.clone();
    bt::Tensor txt_v_c = v_.clone();

    // Q/K/V = concat([txt, img]) — text first.
    concat_rows(txt_q_c, img_q_c, Q_);
    concat_rows(txt_k_c, img_k_c, K_);
    concat_rows(txt_v_c, img_v_c, V_);

    if (trace_out_entry != nullptr || attn_bias != nullptr) {
        // Trace / steering path: materialise the per-head softmax, average over
        // heads, then keep the image-query × text-key block. Joint sequence is
        // [text ; image], so image rows are [txt_len, txt_len+img_len) and
        // text columns are [0, txt_len). `attn_bias`, if non-null, is added to
        // the image→text scores before softmax.
        bt::Tensor attn_avg;
        joint_attention_traced(Q_, K_, V_, cos, sin, HD, NH, attn_, attn_avg,
                               Qr_, Kr_, txt_len, attn_bias);
        if (trace_out_entry != nullptr) {
            slice_block(attn_avg, txt_len, img_len, 0, txt_len,
                        *trace_out_entry);
        }
    } else {
        joint_attention(Q_, K_, V_, cos, sin, HD, NH, attn_, Qr_, Kr_);
    }

    // split attn into text / image parts
    bt::Tensor txt_attn, img_attn;
    slice_rows(attn_, 0, txt_len, txt_attn);
    slice_rows(attn_, txt_len, img_len, img_attn);

    // img = img + gate_msa * to_out(img_attn)
    lin_(blk.to_out, img_attn, proj_);
    bt::broadcast_mul(proj_, i_gate_msa, gated_);
    bt::add_inplace(img_, gated_);
    // txt = txt + gate_msa_ctx * to_add_out(txt_attn)
    lin_(blk.to_add_out, txt_attn, proj_);
    bt::broadcast_mul(proj_, t_gate_msa, gated_);
    bt::add_inplace(txt_, gated_);

    // ── MLP sub-layer ─────────────────────────────────────────────────────
    // img = img + gate_mlp * ff(modulate(LN(img), scale_mlp, shift_mlp))
    detail::layernorm_batched(img_, ada_gamma_, ada_beta_, ln_, 1e-6f);
    bt::modulate(ln_, i_scale_mlp, i_shift_mlp, mod_);
    lin_(blk.ff0, mod_, ff_mid_);
    bt::gelu_forward(ff_mid_, ff_mid_);
    lin_(blk.ff2, ff_mid_, ff_out_);
    bt::broadcast_mul(ff_out_, i_gate_mlp, gated_);
    bt::add_inplace(img_, gated_);

    // txt = txt + gate_mlp_ctx * ff_context(modulate(LN(txt), ...))
    detail::layernorm_batched(txt_, ada_gamma_, ada_beta_, ln_, 1e-6f);
    bt::modulate(ln_, t_scale_mlp, t_shift_mlp, mod_);
    lin_(blk.ffc0, mod_, ff_mid_);
    bt::gelu_forward(ff_mid_, ff_mid_);
    lin_(blk.ffc2, ff_mid_, ff_out_);
    bt::broadcast_mul(ff_out_, t_gate_mlp, gated_);
    bt::add_inplace(txt_, gated_);

    // FP16 saturation guard, matching diffusers' FluxTransformerBlock: the
    // text stream's magnitude grows monotonically through the double-stream
    // stack (real Flux.1 weights reach ~17k by block 17 and overflow FP16
    // inside block 18's ff_context), so clip the stream to the FP16 finite
    // range after the MLP residual — saturation instead of inf/NaN.
    if (dt == bt::Dtype::FP16) {
        bt::clamp(txt_, -65504.0f, 65504.0f);
    }

    (void)dev;
}

// ─── single-stream block ───────────────────────────────────────────────────

void FluxDenoiser::run_single_block_(const SingleBlock& blk,
                                     const bt::Tensor& temb,
                                     const bt::Tensor& cos,
                                     const bt::Tensor& sin,
                                     int txt_len, int img_len,
                                     bt::Tensor* trace_out_entry,
                                     const bt::Tensor* attn_bias) {
    const int D  = cfg_.inner_dim();
    const int HD = cfg_.attention_head_dim;
    const int NH = cfg_.num_attention_heads;
    const int L  = txt_len + img_len;

    bt::silu_forward(temb, silu_);

    // norm = AdaLayerNormZeroSingle → shift, scale, gate
    std::vector<bt::Tensor> mod_chunks;
    lin_(blk.norm, silu_, chunk_row_);
    slice_modulation_chunks(chunk_row_, D, 3, mod_chunks);
    bt::Tensor shift = mod_chunks[0].clone();
    bt::Tensor scale = mod_chunks[1].clone();
    bt::Tensor gate  = mod_chunks[2].clone();

    bt::Tensor residual = x_.clone();

    // x_mod = modulate(LN(x), scale, shift)
    detail::layernorm_batched(x_, ada_gamma_, ada_beta_, ln_, 1e-6f);
    bt::modulate(ln_, scale, shift, mod_);
    bt::Tensor x_mod = mod_.clone();

    // mlp = gelu(proj_mlp(x_mod))   (D → 4D)
    lin_(blk.proj_mlp, x_mod, mlp_);
    bt::gelu_forward(mlp_, mlp_);

    // q/k/v from x_mod; per-head RMSNorm on q,k.
    lin_(blk.to_q, x_mod, q_);
    lin_(blk.to_k, x_mod, k_);
    lin_(blk.to_v, x_mod, v_);

    auto headnorm = [&](bt::Tensor& src, const bt::Tensor& gain,
                        bt::Tensor& dst) -> bt::Tensor {
        const int rows = src.rows;
        bt::Tensor src_v =
            bt::Tensor::view(src.device, src.data, rows * NH, HD, src.dtype);
        bt::rms_norm_forward(src_v, gain, 1e-6f, dst);
        return bt::Tensor::view(dst.device, dst.data, rows, D, dst.dtype);
    };
    bt::Tensor qh = headnorm(q_, blk.norm_q, qn_);
    bt::Tensor kh = headnorm(k_, blk.norm_k, kn_);
    Q_ = qh.clone();
    K_ = kh.clone();
    V_ = v_.clone();

    if (trace_out_entry != nullptr || attn_bias != nullptr) {
        // Trace / steering path: same joint sequence layout as the
        // double-stream block — [text ; image] — so the image-query × text-key
        // block is rows [txt_len, txt_len+img_len) × cols [0, txt_len).
        // `attn_bias`, if non-null, is added to the image→text scores before
        // softmax.
        bt::Tensor attn_avg;
        joint_attention_traced(Q_, K_, V_, cos, sin, HD, NH, attn_, attn_avg,
                               Qr_, Kr_, txt_len, attn_bias);
        if (trace_out_entry != nullptr) {
            slice_block(attn_avg, txt_len, img_len, 0, txt_len,
                        *trace_out_entry);
        }
    } else {
        joint_attention(Q_, K_, V_, cos, sin, HD, NH, attn_, Qr_, Kr_);
    }

    // out = proj_out(concat([attn, mlp]))  — (L, 5D) → (L, D)
    const int FF = 4 * D;
    detail::resize_like(cat5_, L, D + FF, attn_.dtype, attn_.device);
    for (int r = 0; r < L; ++r) {
        bt::copy_d2d(attn_, r * D, cat5_, r * (D + FF), D);
        bt::copy_d2d(mlp_, r * FF, cat5_, r * (D + FF) + D, FF);
    }
    lin_(blk.proj_out, cat5_, proj_);

    // x = residual + gate * out
    bt::broadcast_mul(proj_, gate, gated_);
    x_ = residual;
    bt::add_inplace(x_, gated_);

    // FP16 saturation guard, matching diffusers' FluxSingleTransformerBlock
    // (see the double-stream block for why). Inactive under BF16, the normal
    // CUDA configuration.
    if (flux_compute_dtype() == bt::Dtype::FP16) {
        bt::clamp(x_, -65504.0f, 65504.0f);
    }
}

// ─── forward ───────────────────────────────────────────────────────────────

void FluxDenoiser::forward(const bt::Tensor& latent, int H_lat, int W_lat,
                           float timestep, const PreparedConditioning& prepared,
                           Branch branch, bt::Tensor& out) {
    forward_impl_(latent, H_lat, W_lat, timestep, prepared, branch,
                  /*attn_logit_biases=*/nullptr, /*trace_out=*/nullptr, out);
}

void FluxDenoiser::forward_traced(
        const bt::Tensor& latent, int H_lat, int W_lat, float timestep,
        const PreparedConditioning& prepared, Branch branch,
        const std::vector<const bt::Tensor*>* attn_logit_biases,
        AttentionTrace* trace_out, bt::Tensor& out) {
    if (attn_logit_biases != nullptr &&
        static_cast<int>(attn_logit_biases->size()) != num_xattn_blocks()) {
        fail("forward_traced: attn_logit_biases has " +
             std::to_string(attn_logit_biases->size()) + " entries, expected " +
             std::to_string(num_xattn_blocks()));
    }
    if (trace_out != nullptr) {
        trace_out->resize(static_cast<std::size_t>(num_xattn_blocks()));
    }
    forward_impl_(latent, H_lat, W_lat, timestep, prepared, branch,
                  attn_logit_biases, trace_out, out);
}

void FluxDenoiser::forward_impl_(const bt::Tensor& latent,
                                 int H_lat, int W_lat, float timestep,
                                 const PreparedConditioning& prepared,
                                 Branch branch,
                                 const std::vector<const bt::Tensor*>*
                                     attn_logit_biases,
                                 AttentionTrace* trace_out, bt::Tensor& out) {
    if (branch != Branch::Cond) {
        fail("forward: Flux is single-branch; only Branch::Cond is valid");
    }
    if (!prepared) fail("forward: prepared conditioning is empty");
    const auto* prep = dynamic_cast<const FluxPrepared*>(prepared.get());
    if (!prep) fail("forward: prepared conditioning has the wrong type");
    if (x_embedder_.W.size() == 0) fail("forward: weights not loaded");
    if (H_lat % 2 != 0 || W_lat % 2 != 0) {
        fail("forward: H_lat and W_lat must be even");
    }

    const int D  = cfg_.inner_dim();
    const int HD = cfg_.attention_head_dim;
    const int LC = cfg_.latent_channels();
    const int hp = H_lat / 2;
    const int wp = W_lat / 2;
    const int img_len = hp * wp;
    const int txt_len = prep->txt_len;
    const bt::Device dev = bt::default_device();

    // ── CombinedTimestep embedding → temb (1, D) ──────────────────────────
    // The scheduler passes the continuous timestep sigma*1000 — exactly the
    // value the reference embeds (BFL's timestep_embedding has a built-in
    // time_factor=1000 on sigma; diffusers' transformer multiplies its
    // sigma-valued `timestep` input by 1000 before Timesteps(256)). Embed it
    // as-is: timestep_embedding(sigma*1000, 256, 10000); FP32 output.
    {
        std::vector<float> tval = {timestep};
        ts_ = bt::Tensor::from_host(tval.data(), 1, 1).to(dev);
        bt::timestep_embedding(ts_, /*dim=*/256, /*max_period=*/10000.0f,
                               freq_);
    }
    // freq_ is FP32; on a GPU backend it must be cast to the flux compute
    // dtype before the linear. On CPU that dtype is FP32 — no cast.
    bt::Tensor freq_cd = freq_;
    if (flux_compute_dtype() != bt::Dtype::FP32) {
        bt::cast(freq_, freq_cd, flux_compute_dtype());
    }
    detail::linear_batched(te_time_l1_.W, &te_time_l1_.b, freq_cd, temb_time_);
    bt::silu_forward(temb_time_, temb_time_);
    detail::linear_batched(te_time_l2_.W, &te_time_l2_.b, temb_time_, temb_);
    temb_ = temb_.clone();  // (1, D), timestep part

    if (cfg_.guidance_embeds) {
        // Same 1000x convention as the timestep: the reference embeds
        // guidance_scale * 1000 (flux-dev).
        std::vector<float> gval = {prep->guidance * 1000.0f};
        bt::Tensor gts = bt::Tensor::from_host(gval.data(), 1, 1).to(dev);
        bt::Tensor gfreq;
        bt::timestep_embedding(gts, 256, 10000.0f, gfreq);
        bt::Tensor gfreq_cd = gfreq;
        if (flux_compute_dtype() != bt::Dtype::FP32) {
            bt::cast(gfreq, gfreq_cd, flux_compute_dtype());
        }
        detail::linear_batched(te_guidance_l1_.W, &te_guidance_l1_.b,
                               gfreq_cd, temb_guid_);
        bt::silu_forward(temb_guid_, temb_guid_);
        detail::linear_batched(te_guidance_l2_.W, &te_guidance_l2_.b,
                               temb_guid_, temb_guid_);
        bt::add_inplace(temb_, temb_guid_);
    }
    // temb += text_embedder(pooled)
    bt::add_inplace(temb_, prep->text_emb);

    // ── pack latent + x_embedder ──────────────────────────────────────────
    // pack_latents produces the pipeline dtype; cast to the flux compute
    // dtype (BF16 on CUDA) at the boundary.
    bt::Tensor packed = pack_latents(latent, LC, H_lat, W_lat);  // (img_len,64)
    if (packed.cols != cfg_.in_channels) {
        fail("forward: packed latent width != in_channels");
    }
    bt::Tensor packed_cast;
    const bt::Tensor& packed_cd = to_flux_dtype(packed, packed_cast);
    detail::linear_batched(x_embedder_.W, &x_embedder_.b, packed_cd, img_);
    img_ = img_.clone();                       // (img_len, D)
    txt_ = prep->context.clone();              // (txt_len, D)

    // ── 2D axial RoPE tables (shared by all blocks) ───────────────────────
    RopeTables rope = build_axial_rope_tables(txt_len, hp, wp, HD,
                                              cfg_.axes_dims_rope);

    // Trace traversal: the num_layers double blocks fill entries
    // [0, num_layers), then the num_single_layers single blocks fill the
    // rest — matching num_xattn_blocks()'s declared order. Per-block steering
    // biases (when supplied) follow the same order.
    int trace_idx = 0;

    // Resolve (and shape-check) the steering bias for block `idx`. A non-null
    // bias entry must be an (img_len, txt_len) FP32 tensor.
    auto bias_at = [&](int idx) -> const bt::Tensor* {
        if (attn_logit_biases == nullptr) return nullptr;
        const bt::Tensor* b =
            (*attn_logit_biases)[static_cast<std::size_t>(idx)];
        if (b == nullptr) return nullptr;
        if (b->dtype != bt::Dtype::FP32) {
            fail("forward: attn_logit_biases[" + std::to_string(idx) +
                 "] must be FP32");
        }
        if (b->rows != img_len || b->cols != txt_len) {
            fail("forward: attn_logit_biases[" + std::to_string(idx) +
                 "] must be (img_len, txt_len)");
        }
        return b;
    };

    const bool nancheck = flux_nancheck();
    if (nancheck) probe("pre", -1, img_, &txt_);

    // ── double-stream blocks ──────────────────────────────────────────────
    for (const DoubleBlock& blk : double_blocks_) {
        bt::Tensor* te = trace_out
            ? &(*trace_out)[static_cast<std::size_t>(trace_idx)]
            : nullptr;
        run_double_block_(blk, temb_, rope.cos, rope.sin, txt_len, img_len,
                          te, bias_at(trace_idx));
        if (nancheck) probe("double", trace_idx, img_, &txt_);
        ++trace_idx;
    }

    // ── concat [txt ; img] for single-stream stack ────────────────────────
    concat_rows(txt_, img_, x_);

    for (const SingleBlock& blk : single_blocks_) {
        bt::Tensor* te = trace_out
            ? &(*trace_out)[static_cast<std::size_t>(trace_idx)]
            : nullptr;
        run_single_block_(blk, temb_, rope.cos, rope.sin, txt_len, img_len,
                          te, bias_at(trace_idx));
        if (nancheck) probe("single", trace_idx, x_, nullptr);
        ++trace_idx;
    }

    // ── take image part ───────────────────────────────────────────────────
    bt::Tensor img_part;
    slice_rows(x_, txt_len, img_len, img_part);  // (img_len, D)

    // ── norm_out (AdaLayerNormContinuous) → proj_out ─────────────────────
    bt::silu_forward(temb_, silu_);
    std::vector<bt::Tensor> no_chunks;
    detail::linear_batched(norm_out_.W, &norm_out_.b, silu_, chunk_row_);
    slice_modulation_chunks(chunk_row_, D, 2, no_chunks);  // scale, shift
    bt::Tensor no_scale = no_chunks[0].clone();
    bt::Tensor no_shift = no_chunks[1].clone();
    detail::layernorm_batched(img_part, ada_gamma_, ada_beta_, ln_, 1e-6f);
    bt::modulate(ln_, no_scale, no_shift, mod_);
    detail::linear_batched(proj_out_.W, &proj_out_.b, mod_, proj_);  // (img_len,64)

    // ── unpack → (1, LC*H_lat*W_lat) ──────────────────────────────────────
    unpack_latents(proj_, LC, H_lat, W_lat, out);
    if (nancheck) probe("velocity-out", -1, out, &temb_);
    (void)D;
}

}  // namespace brodiffusion::dit
