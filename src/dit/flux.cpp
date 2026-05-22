#include "brodiffusion/dit/flux.h"

#include "brodiffusion/dit/common.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::dit {

namespace bt = ::brotensor;
namespace st = ::brotensor::safetensors;

using st::upload_compute_checked;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("dit::FluxDenoiser: " + msg);
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

    auto load_lin = [&](const std::string& key, int out, int in, Linear& lin) {
        upload_compute_checked(need(shards, key + ".weight"), out, in, lin.W,
                               key);
        upload_compute_checked(need(shards, key + ".bias"), out, 1, lin.b, key);
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
        load_lin(p + "norm1.linear", 6 * D, D, B.norm1);
        load_lin(p + "norm1_context.linear", 6 * D, D, B.norm1_context);
        load_lin(p + "attn.to_q", D, D, B.to_q);
        load_lin(p + "attn.to_k", D, D, B.to_k);
        load_lin(p + "attn.to_v", D, D, B.to_v);
        load_lin(p + "attn.add_q_proj", D, D, B.add_q);
        load_lin(p + "attn.add_k_proj", D, D, B.add_k);
        load_lin(p + "attn.add_v_proj", D, D, B.add_v);
        load_lin(p + "attn.to_out.0", D, D, B.to_out);
        load_lin(p + "attn.to_add_out", D, D, B.to_add_out);
        upload_compute_checked(need(shards, p + "attn.norm_q.weight"), HD, 1,
                               B.norm_q, "attn.norm_q");
        upload_compute_checked(need(shards, p + "attn.norm_k.weight"), HD, 1,
                               B.norm_k, "attn.norm_k");
        upload_compute_checked(need(shards, p + "attn.norm_added_q.weight"), HD, 1,
                               B.norm_added_q, "attn.norm_added_q");
        upload_compute_checked(need(shards, p + "attn.norm_added_k.weight"), HD, 1,
                               B.norm_added_k, "attn.norm_added_k");
        load_lin(p + "ff.net.0.proj", 4 * D, D, B.ff0);
        load_lin(p + "ff.net.2", D, 4 * D, B.ff2);
        load_lin(p + "ff_context.net.0.proj", 4 * D, D, B.ffc0);
        load_lin(p + "ff_context.net.2", D, 4 * D, B.ffc2);
    }

    for (int i = 0; i < cfg_.num_single_layers; ++i) {
        SingleBlock& B = single_blocks_[static_cast<std::size_t>(i)];
        const std::string p =
            prefix + "single_transformer_blocks." + std::to_string(i) + ".";
        load_lin(p + "norm.linear", 3 * D, D, B.norm);
        load_lin(p + "attn.to_q", D, D, B.to_q);
        load_lin(p + "attn.to_k", D, D, B.to_k);
        load_lin(p + "attn.to_v", D, D, B.to_v);
        upload_compute_checked(need(shards, p + "attn.norm_q.weight"), HD, 1,
                               B.norm_q, "attn.norm_q");
        upload_compute_checked(need(shards, p + "attn.norm_k.weight"), HD, 1,
                               B.norm_k, "attn.norm_k");
        load_lin(p + "proj_mlp", 4 * D, D, B.proj_mlp);
        load_lin(p + "proj_out", D, 5 * D, B.proj_out);
    }

    load_lin(prefix + "norm_out.linear", 2 * D, D, norm_out_);
    load_lin(prefix + "proj_out", IC, D, proj_out_);

    // AdaLN affine-free LayerNorm params: ones gamma, zeros beta (inner_dim,1).
    {
        std::vector<float> ones(static_cast<std::size_t>(D), 1.0f);
        std::vector<float> zeros(static_cast<std::size_t>(D), 0.0f);
        ada_gamma_ = detail::upload_host(ones.data(), D, 1);
        ada_beta_  = detail::upload_host(zeros.data(), D, 1);
    }
}

void FluxDenoiser::finalize_weights() {
    // Flux carries no quantization — finalize is a no-op, idempotent.
    finalized_ = true;
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
    if (cond.text_embeddings.cols != cfg_.joint_attention_dim) {
        fail("prepare: text_embeddings width != joint_attention_dim");
    }
    prep->txt_len = cond.text_embeddings.rows;
    detail::linear_batched(context_embedder_.W, &context_embedder_.b,
                           cond.text_embeddings, prep->context);
    prep->context = prep->context.clone();

    // text_embedder(pooled CLIP) — depends only on the fixed pooled vector.
    if (cond.pooled.cols != cfg_.pooled_projection_dim || cond.pooled.rows != 1) {
        fail("prepare: pooled must be (1, pooled_projection_dim)");
    }
    bt::Tensor h1, h2;
    detail::linear_batched(te_text_l1_.W, &te_text_l1_.b, cond.pooled, h1);
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
                                     bt::Tensor* trace_out_entry) {
    const int D  = cfg_.inner_dim();
    const int HD = cfg_.attention_head_dim;
    const int NH = cfg_.num_attention_heads;
    const bt::Dtype dt = compute_dtype();
    const bt::Device dev = img_.device;

    // silu(temb) once; both AdaLN modulation MLPs consume it.
    bt::silu_forward(temb, silu_);

    // norm1 (image) and norm1_context (text): 6 chunks each.
    std::vector<bt::Tensor> img_mod, txt_mod;
    detail::linear_batched(blk.norm1.W, &blk.norm1.b, silu_, chunk_row_);
    slice_modulation_chunks(chunk_row_, D, 6, img_mod);
    // copy out before reusing chunk_row_
    std::vector<bt::Tensor> img_c;
    img_c.reserve(6);
    for (auto& t : img_mod) img_c.push_back(t.clone());
    detail::linear_batched(blk.norm1_context.W, &blk.norm1_context.b,
                           silu_, chunk_row_);
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
    detail::linear_batched(blk.to_q.W, &blk.to_q.b, img_modulated, q_);
    detail::linear_batched(blk.to_k.W, &blk.to_k.b, img_modulated, k_);
    detail::linear_batched(blk.to_v.W, &blk.to_v.b, img_modulated, v_);
    bt::Tensor img_q = headnorm(q_, blk.norm_q, qn_);
    bt::Tensor img_k = headnorm(k_, blk.norm_k, kn_);
    bt::Tensor img_q_c = img_q.clone();
    bt::Tensor img_k_c = img_k.clone();
    bt::Tensor img_v_c = v_.clone();

    // text q/k/v
    detail::linear_batched(blk.add_q.W, &blk.add_q.b, txt_modulated, q_);
    detail::linear_batched(blk.add_k.W, &blk.add_k.b, txt_modulated, k_);
    detail::linear_batched(blk.add_v.W, &blk.add_v.b, txt_modulated, v_);
    bt::Tensor txt_q = headnorm(q_, blk.norm_added_q, qn_);
    bt::Tensor txt_k = headnorm(k_, blk.norm_added_k, kn_);
    bt::Tensor txt_q_c = txt_q.clone();
    bt::Tensor txt_k_c = txt_k.clone();
    bt::Tensor txt_v_c = v_.clone();

    // Q/K/V = concat([txt, img]) — text first.
    concat_rows(txt_q_c, img_q_c, Q_);
    concat_rows(txt_k_c, img_k_c, K_);
    concat_rows(txt_v_c, img_v_c, V_);

    if (trace_out_entry != nullptr) {
        // Trace path: materialise the per-head softmax, average over heads,
        // then keep the image-query × text-key block. Joint sequence is
        // [text ; image], so image rows are [txt_len, txt_len+img_len) and
        // text columns are [0, txt_len).
        bt::Tensor attn_avg;
        joint_attention_traced(Q_, K_, V_, cos, sin, HD, NH, attn_, attn_avg,
                               Qr_, Kr_);
        slice_block(attn_avg, txt_len, img_len, 0, txt_len, *trace_out_entry);
    } else {
        joint_attention(Q_, K_, V_, cos, sin, HD, NH, attn_, Qr_, Kr_);
    }

    // split attn into text / image parts
    bt::Tensor txt_attn, img_attn;
    slice_rows(attn_, 0, txt_len, txt_attn);
    slice_rows(attn_, txt_len, img_len, img_attn);

    // img = img + gate_msa * to_out(img_attn)
    detail::linear_batched(blk.to_out.W, &blk.to_out.b, img_attn, proj_);
    bt::broadcast_mul(proj_, i_gate_msa, gated_);
    bt::add_inplace(img_, gated_);
    // txt = txt + gate_msa_ctx * to_add_out(txt_attn)
    detail::linear_batched(blk.to_add_out.W, &blk.to_add_out.b,
                           txt_attn, proj_);
    bt::broadcast_mul(proj_, t_gate_msa, gated_);
    bt::add_inplace(txt_, gated_);

    // ── MLP sub-layer ─────────────────────────────────────────────────────
    // img = img + gate_mlp * ff(modulate(LN(img), scale_mlp, shift_mlp))
    detail::layernorm_batched(img_, ada_gamma_, ada_beta_, ln_, 1e-6f);
    bt::modulate(ln_, i_scale_mlp, i_shift_mlp, mod_);
    detail::linear_batched(blk.ff0.W, &blk.ff0.b, mod_, ff_mid_);
    bt::gelu_forward(ff_mid_, ff_mid_);
    detail::linear_batched(blk.ff2.W, &blk.ff2.b, ff_mid_, ff_out_);
    bt::broadcast_mul(ff_out_, i_gate_mlp, gated_);
    bt::add_inplace(img_, gated_);

    // txt = txt + gate_mlp_ctx * ff_context(modulate(LN(txt), ...))
    detail::layernorm_batched(txt_, ada_gamma_, ada_beta_, ln_, 1e-6f);
    bt::modulate(ln_, t_scale_mlp, t_shift_mlp, mod_);
    detail::linear_batched(blk.ffc0.W, &blk.ffc0.b, mod_, ff_mid_);
    bt::gelu_forward(ff_mid_, ff_mid_);
    detail::linear_batched(blk.ffc2.W, &blk.ffc2.b, ff_mid_, ff_out_);
    bt::broadcast_mul(ff_out_, t_gate_mlp, gated_);
    bt::add_inplace(txt_, gated_);

    (void)dt;
    (void)dev;
}

// ─── single-stream block ───────────────────────────────────────────────────

void FluxDenoiser::run_single_block_(const SingleBlock& blk,
                                     const bt::Tensor& temb,
                                     const bt::Tensor& cos,
                                     const bt::Tensor& sin,
                                     int txt_len, int img_len,
                                     bt::Tensor* trace_out_entry) {
    const int D  = cfg_.inner_dim();
    const int HD = cfg_.attention_head_dim;
    const int NH = cfg_.num_attention_heads;
    const int L  = txt_len + img_len;

    bt::silu_forward(temb, silu_);

    // norm = AdaLayerNormZeroSingle → shift, scale, gate
    std::vector<bt::Tensor> mod_chunks;
    detail::linear_batched(blk.norm.W, &blk.norm.b, silu_, chunk_row_);
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
    detail::linear_batched(blk.proj_mlp.W, &blk.proj_mlp.b, x_mod, mlp_);
    bt::gelu_forward(mlp_, mlp_);

    // q/k/v from x_mod; per-head RMSNorm on q,k.
    detail::linear_batched(blk.to_q.W, &blk.to_q.b, x_mod, q_);
    detail::linear_batched(blk.to_k.W, &blk.to_k.b, x_mod, k_);
    detail::linear_batched(blk.to_v.W, &blk.to_v.b, x_mod, v_);

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

    if (trace_out_entry != nullptr) {
        // Trace path: same joint sequence layout as the double-stream block —
        // [text ; image] — so the image-query × text-key block is rows
        // [txt_len, txt_len+img_len) × cols [0, txt_len).
        bt::Tensor attn_avg;
        joint_attention_traced(Q_, K_, V_, cos, sin, HD, NH, attn_, attn_avg,
                               Qr_, Kr_);
        slice_block(attn_avg, txt_len, img_len, 0, txt_len, *trace_out_entry);
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
    detail::linear_batched(blk.proj_out.W, &blk.proj_out.b, cat5_, proj_);

    // x = residual + gate * out
    bt::broadcast_mul(proj_, gate, gated_);
    x_ = residual;
    bt::add_inplace(x_, gated_);
}

// ─── forward ───────────────────────────────────────────────────────────────

void FluxDenoiser::forward(const bt::Tensor& latent, int H_lat, int W_lat,
                           float timestep, const PreparedConditioning& prepared,
                           Branch branch, bt::Tensor& out) {
    forward_impl_(latent, H_lat, W_lat, timestep, prepared, branch,
                  /*trace_out=*/nullptr, out);
}

void FluxDenoiser::forward_traced(
        const bt::Tensor& latent, int H_lat, int W_lat, float timestep,
        const PreparedConditioning& prepared, Branch branch,
        const std::vector<const bt::Tensor*>* attn_logit_biases,
        AttentionTrace* trace_out, bt::Tensor& out) {
    if (attn_logit_biases != nullptr) {
        throw std::runtime_error(
            "FluxDenoiser::forward_traced: attention steering is not yet "
            "implemented for Flux");
    }
    if (trace_out != nullptr) {
        trace_out->resize(static_cast<std::size_t>(num_xattn_blocks()));
    }
    forward_impl_(latent, H_lat, W_lat, timestep, prepared, branch,
                  trace_out, out);
}

void FluxDenoiser::forward_impl_(const bt::Tensor& latent,
                                 int H_lat, int W_lat, float timestep,
                                 const PreparedConditioning& prepared,
                                 Branch branch,
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
    // time_proj: timestep_embedding(timestep/1000, 256, 10000); FP32 output.
    {
        std::vector<float> tval = {timestep / 1000.0f};
        ts_ = bt::Tensor::from_host(tval.data(), 1, 1).to(dev);
        bt::timestep_embedding(ts_, /*dim=*/256, /*max_period=*/10000.0f,
                               freq_);
    }
    // freq_ is FP32; on a GPU backend it must be cast to compute dtype before
    // the linear. On CPU compute dtype is FP32 — no cast.
    bt::Tensor freq_cd = freq_;
    if (compute_dtype() != bt::Dtype::FP32) {
        bt::cast(freq_, freq_cd, compute_dtype());
    }
    detail::linear_batched(te_time_l1_.W, &te_time_l1_.b, freq_cd, temb_time_);
    bt::silu_forward(temb_time_, temb_time_);
    detail::linear_batched(te_time_l2_.W, &te_time_l2_.b, temb_time_, temb_);
    temb_ = temb_.clone();  // (1, D), timestep part

    if (cfg_.guidance_embeds) {
        std::vector<float> gval = {prep->guidance};
        bt::Tensor gts = bt::Tensor::from_host(gval.data(), 1, 1).to(dev);
        bt::Tensor gfreq;
        bt::timestep_embedding(gts, 256, 10000.0f, gfreq);
        bt::Tensor gfreq_cd = gfreq;
        if (compute_dtype() != bt::Dtype::FP32) {
            bt::cast(gfreq, gfreq_cd, compute_dtype());
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
    bt::Tensor packed = pack_latents(latent, LC, H_lat, W_lat);  // (img_len,64)
    if (packed.cols != cfg_.in_channels) {
        fail("forward: packed latent width != in_channels");
    }
    detail::linear_batched(x_embedder_.W, &x_embedder_.b, packed, img_);
    img_ = img_.clone();                       // (img_len, D)
    txt_ = prep->context.clone();              // (txt_len, D)

    // ── 2D axial RoPE tables (shared by all blocks) ───────────────────────
    RopeTables rope = build_axial_rope_tables(txt_len, hp, wp, HD,
                                              cfg_.axes_dims_rope);

    // Trace traversal: the num_layers double blocks fill entries
    // [0, num_layers), then the num_single_layers single blocks fill the
    // rest — matching num_xattn_blocks()'s declared order.
    int trace_idx = 0;

    // ── double-stream blocks ──────────────────────────────────────────────
    for (const DoubleBlock& blk : double_blocks_) {
        bt::Tensor* te = trace_out
            ? &(*trace_out)[static_cast<std::size_t>(trace_idx)]
            : nullptr;
        run_double_block_(blk, temb_, rope.cos, rope.sin, txt_len, img_len,
                          te);
        ++trace_idx;
    }

    // ── concat [txt ; img] for single-stream stack ────────────────────────
    concat_rows(txt_, img_, x_);

    for (const SingleBlock& blk : single_blocks_) {
        bt::Tensor* te = trace_out
            ? &(*trace_out)[static_cast<std::size_t>(trace_idx)]
            : nullptr;
        run_single_block_(blk, temb_, rope.cos, rope.sin, txt_len, img_len,
                          te);
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
    (void)D;
}

}  // namespace brodiffusion::dit
