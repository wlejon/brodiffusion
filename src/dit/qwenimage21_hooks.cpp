// QwenImage21Transformer2DModel — the research control surface.
//
// The setters that arm the hooks (modulation delta, gate scale / mask / gate
// delta, prefix-KV attenuation, norm_out scale delta), the coverage resolver
// and gate folder the forward runs on, the time-mod readout, and the prefix
// KV cache's own research operations. The forward itself lives in
// qwenimage21_forward.cpp; the loader and the shared modulation path in
// qwenimage21.cpp.
//
// Three facts about Qwen-Image 2.1 shape everything here.
//
//   1. ONE modulation vector drives all 32 blocks. A hook that is supposed to
//      apply to a BLOCK RANGE therefore cannot mutate the modulation in
//      place — it builds a SECOND copy (built once per forward, not once per
//      block: the cost is four (1, 4*hidden) rows) that the blocks inside the
//      range read instead.
//   2. Prefix and target rows read DIFFERENT rows of that vector (t = 0 vs
//      the sampled t) under causal_condition. So a delta can address one or
//      the other, and the gate dials have separate prefix / target factors.
//      The catch is the prefix KV cache: prefix rows are computed on the
//      extract step only, so a prefix-side change is invisible until the
//      cache is reset. Every doc comment that can say so, does.
//   3. Every hook holds an ORDERED LIST of bindings. Resolving the lists is a
//      per-forward step, not a per-block one: resolve_coverage_() groups the
//      32 blocks by which bindings cover them — in practice one, two or three
//      groups — and each group gets ONE finished modulation whose gate rows
//      already carry the composed scale factor and the composed post-tanh
//      delta. The block loop is then a lookup, the fused gated-residual
//      kernel keeps its single gate operand, and the activations are still
//      read exactly once per sublayer however many bindings are armed.

#include "brodiffusion/dit/qwenimage21.h"

#include "qwenimage21_detail.h"

#include "brodiffusion/dit/common.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::dit {

namespace bt = ::brotensor;

using qi21::fail;

namespace {

// Upload a caller tensor to the default device at the compute dtype.
bt::Tensor to_compute(const bt::Tensor& src) {
    const bt::Dtype dt = flux_compute_dtype();
    bt::Tensor d = src.to(bt::default_device());
    if (d.dtype != dt) {
        bt::Tensor t;
        bt::cast(d, t, dt);
        d = std::move(t);
    }
    return d;
}

// A binding whose range is empty can never cover a block, so an empty range
// is the same thing as "not armed" everywhere below.
bool empty_range(int lo, int hi) { return hi <= lo; }

}  // namespace

// ─── prefix KV cache research operations ───────────────────────────────────

void QwenImage21PrefixCache::scale_kv(int layer_lo, int layer_hi,
                                      float k_scale, float v_scale) {
    if (!valid()) {
        throw std::runtime_error(
            "dit::QwenImage21PrefixCache::scale_kv: the cache is empty "
            "(nothing has been extracted into it yet)");
    }
    const int n = num_layers();
    const int lo = layer_lo < 0 ? 0 : layer_lo;
    const int hi = layer_hi > n ? n : layer_hi;
    for (int i = lo; i < hi; ++i) {
        if (k_scale != 1.0f) {
            bt::scale_inplace(k_[static_cast<std::size_t>(i)], k_scale);
        }
        if (v_scale != 1.0f) {
            bt::scale_inplace(v_[static_cast<std::size_t>(i)], v_scale);
        }
    }
    bt::sync_all();
}

void QwenImage21PrefixCache::blend_from(const QwenImage21PrefixCache& other,
                                        float alpha) {
    if (!valid() || !other.valid()) {
        throw std::runtime_error(
            "dit::QwenImage21PrefixCache::blend_from: both caches must hold "
            "an extracted prefix");
    }
    if (num_layers() != other.num_layers() ||
        prefix_len_ != other.prefix_len_ || hp_ != other.hp_ ||
        wp_ != other.wp_) {
        throw std::runtime_error(
            "dit::QwenImage21PrefixCache::blend_from: layout mismatch — the "
            "two caches must share the same prefix length, target grid and "
            "layer count (i.e. two prompts of equal token length)");
    }
    if (alpha == 0.0f) return;
    // axpby_inplace(y, x, a, b) computes y = a*y + b*x.
    const float keep = 1.0f - alpha;
    for (int i = 0; i < num_layers(); ++i) {
        const auto ix = static_cast<std::size_t>(i);
        bt::axpby_inplace(k_[ix], other.k_[ix], keep, alpha);
        bt::axpby_inplace(v_[ix], other.v_[ix], keep, alpha);
    }
    bt::sync_all();
}

// ─── modulation delta ──────────────────────────────────────────────────────

int QwenImage21Transformer2DModel::add_mod_delta(const bt::Tensor& delta,
                                                 int block_lo, int block_hi,
                                                 QwenImage21ModTarget target) {
    const int H = cfg_.hidden_size();
    if (static_cast<std::int64_t>(delta.size()) !=
        static_cast<std::int64_t>(4) * H) {
        fail("add_mod_delta: delta must have 4*hidden_size elements "
             "(the [scale1, gate1, scale2, gate2] modulation output)");
    }
    QwenImage21ModDeltaBinding b;
    b.delta = to_compute(delta);
    b.delta.rows = 1;
    b.delta.cols = 4 * H;
    b.block_lo = block_lo;
    b.block_hi = block_hi;
    b.target = target;
    mod_deltas_.push_back(std::move(b));
    return static_cast<int>(mod_deltas_.size()) - 1;
}

void QwenImage21Transformer2DModel::set_mod_delta(const bt::Tensor& delta,
                                                  int block_lo, int block_hi,
                                                  QwenImage21ModTarget target) {
    mod_deltas_.clear();
    if (delta.size() == 0 || empty_range(block_lo, block_hi)) return;
    add_mod_delta(delta, block_lo, block_hi, target);
}

void QwenImage21Transformer2DModel::set_mod_deltas(
    const std::vector<QwenImage21ModDeltaBinding>& list) {
    mod_deltas_.clear();
    for (const auto& b : list) {
        if (b.delta.size() == 0 || empty_range(b.block_lo, b.block_hi)) continue;
        add_mod_delta(b.delta, b.block_lo, b.block_hi, b.target);
    }
}

void QwenImage21Transformer2DModel::clear_mod_deltas() { mod_deltas_.clear(); }

// ─── norm_out scale delta (the one single-valued hook: it is not ranged) ───

void QwenImage21Transformer2DModel::set_norm_out_scale_delta(
    const bt::Tensor& delta) {
    if (delta.size() == 0) {
        norm_out_delta_ = bt::Tensor();
        return;
    }
    const int H = cfg_.hidden_size();
    if (static_cast<std::int64_t>(delta.size()) != H) {
        fail("set_norm_out_scale_delta: delta must have hidden_size elements");
    }
    bt::Tensor d = to_compute(delta);
    d.rows = 1;
    d.cols = H;
    norm_out_delta_ = std::move(d);
}

// ─── gate scale ────────────────────────────────────────────────────────────

int QwenImage21Transformer2DModel::add_gate_scale_rows(float attn_txt,
                                                       float attn_img,
                                                       float mlp_txt,
                                                       float mlp_img,
                                                       int block_lo,
                                                       int block_hi) {
    QwenImage21GateScaleBinding b;
    b.attn_txt = attn_txt;
    b.attn_img = attn_img;
    b.mlp_txt  = mlp_txt;
    b.mlp_img  = mlp_img;
    b.block_lo = block_lo;
    b.block_hi = block_hi;
    gate_scales_.push_back(b);
    return static_cast<int>(gate_scales_.size()) - 1;
}

void QwenImage21Transformer2DModel::set_gate_scale_rows(float attn_txt,
                                                        float attn_img,
                                                        float mlp_txt,
                                                        float mlp_img,
                                                        int block_lo,
                                                        int block_hi) {
    gate_scales_.clear();
    const bool identity = attn_txt == 1.0f && attn_img == 1.0f &&
                          mlp_txt == 1.0f && mlp_img == 1.0f;
    if (identity || empty_range(block_lo, block_hi)) return;
    add_gate_scale_rows(attn_txt, attn_img, mlp_txt, mlp_img, block_lo,
                        block_hi);
}

// The rank-1 sugar: (attn, mlp) x (txt, img) -> the four independent
// multipliers. Exactly what the old single-product hook did, spelled out.
int QwenImage21Transformer2DModel::add_gate_scale(float attn_scale,
                                                  float mlp_scale,
                                                  float txt_scale,
                                                  float img_scale,
                                                  int block_lo, int block_hi) {
    return add_gate_scale_rows(attn_scale * txt_scale, attn_scale * img_scale,
                               mlp_scale * txt_scale, mlp_scale * img_scale,
                               block_lo, block_hi);
}

void QwenImage21Transformer2DModel::set_gate_scale(float attn_scale,
                                                   float mlp_scale,
                                                   float txt_scale,
                                                   float img_scale,
                                                   int block_lo,
                                                   int block_hi) {
    set_gate_scale_rows(attn_scale * txt_scale, attn_scale * img_scale,
                        mlp_scale * txt_scale, mlp_scale * img_scale, block_lo,
                        block_hi);
}

void QwenImage21Transformer2DModel::set_gate_scales(
    const std::vector<QwenImage21GateScaleBinding>& list) {
    gate_scales_.clear();
    for (const auto& b : list) {
        if (empty_range(b.block_lo, b.block_hi)) continue;
        add_gate_scale_rows(b.attn_txt, b.attn_img, b.mlp_txt, b.mlp_img,
                            b.block_lo, b.block_hi);
    }
}

void QwenImage21Transformer2DModel::clear_gate_scales() {
    gate_scales_.clear();
}

// ─── gate delta ────────────────────────────────────────────────────────────

int QwenImage21Transformer2DModel::add_gate_delta(const bt::Tensor& delta,
                                                  int block_lo, int block_hi,
                                                  QwenImage21ModTarget target) {
    const int H = cfg_.hidden_size();
    if (static_cast<std::int64_t>(delta.size()) !=
        static_cast<std::int64_t>(2) * H) {
        fail("add_gate_delta: delta must have 2*hidden_size elements "
             "(the [attn, mlp] post-tanh gate pair)");
    }
    // Split once here rather than slicing per forward: the two halves are
    // added to different rows and each add wants a (1, H) operand.
    bt::Tensor d = to_compute(delta);
    d.rows = 1;
    d.cols = 2 * H;
    const bt::Dtype dt = flux_compute_dtype();
    const bt::Device dev = bt::default_device();
    bt::Tensor attn, mlp;
    detail::resize_like(attn, 1, H, dt, dev);
    detail::resize_like(mlp, 1, H, dt, dev);
    bt::copy_d2d(d, 0, attn, 0, H);
    bt::copy_d2d(d, H, mlp, 0, H);

    QwenImage21GateDeltaBinding b;
    b.delta = std::move(d);
    b.block_lo = block_lo;
    b.block_hi = block_hi;
    b.target = target;
    gate_deltas_.push_back(std::move(b));
    gate_delta_attn_.push_back(std::move(attn));
    gate_delta_mlp_.push_back(std::move(mlp));
    return static_cast<int>(gate_deltas_.size()) - 1;
}

void QwenImage21Transformer2DModel::set_gate_delta(
    const bt::Tensor& delta, int block_lo, int block_hi,
    QwenImage21ModTarget target) {
    clear_gate_deltas();
    if (delta.size() == 0 || empty_range(block_lo, block_hi)) return;
    add_gate_delta(delta, block_lo, block_hi, target);
}

void QwenImage21Transformer2DModel::set_gate_deltas(
    const std::vector<QwenImage21GateDeltaBinding>& list) {
    clear_gate_deltas();
    for (const auto& b : list) {
        if (b.delta.size() == 0 || empty_range(b.block_lo, b.block_hi)) continue;
        add_gate_delta(b.delta, b.block_lo, b.block_hi, b.target);
    }
}

void QwenImage21Transformer2DModel::clear_gate_deltas() {
    gate_deltas_.clear();
    gate_delta_attn_.clear();
    gate_delta_mlp_.clear();
}

// ─── gate mask ─────────────────────────────────────────────────────────────

int QwenImage21Transformer2DModel::add_gate_mask(
    const bt::Tensor& mask, int block_lo, int block_hi,
    QwenImage21GateSublayer which) {
    if (mask.size() == 0) {
        fail("add_gate_mask: an empty mask arms nothing — use "
             "clear_gate_masks()");
    }
    std::vector<float> host;
    {   // keep a host copy so gate capture can report the masked value
        bt::Tensor m32 = mask;
        if (mask.dtype != bt::Dtype::FP32) bt::cast(mask, m32, bt::Dtype::FP32);
        bt::sync_all();
        host = m32.to(bt::Device::CPU).to_host_vector();
    }
    bt::Tensor m = to_compute(mask);
    m.rows = static_cast<int>(m.size());
    m.cols = 1;

    QwenImage21GateMaskBinding b;
    b.mask = std::move(m);
    b.block_lo = block_lo;
    b.block_hi = block_hi;
    b.which = which;
    gate_masks_.push_back(std::move(b));
    gate_mask_host_.push_back(std::move(host));
    return static_cast<int>(gate_masks_.size()) - 1;
}

void QwenImage21Transformer2DModel::set_gate_mask(
    const bt::Tensor& mask, int block_lo, int block_hi,
    QwenImage21GateSublayer which) {
    clear_gate_masks();
    if (mask.size() == 0 || empty_range(block_lo, block_hi)) return;
    add_gate_mask(mask, block_lo, block_hi, which);
}

void QwenImage21Transformer2DModel::set_gate_masks(
    const std::vector<QwenImage21GateMaskBinding>& list) {
    clear_gate_masks();
    for (const auto& b : list) {
        if (b.mask.size() == 0 || empty_range(b.block_lo, b.block_hi)) continue;
        add_gate_mask(b.mask, b.block_lo, b.block_hi, b.which);
    }
}

void QwenImage21Transformer2DModel::clear_gate_masks() {
    gate_masks_.clear();
    gate_mask_host_.clear();
    // Force the per-coverage compositions to rebuild.
    mask_host_attn_.clear();
    mask_host_mlp_.clear();
}

// ─── prefix KV attenuation ─────────────────────────────────────────────────
//
// Composed into a per-layer factor pair here, at bind time, so the forward's
// only cost is one scale over the (prefix_len, hidden) rows it copies out of
// the cache — and only for the layers whose factor is not 1.

void QwenImage21Transformer2DModel::compose_prefix_scales_() {
    prefix_k_scale_.clear();
    prefix_v_scale_.clear();
    prefix_k_row_.clear();
    prefix_v_row_.clear();
    prefix_row_len_ = 0;
    const int n = cfg_.num_layers;
    if (prefix_kv_scales_.empty() || n <= 0) return;

    // One row length for the whole list, so a mismatch is caught once, here,
    // rather than per layer in the forward.
    for (const auto& b : prefix_kv_scales_) {
        if (b.row_scale.size() == 0) continue;
        const int len = static_cast<int>(b.row_scale.size());
        if (prefix_row_len_ != 0 && prefix_row_len_ != len) {
            fail("prefix KV row scales disagree on length (" +
                 std::to_string(prefix_row_len_) + " vs " +
                 std::to_string(len) +
                 ") — every row_scale addresses the same prefix");
        }
        prefix_row_len_ = len;
    }

    const auto nl = static_cast<std::size_t>(n);
    prefix_k_scale_.assign(nl, 1.0f);
    prefix_v_scale_.assign(nl, 1.0f);
    if (prefix_row_len_ > 0) {
        prefix_k_row_.assign(nl, std::vector<float>());
        prefix_v_row_.assign(nl, std::vector<float>());
    }
    bool any = false;
    for (const auto& b : prefix_kv_scales_) {
        const int lo = b.layer_lo < 0 ? 0 : b.layer_lo;
        const int hi = b.layer_hi > n ? n : b.layer_hi;
        std::vector<float> rows;
        if (b.row_scale.size() > 0) {
            bt::Tensor r32 = b.row_scale;
            if (b.row_scale.dtype != bt::Dtype::FP32) {
                bt::cast(b.row_scale, r32, bt::Dtype::FP32);
            }
            bt::sync_all();
            rows = r32.to(bt::Device::CPU).to_host_vector();
        }
        for (int i = lo; i < hi; ++i) {
            const auto ix = static_cast<std::size_t>(i);
            if (b.k_scale != 1.0f || b.v_scale != 1.0f) any = true;
            if (rows.empty()) {
                prefix_k_scale_[ix] *= b.k_scale;
                prefix_v_scale_[ix] *= b.v_scale;
                continue;
            }
            // row_scale is a per-row WEIGHT on this binding's scales, not a
            // second multiplier: row r is scaled by
            //     1 + row_scale[r] * (k_scale - 1)
            // so an all-ones vector reproduces the broadcast exactly (which
            // is what "row_scale empty = all rows" has to mean), a 0 leaves
            // that row untouched, and anything between fades it in.
            any = true;
            auto& kr = prefix_k_row_[ix];
            auto& vr = prefix_v_row_[ix];
            if (kr.empty()) {
                kr.assign(static_cast<std::size_t>(prefix_row_len_), 1.0f);
                vr.assign(static_cast<std::size_t>(prefix_row_len_), 1.0f);
            }
            for (std::size_t r = 0; r < rows.size(); ++r) {
                kr[r] *= 1.0f + rows[r] * (b.k_scale - 1.0f);
                vr[r] *= 1.0f + rows[r] * (b.v_scale - 1.0f);
            }
        }
    }
    if (!any) {   // an all-identity list costs the forward nothing
        prefix_k_scale_.clear();
        prefix_v_scale_.clear();
        prefix_k_row_.clear();
        prefix_v_row_.clear();
        prefix_row_len_ = 0;
        return;
    }
    // Fold the scalars into the row vectors where there is one, so the
    // forward applies exactly one operand per layer.
    for (std::size_t i = 0; i < prefix_k_row_.size(); ++i) {
        if (prefix_k_row_[i].empty()) continue;
        for (float& f : prefix_k_row_[i]) f *= prefix_k_scale_[i];
        for (float& f : prefix_v_row_[i]) f *= prefix_v_scale_[i];
    }
}

void QwenImage21Transformer2DModel::expand_prefix_rows_(
    const std::vector<float>& rows, int n_rows, int cols, bt::Tensor& dst) {
    const auto dev = bt::default_device();
    const bt::Dtype dt = compute_dtype();
    if (gate_ones_row_.rows != 1 || gate_ones_row_.cols != cols ||
        gate_ones_row_.dtype != dt) {
        gate_ones_row_ = bt::Tensor::zeros_on(dev, 1, cols, dt);
        bt::add_scalar_inplace(gate_ones_row_, 1.0f);
    }
    bt::Tensor col = bt::Tensor::from_host(rows.data(), n_rows, 1).to(dev);
    if (col.dtype != dt) {
        bt::Tensor t;
        bt::cast(col, t, dt);
        col = std::move(t);
    }
    bt::matmul(col, gate_ones_row_, dst);
}

int QwenImage21Transformer2DModel::add_prefix_kv_scale(
    int layer_lo, int layer_hi, float k_scale, float v_scale,
    const bt::Tensor& row_scale) {
    QwenImage21PrefixKvBinding b;
    b.layer_lo = layer_lo;
    b.layer_hi = layer_hi;
    b.k_scale = k_scale;
    b.v_scale = v_scale;
    b.row_scale = row_scale;
    prefix_kv_scales_.push_back(std::move(b));
    compose_prefix_scales_();
    return static_cast<int>(prefix_kv_scales_.size()) - 1;
}

void QwenImage21Transformer2DModel::set_prefix_kv_scale(
    int layer_lo, int layer_hi, float k_scale, float v_scale,
    const bt::Tensor& row_scale) {
    prefix_kv_scales_.clear();
    const bool identity =
        k_scale == 1.0f && v_scale == 1.0f && row_scale.size() == 0;
    if (!empty_range(layer_lo, layer_hi) && !identity) {
        QwenImage21PrefixKvBinding b;
        b.layer_lo = layer_lo;
        b.layer_hi = layer_hi;
        b.k_scale = k_scale;
        b.v_scale = v_scale;
        b.row_scale = row_scale;
        prefix_kv_scales_.push_back(std::move(b));
    }
    compose_prefix_scales_();
}

void QwenImage21Transformer2DModel::set_prefix_kv_scales(
    const std::vector<QwenImage21PrefixKvBinding>& list) {
    prefix_kv_scales_.clear();
    for (const auto& b : list) {
        if (empty_range(b.layer_lo, b.layer_hi)) continue;
        prefix_kv_scales_.push_back(b);
    }
    compose_prefix_scales_();
}

void QwenImage21Transformer2DModel::clear_prefix_kv_scales() {
    prefix_kv_scales_.clear();
    prefix_k_scale_.clear();
    prefix_v_scale_.clear();
    prefix_k_row_.clear();
    prefix_v_row_.clear();
    prefix_row_len_ = 0;
}

// ─── capture ───────────────────────────────────────────────────────────────

void QwenImage21Transformer2DModel::capture_gates(std::vector<float>* sink) {
    gate_sink_ = sink;
}

// ─── readout ───────────────────────────────────────────────────────────────

void QwenImage21Transformer2DModel::compute_time_mod(float timestep,
                                                     bt::Tensor& temb_out,
                                                     bt::Tensor& mod_out) {
    if (!loaded_) fail("compute_time_mod: weights not loaded");
    bt::Tensor temb, mod;
    // The raw rows only — no hook resolution, so a readout never disturbs the
    // variants the next forward will replay against.
    build_modulation_(timestep, /*build_variants=*/false, &temb, &mod);
    if (temb.dtype != bt::Dtype::FP32) bt::cast(temb, temb_out, bt::Dtype::FP32);
    else temb_out = temb;
    if (mod.dtype != bt::Dtype::FP32) bt::cast(mod, mod_out, bt::Dtype::FP32);
    else mod_out = mod;
    bt::sync_all();
}

// ─── helpers the forward uses ──────────────────────────────────────────────

float QwenImage21Transformer2DModel::row_mean_(const bt::Tensor& row) const {
    if (row.size() == 0) return 0.0f;
    bt::Tensor f32 = row;
    if (row.dtype != bt::Dtype::FP32) bt::cast(row, f32, bt::Dtype::FP32);
    bt::sync_all();
    std::vector<float> h = f32.to(bt::Device::CPU).to_host_vector();
    double acc = 0.0;
    for (float v : h) acc += v;
    return static_cast<float>(acc / static_cast<double>(h.size()));
}

void QwenImage21Transformer2DModel::resolve_coverage_() {
    const int n = cfg_.num_layers;
    coverages_.clear();
    block_variant_.assign(static_cast<std::size_t>(n < 0 ? 0 : n), 0);

    auto cover = [](int i, int lo, int hi) { return i >= lo && i < hi; };
    BlockCoverage cov;
    for (int i = 0; i < n; ++i) {
        cov.mod_deltas.clear();
        cov.gate_scales.clear();
        cov.gate_deltas.clear();
        cov.gate_masks.clear();
        for (std::size_t j = 0; j < mod_deltas_.size(); ++j) {
            if (cover(i, mod_deltas_[j].block_lo, mod_deltas_[j].block_hi)) {
                cov.mod_deltas.push_back(static_cast<int>(j));
            }
        }
        for (std::size_t j = 0; j < gate_scales_.size(); ++j) {
            if (cover(i, gate_scales_[j].block_lo, gate_scales_[j].block_hi)) {
                cov.gate_scales.push_back(static_cast<int>(j));
            }
        }
        for (std::size_t j = 0; j < gate_deltas_.size(); ++j) {
            if (cover(i, gate_deltas_[j].block_lo, gate_deltas_[j].block_hi)) {
                cov.gate_deltas.push_back(static_cast<int>(j));
            }
        }
        for (std::size_t j = 0; j < gate_masks_.size(); ++j) {
            if (cover(i, gate_masks_[j].block_lo, gate_masks_[j].block_hi)) {
                cov.gate_masks.push_back(static_cast<int>(j));
            }
        }
        int found = -1;
        for (std::size_t k = 0; k < coverages_.size(); ++k) {
            if (coverages_[k] == cov) { found = static_cast<int>(k); break; }
        }
        if (found < 0) {
            coverages_.push_back(cov);
            found = static_cast<int>(coverages_.size()) - 1;
        }
        block_variant_[static_cast<std::size_t>(i)] = found;
    }
    if (coverages_.empty()) coverages_.emplace_back();   // a 0-block model
}

void QwenImage21Transformer2DModel::fold_gate_hooks_(const BlockCoverage& cov,
                                                     Modulation& m) {
    // The composed scale factors: one per (sublayer, row set), each the
    // product of the covering bindings' factor for that pair. Independent by
    // construction — moving the image side cannot disturb the prefix side,
    // which is what keeps an image-row sweep off the re-extract path.
    float f1_0 = 1.0f, f1_t = 1.0f, f2_0 = 1.0f, f2_t = 1.0f;
    for (int j : cov.gate_scales) {
        const auto& b = gate_scales_[static_cast<std::size_t>(j)];
        f1_0 *= b.attn_txt;
        f1_t *= b.attn_img;
        f2_0 *= b.mlp_txt;
        f2_t *= b.mlp_img;
    }
    if (f1_t != 1.0f) bt::scale_inplace(m.gate1_t, f1_t);
    if (f2_t != 1.0f) bt::scale_inplace(m.gate2_t, f2_t);
    if (f1_0 != 1.0f) bt::scale_inplace(m.gate1_0, f1_0);
    if (f2_0 != 1.0f) bt::scale_inplace(m.gate2_0, f2_0);

    // The composed post-tanh deltas, added on top of the scaled gates:
    //     g_eff = (prod of scales) * tanh(gate) + (sum of deltas)
    for (int j : cov.gate_deltas) {
        const auto ix = static_cast<std::size_t>(j);
        const auto& b = gate_deltas_[ix];
        if (b.target != QwenImage21ModTarget::Prefix) {
            bt::add_inplace(m.gate1_t, gate_delta_attn_[ix]);
            bt::add_inplace(m.gate2_t, gate_delta_mlp_[ix]);
        }
        if (b.target != QwenImage21ModTarget::Target) {
            bt::add_inplace(m.gate1_0, gate_delta_attn_[ix]);
            bt::add_inplace(m.gate2_0, gate_delta_mlp_[ix]);
        }
    }

    // Gate means are only read back when a capture sink is armed — the
    // readback is a device sync, so it must not run on the hot path. Reading
    // the FOLDED rows is both simpler and exact: there is no arithmetic
    // reconstruction of what the composition did.
    if (gate_sink_ != nullptr) {
        m.mean_g1_t = row_mean_(m.gate1_t);
        m.mean_g1_0 = row_mean_(m.gate1_0);
    }
}

}  // namespace brodiffusion::dit
