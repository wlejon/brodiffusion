// QwenImage21Transformer2DModel — the research control surface.
//
// The setters that arm the hooks (modulation delta, gate scale / mask / gate
// capture, norm_out scale delta), the time-mod readout, the two small helpers
// the forward uses to realise them, and the prefix KV cache's own research
// operations. The forward itself lives in qwenimage21_forward.cpp; the
// loader and the shared modulation path in qwenimage21.cpp.
//
// Two facts about Qwen-Image 2.1 shape everything here.
//
//   1. ONE modulation vector drives all 32 blocks. A hook that is supposed to
//      apply to a BLOCK RANGE therefore cannot mutate the modulation in
//      place — it builds a SECOND copy (built once per forward, not once per
//      block: the cost is four (1, 4*hidden) rows) that the blocks inside the
//      range read instead. That is what build_modulation_'s `m_delta` output
//      is.
//   2. Prefix and target rows read DIFFERENT rows of that vector (t = 0 vs
//      the sampled t) under causal_condition. So a delta can address one or
//      the other, and the gate dials have separate prefix / target factors.
//      The catch is the prefix KV cache: prefix rows are computed on the
//      extract step only, so a prefix-side change is invisible until the
//      cache is reset. Every doc comment that can say so, does.

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

// ─── hook setters ──────────────────────────────────────────────────────────

void QwenImage21Transformer2DModel::set_mod_delta(const bt::Tensor& delta,
                                                  int block_lo, int block_hi,
                                                  QwenImage21ModTarget target) {
    if (delta.size() == 0) {
        mod_delta_ = bt::Tensor();
        mod_delta_lo_ = mod_delta_hi_ = 0;
        mod_delta_target_ = QwenImage21ModTarget::Target;
        return;
    }
    const int H = cfg_.hidden_size();
    if (static_cast<std::int64_t>(delta.size()) !=
        static_cast<std::int64_t>(4) * H) {
        fail("set_mod_delta: delta must have 4*hidden_size elements "
             "(the [scale1, gate1, scale2, gate2] modulation output)");
    }
    bt::Tensor d = to_compute(delta);
    d.rows = 1;
    d.cols = 4 * H;
    mod_delta_ = std::move(d);
    mod_delta_lo_ = block_lo;
    mod_delta_hi_ = block_hi;
    mod_delta_target_ = target;
}

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

void QwenImage21Transformer2DModel::set_gate_scale(float attn_scale,
                                                   float mlp_scale,
                                                   float txt_scale,
                                                   float img_scale,
                                                   int block_lo,
                                                   int block_hi) {
    gate_attn_scale_ = attn_scale;
    gate_mlp_scale_  = mlp_scale;
    gate_txt_scale_  = txt_scale;
    gate_img_scale_  = img_scale;
    gate_lo_ = block_lo;
    gate_hi_ = block_hi;
}

void QwenImage21Transformer2DModel::set_gate_delta(const bt::Tensor& delta,
                                                    int block_lo, int block_hi,
                                                    QwenImage21ModTarget target) {
    if (delta.size() == 0) {
        gate_delta_attn_ = bt::Tensor();
        gate_delta_mlp_  = bt::Tensor();
        gate_delta_lo_ = gate_delta_hi_ = 0;
        gate_delta_target_ = QwenImage21ModTarget::Target;
        return;
    }
    const int H = cfg_.hidden_size();
    if (static_cast<std::int64_t>(delta.size()) !=
        static_cast<std::int64_t>(2) * H) {
        fail("set_gate_delta: delta must have 2*hidden_size elements "
             "(the [attn, mlp] post-tanh gate pair)");
    }
    // Split once here rather than slicing per forward: the two halves are
    // added to different rows and each add wants a (1, H) operand.
    bt::Tensor d = to_compute(delta);
    d.rows = 1;
    d.cols = 2 * H;
    const bt::Dtype dt = flux_compute_dtype();
    const bt::Device dev = bt::default_device();
    detail::resize_like(gate_delta_attn_, 1, H, dt, dev);
    detail::resize_like(gate_delta_mlp_, 1, H, dt, dev);
    bt::copy_d2d(d, 0, gate_delta_attn_, 0, H);
    bt::copy_d2d(d, H, gate_delta_mlp_, 0, H);
    gate_delta_lo_ = block_lo;
    gate_delta_hi_ = block_hi;
    gate_delta_target_ = target;
}

void QwenImage21Transformer2DModel::set_gate_mask(const bt::Tensor& mask,
                                                  int block_lo, int block_hi) {
    if (mask.size() == 0) {
        gate_mask_ = bt::Tensor();
        gate_mask_host_.clear();
        gate_mask_full_ = bt::Tensor();
        gate_mask_lo_ = gate_mask_hi_ = 0;
        return;
    }
    {   // keep a host copy so gate capture can report the masked value
        bt::Tensor m32 = mask;
        if (mask.dtype != bt::Dtype::FP32) bt::cast(mask, m32, bt::Dtype::FP32);
        bt::sync_all();
        gate_mask_host_ = m32.to(bt::Device::CPU).to_host_vector();
    }
    bt::Tensor m = to_compute(mask);
    m.rows = static_cast<int>(m.size());
    m.cols = 1;
    gate_mask_ = std::move(m);
    gate_mask_full_ = bt::Tensor();   // rebuilt on the next forward
    gate_mask_lo_ = block_lo;
    gate_mask_hi_ = block_hi;
}

void QwenImage21Transformer2DModel::capture_gates(std::vector<float>* sink) {
    gate_sink_ = sink;
}

// ─── readout ───────────────────────────────────────────────────────────────

void QwenImage21Transformer2DModel::compute_time_mod(float timestep,
                                                     bt::Tensor& temb_out,
                                                     bt::Tensor& mod_out) {
    if (!loaded_) fail("compute_time_mod: weights not loaded");
    Modulation m;
    bt::Tensor temb, mod;
    build_modulation_(timestep, m, nullptr, nullptr, &temb, &mod);
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

void QwenImage21Transformer2DModel::scale_gates_(Modulation& m) {
    const bool active = gate_hi_ > gate_lo_ &&
                        (gate_attn_scale_ != 1.0f || gate_mlp_scale_ != 1.0f ||
                         gate_txt_scale_ != 1.0f || gate_img_scale_ != 1.0f);
    const bool delta_on =
        gate_delta_hi_ > gate_delta_lo_ && gate_delta_attn_.size() > 0;
    // Gate means are only read back when a capture sink is armed — the
    // readback is a device sync, so it must not run on the hot path.
    if (gate_sink_ != nullptr) {
        m.mean_g1_t = row_mean_(m.gate1_t);
        m.mean_g1_0 = row_mean_(m.gate1_0);
    }
    if (active) {
        m.gs1_t = m.gate1_t.clone();
        m.gs2_t = m.gate2_t.clone();
        m.gs1_0 = m.gate1_0.clone();
        m.gs2_0 = m.gate2_0.clone();
        bt::scale_inplace(m.gs1_t, gate_attn_scale_ * gate_img_scale_);
        bt::scale_inplace(m.gs2_t, gate_mlp_scale_  * gate_img_scale_);
        bt::scale_inplace(m.gs1_0, gate_attn_scale_ * gate_txt_scale_);
        bt::scale_inplace(m.gs2_0, gate_mlp_scale_  * gate_txt_scale_);
        m.scaled = true;
        if (gate_sink_ != nullptr) {
            m.mean_gs1_t = m.mean_g1_t * gate_attn_scale_ * gate_img_scale_;
            m.mean_gs1_0 = m.mean_g1_0 * gate_attn_scale_ * gate_txt_scale_;
        }
    } else {
        m.scaled = false;
        m.mean_gs1_t = m.mean_g1_t;
        m.mean_gs1_0 = m.mean_g1_0;
    }

    m.deltaed = false;
    if (!delta_on) return;

    // The deltaed variants. Built from the base gates AND (when a block is
    // covered by both hooks) from the scaled ones, so the forward picks one
    // finished row per block instead of composing two operands per residual.
    const bool hit_t = gate_delta_target_ != QwenImage21ModTarget::Prefix;
    const bool hit_0 = gate_delta_target_ != QwenImage21ModTarget::Target;
    auto build = [&](const bt::Tensor& g1_t, const bt::Tensor& g2_t,
                     const bt::Tensor& g1_0, const bt::Tensor& g2_0,
                     bt::Tensor& d1_t, bt::Tensor& d2_t, bt::Tensor& d1_0,
                     bt::Tensor& d2_0) {
        d1_t = g1_t.clone();
        d2_t = g2_t.clone();
        d1_0 = g1_0.clone();
        d2_0 = g2_0.clone();
        if (hit_t) {
            bt::add_inplace(d1_t, gate_delta_attn_);
            bt::add_inplace(d2_t, gate_delta_mlp_);
        }
        if (hit_0) {
            bt::add_inplace(d1_0, gate_delta_attn_);
            bt::add_inplace(d2_0, gate_delta_mlp_);
        }
    };
    build(m.gate1_t, m.gate2_t, m.gate1_0, m.gate2_0, m.gd1_t, m.gd2_t,
          m.gd1_0, m.gd2_0);
    if (m.scaled) {
        build(m.gs1_t, m.gs2_t, m.gs1_0, m.gs2_0, m.gsd1_t, m.gsd2_t,
              m.gsd1_0, m.gsd2_0);
    }
    m.deltaed = true;
    if (gate_sink_ != nullptr) {
        // The delta's contribution to the mean is the mean of the delta row,
        // which is constant across blocks — one readback, not one per block.
        const float da = row_mean_(gate_delta_attn_);
        m.mean_gd1_t  = m.mean_g1_t  + (hit_t ? da : 0.0f);
        m.mean_gd1_0  = m.mean_g1_0  + (hit_0 ? da : 0.0f);
        m.mean_gsd1_t = m.mean_gs1_t + (hit_t ? da : 0.0f);
        m.mean_gsd1_0 = m.mean_gs1_0 + (hit_0 ? da : 0.0f);
    }
}

}  // namespace brodiffusion::dit
