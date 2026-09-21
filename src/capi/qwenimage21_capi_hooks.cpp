// qwenimage21_capi — the research hooks and the prefix KV cache surface.
//
// See include/brodiffusion/qwenimage21_capi.h for the contract and
// qwenimage21_capi_detail.h for the context this shares with the core
// translation unit.
//
// Two shapes repeat here, and both come straight from the DiT:
//
//   * Every ranged hook holds an ORDERED LIST of bindings. qi_set_* replaces
//     the list with one entry (a NULL tensor, an identity scale or an empty
//     range clears it), qi_add_* appends and returns the new binding's index,
//     qi_clear_* empties it. All the bindings in a list apply to the same
//     forward, composing per block the way each hook's semantics imply.
//   * The prefix KV attenuation is a DIAL on the model, not a multiply into
//     the cache. It is applied where a cached forward reads the cache, so it
//     is idempotent, it works through a whole step loop, and it survives
//     qi_reset_cache(). The saved SLOTS still hold real caches, because a
//     blend has to mix two extracted prefixes.

#include "qwenimage21_capi_detail.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace bt = ::brotensor;

using qi_capi::download_fp32;
using qi_capi::guarded;
using qi_capi::gate_sublayer;
using qi_capi::mod_target;

extern "C" {

int qi_reset_cache(qi_ctx* c) {
    return guarded([&] { c->cache.reset(); });
}

// ── modulation delta ───────────────────────────────────────────────────────

int qi_set_mod_delta(qi_ctx* c, const float* delta, int block_lo,
                     int block_hi, int target) {
    return guarded([&] {
        auto& dit = c->need_dit("qi_set_mod_delta");
        const auto mt = mod_target(target, "qi_set_mod_delta");
        if (!delta) {
            dit.set_mod_delta(bt::Tensor(), 0, 0, mt);
            return;
        }
        bt::Tensor d = bt::Tensor::from_host(delta, 1, 4 * c->hidden())
                           .to(bt::default_device());
        dit.set_mod_delta(d, block_lo, block_hi, mt);
    });
}

int qi_add_mod_delta(qi_ctx* c, const float* delta, int block_lo,
                     int block_hi, int target) {
    int slot = -1;
    const int rc = guarded([&] {
        auto& dit = c->need_dit("qi_add_mod_delta");
        const auto mt = mod_target(target, "qi_add_mod_delta");
        if (!delta) {
            throw std::runtime_error("qi_add_mod_delta: delta is NULL — use "
                                     "qi_clear_mod_deltas() to clear");
        }
        bt::Tensor d = bt::Tensor::from_host(delta, 1, 4 * c->hidden())
                           .to(bt::default_device());
        slot = dit.add_mod_delta(d, block_lo, block_hi, mt);
    });
    return rc == 0 ? slot : -1;
}

int qi_clear_mod_deltas(qi_ctx* c) {
    return guarded([&] { c->need_dit("qi_clear_mod_deltas").clear_mod_deltas(); });
}

int qi_mod_delta_count(qi_ctx* c) {
    int n = -1;
    const int rc = guarded([&] {
        n = static_cast<int>(
            c->need_dit("qi_mod_delta_count").mod_deltas().size());
    });
    return rc == 0 ? n : -1;
}

// ── timestep readout ───────────────────────────────────────────────────────

int qi_time_mod(qi_ctx* c, float timestep, float* temb_out, float* mod_out) {
    return guarded([&] {
        auto& dit = c->need_dit("qi_time_mod");
        bt::Tensor temb, mod;
        dit.compute_time_mod(timestep, temb, mod);
        if (temb_out) download_fp32(temb, temb_out);
        if (mod_out)  download_fp32(mod, mod_out);
    });
}

// ── gate scale ─────────────────────────────────────────────────────────────

int qi_set_gate_scale(qi_ctx* c, float attn_scale, float mlp_scale,
                      float txt_scale, float img_scale, int block_lo,
                      int block_hi) {
    return guarded([&] {
        c->need_dit("qi_set_gate_scale")
            .set_gate_scale(attn_scale, mlp_scale, txt_scale, img_scale,
                            block_lo, block_hi);
    });
}

int qi_add_gate_scale(qi_ctx* c, float attn_scale, float mlp_scale,
                      float txt_scale, float img_scale, int block_lo,
                      int block_hi) {
    int slot = -1;
    const int rc = guarded([&] {
        slot = c->need_dit("qi_add_gate_scale")
                   .add_gate_scale(attn_scale, mlp_scale, txt_scale, img_scale,
                                   block_lo, block_hi);
    });
    return rc == 0 ? slot : -1;
}

int qi_set_gate_scale_rows(qi_ctx* c, float attn_txt, float attn_img,
                           float mlp_txt, float mlp_img, int block_lo,
                           int block_hi) {
    return guarded([&] {
        c->need_dit("qi_set_gate_scale_rows")
            .set_gate_scale_rows(attn_txt, attn_img, mlp_txt, mlp_img,
                                 block_lo, block_hi);
    });
}

int qi_add_gate_scale_rows(qi_ctx* c, float attn_txt, float attn_img,
                           float mlp_txt, float mlp_img, int block_lo,
                           int block_hi) {
    int slot = -1;
    const int rc = guarded([&] {
        slot = c->need_dit("qi_add_gate_scale_rows")
                   .add_gate_scale_rows(attn_txt, attn_img, mlp_txt, mlp_img,
                                        block_lo, block_hi);
    });
    return rc == 0 ? slot : -1;
}

int qi_clear_gate_scales(qi_ctx* c) {
    return guarded([&] {
        c->need_dit("qi_clear_gate_scales").clear_gate_scales();
    });
}

int qi_gate_scale_count(qi_ctx* c) {
    int n = -1;
    const int rc = guarded([&] {
        n = static_cast<int>(
            c->need_dit("qi_gate_scale_count").gate_scales().size());
    });
    return rc == 0 ? n : -1;
}

// ── gate delta ─────────────────────────────────────────────────────────────

int qi_set_gate_delta(qi_ctx* c, const float* delta, int block_lo,
                      int block_hi, int target) {
    return guarded([&] {
        auto& dit = c->need_dit("qi_set_gate_delta");
        const auto mt = mod_target(target, "qi_set_gate_delta");
        if (!delta) {
            dit.set_gate_delta(bt::Tensor(), 0, 0, mt);
            return;
        }
        bt::Tensor d = bt::Tensor::from_host(delta, 1, 2 * c->hidden())
                           .to(bt::default_device());
        dit.set_gate_delta(d, block_lo, block_hi, mt);
    });
}

int qi_add_gate_delta(qi_ctx* c, const float* delta, int block_lo,
                      int block_hi, int target) {
    int slot = -1;
    const int rc = guarded([&] {
        auto& dit = c->need_dit("qi_add_gate_delta");
        const auto mt = mod_target(target, "qi_add_gate_delta");
        if (!delta) {
            throw std::runtime_error("qi_add_gate_delta: delta is NULL — use "
                                     "qi_clear_gate_deltas() to clear");
        }
        bt::Tensor d = bt::Tensor::from_host(delta, 1, 2 * c->hidden())
                           .to(bt::default_device());
        slot = dit.add_gate_delta(d, block_lo, block_hi, mt);
    });
    return rc == 0 ? slot : -1;
}

int qi_clear_gate_deltas(qi_ctx* c) {
    return guarded([&] {
        c->need_dit("qi_clear_gate_deltas").clear_gate_deltas();
    });
}

int qi_gate_delta_count(qi_ctx* c) {
    int n = -1;
    const int rc = guarded([&] {
        n = static_cast<int>(
            c->need_dit("qi_gate_delta_count").gate_deltas().size());
    });
    return rc == 0 ? n : -1;
}

// ── gate mask ──────────────────────────────────────────────────────────────

int qi_set_gate_mask(qi_ctx* c, const float* mask, int64_t n, int block_lo,
                     int block_hi, int which) {
    return guarded([&] {
        auto& dit = c->need_dit("qi_set_gate_mask");
        const auto w = gate_sublayer(which, "qi_set_gate_mask");
        if (!mask) {
            dit.set_gate_mask(bt::Tensor(), 0, 0, w);
            return;
        }
        bt::Tensor m = bt::Tensor::from_host(mask, static_cast<int>(n), 1)
                           .to(bt::default_device());
        dit.set_gate_mask(m, block_lo, block_hi, w);
    });
}

int qi_add_gate_mask(qi_ctx* c, const float* mask, int64_t n, int block_lo,
                     int block_hi, int which) {
    int slot = -1;
    const int rc = guarded([&] {
        auto& dit = c->need_dit("qi_add_gate_mask");
        const auto w = gate_sublayer(which, "qi_add_gate_mask");
        if (!mask) {
            throw std::runtime_error("qi_add_gate_mask: mask is NULL — use "
                                     "qi_clear_gate_masks() to clear");
        }
        bt::Tensor m = bt::Tensor::from_host(mask, static_cast<int>(n), 1)
                           .to(bt::default_device());
        slot = dit.add_gate_mask(m, block_lo, block_hi, w);
    });
    return rc == 0 ? slot : -1;
}

int qi_clear_gate_masks(qi_ctx* c) {
    return guarded([&] {
        c->need_dit("qi_clear_gate_masks").clear_gate_masks();
    });
}

int qi_gate_mask_count(qi_ctx* c) {
    int n = -1;
    const int rc = guarded([&] {
        n = static_cast<int>(
            c->need_dit("qi_gate_mask_count").gate_masks().size());
    });
    return rc == 0 ? n : -1;
}

// ── norm_out scale delta ───────────────────────────────────────────────────

int qi_set_norm_out_scale_delta(qi_ctx* c, const float* delta) {
    return guarded([&] {
        auto& dit = c->need_dit("qi_set_norm_out_scale_delta");
        if (!delta) {
            dit.set_norm_out_scale_delta(bt::Tensor());
            return;
        }
        bt::Tensor d = bt::Tensor::from_host(delta, 1, c->hidden())
                           .to(bt::default_device());
        dit.set_norm_out_scale_delta(d);
    });
}

// ── gate capture ───────────────────────────────────────────────────────────

int qi_capture_gates(qi_ctx* c, int enable) {
    return guarded([&] {
        c->need_dit("qi_capture_gates")
            .capture_gates(enable ? &c->gates : nullptr);
        if (!enable) c->gates.clear();
    });
}

int64_t qi_gates_size(qi_ctx* c) {
    return static_cast<int64_t>(c->gates.size());
}

int qi_get_gates(qi_ctx* c, float* out) {
    return guarded([&] {
        if (c->gates.empty()) {
            throw std::runtime_error("qi_get_gates: nothing captured");
        }
        if (!out) throw std::runtime_error("qi_get_gates: out is NULL");
        std::memcpy(out, c->gates.data(), c->gates.size() * sizeof(float));
    });
}

// ── prefix KV attenuation ──────────────────────────────────────────────────
//
// No "has anything been extracted" check any more. The scale is a dial on the
// model rather than a multiply into the cache, so arming it before the first
// forward is legitimate — and, unlike the old in-place version, arming it
// twice with the same value means the same thing as arming it once.

namespace {

// row_scale == NULL -> an empty tensor, which means "uniform over all rows".
bt::Tensor prefix_rows(const float* row_scale, int64_t n_rows) {
    if (!row_scale) return bt::Tensor();
    if (n_rows <= 0) {
        throw std::runtime_error("qi_scale_prefix_kv: row_scale is non-NULL "
                                 "but n_rows is not positive");
    }
    return bt::Tensor::from_host(row_scale, static_cast<int>(n_rows), 1)
        .to(bt::default_device());
}

}  // namespace

int qi_scale_prefix_kv(qi_ctx* c, int layer_lo, int layer_hi, float k_scale,
                       float v_scale, const float* row_scale, int64_t n_rows) {
    return guarded([&] {
        c->need_dit("qi_scale_prefix_kv")
            .set_prefix_kv_scale(layer_lo, layer_hi, k_scale, v_scale,
                                 prefix_rows(row_scale, n_rows));
    });
}

int qi_add_prefix_kv_scale(qi_ctx* c, int layer_lo, int layer_hi,
                           float k_scale, float v_scale,
                           const float* row_scale, int64_t n_rows) {
    int slot = -1;
    const int rc = guarded([&] {
        slot = c->need_dit("qi_add_prefix_kv_scale")
                   .add_prefix_kv_scale(layer_lo, layer_hi, k_scale, v_scale,
                                        prefix_rows(row_scale, n_rows));
    });
    return rc == 0 ? slot : -1;
}

int qi_clear_prefix_kv_scales(qi_ctx* c) {
    return guarded([&] {
        c->need_dit("qi_clear_prefix_kv_scales").clear_prefix_kv_scales();
    });
}

int qi_prefix_kv_scale_count(qi_ctx* c) {
    int n = -1;
    const int rc = guarded([&] {
        n = static_cast<int>(
            c->need_dit("qi_prefix_kv_scale_count").prefix_kv_scales().size());
    });
    return rc == 0 ? n : -1;
}

// ── prefix cache slots ─────────────────────────────────────────────────────

int qi_save_prefix(qi_ctx* c, int slot) {
    return guarded([&] {
        if (slot < 0 || slot >= QI_PREFIX_SLOTS) {
            throw std::runtime_error("qi_save_prefix: slot out of range");
        }
        if (!c->cache.valid()) {
            throw std::runtime_error("qi_save_prefix: nothing extracted yet — "
                                     "run one qi_forward first");
        }
        c->slots[static_cast<std::size_t>(slot)] = c->cache;
    });
}

int qi_blend_prefix(qi_ctx* c, int slot, float alpha) {
    return guarded([&] {
        if (slot < 0 || slot >= QI_PREFIX_SLOTS) {
            throw std::runtime_error("qi_blend_prefix: slot out of range");
        }
        if (!c->cache.valid()) {
            throw std::runtime_error("qi_blend_prefix: nothing extracted yet — "
                                     "run one qi_forward first");
        }
        const auto ix = static_cast<std::size_t>(slot);
        if (!c->slots[ix].valid()) {
            throw std::runtime_error("qi_blend_prefix: slot " +
                                     std::to_string(slot) +
                                     " is empty — qi_save_prefix() first");
        }
        c->cache.blend_from(c->slots[ix], alpha);
    });
}

int qi_clear_prefix_slots(qi_ctx* c) {
    return guarded([&] {
        for (auto& s : c->slots) s.reset();
        bt::device_mem_trim(bt::default_device());
    });
}

int qi_prefix_slot_valid(qi_ctx* c, int slot) {
    if (slot < 0 || slot >= QI_PREFIX_SLOTS) return 0;
    return c->slots[static_cast<std::size_t>(slot)].valid() ? 1 : 0;
}

}  // extern "C"
