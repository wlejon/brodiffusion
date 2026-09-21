// QwenImage21Transformer2DModel — the joint-sequence forward: 3-axis RoPE
// table construction, the block-causal prefill / cached-decode attention, the
// 32 single-stream blocks, and the prefix KV cache.
//
// Attention, in the two modes the denoising loop alternates between:
//
//   prefill  (kv_cache_mode "extract", or no cache at all) — the whole joint
//     sequence runs. The block-causal mask decomposes exactly into one
//     attention call per prefix segment plus one for the target image:
//       * a TEXT segment [s, e) attends keys [0, e) causally within its own
//         block and fully before it — which is precisely
//         flash_attention_windowed_forward's unbounded-causal semantics with
//         q_offset = Lk - Lq = s.
//       * an IMAGE segment [s, e) attends keys [0, e) with no mask at all
//         (bidirectional inside its block, causal-by-construction outside).
//       * the target image attends every key, unmasked.
//     An "extract" prefill additionally snapshots each layer's post-RoPE K/V
//     over rows [0, prefix_len).
//
//   cached (kv_cache_mode "cached") — only the target image's rows are
//     pushed through the network. Target queries see the entire sequence
//     under the block-causal rule, so this is plain non-causal attention over
//     [cached prefix K/V ; this step's target K/V].

#include "brodiffusion/dit/qwenimage21.h"

#include "qwenimage21_detail.h"

#include "brodiffusion/dit/common.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"
#include "brodiffusion/detail/jit_fusion.h"

#include "brotensor/jit/trace.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace brodiffusion::dit {

namespace bt = ::brotensor;

using qi21::fail;
using qi21::row_view;

namespace {

// Per-head RMSNorm: view (L, nh*hd) as (L*nh, hd), norm, relabel (L, nh*hd).
bt::Tensor headnorm(const bt::Tensor& X, const bt::Tensor& gain, float eps,
                    int nh, int hd) {
    const int L = X.rows;
    bt::Tensor xv = bt::Tensor::view(X.device, X.data, L * nh, hd, X.dtype);
    bt::Tensor normed;
    bt::rms_norm_forward(xv, gain, eps, normed);   // owns (L*nh, hd)
    normed.rows = L;
    normed.cols = nh * hd;
    return normed;
}

// ── trace-JIT seams ────────────────────────────────────────────────────────
//
// Three places in a block write a whole (L, hidden) activation to HBM only to
// read it straight back. Each becomes one kernel here, with the eager pair
// left in place as the fallback try_fused() returns to.

// dst = LN(src) * (1 + scale). Eager is layernorm + modulate: two kernels,
// two reads and two writes of (L, hidden). Fused it is one of each, and the
// normalized intermediate never exists.
bool fuse_norm_modulate(detail::JitSite& site, const bt::Tensor& src,
                        const bt::Tensor& scale, const bt::Tensor& shift,
                        bt::Tensor& dst, float eps) {
    return detail::try_fused(
        site, {src.data, scale.data, shift.data, dst.data}, [&] {
            // One expression, no named intermediates: every traced temporary
            // stays alive to the semicolon, so no two DAG nodes can land on
            // the same freed device buffer.
            bt::store(dst, bt::modulate(
                               bt::layernorm(src, bt::Tensor(), bt::Tensor(), eps),
                               scale, shift));
        });
}

// x += gate * y, with the optional research mask folded in. Eager is a
// broadcast_mul into a scratch buffer, an optional mul_inplace, and an
// add_inplace — two or three kernels and a (L, hidden) temporary that the
// fused form does not need at all.
bool fuse_gated_residual(detail::JitSite& site, bt::Tensor& x,
                         const bt::Tensor& gate, const bt::Tensor& y,
                         const bt::Tensor* mask) {
    const void* mp = mask ? mask->data : nullptr;
    return detail::try_fused(site, {x.data, gate.data, y.data, mp}, [&] {
        if (mask) {
            x += gate * y * (*mask);
        } else {
            x += gate * y;
        }
    });
}

}  // namespace

// ─── 3-axis RoPE tables ────────────────────────────────────────────────────

void QwenImage21Transformer2DModel::build_rope_tables_(
    const std::vector<QwenImage21Segment>& prefix, int hp, int wp,
    bt::Tensor& cos_t, bt::Tensor& sin_t) {
    const int hd = cfg_.attention_head_dim;
    const int half = hd / 2;

    // A shared running position walks the sequence. Text tokens advance it by
    // one and sit at (p, p, p). An image block freezes the frame axis at the
    // position the preceding text reached, lays its tokens on an h/w grid
    // centred on zero (negative indices included — the reference's frequency
    // table carries positions [-1024, 8192)), then advances the shared
    // position by max(h, w).
    std::vector<int> ax_f, ax_h, ax_w;
    int pos = 0;
    auto add_text = [&](int n) {
        for (int i = 0; i < n; ++i) {
            ax_f.push_back(pos); ax_h.push_back(pos); ax_w.push_back(pos);
            ++pos;
        }
    };
    auto add_image = [&](int h, int w) {
        const int frame = pos;
        const int h0 = -(h - h / 2);
        const int w0 = -(w - w / 2);
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                ax_f.push_back(frame);
                ax_h.push_back(h0 + i);
                ax_w.push_back(w0 + j);
            }
        }
        pos += (h > w ? h : w);
    };
    for (const QwenImage21Segment& s : prefix) {
        if (s.kind == QwenImage21Segment::Kind::Text) add_text(s.n_tokens);
        else add_image(s.h, s.w);
    }
    add_image(hp, wp);

    const int L = static_cast<int>(ax_f.size());
    std::vector<float> ch(static_cast<std::size_t>(L) * half);
    std::vector<float> sh(static_cast<std::size_t>(L) * half);
    const double theta = static_cast<double>(cfg_.rope_theta);
    const int* axes[3] = {ax_f.data(), ax_h.data(), ax_w.data()};

    // Pair p of axis a uses freq = theta^(-2p/axes_dims_rope[a]); the three
    // axes' pairs are concatenated in (frame, height, width) order.
    std::vector<double> freqs(static_cast<std::size_t>(half));
    std::vector<int> pair_axis(static_cast<std::size_t>(half));
    {
        int off = 0;
        for (int a = 0; a < 3; ++a) {
            const int dim = cfg_.axes_dims_rope[static_cast<std::size_t>(a)];
            for (int p = 0; p < dim / 2; ++p) {
                freqs[static_cast<std::size_t>(off + p)] =
                    std::pow(theta, -2.0 * static_cast<double>(p) /
                                        static_cast<double>(dim));
                pair_axis[static_cast<std::size_t>(off + p)] = a;
            }
            off += dim / 2;
        }
    }
    for (int r = 0; r < L; ++r) {
        float* crow = ch.data() + static_cast<std::size_t>(r) * half;
        float* srow = sh.data() + static_cast<std::size_t>(r) * half;
        for (int p = 0; p < half; ++p) {
            const double ang =
                static_cast<double>(axes[pair_axis[static_cast<std::size_t>(p)]][r]) *
                freqs[static_cast<std::size_t>(p)];
            crow[p] = static_cast<float>(std::cos(ang));
            srow[p] = static_cast<float>(std::sin(ang));
        }
    }
    // rope_apply requires FP32 tables on every backend.
    cos_t = bt::Tensor::from_host(ch.data(), L, half).to(bt::default_device());
    sin_t = bt::Tensor::from_host(sh.data(), L, half).to(bt::default_device());
}

// ─── forward ───────────────────────────────────────────────────────────────

void QwenImage21Transformer2DModel::forward(const bt::Tensor& packed_latent,
                                            int hp, int wp,
                                            const bt::Tensor& txt,
                                            float timestep,
                                            QwenImage21PrefixCache* cache,
                                            bt::Tensor& out) {
    std::vector<QwenImage21Segment> prefix(1);
    prefix[0].kind = QwenImage21Segment::Kind::Text;
    prefix[0].n_tokens = txt.rows;
    forward_joint(packed_latent, hp, wp, txt, nullptr, prefix, timestep, cache,
                  out);
}

void QwenImage21Transformer2DModel::forward_joint(
    const bt::Tensor& packed_latent, int hp, int wp, const bt::Tensor& txt,
    const bt::Tensor* cond_latents,
    const std::vector<QwenImage21Segment>& prefix, float timestep,
    QwenImage21PrefixCache* cache, bt::Tensor& out) {
    if (!loaded_) fail("forward: weights not loaded");
    const bt::Dtype dt = flux_compute_dtype();
    const bt::Device dev = bt::default_device();
    const int H = cfg_.hidden_size();
    const int hd = cfg_.attention_head_dim;
    const int nh = cfg_.num_attention_heads;
    const int img_len = hp * wp;
    if (hp <= 0 || wp <= 0) fail("forward: hp and wp must be positive");
    if (packed_latent.rows != img_len ||
        packed_latent.cols != cfg_.in_channels) {
        fail("forward: packed_latent must be (hp*wp, in_channels)");
    }

    // ── prefix layout ────────────────────────────────────────────────────
    int prefix_len = 0, n_text = 0, n_cond = 0;
    for (const QwenImage21Segment& s : prefix) {
        if (s.kind == QwenImage21Segment::Kind::Text) {
            if (s.n_tokens <= 0) fail("forward: empty text segment");
            n_text += s.n_tokens;
        } else {
            if (s.h <= 0 || s.w <= 0) fail("forward: image segment needs h,w");
            if (s.n_tokens != s.h * s.w) {
                fail("forward: image segment n_tokens must equal h*w");
            }
            n_cond += s.n_tokens;
        }
        prefix_len += s.n_tokens;
    }
    if (prefix_len <= 0) fail("forward: the prefix must hold at least one token");
    if (txt.rows != n_text || txt.cols != H || txt.dtype != dt) {
        fail("forward: txt must be (n_text_rows, hidden) at the compute dtype "
             "(see encode_text)");
    }
    if (n_cond > 0 &&
        (cond_latents == nullptr || cond_latents->rows != n_cond ||
         cond_latents->cols != cfg_.in_channels)) {
        fail("forward: cond_latents must be (n_cond_tokens, in_channels)");
    }
    const int L = prefix_len + img_len;

    // ── cache mode ───────────────────────────────────────────────────────
    bool cached_mode = false, extract = false;
    if (cache != nullptr) {
        if (!cfg_.causal_condition) {
            fail("forward: the prefix KV cache requires causal_condition — "
                 "without it the prefix activations depend on the timestep");
        }
        if (cache->valid()) {
            if (!cache->matches(prefix_len, hp, wp)) {
                fail("forward: the prefix cache was extracted for a different "
                     "joint layout — reset() it first");
            }
            cached_mode = true;
        } else {
            extract = true;
            cache->k_.assign(static_cast<std::size_t>(cfg_.num_layers),
                             bt::Tensor());
            cache->v_.assign(static_cast<std::size_t>(cfg_.num_layers),
                             bt::Tensor());
            cache->prefix_len_ = prefix_len;
            cache->hp_ = hp;
            cache->wp_ = wp;
        }
    }
    const int Lq = cached_mode ? img_len : L;

    // ── timestep embedding + shared modulation ───────────────────────────
    // `mod_d` is the set_mod_delta() variant; blocks inside the hook's range
    // read it instead of `mod`. Built once, not per block — the modulation is
    // four (1, hidden) rows.
    Modulation mod, mod_d;
    bool has_mod_delta = false;
    build_modulation_(timestep, mod, &mod_d, &has_mod_delta);

    // Rows [0, prefix_len) of THIS forward's query block carry the t = 0
    // modulation; the rest carry the sampled t. A cached step runs target rows
    // only, so its split degenerates to "all sampled".
    const int mod_split = cached_mode ? 0 : prefix_len;

    // ── RoPE tables ──────────────────────────────────────────────────────
    bt::Tensor cos_full, sin_full;
    build_rope_tables_(prefix, hp, wp, cos_full, sin_full);
    // Non-owning row slices — Tensor's copy ctor deep-clones, so never build
    // these with a `cond ? view : whole` ternary (that clones the table).
    const int rope_off = cached_mode ? prefix_len : 0;
    bt::Tensor cos_q = row_view(cos_full, rope_off, Lq);
    bt::Tensor sin_q = row_view(sin_full, rope_off, Lq);

    // ── joint hidden states ──────────────────────────────────────────────
    bt::Tensor lat;
    if (packed_latent.dtype != dt) {
        bt::cast(packed_latent.to(dev), lat, dt);
    } else {
        lat = packed_latent.to(dev);
    }
    bt::Tensor img = lin_(img_in_, lat);          // (img_len, H)

    bt::Tensor x;
    detail::resize_like(x, Lq, H, dt, dev);
    if (cached_mode) {
        bt::copy_d2d(img, 0, x, 0, img_len * H);
    } else {
        bt::Tensor cimg;
        if (n_cond > 0) {
            bt::Tensor cl;
            if (cond_latents->dtype != dt) {
                bt::cast(cond_latents->to(dev), cl, dt);
            } else {
                cl = cond_latents->to(dev);
            }
            cimg = lin_(img_in_, cl);             // (n_cond, H)
        }
        int row = 0, tcur = 0, ccur = 0;
        for (const QwenImage21Segment& s : prefix) {
            if (s.kind == QwenImage21Segment::Kind::Text) {
                bt::copy_d2d(txt, tcur * H, x, row * H, s.n_tokens * H);
                tcur += s.n_tokens;
            } else {
                bt::copy_d2d(cimg, ccur * H, x, row * H, s.n_tokens * H);
                ccur += s.n_tokens;
            }
            row += s.n_tokens;
        }
        bt::copy_d2d(img, 0, x, prefix_len * H, img_len * H);
    }

    // ── blocks ───────────────────────────────────────────────────────────
    bt::Tensor ln, xm, gated, attn_cat, qr, kr;
    detail::resize_like(xm, Lq, H, dt, dev);
    detail::resize_like(gated, Lq, H, dt, dev);
    detail::resize_like(attn_cat, Lq, H, dt, dev);

    // The linear outputs a JIT site consumes, hoisted so their addresses hold
    // for all 32 blocks. They also have to be pinned to the device before the
    // first linear writes them: a default Tensor is Device::CPU and resize
    // preserves that, so an unpinned output would allocate on the host.
    bt::Tensor ao, mlp_g, mlp_p, mo;
    detail::resize_like(ao, Lq, H, dt, dev);
    detail::resize_like(mo, Lq, H, dt, dev);
    detail::resize_like(mlp_g, Lq, cfg_.mlp_hidden_size(), dt, dev);
    detail::resize_like(mlp_p, Lq, cfg_.mlp_hidden_size(), dt, dev);

    // LN(src) * (1 + scale) into dst, with a (1, H) scale for rows
    // [0, mod_split) and another for the rest. Fused where the JIT takes it;
    // otherwise the eager layernorm + modulate pair, which is why `ln` is
    // only materialised on that branch.
    auto norm_modulate_rows = [&](const bt::Tensor& src, const bt::Tensor& s_pre,
                                  const bt::Tensor& s_post, bt::Tensor& dst,
                                  detail::JitSite& site_pre,
                                  detail::JitSite& site_post) {
        if (mod_split <= 0) {
            if (fuse_norm_modulate(site_post, src, s_post, zero_shift_, dst,
                                   cfg_.eps)) {
                return;
            }
            layernorm_(src, ln);
            bt::modulate(ln, s_post, zero_shift_, dst);
            return;
        }
        bt::Tensor s0 = row_view(src, 0, mod_split);
        bt::Tensor d0 = row_view(dst, 0, mod_split);
        bt::Tensor s1 = row_view(src, mod_split, Lq - mod_split);
        bt::Tensor d1 = row_view(dst, mod_split, Lq - mod_split);
        const bool f0 =
            fuse_norm_modulate(site_pre, s0, s_pre, zero_shift_, d0, cfg_.eps);
        const bool f1 =
            fuse_norm_modulate(site_post, s1, s_post, zero_shift_, d1, cfg_.eps);
        if (f0 && f1) return;
        layernorm_(src, ln);
        if (!f0) bt::modulate(row_view(ln, 0, mod_split), s_pre, zero_shift_, d0);
        if (!f1) {
            bt::modulate(row_view(ln, mod_split, Lq - mod_split), s_post,
                         zero_shift_, d1);
        }
    };

    // x += gate * y (plus the research mask), over the same two row ranges.
    auto gated_residual_rows = [&](bt::Tensor& dst, const bt::Tensor& g_pre,
                                   const bt::Tensor& g_post,
                                   const bt::Tensor& y, const bt::Tensor* mask,
                                   detail::JitSite& site_pre,
                                   detail::JitSite& site_post) {
        auto eager = [&](const bt::Tensor& g, const bt::Tensor& src,
                         bt::Tensor& d, int rows) {
            bt::Tensor gv = row_view(gated, 0, rows);
            bt::broadcast_mul(src, g, gv);
            if (mask) bt::mul_inplace(gv, *mask);
            bt::add_inplace(d, gv);
        };
        if (mod_split <= 0) {
            if (fuse_gated_residual(site_post, dst, g_post, y, mask)) return;
            eager(g_post, y, dst, Lq);
            return;
        }
        bt::Tensor d0 = row_view(dst, 0, mod_split);
        bt::Tensor y0 = row_view(y, 0, mod_split);
        bt::Tensor d1 = row_view(dst, mod_split, Lq - mod_split);
        bt::Tensor y1 = row_view(y, mod_split, Lq - mod_split);
        // The mask is (Lq, H); each range needs its own slice of it.
        bt::Tensor m0, m1;
        if (mask) {
            m0 = row_view(*mask, 0, mod_split);
            m1 = row_view(*mask, mod_split, Lq - mod_split);
        }
        if (!fuse_gated_residual(site_pre, d0, g_pre, y0, mask ? &m0 : nullptr)) {
            eager(g_pre, y0, d0, mod_split);
        }
        if (!fuse_gated_residual(site_post, d1, g_post, y1, mask ? &m1 : nullptr)) {
            eager(g_post, y1, d1, Lq - mod_split);
        }
    };

    // Research hook (set_gate_mask): the rank-1 (Lq, hidden) expansion of the
    // active slice of the per-token mask, built once here and reused by every
    // block inside the hook's range. A cached step reads the mask's TARGET
    // slice — the prefix rows it describes are not being recomputed.
    const bool mask_on =
        gate_mask_.size() == static_cast<std::size_t>(L) &&
        gate_mask_hi_ > gate_mask_lo_;
    if (mask_on) {
        if (gate_ones_row_.rows != 1 || gate_ones_row_.cols != H ||
            gate_ones_row_.dtype != dt) {
            gate_ones_row_ = bt::Tensor::zeros_on(dev, 1, H, dt);
            bt::add_scalar_inplace(gate_ones_row_, 1.0f);
        }
        const bt::Tensor mcol = row_view(gate_mask_, rope_off, Lq);
        bt::matmul(mcol, gate_ones_row_, gate_mask_full_);
    }
    // Research hook (capture_gates): rows = blocks, cols = the FULL joint
    // sequence, so the layout is stable whether the step extracted or decoded.
    if (gate_sink_ != nullptr) {
        gate_sink_->assign(
            static_cast<std::size_t>(cfg_.num_layers) * static_cast<std::size_t>(L),
            0.0f);
    }

    for (int i = 0; i < cfg_.num_layers; ++i) {
        const Block& b = blocks_[static_cast<std::size_t>(i)];

        // Which modulation this block reads (set_mod_delta), and whether its
        // gates carry the set_gate_scale factors / the set_gate_mask.
        const Modulation& M =
            (has_mod_delta && i >= mod_delta_lo_ && i < mod_delta_hi_) ? mod_d
                                                                      : mod;
        const bool gscale = M.scaled && i >= gate_lo_ && i < gate_hi_;
        const bt::Tensor& g1_0 = gscale ? M.gs1_0 : M.gate1_0;
        const bt::Tensor& g1_t = gscale ? M.gs1_t : M.gate1_t;
        const bt::Tensor& g2_0 = gscale ? M.gs2_0 : M.gate2_0;
        const bt::Tensor& g2_t = gscale ? M.gs2_t : M.gate2_t;
        const bool gmask = mask_on && i >= gate_mask_lo_ && i < gate_mask_hi_;

        if (gate_sink_ != nullptr) {
            float* dst = gate_sink_->data() +
                         static_cast<std::size_t>(i) * static_cast<std::size_t>(L);
            const float gp = gscale ? M.mean_gs1_0 : M.mean_g1_0;
            const float gt = gscale ? M.mean_gs1_t : M.mean_g1_t;
            for (int r = 0; r < prefix_len; ++r) dst[r] = gp;
            for (int r = prefix_len; r < L; ++r) dst[r] = gt;
            if (gmask && gate_mask_host_.size() == static_cast<std::size_t>(L)) {
                for (int r = 0; r < L; ++r) {
                    dst[r] *= gate_mask_host_[static_cast<std::size_t>(r)];
                }
            }
        }

        // ── attention sublayer ───────────────────────────────────────────
        norm_modulate_rows(x, M.scale1_0, M.scale1_t, xm, jit_.attn_norm_pre,
                           jit_.attn_norm_post);

        bt::Tensor q = lin_(b.to_q, xm);
        bt::Tensor k = lin_(b.to_k, xm);
        bt::Tensor v = lin_(b.to_v, xm);
        q = headnorm(q, b.norm_q, cfg_.eps, nh, hd);
        k = headnorm(k, b.norm_k, cfg_.eps, nh, hd);
        bt::rope_apply(q, cos_q, sin_q, hd, nh, qr);
        bt::rope_apply(k, cos_q, sin_q, hd, nh, kr);

        if (extract) {
            // Snapshot the prefix's post-RoPE K/V. Only the prefix rows are
            // kept: the target rows are recomputed every step anyway, and at
            // 1024x1024 storing them too would cost ~2.2 GB across 32 layers.
            bt::Tensor& ck = cache->k_[static_cast<std::size_t>(i)];
            bt::Tensor& cv = cache->v_[static_cast<std::size_t>(i)];
            detail::resize_like(ck, prefix_len, H, dt, dev);
            detail::resize_like(cv, prefix_len, H, dt, dev);
            bt::copy_d2d(kr, 0, ck, 0, prefix_len * H);
            bt::copy_d2d(v, 0, cv, 0, prefix_len * H);
        }

        if (cached_mode) {
            detail::resize_like(k_full_, L, H, dt, dev);
            detail::resize_like(v_full_, L, H, dt, dev);
            bt::copy_d2d(cache->k_[static_cast<std::size_t>(i)], 0, k_full_, 0,
                         prefix_len * H);
            bt::copy_d2d(cache->v_[static_cast<std::size_t>(i)], 0, v_full_, 0,
                         prefix_len * H);
            bt::copy_d2d(kr, 0, k_full_, prefix_len * H, img_len * H);
            bt::copy_d2d(v, 0, v_full_, prefix_len * H, img_len * H);
            bt::flash_attention_forward(qr, k_full_, v_full_, nullptr, nh,
                                        /*causal=*/false, attn_cat);
        } else {
            int start = 0;
            for (const QwenImage21Segment& s : prefix) {
                const int end = start + s.n_tokens;
                bt::Tensor Os = row_view(attn_cat, start, s.n_tokens);
                const bt::Tensor Qs = row_view(qr, start, s.n_tokens);
                const bt::Tensor Ks = row_view(kr, 0, end);
                const bt::Tensor Vs = row_view(v, 0, end);
                if (s.kind == QwenImage21Segment::Kind::Text) {
                    // Unbounded causal with q_offset = Lk - Lq = start: query
                    // row r attends keys [0, start + r] — the block-causal
                    // rule restricted to a text run.
                    bt::flash_attention_windowed_forward(Qs, Ks, Vs, nullptr,
                                                         nh, /*window=*/0, Os);
                } else {
                    // A condition image is bidirectional inside its own block
                    // and sees everything before it: unmasked over [0, end).
                    bt::flash_attention_forward(Qs, Ks, Vs, nullptr, nh,
                                                /*causal=*/false, Os);
                }
                start = end;
            }
            bt::Tensor Ot = row_view(attn_cat, prefix_len, img_len);
            const bt::Tensor Qt = row_view(qr, prefix_len, img_len);
            bt::flash_attention_forward(Qt, kr, v, nullptr, nh,
                                        /*causal=*/false, Ot);
        }

        lin_into_(b.to_out, attn_cat, ao);
        gated_residual_rows(x, g1_0, g1_t, ao,
                            gmask ? &gate_mask_full_ : nullptr,
                            jit_.attn_gate_pre, jit_.attn_gate_post);

        // ── SwiGLU feed-forward ──────────────────────────────────────────
        norm_modulate_rows(x, M.scale2_0, M.scale2_t, xm, jit_.mlp_norm_pre,
                           jit_.mlp_norm_post);
        lin_into_(b.mlp_gate, xm, mlp_g);
        lin_into_(b.mlp_proj, xm, mlp_p);
        // g = silu(g) * p: one kernel rather than silu then multiply.
        if (!detail::try_fused(jit_.swiglu, {mlp_g.data, mlp_p.data}, [&] {
                bt::store(mlp_g, bt::silu(mlp_g) * mlp_p);
            })) {
            bt::silu_forward(mlp_g, mlp_g);
            bt::mul_inplace(mlp_g, mlp_p);
        }
        lin_into_(b.mlp_out, mlp_g, mo);
        gated_residual_rows(x, g2_0, g2_t, mo,
                            gmask ? &gate_mask_full_ : nullptr,
                            jit_.mlp_gate_pre, jit_.mlp_gate_post);
    }

    // ── norm_out / proj_out over the target rows only ────────────────────
    bt::Tensor tgt = row_view(x, cached_mode ? 0 : prefix_len, img_len);
    bt::Tensor fnm;
    detail::resize_like(fnm, img_len, H, dt, dev);
    if (!fuse_norm_modulate(jit_.final_norm, tgt, mod.final_scale, zero_shift_,
                            fnm, cfg_.eps)) {
        bt::Tensor fn;
        layernorm_(tgt, fn);
        bt::modulate(fn, mod.final_scale, zero_shift_, fnm);
    }
    out = lin_(proj_out_, fnm);          // (img_len, out_channels)
    bt::sync_all();
}

}  // namespace brodiffusion::dit
