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
    Modulation mod;
    build_modulation_(timestep, mod);

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

    // Apply a (1, H) scale to rows [0, mod_split) and another to the rest.
    auto modulate_rows = [&](const bt::Tensor& src, const bt::Tensor& s_pre,
                             const bt::Tensor& s_post, bt::Tensor& dst) {
        if (mod_split <= 0) {
            bt::modulate(src, s_post, zero_shift_, dst);
            return;
        }
        bt::Tensor d0 = row_view(dst, 0, mod_split);
        bt::modulate(row_view(src, 0, mod_split), s_pre, zero_shift_, d0);
        bt::Tensor d1 = row_view(dst, mod_split, Lq - mod_split);
        bt::modulate(row_view(src, mod_split, Lq - mod_split), s_post,
                     zero_shift_, d1);
    };
    auto gate_rows = [&](const bt::Tensor& src, const bt::Tensor& g_pre,
                         const bt::Tensor& g_post, bt::Tensor& dst) {
        if (mod_split <= 0) {
            bt::broadcast_mul(src, g_post, dst);
            return;
        }
        bt::Tensor d0 = row_view(dst, 0, mod_split);
        bt::broadcast_mul(row_view(src, 0, mod_split), g_pre, d0);
        bt::Tensor d1 = row_view(dst, mod_split, Lq - mod_split);
        bt::broadcast_mul(row_view(src, mod_split, Lq - mod_split), g_post, d1);
    };

    for (int i = 0; i < cfg_.num_layers; ++i) {
        const Block& b = blocks_[static_cast<std::size_t>(i)];

        // ── attention sublayer ───────────────────────────────────────────
        layernorm_(x, ln);
        modulate_rows(ln, mod.scale1_0, mod.scale1_t, xm);

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

        bt::Tensor ao = lin_(b.to_out, attn_cat);
        gate_rows(ao, mod.gate1_0, mod.gate1_t, gated);
        bt::add_inplace(x, gated);

        // ── SwiGLU feed-forward ──────────────────────────────────────────
        layernorm_(x, ln);
        modulate_rows(ln, mod.scale2_0, mod.scale2_t, xm);
        bt::Tensor g = lin_(b.mlp_gate, xm);
        bt::silu_forward(g, g);
        bt::Tensor p = lin_(b.mlp_proj, xm);
        bt::mul_inplace(g, p);
        bt::Tensor mo = lin_(b.mlp_out, g);
        gate_rows(mo, mod.gate2_0, mod.gate2_t, gated);
        bt::add_inplace(x, gated);
    }

    // ── norm_out / proj_out over the target rows only ────────────────────
    bt::Tensor tgt = row_view(x, cached_mode ? 0 : prefix_len, img_len);
    bt::Tensor fn;
    layernorm_(tgt, fn);
    bt::Tensor fnm;
    bt::modulate(fn, mod.final_scale, zero_shift_, fnm);
    out = lin_(proj_out_, fnm);          // (img_len, out_channels)
    bt::sync_all();
}

}  // namespace brodiffusion::dit
