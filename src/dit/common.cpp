#include "brodiffusion/dit/common.h"

#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace brodiffusion::dit {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("dit::common: " + msg);
}

// Download an FP32/FP16/BF16 tensor into a host FP32 vector — same pattern
// as pipeline.cpp's decode(), plus the BF16 case (Flux's internal dtype on
// CUDA).
std::vector<float> download_fp32(const bt::Tensor& t) {
    const std::size_t n = static_cast<std::size_t>(t.size());
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(n);
        t.copy_to_host_fp16(bits.data());
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = bt::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    if (t.dtype == bt::Dtype::BF16) {
        std::vector<std::uint16_t> bits(n);
        t.copy_to_host_bf16(bits.data());
        std::vector<float> out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = bt::bf16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return t.to_host_vector();
}

}  // namespace

brotensor::Dtype flux_compute_dtype() {
    return bt::default_device() == bt::Device::CUDA
               ? bt::Dtype::BF16
               : brodiffusion::compute_dtype();
}

// ─── 2x2 latent patch packing ──────────────────────────────────────────────

bt::Tensor pack_latents(const bt::Tensor& latent, int latent_channels,
                        int H_lat, int W_lat) {
    if (H_lat % 2 != 0 || W_lat % 2 != 0) {
        fail("pack_latents: H_lat and W_lat must be even");
    }
    const int hp = H_lat / 2;
    const int wp = W_lat / 2;
    const int img_len = hp * wp;
    const int patch = latent_channels * 4;
    const std::size_t expect =
        static_cast<std::size_t>(latent_channels) * H_lat * W_lat;
    if (static_cast<std::size_t>(latent.size()) != expect) {
        fail("pack_latents: latent has unexpected element count");
    }

    bt::sync_all();
    std::vector<float> src = download_fp32(latent);

    std::vector<float> packed(static_cast<std::size_t>(img_len) * patch);
    const std::size_t plane = static_cast<std::size_t>(H_lat) * W_lat;
    for (int i = 0; i < hp; ++i) {
        for (int j = 0; j < wp; ++j) {
            const std::size_t tok =
                static_cast<std::size_t>(i) * wp + j;
            for (int c = 0; c < latent_channels; ++c) {
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const std::size_t si =
                            static_cast<std::size_t>(c) * plane +
                            static_cast<std::size_t>(2 * i + dy) * W_lat +
                            static_cast<std::size_t>(2 * j + dx);
                        const std::size_t di =
                            tok * patch +
                            static_cast<std::size_t>(c) * 4 + dy * 2 + dx;
                        packed[di] = src[si];
                    }
                }
            }
        }
    }
    return detail::upload_host(packed.data(), img_len, patch);
}

void unpack_latents(const bt::Tensor& packed, int latent_channels,
                    int H_lat, int W_lat, bt::Tensor& out) {
    if (H_lat % 2 != 0 || W_lat % 2 != 0) {
        fail("unpack_latents: H_lat and W_lat must be even");
    }
    const int hp = H_lat / 2;
    const int wp = W_lat / 2;
    const int img_len = hp * wp;
    const int patch = latent_channels * 4;
    if (packed.rows != img_len || packed.cols != patch) {
        fail("unpack_latents: packed has unexpected shape");
    }

    bt::sync_all();
    std::vector<float> src = download_fp32(packed);

    const std::size_t plane = static_cast<std::size_t>(H_lat) * W_lat;
    std::vector<float> latent(
        static_cast<std::size_t>(latent_channels) * plane);
    for (int i = 0; i < hp; ++i) {
        for (int j = 0; j < wp; ++j) {
            const std::size_t tok =
                static_cast<std::size_t>(i) * wp + j;
            for (int c = 0; c < latent_channels; ++c) {
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const std::size_t si =
                            tok * patch +
                            static_cast<std::size_t>(c) * 4 + dy * 2 + dx;
                        const std::size_t di =
                            static_cast<std::size_t>(c) * plane +
                            static_cast<std::size_t>(2 * i + dy) * W_lat +
                            static_cast<std::size_t>(2 * j + dx);
                        latent[di] = src[si];
                    }
                }
            }
        }
    }
    bt::Tensor t = detail::upload_host(
        latent.data(), 1,
        static_cast<int>(latent.size()));
    out = t;
}

// ─── 2D axial RoPE tables ──────────────────────────────────────────────────

RopeTables build_axial_rope_tables(int txt_len, int hp, int wp, int head_dim,
                                   const std::vector<int>& axes_dims_rope,
                                   float theta) {
    if (head_dim % 2 != 0) fail("build_axial_rope_tables: head_dim must be even");
    int sum = 0;
    for (int d : axes_dims_rope) sum += d;
    if (sum != head_dim) {
        fail("build_axial_rope_tables: axes_dims_rope must sum to head_dim");
    }
    const int half = head_dim / 2;          // total pair count
    const int img_len = hp * wp;
    const int L = txt_len + img_len;

    // Per-axis pair offsets / counts. axis a owns pairs
    // [pair_off[a], pair_off[a] + axes_dims_rope[a]/2).
    const int n_axes = static_cast<int>(axes_dims_rope.size());
    std::vector<int> pair_off(static_cast<std::size_t>(n_axes));
    std::vector<int> pair_cnt(static_cast<std::size_t>(n_axes));
    {
        int off = 0;
        for (int a = 0; a < n_axes; ++a) {
            pair_off[static_cast<std::size_t>(a)] = off;
            pair_cnt[static_cast<std::size_t>(a)] = axes_dims_rope[a] / 2;
            off += pair_cnt[static_cast<std::size_t>(a)];
        }
    }

    std::vector<float> cos(static_cast<std::size_t>(L) * half, 1.0f);
    std::vector<float> sin(static_cast<std::size_t>(L) * half, 0.0f);

    // Fill one (row, pair) cell for axis `a`, axis position `pos`.
    auto fill_axis = [&](std::size_t row, int a, int pos) {
        const int D_a = axes_dims_rope[a];
        const int off = pair_off[static_cast<std::size_t>(a)];
        const int cnt = pair_cnt[static_cast<std::size_t>(a)];
        for (int p = 0; p < cnt; ++p) {
            const double omega =
                1.0 / std::pow(static_cast<double>(theta),
                               (2.0 * p) / static_cast<double>(D_a));
            const double angle = static_cast<double>(pos) * omega;
            const std::size_t idx =
                row * half + static_cast<std::size_t>(off + p);
            cos[idx] = static_cast<float>(std::cos(angle));
            sin[idx] = static_cast<float>(std::sin(angle));
        }
    };

    // Text tokens: position (0,0,0) → identity (already initialized).
    // Image tokens: grid (i,j) → axial position (0, i, j).
    for (int i = 0; i < hp; ++i) {
        for (int j = 0; j < wp; ++j) {
            const std::size_t row =
                static_cast<std::size_t>(txt_len) +
                static_cast<std::size_t>(i) * wp + j;
            const int pos[3] = {0, i, j};
            for (int a = 0; a < n_axes && a < 3; ++a) {
                fill_axis(row, a, pos[a]);
            }
        }
    }

    RopeTables out;
    // rope_apply requires FP32 cos/sin tables on every backend.
    out.cos = bt::Tensor::from_host(cos.data(), L, half)
                  .to(bt::default_device());
    out.sin = bt::Tensor::from_host(sin.data(), L, half)
                  .to(bt::default_device());
    return out;
}

// ─── joint attention ───────────────────────────────────────────────────────

void joint_attention(const bt::Tensor& Q, const bt::Tensor& K,
                     const bt::Tensor& V, const bt::Tensor& cos,
                     const bt::Tensor& sin, int head_dim, int num_heads,
                     bt::Tensor& out, bt::Tensor& Qr, bt::Tensor& Kr) {
    bt::rope_apply(Q, cos, sin, head_dim, num_heads, Qr);
    bt::rope_apply(K, cos, sin, head_dim, num_heads, Kr);
    bt::flash_attention_forward(Qr, Kr, V, /*d_mask=*/nullptr, num_heads,
                                /*causal=*/false, out);
}

void joint_attention_traced(const bt::Tensor& Q, const bt::Tensor& K,
                            const bt::Tensor& V, const bt::Tensor& cos,
                            const bt::Tensor& sin, int head_dim, int num_heads,
                            bt::Tensor& out, bt::Tensor& attn_avg,
                            bt::Tensor& Qr, bt::Tensor& Kr,
                            int txt_len,
                            const bt::Tensor* image_text_bias) {
    // RoPE on Q and K — identical to the fused path.
    bt::rope_apply(Q, cos, sin, head_dim, num_heads, Qr);
    bt::rope_apply(K, cos, sin, head_dim, num_heads, Kr);

    const int D = Q.cols;
    if (head_dim <= 0 || num_heads <= 0 || D != head_dim * num_heads) {
        fail("joint_attention_traced: D must equal head_dim * num_heads");
    }
    const int L = Q.rows;
    if (K.rows != L || V.rows != L || K.cols != D || V.cols != D) {
        fail("joint_attention_traced: Q/K/V must share shape (L, D)");
    }

    // Optional pre-softmax image→text steering bias. The joint sequence is
    // [text ; image]: image rows are [txt_len, L), text cols [0, txt_len).
    std::vector<float> bias_host;  // (img_len, txt_len) FP32, empty when unused
    if (image_text_bias != nullptr) {
        if (txt_len < 0 || txt_len > L) {
            fail("joint_attention_traced: txt_len out of range for bias");
        }
        const int img_len = L - txt_len;
        if (image_text_bias->dtype != bt::Dtype::FP32) {
            fail("joint_attention_traced: image_text_bias must be FP32");
        }
        if (image_text_bias->rows != img_len ||
            image_text_bias->cols != txt_len) {
            fail("joint_attention_traced: image_text_bias must be "
                 "(img_len, txt_len)");
        }
        bias_host = image_text_bias->to_host_vector();
    }

    // Materialise the per-head softmax on host in FP32. This is the slow
    // "experiment path": the fused flash kernel never exposes the softmax, so
    // trace mode recomputes attention with plain primitives. FP32 accumulation
    // throughout — matches the fused kernel's internal accumulation dtype.
    bt::sync_all();
    const std::vector<float> q = download_fp32(Qr);  // (L, D), RoPE-rotated
    const std::vector<float> k = download_fp32(Kr);  // (L, D), RoPE-rotated
    const std::vector<float> v = download_fp32(V);   // (L, D)

    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    const std::size_t LD = static_cast<std::size_t>(L) * D;
    const std::size_t LL = static_cast<std::size_t>(L) * L;

    std::vector<float> o(LD, 0.0f);        // attention output (L, D)
    std::vector<double> avg(LL, 0.0);      // head-summed softmax map (L, L)
    std::vector<double> s(static_cast<std::size_t>(L));  // per-row score scratch

    for (int h = 0; h < num_heads; ++h) {
        const int hd0 = h * head_dim;
        for (int qi = 0; qi < L; ++qi) {
            const float* qrow = q.data() +
                static_cast<std::size_t>(qi) * D + hd0;
            // Image-query rows [txt_len, L) carry a per-text-key bias added to
            // the scaled score before softmax; null bias / non-image rows add
            // nothing.
            const bool biased_row =
                !bias_host.empty() && qi >= txt_len;
            const float* brow = biased_row
                ? bias_host.data() +
                      static_cast<std::size_t>(qi - txt_len) * txt_len
                : nullptr;
            // S[qi, kk] = scale * <Qr_head[qi], Kr_head[kk]>  (+ bias)
            double smax = -std::numeric_limits<double>::infinity();
            for (int kk = 0; kk < L; ++kk) {
                const float* krow = k.data() +
                    static_cast<std::size_t>(kk) * D + hd0;
                double dot = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    dot += static_cast<double>(qrow[d]) *
                           static_cast<double>(krow[d]);
                }
                double sc = dot * scale;
                if (brow != nullptr && kk < txt_len) {
                    sc += static_cast<double>(brow[kk]);
                }
                s[static_cast<std::size_t>(kk)] = sc;
                if (sc > smax) smax = sc;
            }
            // Numerically stable softmax over keys.
            double denom = 0.0;
            for (int kk = 0; kk < L; ++kk) {
                const double e =
                    std::exp(s[static_cast<std::size_t>(kk)] - smax);
                s[static_cast<std::size_t>(kk)] = e;
                denom += e;
            }
            const double inv = (denom > 0.0) ? (1.0 / denom) : 0.0;
            float* orow = o.data() + static_cast<std::size_t>(qi) * D + hd0;
            for (int kk = 0; kk < L; ++kk) {
                const double p = s[static_cast<std::size_t>(kk)] * inv;
                avg[static_cast<std::size_t>(qi) * L + kk] += p;
                const float* vrow = v.data() +
                    static_cast<std::size_t>(kk) * D + hd0;
                for (int d = 0; d < head_dim; ++d) {
                    orow[d] += static_cast<float>(p) * vrow[d];
                }
            }
        }
    }

    // Average the per-head softmax maps.
    std::vector<float> avg_f(LL);
    const float inv_h = 1.0f / static_cast<float>(num_heads);
    for (std::size_t i = 0; i < LL; ++i) {
        avg_f[i] = static_cast<float>(avg[i]) * inv_h;
    }

    // The attention output feeds straight back into the denoiser's compute
    // stream, so it must carry Q's dtype (BF16 for Flux on CUDA — the
    // pipeline-dtype upload_host would mismatch). The head-averaged map is a
    // host-facing trace artifact and keeps the pipeline compute dtype.
    if (Q.dtype == bt::Dtype::BF16) {
        const std::size_t n = o.size();
        std::vector<std::uint16_t> bits(n);
        for (std::size_t i = 0; i < n; ++i) {
            bits[i] = bt::fp32_to_bf16_bits(o[i]);
        }
        out = bt::Tensor::from_host_bf16(bits.data(), L, D);
    } else {
        out = detail::upload_host(o.data(), L, D);
    }
    attn_avg = detail::upload_host(avg_f.data(), L, L);
}

// ─── grouped-query masked attention (Krea 2) ───────────────────────────────

void gqa_attention_masked(const bt::Tensor& Q, const bt::Tensor& K,
                          const bt::Tensor& V, const bt::Tensor* cos,
                          const bt::Tensor* sin, int head_dim,
                          int num_q_heads, int num_kv_heads,
                          const float* d_mask, bt::Tensor& out,
                          bt::Tensor& Qr, bt::Tensor& Kr,
                          bt::Tensor& Krep, bt::Tensor& Vrep) {
    if (num_q_heads <= 0 || num_kv_heads <= 0 ||
        num_q_heads % num_kv_heads != 0) {
        fail("gqa_attention_masked: num_kv_heads must divide num_q_heads");
    }
    const int L = Q.rows;

    const bt::Tensor* Qp = &Q;
    const bt::Tensor* Kp = &K;
    if (cos != nullptr && sin != nullptr) {
        bt::rope_apply(Q, *cos, *sin, head_dim, num_q_heads, Qr);
        bt::rope_apply(K, *cos, *sin, head_dim, num_kv_heads, Kr);
        Qp = &Qr;
        Kp = &Kr;
    }

    if (num_q_heads == num_kv_heads) {
        bt::flash_attention_forward(*Qp, *Kp, V, d_mask, num_q_heads,
                                    /*causal=*/false, out);
        return;
    }

    // Widen K/V heads to the query-head count (repeat_interleave): query head qh
    // reads KV head qh/group. flash_attention_forward is not GQA-aware, so we
    // materialise the widened K/V once per call via strided column copies.
    const int group = num_q_heads / num_kv_heads;
    const int qcols = num_q_heads * head_dim;
    const int kvcols = num_kv_heads * head_dim;
    detail::resize_like(Krep, L, qcols, Kp->dtype, Kp->device);
    detail::resize_like(Vrep, L, qcols, V.dtype, V.device);
    for (int qh = 0; qh < num_q_heads; ++qh) {
        const int kvh = qh / group;
        bt::copy_d2d_strided(*Kp, kvh * head_dim, kvcols,
                             Krep, qh * head_dim, qcols, head_dim, L);
        bt::copy_d2d_strided(V, kvh * head_dim, kvcols,
                             Vrep, qh * head_dim, qcols, head_dim, L);
    }
    bt::flash_attention_forward(*Qp, Krep, Vrep, d_mask, num_q_heads,
                                /*causal=*/false, out);
}

// ─── AdaLN modulation chunks ───────────────────────────────────────────────

void slice_modulation_chunks(const bt::Tensor& row, int inner_dim,
                             int n_chunks,
                             std::vector<bt::Tensor>& chunks) {
    if (row.cols != inner_dim * n_chunks) {
        fail("slice_modulation_chunks: row width mismatch");
    }
    chunks.resize(static_cast<std::size_t>(n_chunks));
    for (int k = 0; k < n_chunks; ++k) {
        bt::Tensor& dst = chunks[static_cast<std::size_t>(k)];
        detail::resize_like(dst, 1, inner_dim, row.dtype, row.device);
        bt::copy_d2d(row, k * inner_dim, dst, 0, inner_dim);
    }
}

}  // namespace brodiffusion::dit
