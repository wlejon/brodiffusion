// SD1.5 UNet inlet — depthwise-separable down path with FiLM conditioning,
// one cross-attn at 8x8, and 12 1x1 tap heads.
//
// Forward only. FP16 throughout. N=1, latent 4x64x64, ctx (77, 768).
//
// The down path computes a chain of features t64a..t8d; each is then
// passed through a per-tap 1x1 conv to produce the corresponding skip
// at the channel width the teacher's up path expects.

#include "brodiffusion/inlet.h"
#include "brodiffusion/safetensors.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <array>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::inlet {

namespace bt = ::brotensor;
namespace st = ::brodiffusion::safetensors;

namespace {

[[noreturn]] void fail(const std::string& m) {
    throw std::runtime_error("inlet::Inlet: " + m);
}

// ─── FiLM kernel ───────────────────────────────────────────────────────────
//
// Given x (1, C, H, W) and a per-channel scale/shift packed as
// params (1, 2*C) = [scale(C) | shift(C)], compute in-place:
//   y[c, h, w] = x[c, h, w] * (1 + scale[c]) + shift[c].
__global__ void film_kernel(__half* __restrict__ x,
                            const __half* __restrict__ params,
                            int C, int spatial) {
    const int c   = blockIdx.x;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
    if (c >= C) return;

    const float scale = __half2float(params[c]) + 1.0f;
    const float shift = __half2float(params[C + c]);

    __half* xc = x + c * spatial;
    for (int i = tid; i < spatial; i += bs) {
        float v = __half2float(xc[i]);
        xc[i] = __float2half(v * scale + shift);
    }
}

void launch_film(bt::GpuTensor& x_nchw, const bt::GpuTensor& params, int C, int H, int W) {
    const int spatial = H * W;
    dim3 grid(C);
    dim3 block(256);
    film_kernel<<<grid, block>>>(
        reinterpret_cast<__half*>(x_nchw.data_fp16()),
        reinterpret_cast<const __half*>(params.data_fp16()),
        C, spatial);
}

// ─── NCHW <-> seq layout helpers (for the 8x8 xattn) ──────────────────────
//
// x_nchw (1, C, H, W) -> x_seq (H*W, C)
// Implemented as a transpose: C is the contiguous axis in seq form.
__global__ void nchw_to_seq_kernel(const __half* __restrict__ src,
                                   __half* __restrict__ dst,
                                   int C, int spatial) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    if (s >= spatial || c >= C) return;
    dst[s * C + c] = src[c * spatial + s];
}

__global__ void seq_to_nchw_kernel(const __half* __restrict__ src,
                                   __half* __restrict__ dst,
                                   int C, int spatial) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    if (s >= spatial || c >= C) return;
    dst[c * spatial + s] = src[s * C + c];
}

void launch_nchw_to_seq(const bt::GpuTensor& src, bt::GpuTensor& dst, int C, int spatial) {
    dst.resize(spatial, C, bt::Dtype::FP16);
    dim3 block(64);
    dim3 grid((spatial + 63) / 64, C);
    nchw_to_seq_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(src.data_fp16()),
        reinterpret_cast<__half*>(dst.data_fp16()),
        C, spatial);
}

void launch_seq_to_nchw(const bt::GpuTensor& src, bt::GpuTensor& dst, int C, int spatial) {
    dst.resize(1, C * spatial, bt::Dtype::FP16);
    dim3 block(64);
    dim3 grid((spatial + 63) / 64, C);
    seq_to_nchw_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(src.data_fp16()),
        reinterpret_cast<__half*>(dst.data_fp16()),
        C, spatial);
}

// In-place add: a += b (same shape, FP16).
__global__ void add_inplace_kernel(__half* __restrict__ a, const __half* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

void launch_add_inplace(bt::GpuTensor& a, const bt::GpuTensor& b) {
    const int n = a.size();
    const int bs = 256;
    add_inplace_kernel<<<(n + bs - 1) / bs, bs>>>(
        reinterpret_cast<__half*>(a.data_fp16()),
        reinterpret_cast<const __half*>(b.data_fp16()),
        n);
}

// ─── Shape tables ──────────────────────────────────────────────────────────
//
// FiLM-conditioned DWS stages: 0..7.
struct FilmShape {
    int C_in, C_out, H;  // H == W
};
constexpr std::array<FilmShape, 8> kFilm = {{
    {320,  320,  64}, {320,  320,  64},
    {320,  640,  32}, {640,  640,  32},
    {640,  1280, 16}, {1280, 1280, 16},
    {1280, 1280, 8},  {1280, 1280, 8},
}};

// Stride-2 DWS stages: 0..2 (downto32, downto16, downto8). C_in == C_out.
struct DownShape {
    int C, H_in;
};
constexpr std::array<DownShape, 3> kDown = {{
    {320,  64}, {640,  32}, {1280, 16},
}};

constexpr int kTapC[12] = {320,320,320,320, 640,640,640, 1280,1280,1280,1280,1280};
constexpr int kTapH[12] = { 64, 64, 64, 32,  32, 32, 16,   16,   16,    8,    8,    8};

// ─── Safetensors helpers ───────────────────────────────────────────────────

void upload_fp16_or_zero(const st::File& f, const std::string& key,
                         int rows, int cols, bt::GpuTensor& dst, bool required) {
    const auto* v = f.find(key);
    if (!v) {
        if (required) fail("missing tensor '" + key + "'");
        dst.resize(rows, cols, bt::Dtype::FP16);
        dst.zero();
        return;
    }
    int64_t expected = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    if (v->numel() != expected) {
        fail("shape mismatch for '" + key + "' (expected " +
             std::to_string(rows) + "x" + std::to_string(cols) + ")");
    }
    st::upload_fp16(*v, rows, cols, dst);
}

}  // namespace

// ─── allocate ──────────────────────────────────────────────────────────────

void Inlet::allocate() {
    // stem: 320 out, 4 in, 3x3 -> Wt (320, 4*9), b (320, 1).
    stem_w.resize(320, 4 * 9, bt::Dtype::FP16);
    stem_b.resize(320, 1, bt::Dtype::FP16);

    for (int i = 0; i < 8; ++i) {
        const auto& s = kFilm[i];
        dws_film[i].dw_w.resize(s.C_in,  9, bt::Dtype::FP16);
        dws_film[i].dw_b.resize(s.C_in,  1, bt::Dtype::FP16);
        dws_film[i].pw_w.resize(s.C_out, s.C_in, bt::Dtype::FP16);
        dws_film[i].pw_b.resize(s.C_out, 1, bt::Dtype::FP16);
        film_w[i].resize(2 * s.C_out, 1280, bt::Dtype::FP16);
        film_b[i].resize(2 * s.C_out, 1, bt::Dtype::FP16);
    }

    for (int i = 0; i < 3; ++i) {
        const int C = kDown[i].C;
        dws_down[i].dw_w.resize(C, 9, bt::Dtype::FP16);
        dws_down[i].dw_b.resize(C, 1, bt::Dtype::FP16);
        dws_down[i].pw_w.resize(C, C, bt::Dtype::FP16);
        dws_down[i].pw_b.resize(C, 1, bt::Dtype::FP16);
    }

    xattn_ln_g.resize(1280, 1, bt::Dtype::FP16);
    xattn_ln_b.resize(1280, 1, bt::Dtype::FP16);
    xattn_wq.resize(1280, 1280, bt::Dtype::FP16);
    xattn_bq.resize(1280, 1,    bt::Dtype::FP16);
    xattn_wk.resize(1280, 768,  bt::Dtype::FP16);
    xattn_bk.resize(1280, 1,    bt::Dtype::FP16);
    xattn_wv.resize(1280, 768,  bt::Dtype::FP16);
    xattn_bv.resize(1280, 1,    bt::Dtype::FP16);
    xattn_wo.resize(1280, 1280, bt::Dtype::FP16);
    xattn_bo.resize(1280, 1,    bt::Dtype::FP16);

    for (int i = 0; i < 12; ++i) {
        const int C = kTapC[i];
        tap_w[i].resize(C, C, bt::Dtype::FP16);
        tap_b[i].resize(C, 1, bt::Dtype::FP16);
    }
}

void Inlet::zero_init() {
    auto z = [](bt::GpuTensor& t) { if (t.size() > 0) t.zero(); };
    z(stem_w); z(stem_b);
    for (auto& d : dws_film) { z(d.dw_w); z(d.dw_b); z(d.pw_w); z(d.pw_b); }
    for (auto& d : dws_down) { z(d.dw_w); z(d.dw_b); z(d.pw_w); z(d.pw_b); }
    for (auto& t : film_w) z(t);
    for (auto& t : film_b) z(t);
    z(xattn_ln_g); z(xattn_ln_b);
    z(xattn_wq); z(xattn_bq);
    z(xattn_wk); z(xattn_bk);
    z(xattn_wv); z(xattn_bv);
    z(xattn_wo); z(xattn_bo);
    for (auto& t : tap_w) z(t);
    for (auto& t : tap_b) z(t);
}

void Inlet::load_from_safetensors(const st::File& f, const std::string& prefix) {
    allocate();

    upload_fp16_or_zero(f, prefix + "stem.weight", 320, 4 * 9, stem_w, true);
    upload_fp16_or_zero(f, prefix + "stem.bias",   320, 1, stem_b, true);

    for (int i = 0; i < 8; ++i) {
        const auto& s = kFilm[i];
        const std::string p = prefix + "dws_film." + std::to_string(i) + ".";
        upload_fp16_or_zero(f, p + "dw_w", s.C_in,  9,        dws_film[i].dw_w, true);
        upload_fp16_or_zero(f, p + "dw_b", s.C_in,  1,        dws_film[i].dw_b, true);
        upload_fp16_or_zero(f, p + "pw_w", s.C_out, s.C_in,   dws_film[i].pw_w, true);
        upload_fp16_or_zero(f, p + "pw_b", s.C_out, 1,        dws_film[i].pw_b, true);

        const std::string pf = prefix + "film." + std::to_string(i) + ".";
        upload_fp16_or_zero(f, pf + "weight", 2 * s.C_out, 1280, film_w[i], true);
        upload_fp16_or_zero(f, pf + "bias",   2 * s.C_out, 1,    film_b[i], true);
    }

    for (int i = 0; i < 3; ++i) {
        const int C = kDown[i].C;
        const std::string p = prefix + "dws_down." + std::to_string(i) + ".";
        upload_fp16_or_zero(f, p + "dw_w", C, 9, dws_down[i].dw_w, true);
        upload_fp16_or_zero(f, p + "dw_b", C, 1, dws_down[i].dw_b, true);
        upload_fp16_or_zero(f, p + "pw_w", C, C, dws_down[i].pw_w, true);
        upload_fp16_or_zero(f, p + "pw_b", C, 1, dws_down[i].pw_b, true);
    }

    upload_fp16_or_zero(f, prefix + "xattn.ln_g", 1280, 1,    xattn_ln_g, true);
    upload_fp16_or_zero(f, prefix + "xattn.ln_b", 1280, 1,    xattn_ln_b, true);
    upload_fp16_or_zero(f, prefix + "xattn.wq",   1280, 1280, xattn_wq, true);
    upload_fp16_or_zero(f, prefix + "xattn.bq",   1280, 1,    xattn_bq, true);
    upload_fp16_or_zero(f, prefix + "xattn.wk",   1280, 768,  xattn_wk, true);
    upload_fp16_or_zero(f, prefix + "xattn.bk",   1280, 1,    xattn_bk, true);
    upload_fp16_or_zero(f, prefix + "xattn.wv",   1280, 768,  xattn_wv, true);
    upload_fp16_or_zero(f, prefix + "xattn.bv",   1280, 1,    xattn_bv, true);
    upload_fp16_or_zero(f, prefix + "xattn.wo",   1280, 1280, xattn_wo, true);
    upload_fp16_or_zero(f, prefix + "xattn.bo",   1280, 1,    xattn_bo, true);

    for (int i = 0; i < 12; ++i) {
        const int C = kTapC[i];
        const std::string p = prefix + "tap." + std::to_string(i) + ".";
        upload_fp16_or_zero(f, p + "weight", C, C, tap_w[i], true);
        upload_fp16_or_zero(f, p + "bias",   C, 1, tap_b[i], true);
    }
}

// ─── forward ───────────────────────────────────────────────────────────────

void Inlet::forward(const bt::GpuTensor& sample,
                    const bt::GpuTensor& t_emb,
                    const bt::GpuTensor& ctx,
                    std::array<bt::GpuTensor, 12>& skips,
                    bt::GpuTensor& bottleneck) const {
    if (stem_w.size() == 0) fail("forward: weights not allocated");

    // ─── scratch ──────────────────────────────────────────────────────────
    // Single rotating-buffer pattern: t_main holds the current activation.
    // dw_buf holds the depthwise intermediate; t_next holds the pointwise out.
    thread_local static bt::GpuTensor t_main;
    thread_local static bt::GpuTensor dw_buf;
    thread_local static bt::GpuTensor t_next;
    thread_local static bt::GpuTensor temb_silu_local;
    thread_local static bt::GpuTensor film_proj;     // (1, 2*C)
    thread_local static bt::GpuTensor seq_buf;       // (Lq, D) for xattn
    thread_local static bt::GpuTensor attn_out;      // (Lq, D)

    // The 12 "feature" tensors that get fed to the 12 tap heads. We materialise
    // them as separate buffers because they're used out-of-order at the end.
    // (Could be optimised — but this is scaffolding.)
    thread_local static std::array<bt::GpuTensor, 12> feats;

    // ─── 1. stem (4 -> 320, 3x3 s1 p1) ────────────────────────────────────
    bt::conv2d_forward_gpu(sample, stem_w, &stem_b,
                           /*N=*/1, /*C_in=*/4, /*H=*/64, /*W=*/64,
                           /*C_out=*/320, /*kH=*/3, /*kW=*/3,
                           /*stride*/1, 1, /*pad*/1, 1, /*dil*/1, 1,
                           feats[0]);  // t64a

    auto apply_dws = [&](const DWSWeights& w,
                         int C_in, int C_out, int H_in, int W_in,
                         int stride,
                         const bt::GpuTensor& in,
                         bt::GpuTensor& out) {
        // depthwise 3x3
        bt::conv2d_forward_gpu(in, w.dw_w, &w.dw_b,
                               1, C_in, H_in, W_in,
                               C_in, 3, 3,
                               stride, stride, 1, 1, 1, 1,
                               /*groups=*/C_in,
                               dw_buf);
        bt::silu_forward_gpu(dw_buf, dw_buf);
        const int H_out = H_in / stride;
        const int W_out = W_in / stride;
        // pointwise 1x1
        bt::conv2d_forward_gpu(dw_buf, w.pw_w, &w.pw_b,
                               1, C_in, H_out, W_out,
                               C_out, 1, 1,
                               1, 1, 0, 0, 1, 1,
                               out);
    };

    auto apply_film = [&](int film_idx, int C, int H, bt::GpuTensor& x_nchw) {
        // film_proj = Linear(film_w[i], film_b[i]) (silu(t_emb))
        // silu(t_emb) into temb_silu_local
        bt::silu_forward_gpu(t_emb, temb_silu_local);
        bt::linear_forward_batched_fp16_gpu(film_w[film_idx], &film_b[film_idx],
                                            temb_silu_local, film_proj);
        launch_film(x_nchw, film_proj, C, H, H);
    };

    // ─── 2. stage64 (DWS film 0,1) ────────────────────────────────────────
    // feats[0] = t64a (320, 64, 64)
    // -> DWS#0 + FiLM#0 -> t64b (feats[1])
    apply_dws(dws_film[0], 320, 320, 64, 64, 1, feats[0], feats[1]);
    apply_film(0, 320, 64, feats[1]);
    // -> DWS#1 + FiLM#1 -> t64c (feats[2])
    apply_dws(dws_film[1], 320, 320, 64, 64, 1, feats[1], feats[2]);
    apply_film(1, 320, 64, feats[2]);

    // ─── 3. downto32 (stride-2 DWS#0) -> t32a = feats[3] ──────────────────
    apply_dws(dws_down[0], 320, 320, 64, 64, 2, feats[2], feats[3]);

    // ─── 4. stage32 (DWS film 2,3) ────────────────────────────────────────
    // -> DWS#2 (320->640) + FiLM#2 -> t32b feats[4]
    apply_dws(dws_film[2], 320, 640, 32, 32, 1, feats[3], feats[4]);
    apply_film(2, 640, 32, feats[4]);
    // -> DWS#3 (640->640) + FiLM#3 -> t32c feats[5]
    apply_dws(dws_film[3], 640, 640, 32, 32, 1, feats[4], feats[5]);
    apply_film(3, 640, 32, feats[5]);

    // ─── 5. downto16 (stride-2 DWS#1) -> t16a feats[6] ────────────────────
    apply_dws(dws_down[1], 640, 640, 32, 32, 2, feats[5], feats[6]);

    // ─── 6. stage16 (DWS film 4,5) ────────────────────────────────────────
    apply_dws(dws_film[4], 640, 1280, 16, 16, 1, feats[6], feats[7]);
    apply_film(4, 1280, 16, feats[7]);
    apply_dws(dws_film[5], 1280, 1280, 16, 16, 1, feats[7], feats[8]);
    apply_film(5, 1280, 16, feats[8]);

    // ─── 7. downto8 (stride-2 DWS#2) -> t8a feats[9] ──────────────────────
    apply_dws(dws_down[2], 1280, 1280, 16, 16, 2, feats[8], feats[9]);

    // ─── 8. xattn8 ────────────────────────────────────────────────────────
    // Treat feats[9] (1,1280,8,8) as a length-64 sequence of 1280-dim tokens.
    // Pre-LN (over D), flash_attention_qkvo cross-attn with ctx (77,768),
    // residual add, then back to NCHW.
    constexpr int Lq = 64;
    constexpr int D  = 1280;
    launch_nchw_to_seq(feats[9], seq_buf, D, Lq);  // seq_buf : (64, 1280)

    // pre-LN (over the D axis): batched-row layernorm
    bt::layernorm_forward_inference_batched_fp16_gpu(seq_buf,
                                                     xattn_ln_g, xattn_ln_b,
                                                     seq_buf, /*eps=*/1e-5f);

    // cross-attention: Q from seq_buf, K/V from ctx; 8 heads, head_dim=160.
    bt::flash_attention_qkvo_forward_gpu(seq_buf, /*Ctx=*/&ctx,
                                         xattn_wq, &xattn_bq,
                                         xattn_wk, &xattn_bk,
                                         xattn_wv, &xattn_bv,
                                         xattn_wo, &xattn_bo,
                                         /*d_mask=*/nullptr,
                                         /*num_heads=*/8,
                                         /*causal=*/false,
                                         attn_out);
    // residual add (seq layout): attn_out += seq_buf
    launch_add_inplace(attn_out, seq_buf);
    launch_seq_to_nchw(attn_out, feats[10], D, Lq);   // t8b -> feats[10]
    // Wait — we still need t8c, t8d. We've reused indices wrong: t8b is the
    // post-xattn tensor, then stage8 produces t8c, t8d which feed into the
    // s10/s11 taps. But feats[9] holds t8a (already, for s9). We need a
    // scratch t8b -> apply DWS#6 to get t8c -> apply DWS#7 to get t8d.
    // feats[10] and feats[11] correspond to t8c, t8d (the tap inputs for s10
    // and s11). t8b is intermediate, not tapped — overwrite via scratch.

    // Use t_main as t8b scratch.
    t_main = std::move(feats[10]);  // t8b

    // ─── 9. stage8 (DWS film 6,7) ─────────────────────────────────────────
    apply_dws(dws_film[6], 1280, 1280, 8, 8, 1, t_main, feats[10]);  // t8c
    apply_film(6, 1280, 8, feats[10]);
    apply_dws(dws_film[7], 1280, 1280, 8, 8, 1, feats[10], feats[11]); // t8d
    apply_film(7, 1280, 8, feats[11]);

    // ─── 10. Tap heads (1x1 conv per skip) ────────────────────────────────
    for (int i = 0; i < 12; ++i) {
        const int C = kTapC[i];
        const int H = kTapH[i];
        bt::conv2d_forward_gpu(feats[i], tap_w[i], &tap_b[i],
                               1, C, H, H,
                               C, 1, 1,
                               1, 1, 0, 0, 1, 1,
                               skips[i]);
    }

    // Bottleneck is a separate buffer identical (semantically) to skips[11].
    bottleneck = skips[11].clone();
}

}  // namespace brodiffusion::inlet
