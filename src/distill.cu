// In-tree distillation trainer for the SD1.5 UNet Inlet.
//
// Mirrors Inlet::forward in src/inlet.cu, caching every intermediate the
// backward needs, then runs a hand-rolled backward through the graph in
// reverse order. FP16 weights with FP32 master + FP32 Adam state.

#include "brodiffusion/distill.h"
#include "brodiffusion/inlet.h"
#include "brodiffusion/safetensors.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::distill {

namespace bt = ::brotensor;
namespace st = ::brodiffusion::safetensors;
using inlet::Inlet;

namespace {

[[noreturn]] void fail(const std::string& m) {
    throw std::runtime_error("distill: " + m);
}

// ─── Shape tables (mirror inlet.cu) ────────────────────────────────────────
struct FilmShape { int C_in, C_out, H; };
constexpr std::array<FilmShape, 8> kFilm = {{
    {320,  320,  64}, {320,  320,  64},
    {320,  640,  32}, {640,  640,  32},
    {640,  1280, 16}, {1280, 1280, 16},
    {1280, 1280, 8},  {1280, 1280, 8},
}};
struct DownShape { int C, H_in; };
constexpr std::array<DownShape, 3> kDown = {{
    {320,  64}, {640,  32}, {1280, 16},
}};
constexpr int kTapC[12] = {320,320,320,320, 640,640,640, 1280,1280,1280,1280,1280};
constexpr int kTapH[12] = { 64, 64, 64, 32,  32, 32, 16,   16,   16,    8,    8,    8};

// ─── Kernels ───────────────────────────────────────────────────────────────

// FiLM forward.
__global__ void film_fwd_k(__half* x, const __half* params, int C, int spatial) {
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

// FiLM backward (writes dx, atomic-adds into dParams FP32).
__global__ void film_bwd_k(const __half* x, const __half* dy, const __half* params,
                           __half* dx, float* dParams, int C, int spatial) {
    const int c   = blockIdx.x;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
    if (c >= C) return;
    const float scale1 = __half2float(params[c]) + 1.0f;
    const __half* xc  = x  + c * spatial;
    const __half* dyc = dy + c * spatial;
    __half*       dxc = dx + c * spatial;
    float ds = 0.0f, dsh = 0.0f;
    for (int i = tid; i < spatial; i += bs) {
        float xv = __half2float(xc[i]);
        float dv = __half2float(dyc[i]);
        ds += xv * dv;
        dsh += dv;
        dxc[i] = __float2half(dv * scale1);
    }
    __shared__ float s_ds[256], s_dsh[256];
    s_ds[tid] = ds; s_dsh[tid] = dsh;
    __syncthreads();
    for (int off = bs / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_ds[tid]  += s_ds[tid + off];
            s_dsh[tid] += s_dsh[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&dParams[c],     s_ds[0]);
        atomicAdd(&dParams[C + c], s_dsh[0]);
    }
}

// NCHW ↔ seq transpose.
__global__ void nchw_to_seq_k(const __half* src, __half* dst, int C, int spatial) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    if (s >= spatial || c >= C) return;
    dst[s * C + c] = src[c * spatial + s];
}
__global__ void seq_to_nchw_k(const __half* src, __half* dst, int C, int spatial) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    if (s >= spatial || c >= C) return;
    dst[c * spatial + s] = src[s * C + c];
}
void nchw_to_seq(const bt::GpuTensor& src, bt::GpuTensor& dst, int C, int spatial) {
    dst.resize(spatial, C, bt::Dtype::FP16);
    nchw_to_seq_k<<<dim3((spatial + 63) / 64, C), dim3(64)>>>(
        reinterpret_cast<const __half*>(src.data_fp16()),
        reinterpret_cast<__half*>(dst.data_fp16()),
        C, spatial);
}
void seq_to_nchw(const bt::GpuTensor& src, bt::GpuTensor& dst, int C, int spatial) {
    dst.resize(1, C * spatial, bt::Dtype::FP16);
    seq_to_nchw_k<<<dim3((spatial + 63) / 64, C), dim3(64)>>>(
        reinterpret_cast<const __half*>(src.data_fp16()),
        reinterpret_cast<__half*>(dst.data_fp16()),
        C, spatial);
}

// In-place FP16 a += b.
__global__ void add_h_k(__half* a, const __half* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}
void add_h_inplace(bt::GpuTensor& a, const bt::GpuTensor& b) {
    int n = a.size();
    add_h_k<<<(n + 255) / 256, 256>>>(
        reinterpret_cast<__half*>(a.data_fp16()),
        reinterpret_cast<const __half*>(b.data_fp16()), n);
}

// Accumulate FP16 grad into FP32 grad.
__global__ void acc_h2f_k(float* g, const __half* dW, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    g[i] += __half2float(dW[i]);
}
void acc_h_into_f(bt::GpuTensor& f, const bt::GpuTensor& h) {
    int n = f.size();
    acc_h2f_k<<<(n + 255) / 256, 256>>>(
        f.data, reinterpret_cast<const __half*>(h.data_fp16()), n);
}

__global__ void add_ff_k(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += src[i];
}
void add_f_into_f(bt::GpuTensor& dst, const bt::GpuTensor& src) {
    int n = dst.size();
    add_ff_k<<<(n + 255) / 256, 256>>>(dst.data, src.data, n);
}

// FP16 → FP32.
__global__ void cast_h2f_k(const __half* s, float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d[i] = __half2float(s[i]);
}
void cast_h_to_f(const bt::GpuTensor& src, bt::GpuTensor& dst) {
    dst.resize(src.rows, src.cols, bt::Dtype::FP32);
    int n = src.size();
    cast_h2f_k<<<(n + 255) / 256, 256>>>(
        reinterpret_cast<const __half*>(src.data_fp16()), dst.data, n);
}

// FP32 → FP16.
__global__ void cast_f2h_k(const float* s, __half* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d[i] = __float2half(s[i]);
}
void cast_f_to_h(const bt::GpuTensor& src, bt::GpuTensor& dst) {
    if (dst.size() != src.size() || dst.dtype != bt::Dtype::FP16) {
        dst.resize(src.rows, src.cols, bt::Dtype::FP16);
    }
    int n = src.size();
    cast_f2h_k<<<(n + 255) / 256, 256>>>(
        src.data, reinterpret_cast<__half*>(dst.data_fp16()), n);
}

// MSE forward + backward fused (returns scalar loss, writes dPred).
__global__ void mse_k(const __half* pred, const __half* tgt, __half* dPred,
                      float* loss_acc, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (i < N) {
        float p = __half2float(pred[i]);
        float t = __half2float(tgt[i]);
        float d = p - t;
        sum = d * d;
        dPred[i] = __float2half((2.0f / static_cast<float>(N)) * d);
    }
    __shared__ float s[256];
    int tid = threadIdx.x;
    s[tid] = sum;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(loss_acc, s[0]);
}

float mse_fwbw(const bt::GpuTensor& pred, const bt::GpuTensor& tgt,
               bt::GpuTensor& dPred, float* d_loss) {
    int N = pred.size();
    if (dPred.size() != N || dPred.dtype != bt::Dtype::FP16) {
        dPred.resize(pred.rows, pred.cols, bt::Dtype::FP16);
    }
    BROTENSOR_CUDA_CHECK(cudaMemsetAsync(d_loss, 0, sizeof(float)));
    mse_k<<<(N + 255) / 256, 256>>>(
        reinterpret_cast<const __half*>(pred.data_fp16()),
        reinterpret_cast<const __half*>(tgt.data_fp16()),
        reinterpret_cast<__half*>(dPred.data_fp16()), d_loss, N);
    float host = 0.0f;
    BROTENSOR_CUDA_CHECK(cudaMemcpy(&host, d_loss, sizeof(float),
                                    cudaMemcpyDeviceToHost));
    return host / static_cast<float>(N);
}

// Batched LayerNorm (FP16, recompute, with xhat/rstd caches).
__global__ void ln_fwd_k(const __half* X, const __half* g, const __half* b,
                         __half* Y, __half* Xhat, float* Rstd,
                         int D, float eps) {
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const __half* xr = X + r * D;
    __half* yr  = Y + r * D;
    __half* xhr = Xhat + r * D;
    __shared__ float s_sum[256], s_sq[256];
    float ss = 0.0f, sq = 0.0f;
    for (int i = tid; i < D; i += bs) {
        float v = __half2float(xr[i]);
        ss += v; sq += v * v;
    }
    s_sum[tid] = ss; s_sq[tid] = sq;
    __syncthreads();
    for (int off = bs / 2; off > 0; off >>= 1) {
        if (tid < off) { s_sum[tid] += s_sum[tid+off]; s_sq[tid] += s_sq[tid+off]; }
        __syncthreads();
    }
    float mean = s_sum[0] / D;
    float var  = s_sq[0] / D - mean * mean;
    float rstd = rsqrtf(var + eps);
    if (tid == 0) Rstd[r] = rstd;
    for (int i = tid; i < D; i += bs) {
        float v = __half2float(xr[i]);
        float xh = (v - mean) * rstd;
        xhr[i] = __float2half(xh);
        float gg = __half2float(g[i]);
        float bb = __half2float(b[i]);
        yr[i] = __float2half(xh * gg + bb);
    }
}
void ln_forward(const bt::GpuTensor& X, const bt::GpuTensor& g,
                const bt::GpuTensor& b, bt::GpuTensor& Y,
                bt::GpuTensor& Xhat, bt::GpuTensor& Rstd, float eps) {
    int R = X.rows, D = X.cols;
    Y.resize(R, D, bt::Dtype::FP16);
    Xhat.resize(R, D, bt::Dtype::FP16);
    Rstd.resize(R, 1, bt::Dtype::FP32);
    ln_fwd_k<<<R, 256>>>(
        reinterpret_cast<const __half*>(X.data_fp16()),
        reinterpret_cast<const __half*>(g.data_fp16()),
        reinterpret_cast<const __half*>(b.data_fp16()),
        reinterpret_cast<__half*>(Y.data_fp16()),
        reinterpret_cast<__half*>(Xhat.data_fp16()),
        Rstd.data, D, eps);
}

__global__ void ln_bwd_k(const __half* dY, const __half* Xhat, const __half* g,
                         const float* Rstd, __half* dX,
                         float* dG, float* dB, int D) {
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const __half* dyr = dY + r * D;
    const __half* xhr = Xhat + r * D;
    __half* dxr = dX + r * D;
    float rstd = Rstd[r];
    __shared__ float s1[256], s2[256];
    float sum_dxh = 0.0f, sum_dxh_xh = 0.0f;
    for (int i = tid; i < D; i += bs) {
        float gg = __half2float(g[i]);
        float dy = __half2float(dyr[i]);
        float xh = __half2float(xhr[i]);
        float dxh = dy * gg;
        sum_dxh += dxh;
        sum_dxh_xh += dxh * xh;
    }
    s1[tid] = sum_dxh; s2[tid] = sum_dxh_xh;
    __syncthreads();
    for (int off = bs / 2; off > 0; off >>= 1) {
        if (tid < off) { s1[tid] += s1[tid+off]; s2[tid] += s2[tid+off]; }
        __syncthreads();
    }
    float m1 = s1[0] / D;
    float m2 = s2[0] / D;
    for (int i = tid; i < D; i += bs) {
        float gg = __half2float(g[i]);
        float dy = __half2float(dyr[i]);
        float xh = __half2float(xhr[i]);
        float dxh = dy * gg;
        float dx_v = (dxh - m1 - xh * m2) * rstd;
        dxr[i] = __float2half(dx_v);
        atomicAdd(&dG[i], dy * xh);
        atomicAdd(&dB[i], dy);
    }
}
void ln_backward(const bt::GpuTensor& dY, const bt::GpuTensor& Xhat,
                 const bt::GpuTensor& g, const bt::GpuTensor& Rstd,
                 bt::GpuTensor& dX, bt::GpuTensor& dG_f, bt::GpuTensor& dB_f) {
    int R = dY.rows, D = dY.cols;
    dX.resize(R, D, bt::Dtype::FP16);
    ln_bwd_k<<<R, 256>>>(
        reinterpret_cast<const __half*>(dY.data_fp16()),
        reinterpret_cast<const __half*>(Xhat.data_fp16()),
        reinterpret_cast<const __half*>(g.data_fp16()),
        Rstd.data,
        reinterpret_cast<__half*>(dX.data_fp16()),
        dG_f.data, dB_f.data, D);
}

void launch_film(bt::GpuTensor& x, const bt::GpuTensor& params, int C, int spatial) {
    film_fwd_k<<<C, 256>>>(
        reinterpret_cast<__half*>(x.data_fp16()),
        reinterpret_cast<const __half*>(params.data_fp16()),
        C, spatial);
}

// ─── Caches ────────────────────────────────────────────────────────────────

struct DwsCache {
    bt::GpuTensor in_clone;   // forward input (kept for dw weight backward)
    bt::GpuTensor dw_pre;     // raw depthwise output (pre-silu)
    bt::GpuTensor dw_post;    // silu(dw_pre); pointwise input
    int H_in, W_in, H_out, W_out, C_in, C_out, stride;
};

struct FilmCache {
    bt::GpuTensor temb_silu;  // (1, 1280)
    bt::GpuTensor proj;       // (1, 2*C)
    bt::GpuTensor pre_film;   // (1, C*H*W) pre-FiLM activation (cloned)
};

struct XAttnCache {
    bt::GpuTensor seq_pre_ln;
    bt::GpuTensor seq_post_ln;
    bt::GpuTensor xhat;
    bt::GpuTensor rstd;
    bt::GpuTensor attn_out;
    bt::GpuTensor post_resid;
    bt::GpuTensor t8b_nchw;    // post-xattn NCHW, fed into DWS#6
};

struct Cache {
    std::array<bt::GpuTensor, 12> feats;
    std::array<DwsCache, 8> dws_film_c;
    std::array<DwsCache, 3> dws_down_c;
    std::array<FilmCache, 8> film_c;
    bt::GpuTensor stem_input;
    XAttnCache xattn;
};

// ─── Param registry ────────────────────────────────────────────────────────

struct ParamRef {
    bt::GpuTensor* param_fp16;
    bt::GpuTensor  param_fp32;
    bt::GpuTensor  grad_fp32;
    bt::GpuTensor  m_fp32, v_fp32;
    std::string    key;
    int rows, cols;
};

// ─── Trainer ────────────────────────────────────────────────────────────────

struct Trainer {
    Inlet net;
    std::vector<ParamRef> params;
    Cache cache;
    int adam_step_n = 0;
    bt::GpuTensor d_temb_silu;   // accumulator for grads to silu(t_emb)
    std::array<bt::GpuTensor, 12> d_skips;
    std::array<bt::GpuTensor, 12> d_feats;
    bt::GpuTensor d_loss_scratch;

    void reg(bt::GpuTensor& p, const std::string& key) {
        ParamRef r;
        r.param_fp16 = &p;
        r.rows = p.rows; r.cols = p.cols;
        r.key = key;
        cast_h_to_f(p, r.param_fp32);
        r.grad_fp32.resize(p.rows, p.cols, bt::Dtype::FP32);
        r.grad_fp32.zero();
        r.m_fp32.resize(p.rows, p.cols, bt::Dtype::FP32); r.m_fp32.zero();
        r.v_fp32.resize(p.rows, p.cols, bt::Dtype::FP32); r.v_fp32.zero();
        params.push_back(std::move(r));
    }

    void build_params() {
        params.clear();
        reg(net.stem_w, "stem.weight");
        reg(net.stem_b, "stem.bias");
        for (int i = 0; i < 8; ++i) {
            std::string p = "dws_film." + std::to_string(i) + ".";
            reg(net.dws_film[i].dw_w, p + "dw_w");
            reg(net.dws_film[i].dw_b, p + "dw_b");
            reg(net.dws_film[i].pw_w, p + "pw_w");
            reg(net.dws_film[i].pw_b, p + "pw_b");
            std::string pf = "film." + std::to_string(i) + ".";
            reg(net.film_w[i], pf + "weight");
            reg(net.film_b[i], pf + "bias");
        }
        for (int i = 0; i < 3; ++i) {
            std::string p = "dws_down." + std::to_string(i) + ".";
            reg(net.dws_down[i].dw_w, p + "dw_w");
            reg(net.dws_down[i].dw_b, p + "dw_b");
            reg(net.dws_down[i].pw_w, p + "pw_w");
            reg(net.dws_down[i].pw_b, p + "pw_b");
        }
        reg(net.xattn_ln_g, "xattn.ln_g");
        reg(net.xattn_ln_b, "xattn.ln_b");
        reg(net.xattn_wq,   "xattn.wq");
        reg(net.xattn_bq,   "xattn.bq");
        reg(net.xattn_wk,   "xattn.wk");
        reg(net.xattn_bk,   "xattn.bk");
        reg(net.xattn_wv,   "xattn.wv");
        reg(net.xattn_bv,   "xattn.bv");
        reg(net.xattn_wo,   "xattn.wo");
        reg(net.xattn_bo,   "xattn.bo");
        for (int i = 0; i < 12; ++i) {
            std::string p = "tap." + std::to_string(i) + ".";
            reg(net.tap_w[i], p + "weight");
            reg(net.tap_b[i], p + "bias");
        }
        d_loss_scratch.resize(1, 1, bt::Dtype::FP32);
    }

    ParamRef* find(bt::GpuTensor& p) {
        for (auto& r : params) if (r.param_fp16 == &p) return &r;
        return nullptr;
    }

    // Accumulate FP16 grad-tensor into the FP32 grad of given param.
    void grad_acc(bt::GpuTensor& param_fp16, const bt::GpuTensor& dH) {
        ParamRef* r = find(param_fp16);
        if (!r) fail("grad_acc: param not found");
        acc_h_into_f(r->grad_fp32, dH);
    }
    void grad_acc_f(bt::GpuTensor& param_fp16, const bt::GpuTensor& dF) {
        ParamRef* r = find(param_fp16);
        if (!r) fail("grad_acc_f: param not found");
        add_f_into_f(r->grad_fp32, dF);
    }

    void zero_grads() { for (auto& r : params) r.grad_fp32.zero(); }

    void random_init(std::uint64_t seed) {
        std::mt19937_64 rng(seed);
        for (auto& r : params) {
            int n = r.rows * r.cols;
            int fan_in = std::max(1, r.cols);
            float scale = std::sqrt(2.0f / static_cast<float>(fan_in));
            bool is_bias = (r.cols == 1);
            bool is_lng  = (r.key == "xattn.ln_g");
            std::vector<uint16_t> host(n);
            std::normal_distribution<float> nd(0.0f, scale);
            for (int i = 0; i < n; ++i) {
                float v = 0.0f;
                if (is_lng) v = 1.0f;
                else if (!is_bias) v = nd(rng);
                host[i] = bt::fp32_to_fp16_bits(v);
            }
            bt::upload_fp16(host.data(), r.rows, r.cols, *r.param_fp16);
            cast_h_to_f(*r.param_fp16, r.param_fp32);
        }
    }

    // ─── DWS forward ──────────────────────────────────────────────────────
    void dws_forward(const Inlet::DWSWeights& w,
                     int C_in, int C_out, int H_in, int W_in, int stride,
                     const bt::GpuTensor& in, bt::GpuTensor& out, DwsCache& c) {
        c.C_in = C_in; c.C_out = C_out;
        c.H_in = H_in; c.W_in = W_in;
        c.H_out = H_in / stride; c.W_out = W_in / stride;
        c.stride = stride;
        c.in_clone = in.clone();
        bt::conv2d_forward_gpu(in, w.dw_w, &w.dw_b,
                               1, C_in, H_in, W_in,
                               C_in, 3, 3, stride, stride, 1, 1, 1, 1, C_in,
                               c.dw_pre);
        bt::silu_forward_gpu(c.dw_pre, c.dw_post);
        bt::conv2d_forward_gpu(c.dw_post, w.pw_w, &w.pw_b,
                               1, C_in, c.H_out, c.W_out,
                               C_out, 1, 1, 1, 1, 0, 0, 1, 1, 1,
                               out);
    }

    // ─── FiLM forward (writes back into x_nchw in place) ──────────────────
    void film_forward(int i, int C, int H, const bt::GpuTensor& t_emb,
                      bt::GpuTensor& x_nchw, FilmCache& c) {
        bt::silu_forward_gpu(t_emb, c.temb_silu);
        bt::linear_forward_batched_fp16_gpu(net.film_w[i], &net.film_b[i],
                                            c.temb_silu, c.proj);
        c.pre_film = x_nchw.clone();
        launch_film(x_nchw, c.proj, C, H * H);
    }

    // ─── Full forward ─────────────────────────────────────────────────────
    void forward(const bt::GpuTensor& sample,
                 const bt::GpuTensor& t_emb,
                 const bt::GpuTensor& ctx) {
        cache.stem_input = sample.clone();
        bt::conv2d_forward_gpu(sample, net.stem_w, &net.stem_b,
                               1, 4, 64, 64, 320, 3, 3,
                               1, 1, 1, 1, 1, 1, cache.feats[0]);

        dws_forward(net.dws_film[0], 320, 320, 64, 64, 1, cache.feats[0],
                    cache.feats[1], cache.dws_film_c[0]);
        film_forward(0, 320, 64, t_emb, cache.feats[1], cache.film_c[0]);
        dws_forward(net.dws_film[1], 320, 320, 64, 64, 1, cache.feats[1],
                    cache.feats[2], cache.dws_film_c[1]);
        film_forward(1, 320, 64, t_emb, cache.feats[2], cache.film_c[1]);

        dws_forward(net.dws_down[0], 320, 320, 64, 64, 2, cache.feats[2],
                    cache.feats[3], cache.dws_down_c[0]);

        dws_forward(net.dws_film[2], 320, 640, 32, 32, 1, cache.feats[3],
                    cache.feats[4], cache.dws_film_c[2]);
        film_forward(2, 640, 32, t_emb, cache.feats[4], cache.film_c[2]);
        dws_forward(net.dws_film[3], 640, 640, 32, 32, 1, cache.feats[4],
                    cache.feats[5], cache.dws_film_c[3]);
        film_forward(3, 640, 32, t_emb, cache.feats[5], cache.film_c[3]);

        dws_forward(net.dws_down[1], 640, 640, 32, 32, 2, cache.feats[5],
                    cache.feats[6], cache.dws_down_c[1]);

        dws_forward(net.dws_film[4], 640, 1280, 16, 16, 1, cache.feats[6],
                    cache.feats[7], cache.dws_film_c[4]);
        film_forward(4, 1280, 16, t_emb, cache.feats[7], cache.film_c[4]);
        dws_forward(net.dws_film[5], 1280, 1280, 16, 16, 1, cache.feats[7],
                    cache.feats[8], cache.dws_film_c[5]);
        film_forward(5, 1280, 16, t_emb, cache.feats[8], cache.film_c[5]);

        dws_forward(net.dws_down[2], 1280, 1280, 16, 16, 2, cache.feats[8],
                    cache.feats[9], cache.dws_down_c[2]);

        // xattn8
        constexpr int Lq = 64, D = 1280;
        nchw_to_seq(cache.feats[9], cache.xattn.seq_pre_ln, D, Lq);
        ln_forward(cache.xattn.seq_pre_ln, net.xattn_ln_g, net.xattn_ln_b,
                   cache.xattn.seq_post_ln, cache.xattn.xhat,
                   cache.xattn.rstd, 1e-5f);
        bt::flash_attention_qkvo_forward_gpu(
            cache.xattn.seq_post_ln, &ctx,
            net.xattn_wq, &net.xattn_bq, net.xattn_wk, &net.xattn_bk,
            net.xattn_wv, &net.xattn_bv, net.xattn_wo, &net.xattn_bo,
            nullptr, 8, false, cache.xattn.attn_out);
        cache.xattn.post_resid = cache.xattn.attn_out.clone();
        add_h_inplace(cache.xattn.post_resid, cache.xattn.seq_post_ln);
        seq_to_nchw(cache.xattn.post_resid, cache.xattn.t8b_nchw, D, Lq);

        // stage8
        dws_forward(net.dws_film[6], 1280, 1280, 8, 8, 1, cache.xattn.t8b_nchw,
                    cache.feats[10], cache.dws_film_c[6]);
        film_forward(6, 1280, 8, t_emb, cache.feats[10], cache.film_c[6]);
        dws_forward(net.dws_film[7], 1280, 1280, 8, 8, 1, cache.feats[10],
                    cache.feats[11], cache.dws_film_c[7]);
        film_forward(7, 1280, 8, t_emb, cache.feats[11], cache.film_c[7]);
    }

    // Tap heads + MSE + per-tap backward. Returns total weighted loss; fills
    // per_tap_loss with unweighted MSEs. Initialises d_feats[i].
    float taps_and_loss(const std::array<const bt::GpuTensor*, 12>& targets,
                        const std::array<float, 12>& weights,
                        std::array<float, 12>& per_tap_loss) {
        std::array<bt::GpuTensor, 12> pred;
        float total = 0.0f;
        for (int i = 0; i < 12; ++i) {
            int C = kTapC[i], H = kTapH[i];
            bt::conv2d_forward_gpu(cache.feats[i], net.tap_w[i], &net.tap_b[i],
                                   1, C, H, H, C, 1, 1,
                                   1, 1, 0, 0, 1, 1, pred[i]);
            per_tap_loss[i] = mse_fwbw(pred[i], *targets[i], d_skips[i],
                                       d_loss_scratch.data);
            if (weights[i] != 1.0f) bt::scale_inplace_gpu(d_skips[i], weights[i]);
            total += weights[i] * per_tap_loss[i];
        }
        // tap backward -> d_feats[i]
        for (int i = 0; i < 12; ++i) {
            int C = kTapC[i], H = kTapH[i];
            bt::conv2d_backward_input_gpu(net.tap_w[i], d_skips[i],
                                          1, C, H, H, C, 1, 1,
                                          1, 1, 0, 0, 1, 1, 1, d_feats[i]);
            bt::GpuTensor dW; dW.resize(C, C, bt::Dtype::FP16); dW.zero();
            bt::GpuTensor dB; dB.resize(C, 1, bt::Dtype::FP16); dB.zero();
            bt::conv2d_backward_weight_gpu(cache.feats[i], d_skips[i],
                                           1, C, H, H, C, 1, 1,
                                           1, 1, 0, 0, 1, 1, 1, dW);
            bt::conv2d_backward_bias_gpu(d_skips[i], 1, C, H, H, dB);
            grad_acc(net.tap_w[i], dW);
            grad_acc(net.tap_b[i], dB);
        }
        return total;
    }

    // FiLM backward in-place: input d holds dy of FiLM output; on return, d
    // holds d(pre-FiLM activation). Accumulates grads on film_w/b and adds to
    // d_temb_silu.
    void film_backward(int i, int C, int H, const FilmCache& fc, bt::GpuTensor& d) {
        int spatial = H * H;
        bt::GpuTensor dParams_f; dParams_f.resize(2 * C, 1, bt::Dtype::FP32);
        dParams_f.zero();
        bt::GpuTensor dx_tmp; dx_tmp.resize(d.rows, d.cols, bt::Dtype::FP16);
        film_bwd_k<<<C, 256>>>(
            reinterpret_cast<const __half*>(fc.pre_film.data_fp16()),
            reinterpret_cast<const __half*>(d.data_fp16()),
            reinterpret_cast<const __half*>(fc.proj.data_fp16()),
            reinterpret_cast<__half*>(dx_tmp.data_fp16()),
            dParams_f.data, C, spatial);
        BROTENSOR_CUDA_CHECK(cudaMemcpy(d.data, dx_tmp.data, d.bytes(),
                                        cudaMemcpyDeviceToDevice));

        // dProj (1, 2*C) = cast(dParams_f). dParams_f is laid out flat as
        // [scale(C) | shift(C)] — same as proj's (1, 2*C) row.
        bt::GpuTensor dProj_h; dProj_h.resize(1, 2 * C, bt::Dtype::FP16);
        {
            int n = 2 * C;
            cast_f2h_k<<<(n + 255) / 256, 256>>>(
                dParams_f.data,
                reinterpret_cast<__half*>(dProj_h.data_fp16()), n);
        }

        bt::GpuTensor dWf, dBf, dX_silu;
        dWf.resize(2 * C, 1280, bt::Dtype::FP16); dWf.zero();
        dBf.resize(2 * C, 1,    bt::Dtype::FP16); dBf.zero();
        bt::linear_backward_batched_gpu(net.film_w[i], fc.temb_silu, dProj_h,
                                        dX_silu, dWf, dBf);
        grad_acc(net.film_w[i], dWf);
        grad_acc(net.film_b[i], dBf);
        add_h_inplace(d_temb_silu, dX_silu);
    }

    // DWS backward: given d_out (grad of pointwise output), compute d_in (grad
    // of forward input) and accumulate grads on dw_w/dw_b/pw_w/pw_b.
    void dws_backward(Inlet::DWSWeights& w, const DwsCache& c,
                      const bt::GpuTensor& d_out, bt::GpuTensor& d_in) {
        // PW backward
        bt::GpuTensor d_dw_post;
        bt::GpuTensor dWpw; dWpw.resize(c.C_out, c.C_in, bt::Dtype::FP16); dWpw.zero();
        bt::GpuTensor dBpw; dBpw.resize(c.C_out, 1,      bt::Dtype::FP16); dBpw.zero();
        bt::conv2d_backward_input_gpu(w.pw_w, d_out,
                                      1, c.C_in, c.H_out, c.W_out,
                                      c.C_out, 1, 1, 1, 1, 0, 0, 1, 1, 1,
                                      d_dw_post);
        bt::conv2d_backward_weight_gpu(c.dw_post, d_out,
                                       1, c.C_in, c.H_out, c.W_out,
                                       c.C_out, 1, 1, 1, 1, 0, 0, 1, 1, 1,
                                       dWpw);
        bt::conv2d_backward_bias_gpu(d_out, 1, c.C_out, c.H_out, c.W_out, dBpw);
        grad_acc(w.pw_w, dWpw);
        grad_acc(w.pw_b, dBpw);

        // SiLU backward
        bt::GpuTensor d_dw_pre;
        bt::silu_backward_gpu(c.dw_pre, d_dw_post, d_dw_pre);

        // DW backward
        bt::GpuTensor dWdw; dWdw.resize(c.C_in, 9, bt::Dtype::FP16); dWdw.zero();
        bt::GpuTensor dBdw; dBdw.resize(c.C_in, 1, bt::Dtype::FP16); dBdw.zero();
        bt::conv2d_backward_input_gpu(w.dw_w, d_dw_pre,
                                      1, c.C_in, c.H_in, c.W_in,
                                      c.C_in, 3, 3,
                                      c.stride, c.stride, 1, 1, 1, 1, c.C_in,
                                      d_in);
        bt::conv2d_backward_weight_gpu(c.in_clone, d_dw_pre,
                                       1, c.C_in, c.H_in, c.W_in,
                                       c.C_in, 3, 3,
                                       c.stride, c.stride, 1, 1, 1, 1, c.C_in,
                                       dWdw);
        bt::conv2d_backward_bias_gpu(d_dw_pre, 1, c.C_in, c.H_out, c.W_out, dBdw);
        grad_acc(w.dw_w, dWdw);
        grad_acc(w.dw_b, dBdw);
    }

    void backward(const bt::GpuTensor& ctx) {
        d_temb_silu.resize(1, 1280, bt::Dtype::FP16);
        d_temb_silu.zero();

        // Stage8 reverse
        film_backward(7, 1280, 8, cache.film_c[7], d_feats[11]);
        bt::GpuTensor d_in7;
        dws_backward(net.dws_film[7], cache.dws_film_c[7], d_feats[11], d_in7);
        add_h_inplace(d_feats[10], d_in7);
        film_backward(6, 1280, 8, cache.film_c[6], d_feats[10]);
        bt::GpuTensor d_t8b;
        dws_backward(net.dws_film[6], cache.dws_film_c[6], d_feats[10], d_t8b);

        // xattn8 reverse
        constexpr int Lq = 64, D = 1280;
        bt::GpuTensor d_post_resid;
        nchw_to_seq(d_t8b, d_post_resid, D, Lq);
        bt::GpuTensor d_attn_out = d_post_resid.clone();
        bt::GpuTensor d_seq_resid = d_post_resid.clone();

        bt::GpuTensor d_seq_attn, d_ctx;
        d_ctx.resize(ctx.rows, ctx.cols, bt::Dtype::FP16); d_ctx.zero();
        bt::GpuTensor dWq, dWk, dWv, dWo, dbq, dbk, dbv, dbo;
        dWq.resize(1280, 1280, bt::Dtype::FP16); dWq.zero();
        dWk.resize(1280, 768,  bt::Dtype::FP16); dWk.zero();
        dWv.resize(1280, 768,  bt::Dtype::FP16); dWv.zero();
        dWo.resize(1280, 1280, bt::Dtype::FP16); dWo.zero();
        dbq.resize(1280, 1,    bt::Dtype::FP16); dbq.zero();
        dbk.resize(1280, 1,    bt::Dtype::FP16); dbk.zero();
        dbv.resize(1280, 1,    bt::Dtype::FP16); dbv.zero();
        dbo.resize(1280, 1,    bt::Dtype::FP16); dbo.zero();
        bt::flash_attention_qkvo_backward_gpu(
            cache.xattn.seq_post_ln, &ctx,
            net.xattn_wq, &net.xattn_bq, net.xattn_wk, &net.xattn_bk,
            net.xattn_wv, &net.xattn_bv, net.xattn_wo, &net.xattn_bo,
            nullptr, 8, false, d_attn_out,
            d_seq_attn, &d_ctx,
            dWq, &dbq, dWk, &dbk, dWv, &dbv, dWo, &dbo);
        grad_acc(net.xattn_wq, dWq); grad_acc(net.xattn_bq, dbq);
        grad_acc(net.xattn_wk, dWk); grad_acc(net.xattn_bk, dbk);
        grad_acc(net.xattn_wv, dWv); grad_acc(net.xattn_bv, dbv);
        grad_acc(net.xattn_wo, dWo); grad_acc(net.xattn_bo, dbo);

        add_h_inplace(d_seq_attn, d_seq_resid);

        bt::GpuTensor d_seq_pre_ln;
        bt::GpuTensor dG_f; dG_f.resize(1280, 1, bt::Dtype::FP32); dG_f.zero();
        bt::GpuTensor dB_f; dB_f.resize(1280, 1, bt::Dtype::FP32); dB_f.zero();
        ln_backward(d_seq_attn, cache.xattn.xhat, net.xattn_ln_g,
                    cache.xattn.rstd, d_seq_pre_ln, dG_f, dB_f);
        grad_acc_f(net.xattn_ln_g, dG_f);
        grad_acc_f(net.xattn_ln_b, dB_f);

        bt::GpuTensor d_f9_xattn;
        seq_to_nchw(d_seq_pre_ln, d_f9_xattn, D, Lq);
        add_h_inplace(d_feats[9], d_f9_xattn);

        // downto8 reverse
        bt::GpuTensor d_in_d2;
        dws_backward(net.dws_down[2], cache.dws_down_c[2], d_feats[9], d_in_d2);
        add_h_inplace(d_feats[8], d_in_d2);

        // stage16 reverse
        film_backward(5, 1280, 16, cache.film_c[5], d_feats[8]);
        bt::GpuTensor d_in5;
        dws_backward(net.dws_film[5], cache.dws_film_c[5], d_feats[8], d_in5);
        add_h_inplace(d_feats[7], d_in5);
        film_backward(4, 1280, 16, cache.film_c[4], d_feats[7]);
        bt::GpuTensor d_in4;
        dws_backward(net.dws_film[4], cache.dws_film_c[4], d_feats[7], d_in4);
        add_h_inplace(d_feats[6], d_in4);

        // downto16 reverse
        bt::GpuTensor d_in_d1;
        dws_backward(net.dws_down[1], cache.dws_down_c[1], d_feats[6], d_in_d1);
        add_h_inplace(d_feats[5], d_in_d1);

        // stage32 reverse
        film_backward(3, 640, 32, cache.film_c[3], d_feats[5]);
        bt::GpuTensor d_in3;
        dws_backward(net.dws_film[3], cache.dws_film_c[3], d_feats[5], d_in3);
        add_h_inplace(d_feats[4], d_in3);
        film_backward(2, 640, 32, cache.film_c[2], d_feats[4]);
        bt::GpuTensor d_in2;
        dws_backward(net.dws_film[2], cache.dws_film_c[2], d_feats[4], d_in2);
        add_h_inplace(d_feats[3], d_in2);

        // downto32 reverse
        bt::GpuTensor d_in_d0;
        dws_backward(net.dws_down[0], cache.dws_down_c[0], d_feats[3], d_in_d0);
        add_h_inplace(d_feats[2], d_in_d0);

        // stage64 reverse
        film_backward(1, 320, 64, cache.film_c[1], d_feats[2]);
        bt::GpuTensor d_in1;
        dws_backward(net.dws_film[1], cache.dws_film_c[1], d_feats[2], d_in1);
        add_h_inplace(d_feats[1], d_in1);
        film_backward(0, 320, 64, cache.film_c[0], d_feats[1]);
        bt::GpuTensor d_in0;
        dws_backward(net.dws_film[0], cache.dws_film_c[0], d_feats[1], d_in0);
        add_h_inplace(d_feats[0], d_in0);

        // stem reverse: feats[0] = conv2d(sample, stem_w, stem_b)
        bt::GpuTensor dWs; dWs.resize(320, 4 * 9, bt::Dtype::FP16); dWs.zero();
        bt::GpuTensor dBs; dBs.resize(320, 1, bt::Dtype::FP16); dBs.zero();
        bt::conv2d_backward_weight_gpu(cache.stem_input, d_feats[0],
                                       1, 4, 64, 64, 320, 3, 3,
                                       1, 1, 1, 1, 1, 1, 1, dWs);
        bt::conv2d_backward_bias_gpu(d_feats[0], 1, 320, 64, 64, dBs);
        grad_acc(net.stem_w, dWs);
        grad_acc(net.stem_b, dBs);
        // d_temb_silu / dCtx discarded (inputs, not parameters).
    }

    void adam_step(const TrainOptions& opts) {
        ++adam_step_n;
        for (auto& r : params) {
            bt::adam_step_gpu(r.param_fp32, r.grad_fp32, r.m_fp32, r.v_fp32,
                              opts.lr, opts.beta1, opts.beta2, opts.eps,
                              adam_step_n);
            cast_f_to_h(r.param_fp32, *r.param_fp16);
        }
    }
};

// ─── Capture file IO ────────────────────────────────────────────────────────

struct Capture {
    bt::GpuTensor sample, t_emb_raw, ctx;
    std::array<bt::GpuTensor, 12> targets;
    bt::GpuTensor t_emb_silu;  // we compute silu(t_emb_raw) once and reuse
};

void load_capture(const std::string& path, Capture& cap) {
    auto f = st::File::open(path);
    auto get = [&](const char* name) -> const st::TensorView& {
        const auto* v = f.find(name);
        if (!v) fail("missing tensor '" + std::string(name) + "' in " + path);
        return *v;
    };
    st::upload_fp16(get("sample"),    1, 4 * 64 * 64,   cap.sample);
    st::upload_fp16(get("t_emb_raw"), 1, 1280,          cap.t_emb_raw);
    st::upload_fp16(get("ctx"),       77, 768,          cap.ctx);
    static const int kC[12] = {320,320,320,320, 640,640,640, 1280,1280,1280,1280,1280};
    static const int kH[12] = { 64, 64, 64, 32,  32, 32, 16,   16,   16,    8,    8,    8};
    for (int i = 0; i < 12; ++i) {
        std::string name = "s" + std::to_string(i);
        st::upload_fp16(get(name.c_str()), 1, kC[i] * kH[i] * kH[i],
                        cap.targets[i]);
    }
}

// ─── Save Inlet weights ─────────────────────────────────────────────────────

void download_h(const bt::GpuTensor& g, std::vector<uint16_t>& host) {
    host.resize(g.size());
    bt::download_fp16(g, host.data());
}

}  // namespace

void save_inlet_safetensors(const Inlet& net, const std::string& path) {
    std::vector<st::WriteEntry> ents;
    std::vector<std::vector<uint16_t>> bufs;
    bufs.reserve(200);
    auto add = [&](const std::string& name, const bt::GpuTensor& g,
                   std::vector<int64_t> shape) {
        bufs.emplace_back();
        download_h(g, bufs.back());
        st::WriteEntry e;
        e.name = name; e.dtype = st::Dtype::F16; e.shape = std::move(shape);
        e.host_data = bufs.back().data();
        e.bytes = bufs.back().size() * sizeof(uint16_t);
        ents.push_back(std::move(e));
    };
    add("stem.weight", net.stem_w, {320, 4, 3, 3});
    add("stem.bias",   net.stem_b, {320});
    for (int i = 0; i < 8; ++i) {
        const auto& s = kFilm[i];
        std::string p = "dws_film." + std::to_string(i) + ".";
        add(p + "dw_w", net.dws_film[i].dw_w, {s.C_in, 1, 3, 3});
        add(p + "dw_b", net.dws_film[i].dw_b, {s.C_in});
        add(p + "pw_w", net.dws_film[i].pw_w, {s.C_out, s.C_in, 1, 1});
        add(p + "pw_b", net.dws_film[i].pw_b, {s.C_out});
        std::string pf = "film." + std::to_string(i) + ".";
        add(pf + "weight", net.film_w[i], {2 * s.C_out, 1280});
        add(pf + "bias",   net.film_b[i], {2 * s.C_out});
    }
    for (int i = 0; i < 3; ++i) {
        const int C = kDown[i].C;
        std::string p = "dws_down." + std::to_string(i) + ".";
        add(p + "dw_w", net.dws_down[i].dw_w, {C, 1, 3, 3});
        add(p + "dw_b", net.dws_down[i].dw_b, {C});
        add(p + "pw_w", net.dws_down[i].pw_w, {C, C, 1, 1});
        add(p + "pw_b", net.dws_down[i].pw_b, {C});
    }
    add("xattn.ln_g", net.xattn_ln_g, {1280});
    add("xattn.ln_b", net.xattn_ln_b, {1280});
    add("xattn.wq",   net.xattn_wq,   {1280, 1280});
    add("xattn.bq",   net.xattn_bq,   {1280});
    add("xattn.wk",   net.xattn_wk,   {1280, 768});
    add("xattn.bk",   net.xattn_bk,   {1280});
    add("xattn.wv",   net.xattn_wv,   {1280, 768});
    add("xattn.bv",   net.xattn_bv,   {1280});
    add("xattn.wo",   net.xattn_wo,   {1280, 1280});
    add("xattn.bo",   net.xattn_bo,   {1280});
    for (int i = 0; i < 12; ++i) {
        const int C = kTapC[i];
        std::string p = "tap." + std::to_string(i) + ".";
        add(p + "weight", net.tap_w[i], {C, C, 1, 1});
        add(p + "bias",   net.tap_b[i], {C});
    }
    st::write_file(path, ents);
}

int run_distill(const std::string& capture_dir,
                const std::string& out_path,
                const std::string& init_path,
                const TrainOptions& opts) {
    bt::cuda_init();

    // Enumerate capture files.
    std::vector<std::string> files;
    {
        std::error_code ec;
        for (auto& e : std::filesystem::directory_iterator(capture_dir, ec)) {
            if (!e.is_regular_file()) continue;
            auto p = e.path();
            if (p.extension() == ".safetensors") files.push_back(p.string());
        }
        if (ec || files.empty()) fail("no .safetensors files in " + capture_dir);
        std::sort(files.begin(), files.end());
    }
    std::fprintf(stderr, "distill: found %zu capture files in %s\n",
                 files.size(), capture_dir.c_str());
    std::fflush(stderr);

    // Build trainer.
    Trainer T;
    T.net.allocate();
    if (!init_path.empty()) {
        auto f = st::File::open(init_path);
        T.net.load_from_safetensors(f, "");
        std::fprintf(stderr, "distill: warm-started from %s\n", init_path.c_str());
    } else {
        T.net.zero_init();
    }
    T.build_params();
    if (init_path.empty()) T.random_init(opts.shuffle_seed ^ 0x9E3779B97F4A7C15ULL);
    std::fprintf(stderr, "distill: %zu trainable tensors\n", T.params.size());
    std::fflush(stderr);

    // Shuffle order (per-epoch).
    std::mt19937_64 rng(opts.shuffle_seed);
    std::vector<int> order(files.size());
    for (size_t i = 0; i < files.size(); ++i) order[i] = static_cast<int>(i);
    auto reshuffle = [&]() { std::shuffle(order.begin(), order.end(), rng); };
    reshuffle();
    size_t cursor = 0;

    Capture cap;
    bt::GpuTensor t_emb;   // silu(t_emb_raw) — Inlet forward expects raw t_emb,
                            // which is what we pass; inlet's film_forward
                            // applies silu internally.
    (void)t_emb;

    for (int step = 1; step <= opts.steps; ++step) {
        if (cursor >= order.size()) { reshuffle(); cursor = 0; }
        const std::string& path = files[order[cursor++]];
        load_capture(path, cap);

        T.zero_grads();
        T.forward(cap.sample, cap.t_emb_raw, cap.ctx);

        std::array<const bt::GpuTensor*, 12> tgts;
        for (int i = 0; i < 12; ++i) tgts[i] = &cap.targets[i];
        std::array<float, 12> per_tap{};
        float loss = T.taps_and_loss(tgts, opts.loss_weights, per_tap);

        T.backward(cap.ctx);
        T.adam_step(opts);

        if (step == 1 || (step % opts.log_every) == 0) {
            std::fprintf(stderr, "[step %d] loss=%.6f  per-tap=[", step, loss);
            for (int i = 0; i < 12; ++i) {
                std::fprintf(stderr, "%s%.4g", i ? "," : "", per_tap[i]);
            }
            std::fprintf(stderr, "]\n");
            std::fflush(stderr);
        }
        if (opts.ckpt_every > 0 && (step % opts.ckpt_every) == 0) {
            std::string p = out_path + ".step" + std::to_string(step);
            save_inlet_safetensors(T.net, p);
            std::fprintf(stderr, "distill: ckpt -> %s\n", p.c_str());
            std::fflush(stderr);
        }
    }
    save_inlet_safetensors(T.net, out_path);
    std::fprintf(stderr, "distill: final weights -> %s\n", out_path.c_str());
    std::fflush(stderr);

    // Verify roundtrip: reopen with the safetensors reader and feed through
    // Inlet::load_from_safetensors(prefix=""). Any missing or mis-shaped key
    // will throw, surfacing here.
    try {
        auto f = st::File::open(out_path);
        Inlet check;
        check.load_from_safetensors(f, "");
        std::fprintf(stderr, "distill: roundtrip OK (%zu tensors)\n", f.size());
        std::fflush(stderr);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "distill: roundtrip FAILED: %s\n", e.what());
        return 1;
    }
    return 0;
}

}  // namespace brodiffusion::distill
