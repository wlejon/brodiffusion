// SD1.5-specific fused resblock forward.
//
// Variant of brotensor's resblock_forward that folds:
//   * per-channel t_emb shift into conv1's epilogue
//   * residual (or skip) add into conv2's epilogue
//
// Hard-coded for the SD1.5 inference path:
//   N=1, kH=kW=3, pad=1, stride=1, dil=1, FP16 storage, FP32 accumulator.
//
// The WMMA implicit-GEMM kernel is structurally a copy of
// brotensor/src/cuda/conv2d_wmma.cu (BM=BN=64, BK=32, 4-warp CTA, FP32
// accumulators, im2col-on-the-fly), specialised to N=1 + 3x3 s1 p1 d1, with
// an extended epilogue parameterised by two optional pointers:
//   shift_C   : (C_out,) per-channel bias added to every output element
//   skip_NCHW : (C_out*H*W,) added to every output element (only when
//               non-null; semantics match resblock's post-conv2 residual add)
//
// We deliberately leave the brotensor kernel alone — brotensor is the
// generic GPU lib and per project policy SD1.5-specific optimisations live
// in brodiffusion.

#include "brodiffusion/fused_resblock.h"
#include "brodiffusion/detail/cuda_check.cuh"
#include "brodiffusion/detail/device.h"

#include <brotensor/ops.h>
#include <brotensor/runtime.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <stdexcept>

namespace brodiffusion {

namespace bt = ::brotensor;

namespace {

using namespace nvcuda;

// ─── WMMA implicit-GEMM conv2d, N=1, 3x3 s1 p1, with extended epilogue ────

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int BM = 64;
static constexpr int BN = 64;
static constexpr int BK = 32;
static constexpr int WARPS_M = 2;
static constexpr int WARPS_N = 2;
static constexpr int WARPS_PER_CTA = WARPS_M * WARPS_N;       // 4
static constexpr int THREADS_PER_CTA = WARPS_PER_CTA * 32;    // 128
static constexpr int WM = BM / WARPS_M;                       // 32
static constexpr int WN = BN / WARPS_N;                       // 32
static constexpr int FRAGS_M = WM / WMMA_M;                   // 2
static constexpr int FRAGS_N = WN / WMMA_N;                   // 2

static constexpr int LDA_SMEM = BK + 8;
static constexpr int LDB_SMEM = BK + 8;

// N=1, KH=KW=3, PAD=1, STRIDE=1, DIL=1, with epilogue.
//   shift : (C_out,) — per-channel scalar added when shift != nullptr.
//   skip  : (C_out, H, W) row-major (== (C_out*H*W,) since N=1) added when
//           skip != nullptr.
__launch_bounds__(THREADS_PER_CTA)
__global__ void conv3x3_fused_epilogue_kernel(
        const __half* __restrict__ X,
        const __half* __restrict__ Wt,
        const __half* __restrict__ bias,
        const __half* __restrict__ shift,
        const __half* __restrict__ skip,
        __half* __restrict__ Y,
        int C_in, int H, int W,
        int C_out) {
    constexpr int KH = 3;
    constexpr int KW = 3;
    constexpr int PAD_H = 1;
    constexpr int PAD_W = 1;
    constexpr int KHW = KH * KW;

    __shared__ __half As[BM][LDA_SMEM];
    __shared__ __half Bs[BN][LDB_SMEM];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / WARPS_N;
    const int warp_n  = warp_id % WARPS_N;

    const int block_m = blockIdx.y * BM;   // output-pixel index (oh*W+ow) since N=1
    const int block_n = blockIdx.x * BN;   // output channel

    const int H_out = H;
    const int W_out = W;
    const int HW_out  = H_out * W_out;
    const int K_total = C_in * KHW;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[FRAGS_M][FRAGS_N];
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    for (int k0 = 0; k0 < K_total; k0 += BK) {
        // ---- Load A tile ----
        {
            constexpr int kElemsPerRow = BK;
            constexpr int kElemsTotal  = BM * BK;
            constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_CTA;

            #pragma unroll
            for (int li = 0; li < kElemsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / kElemsPerRow;
                const int col = lin - row * kElemsPerRow;
                const int gk  = k0 + col;
                const int m_g = block_m + row;

                __half v = __float2half(0.0f);
                if (m_g < HW_out && gk < K_total) {
                    // N=1, so m_g == oh*W_out + ow directly.
                    const int oh    = m_g / W_out;
                    const int ow    = m_g - oh * W_out;
                    const int ic    = gk / KHW;
                    const int khw   = gk - ic * KHW;
                    const int kh    = khw / KW;
                    const int kw    = khw - kh * KW;

                    const int in_h  = oh - PAD_H + kh;
                    const int in_w  = ow - PAD_W + kw;
                    if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                        v = X[(ic * H + in_h) * W + in_w];
                    }
                }
                As[row][col] = v;
            }
        }

        // ---- Load B tile ----
        {
            constexpr int kHalvesPerLoad = 8;
            constexpr int kTotalHalves   = BN * BK;
            constexpr int kLoadsTotal    = kTotalHalves / kHalvesPerLoad;
            constexpr int kLoadsPerThr   = kLoadsTotal / THREADS_PER_CTA;

            const bool k_aligned8 = ((K_total & 7) == 0);

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / (BK / kHalvesPerLoad);
                const int col_grp = lin % (BK / kHalvesPerLoad);
                const int gcol = col_grp * kHalvesPerLoad;
                const int grow = block_n + row;
                const int gk   = k0 + gcol;

                __half tmp[kHalvesPerLoad];
                if (k_aligned8 && grow < C_out && gk + kHalvesPerLoad <= K_total) {
                    const int4* src = reinterpret_cast<const int4*>(&Wt[grow * K_total + gk]);
                    *reinterpret_cast<int4*>(tmp) = *src;
                } else {
                    #pragma unroll
                    for (int q = 0; q < kHalvesPerLoad; ++q) {
                        const int gk_q = gk + q;
                        if (grow < C_out && gk_q < K_total) {
                            tmp[q] = Wt[grow * K_total + gk_q];
                        } else {
                            tmp[q] = __float2half(0.0f);
                        }
                    }
                }
                *reinterpret_cast<int4*>(&Bs[row][gcol]) = *reinterpret_cast<int4*>(tmp);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[FRAGS_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag[FRAGS_N];

            #pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                const __half* a_ptr = &As[warp_m * WM + i * WMMA_M][kk];
                wmma::load_matrix_sync(a_frag[i], a_ptr, LDA_SMEM);
            }
            #pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                const __half* b_ptr = &Bs[warp_n * WN + j * WMMA_N][kk];
                wmma::load_matrix_sync(b_frag[j], b_ptr, LDB_SMEM);
            }
            #pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                #pragma unroll
                for (int j = 0; j < FRAGS_N; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // ---- Epilogue: store via shared mem, fold bias/shift/skip into FP32. ----
    __shared__ __half Cs[BM][BN + 8];
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_h;
            #pragma unroll
            for (int e = 0; e < c_frag[i][j].num_elements; ++e) {
                c_h.x[e] = __float2half(c_frag[i][j].x[e]);
            }
            __half* c_ptr = &Cs[warp_m * WM + i * WMMA_M][warp_n * WN + j * WMMA_N];
            wmma::store_matrix_sync(c_ptr, c_h, BN + 8, wmma::mem_row_major);
        }
    }
    __syncthreads();

    {
        constexpr int kElemsPerCol = BN;
        constexpr int kElemsTotal  = BM * BN;
        constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_CTA;

        #pragma unroll
        for (int si = 0; si < kElemsPerThr; ++si) {
            const int lin = tid + si * THREADS_PER_CTA;
            const int row = lin / kElemsPerCol;
            const int col = lin - row * kElemsPerCol;

            const int m_g = block_m + row;
            const int oc  = block_n + col;
            if (oc >= C_out) continue;
            if (m_g >= HW_out) continue;

            float v = __half2float(Cs[row][col]);
            if (bias)  v += __half2float(bias[oc]);
            if (shift) v += __half2float(shift[oc]);
            const int y_idx = oc * HW_out + m_g;
            if (skip)  v += __half2float(skip[y_idx]);
            Y[y_idx] = __float2half(v);
        }
    }
}

// ─── Standalone fused GN + SiLU (N=1) ────────────────────────────────────
// Same partitioning as brotensor's gn_silu_fused_kernel: one block per
// (n=0, group); FP32 accumulation, FP16 storage.

constexpr int GN_BLOCK = 256;

__global__ void gn_silu_n1_kernel(const __half* __restrict__ X,
                                  const __half* __restrict__ gamma,
                                  const __half* __restrict__ beta,
                                  __half* __restrict__ Y,
                                  int C, int spatial,
                                  int channels_per_group,
                                  float eps) {
    const int g = blockIdx.x;
    const int tid = threadIdx.x;
    const int tile_size = channels_per_group * spatial;
    const int chan_base = g * channels_per_group;
    const __half* x_tile = X + chan_base * spatial;
    __half*       y_tile = Y + chan_base * spatial;

    float sum = 0.0f, sumsq = 0.0f;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        const float v = __half2float(x_tile[i]);
        sum += v;
        sumsq += v * v;
    }
    __shared__ float s_sum[GN_BLOCK];
    __shared__ float s_sumsq[GN_BLOCK];
    s_sum[tid] = sum;
    s_sumsq[tid] = sumsq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid]   += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }
    __shared__ float s_mean, s_rstd;
    if (tid == 0) {
        const float inv_n = 1.0f / static_cast<float>(tile_size);
        const float mean  = s_sum[0] * inv_n;
        const float var   = s_sumsq[0] * inv_n - mean * mean;
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    const float mean = s_mean;
    const float rstd = s_rstd;

    for (int i = tid; i < tile_size; i += blockDim.x) {
        const int local_c = i / spatial;
        const int channel = chan_base + local_c;
        const float gv = __half2float(gamma[channel]);
        const float bv = __half2float(beta[channel]);
        const float v  = __half2float(x_tile[i]);
        const float yn = (v - mean) * rstd * gv + bv;
        const float silu = yn / (1.0f + __expf(-yn));
        y_tile[i] = __float2half(silu);
    }
}

inline void launch_conv3x3_fused(const __half* X, const __half* Wt,
                                 const __half* bias,
                                 const __half* shift,
                                 const __half* skip,
                                 __half* Y,
                                 int C_in, int H, int W, int C_out) {
    const int M = H * W;          // N=1
    dim3 block(THREADS_PER_CTA);
    dim3 grid((C_out + BN - 1) / BN, (M + BM - 1) / BM);
    conv3x3_fused_epilogue_kernel<<<grid, block>>>(
        X, Wt, bias, shift, skip, Y, C_in, H, W, C_out);
    BRODIFFUSION_CUDA_CHECK(cudaGetLastError());
}

} // anonymous namespace


void fused_resblock_forward(
    const bt::Tensor& X,
    const bt::Tensor& gn1_g, const bt::Tensor& gn1_b,
    const bt::Tensor& W1,    const bt::Tensor& b1,
    const bt::Tensor& t_emb_shift,
    const bt::Tensor& gn2_g, const bt::Tensor& gn2_b,
    const bt::Tensor& W2,    const bt::Tensor& b2,
    const bt::Tensor* Wskip, const bt::Tensor* bskip,
    int C_in, int C_out, int H, int W,
    int num_groups, float eps,
    bt::Tensor& Y) {
    if (X.dtype != bt::Dtype::FP16) {
        throw std::runtime_error("fused_resblock_forward: X must be FP16");
    }
    if (num_groups <= 0 || C_in % num_groups != 0 || C_out % num_groups != 0) {
        throw std::runtime_error("fused_resblock_forward: num_groups must divide C_in and C_out");
    }
    if (Wskip == nullptr && C_in != C_out) {
        throw std::runtime_error("fused_resblock_forward: Wskip required when C_in != C_out");
    }
    const int spatial = H * W;
    const int N = 1;
    const int out_cols = C_out * spatial;
    detail::resize_like(Y, N, out_cols, bt::Dtype::FP16, X.device);
    if (spatial == 0) return;

    // Validate t_emb_shift shape: accept (1, C_out), (C_out, 1), or flat C_out.
    if (t_emb_shift.dtype != bt::Dtype::FP16 ||
        static_cast<int>(t_emb_shift.size()) != C_out) {
        throw std::runtime_error("fused_resblock_forward: t_emb_shift must be FP16, length C_out");
    }

    thread_local static bt::Tensor h1;
    thread_local static bt::Tensor h2;
    thread_local static bt::Tensor h3;
    thread_local static bt::Tensor skip_buf;
    detail::resize_like(h1, N, C_in  * spatial, bt::Dtype::FP16, X.device);
    detail::resize_like(h2, N, C_out * spatial, bt::Dtype::FP16, X.device);
    detail::resize_like(h3, N, C_out * spatial, bt::Dtype::FP16, X.device);

    // GN1 + SiLU on X -> h1.
    {
        dim3 grid(num_groups, 1, 1);
        gn_silu_n1_kernel<<<grid, GN_BLOCK>>>(
            reinterpret_cast<const __half*>(X.data),
            reinterpret_cast<const __half*>(gn1_g.data),
            reinterpret_cast<const __half*>(gn1_b.data),
            reinterpret_cast<__half*>(h1.data),
            C_in, spatial, C_in / num_groups, eps);
        BRODIFFUSION_CUDA_CHECK(cudaGetLastError());
    }

    // conv1 (3x3 s1 p1) + bias + t_emb_shift -> h2.
    launch_conv3x3_fused(
        reinterpret_cast<const __half*>(h1.data),
        reinterpret_cast<const __half*>(W1.data),
        reinterpret_cast<const __half*>(b1.data),
        reinterpret_cast<const __half*>(t_emb_shift.data),
        /*skip=*/nullptr,
        reinterpret_cast<__half*>(h2.data),
        C_in, H, W, C_out);

    // GN2 + SiLU on h2 -> h3.
    {
        dim3 grid(num_groups, 1, 1);
        gn_silu_n1_kernel<<<grid, GN_BLOCK>>>(
            reinterpret_cast<const __half*>(h2.data),
            reinterpret_cast<const __half*>(gn2_g.data),
            reinterpret_cast<const __half*>(gn2_b.data),
            reinterpret_cast<__half*>(h3.data),
            C_out, spatial, C_out / num_groups, eps);
        BRODIFFUSION_CUDA_CHECK(cudaGetLastError());
    }

    // 1x1 skip if needed, via brotensor's conv2d (WMMA path).
    const __half* skip_ptr = nullptr;
    if (Wskip != nullptr) {
        bt::conv2d_forward(X, *Wskip, bskip,
                               N, C_in, H, W,
                               C_out, 1, 1,
                               /*stride*/1, 1,
                               /*pad*/0, 0,
                               /*dil*/1, 1,
                               skip_buf);
        skip_ptr = reinterpret_cast<const __half*>(skip_buf.data);
    } else {
        // skip == X directly (C_in == C_out).
        skip_ptr = reinterpret_cast<const __half*>(X.data);
    }

    // conv2 (3x3 s1 p1) + bias + skip -> Y (no shift on conv2).
    launch_conv3x3_fused(
        reinterpret_cast<const __half*>(h3.data),
        reinterpret_cast<const __half*>(W2.data),
        reinterpret_cast<const __half*>(b2.data),
        /*shift=*/nullptr,
        skip_ptr,
        reinterpret_cast<__half*>(Y.data),
        C_out, H, W, C_out);
}

// ─── INT8 (W8A16) overload ────────────────────────────────────────────────
//
// Delegates to brotensor::resblock_forward_int8w_fp16, which mirrors the
// FP16 fusion (GN+SiLU kernels, WMMA INT8 convs, t_emb add, residual fold) in
// one op-layer call. The earlier brodiffusion-side composition lost the
// epilogue-fusion launches and ran ~1.5x slower than FP16; this delegation
// closes that gap.
void fused_resblock_forward(
    const bt::Tensor& X,
    const bt::Tensor& gn1_g, const bt::Tensor& gn1_b,
    const bt::Tensor& W1_int8, const bt::Tensor& W1_scales,
    const bt::Tensor& b1,
    const bt::Tensor& t_emb_shift,
    const bt::Tensor& gn2_g, const bt::Tensor& gn2_b,
    const bt::Tensor& W2_int8, const bt::Tensor& W2_scales,
    const bt::Tensor& b2,
    const bt::Tensor* Wskip_int8,
    const bt::Tensor* Wskip_scales,
    const bt::Tensor* bskip,
    int C_in, int C_out, int H, int W,
    int num_groups, float eps,
    bt::Tensor& Y) {
    bt::resblock_forward_int8w_fp16(
        X,
        gn1_g, gn1_b,
        W1_int8, W1_scales, &b1,
        &t_emb_shift,
        gn2_g, gn2_b,
        W2_int8, W2_scales, &b2,
        Wskip_int8, Wskip_scales, bskip,
        /*N=*/1, C_in, C_out, H, W,
        num_groups, eps,
        Y);
}

} // namespace brodiffusion
