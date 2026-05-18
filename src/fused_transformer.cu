// SD1.5-specific fused transformer ops.
//
// Fusions:
//   1. fused_linear_geglu:
//        Y(B, D) = a * gelu_exact(g)
//        where T(B, 2D) = X @ W^T + b, a = T[:, :D], g = T[:, D:].
//      Replaces (linear_forward_batched_fp16_gpu + geglu_exact_forward_gpu)
//      i.e. skips materialising the (B, 2D) FF1 intermediate (~10 MB at the
//      bottom level), which is a real memory pass on the 4090.
//
//   2. add_inplace_fp16_vec:
//        Vectorised FP16 elementwise add. brotensor's add_inplace_gpu FP16
//        path goes through __half2float per element; we use __half2 adds and
//        int4 vector loads, which matters because the transformer block does
//        3 of these per call (self-attn residual, cross-attn residual, FF
//        residual), 16 transformer blocks per step.
//
// Both ops assume FP16 storage, FP32 accumulators where relevant, and the
// SD1.5 inference shape constraints documented in the public header.

#include "brodiffusion/fused_transformer.h"

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

// ─── WMMA fused matmul + GEGLU ────────────────────────────────────────────
//
// Structurally a copy of brotensor::fp16_internal::matmul_ABT_wmma_kernel
// (BM=BN=64, BK=32, 4-warp CTA, FP32 accumulators), with two changes:
//
//   * Per output-tile (M=BM rows, N_out=BN cols in D_out space) we compute
//     TWO BM×BN tiles of T: the value half at W-row offset (block_n + 0)
//     and the gate half at W-row offset (block_n + D_out). We carry two
//     accumulator-fragment sets and load two B tiles per K-loop iteration.
//
//   * Epilogue applies bias to each half, then computes
//       Y[m, n] = val * 0.5 * gate * (1 + erff(gate * 0.7071067812))
//     and writes Y(B, D_out) (no 2D intermediate ever lands in DRAM).

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

__device__ __forceinline__ float gelu_exact_scalar(float x) {
    // y = 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

__launch_bounds__(THREADS_PER_CTA)
__global__ void fused_linear_geglu_kernel(const __half* __restrict__ X,
                                          const __half* __restrict__ W,
                                          const __half* __restrict__ bias,
                                          __half* __restrict__ Y,
                                          int B, int D_out, int D_in) {
    // Shared A tile (BM rows of X), two shared B tiles (value/gate halves of W).
    __shared__ __half As[BM][LDA_SMEM];
    __shared__ __half Bs_v[BN][LDB_SMEM];
    __shared__ __half Bs_g[BN][LDB_SMEM];

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / WARPS_N;
    const int warp_n  = warp_id % WARPS_N;

    const int block_m   = blockIdx.y * BM;   // batch/row in (B,)
    const int block_n   = blockIdx.x * BN;   // output column in (D_out,)
    const int two_D     = 2 * D_out;
    const int K         = D_in;

    // Two accumulator-fragment sets: value half and gate half.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_v[FRAGS_M][FRAGS_N];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_g[FRAGS_M][FRAGS_N];
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::fill_fragment(c_v[i][j], 0.0f);
            wmma::fill_fragment(c_g[i][j], 0.0f);
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- Load A tile from X(B, D_in) ----
        {
            constexpr int kHalvesPerLoad = 8;
            constexpr int kTotalHalves   = BM * BK;
            constexpr int kLoadsTotal    = kTotalHalves / kHalvesPerLoad;
            constexpr int kLoadsPerThr   = kLoadsTotal / THREADS_PER_CTA;

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / (BK / kHalvesPerLoad);
                const int col_grp = lin % (BK / kHalvesPerLoad);
                const int gcol = col_grp * kHalvesPerLoad;
                const int grow = block_m + row;
                const int gk   = k0 + gcol;

                __half tmp[kHalvesPerLoad];
                if (grow < B && gk + kHalvesPerLoad <= K) {
                    const int4* src = reinterpret_cast<const int4*>(&X[grow * K + gk]);
                    *reinterpret_cast<int4*>(tmp) = *src;
                } else {
                    #pragma unroll
                    for (int q = 0; q < kHalvesPerLoad; ++q) {
                        const int gk_q = gk + q;
                        if (grow < B && gk_q < K) tmp[q] = X[grow * K + gk_q];
                        else tmp[q] = __float2half(0.0f);
                    }
                }
                *reinterpret_cast<int4*>(&As[row][gcol]) = *reinterpret_cast<int4*>(tmp);
            }
        }

        // Helper to load a B tile (BN rows of W, starting at row b0).
        auto load_B = [&](__half Bs[BN][LDB_SMEM], int b0) {
            constexpr int kHalvesPerLoad = 8;
            constexpr int kTotalHalves   = BN * BK;
            constexpr int kLoadsTotal    = kTotalHalves / kHalvesPerLoad;
            constexpr int kLoadsPerThr   = kLoadsTotal / THREADS_PER_CTA;

            #pragma unroll
            for (int li = 0; li < kLoadsPerThr; ++li) {
                const int lin = tid + li * THREADS_PER_CTA;
                const int row = lin / (BK / kHalvesPerLoad);
                const int col_grp = lin % (BK / kHalvesPerLoad);
                const int gcol = col_grp * kHalvesPerLoad;
                const int grow = b0 + row;
                const int gk   = k0 + gcol;

                __half tmp[kHalvesPerLoad];
                if (grow < two_D && gk + kHalvesPerLoad <= K) {
                    const int4* src = reinterpret_cast<const int4*>(&W[grow * K + gk]);
                    *reinterpret_cast<int4*>(tmp) = *src;
                } else {
                    #pragma unroll
                    for (int q = 0; q < kHalvesPerLoad; ++q) {
                        const int gk_q = gk + q;
                        if (grow < two_D && gk_q < K) tmp[q] = W[grow * K + gk_q];
                        else tmp[q] = __float2half(0.0f);
                    }
                }
                *reinterpret_cast<int4*>(&Bs[row][gcol]) = *reinterpret_cast<int4*>(tmp);
            }
        };

        // ---- Load value-half B tile: W rows [block_n, block_n+BN) ----
        load_B(Bs_v, block_n);
        // ---- Load gate-half B tile:  W rows [block_n + D_out, +BN) ----
        load_B(Bs_g, block_n + D_out);

        __syncthreads();

        // ---- Compute on shared mem tiles ----
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag[FRAGS_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> bv_frag[FRAGS_N];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> bg_frag[FRAGS_N];

            #pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                const __half* a_ptr = &As[warp_m * WM + i * WMMA_M][kk];
                wmma::load_matrix_sync(a_frag[i], a_ptr, LDA_SMEM);
            }
            #pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                const __half* bv_ptr = &Bs_v[warp_n * WN + j * WMMA_N][kk];
                const __half* bg_ptr = &Bs_g[warp_n * WN + j * WMMA_N][kk];
                wmma::load_matrix_sync(bv_frag[j], bv_ptr, LDB_SMEM);
                wmma::load_matrix_sync(bg_frag[j], bg_ptr, LDB_SMEM);
            }
            #pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                #pragma unroll
                for (int j = 0; j < FRAGS_N; ++j) {
                    wmma::mma_sync(c_v[i][j], a_frag[i], bv_frag[j], c_v[i][j]);
                    wmma::mma_sync(c_g[i][j], a_frag[i], bg_frag[j], c_g[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // ---- Epilogue: stage value/gate accumulators in shared as FP16, then
    // apply bias + GEGLU and write Y(B, D_out). We reuse the two B-tile
    // shared buffers (Bs_v / Bs_g) as staging since they fit BM×BN halves
    // (BN+8 stride). They are BN rows × (BK+8) halves; we need BM rows ×
    // (BN+8) halves. BN=64 > BM=64, so Bs_v[BN][BK+8] has 64*40 = 2560
    // halves total — not enough to hold a 64*72 = 4608 staging tile.
    // Use a dedicated staging tile (same as resblock kernel).
    __shared__ __half Cs_v[BM][BN + 8];
    __shared__ __half Cs_g[BM][BN + 8];

    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> v_h, g_h;
            #pragma unroll
            for (int e = 0; e < c_v[i][j].num_elements; ++e) {
                v_h.x[e] = __float2half(c_v[i][j].x[e]);
                g_h.x[e] = __float2half(c_g[i][j].x[e]);
            }
            __half* v_ptr = &Cs_v[warp_m * WM + i * WMMA_M][warp_n * WN + j * WMMA_N];
            __half* g_ptr = &Cs_g[warp_m * WM + i * WMMA_M][warp_n * WN + j * WMMA_N];
            wmma::store_matrix_sync(v_ptr, v_h, BN + 8, wmma::mem_row_major);
            wmma::store_matrix_sync(g_ptr, g_h, BN + 8, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Write Y(B, D_out) with bias + GEGLU fusion.
    {
        constexpr int kElemsPerRow = BN;
        constexpr int kElemsTotal  = BM * BN;
        constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_CTA;

        #pragma unroll
        for (int si = 0; si < kElemsPerThr; ++si) {
            const int lin = tid + si * THREADS_PER_CTA;
            const int row = lin / kElemsPerRow;
            const int col = lin - row * kElemsPerRow;
            const int grow = block_m + row;
            const int gcol = block_n + col;
            if (grow >= B) continue;
            if (gcol >= D_out) continue;

            float val  = __half2float(Cs_v[row][col]);
            float gate = __half2float(Cs_g[row][col]);
            if (bias) {
                val  += __half2float(bias[gcol]);
                gate += __half2float(bias[gcol + D_out]);
            }
            const float y = val * gelu_exact_scalar(gate);
            Y[grow * D_out + gcol] = __float2half(y);
        }
    }
}

// ─── Vectorised FP16 add_inplace ──────────────────────────────────────────
// One int4 (8 halves) per thread when alignment/length permit.

__global__ void add_inplace_fp16_vec_kernel(__half* __restrict__ Y,
                                            const __half* __restrict__ X,
                                            int n) {
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int n_vec  = n / 8;

    // Bulk: 8 halves per thread via int4 + __half2 adds.
    for (int i = tid; i < n_vec; i += stride) {
        int4 yv = reinterpret_cast<int4*>(Y)[i];
        int4 xv = reinterpret_cast<const int4*>(X)[i];
        __half2* y2 = reinterpret_cast<__half2*>(&yv);
        __half2* x2 = reinterpret_cast<__half2*>(&xv);
        y2[0] = __hadd2(y2[0], x2[0]);
        y2[1] = __hadd2(y2[1], x2[1]);
        y2[2] = __hadd2(y2[2], x2[2]);
        y2[3] = __hadd2(y2[3], x2[3]);
        reinterpret_cast<int4*>(Y)[i] = yv;
    }
    // Tail.
    const int tail_base = n_vec * 8;
    for (int i = tail_base + tid; i < n; i += stride) {
        Y[i] = __hadd(Y[i], X[i]);
    }
}

} // anonymous namespace


void fused_linear_geglu(const bt::GpuTensor& X,
                        const bt::GpuTensor& W,
                        const bt::GpuTensor& b,
                        bt::GpuTensor& Y) {
    if (X.dtype != bt::Dtype::FP16 || W.dtype != bt::Dtype::FP16 ||
        b.dtype != bt::Dtype::FP16) {
        throw std::runtime_error("fused_linear_geglu: all inputs must be FP16");
    }
    const int B     = X.rows;
    const int D_in  = X.cols;
    const int two_D = W.rows;
    if (W.cols != D_in) {
        throw std::runtime_error("fused_linear_geglu: W.cols != X.cols");
    }
    if ((two_D & 1) != 0) {
        throw std::runtime_error("fused_linear_geglu: W.rows must be even (2*D_out)");
    }
    if (static_cast<int>(b.size()) != two_D) {
        throw std::runtime_error("fused_linear_geglu: bias length must equal W.rows");
    }
    const int D_out = two_D / 2;
    if (Y.rows != B || Y.cols != D_out || Y.dtype != bt::Dtype::FP16) {
        Y.resize(B, D_out, bt::Dtype::FP16);
    }
    if (B == 0 || D_out == 0) return;

    // The vectorised WMMA path needs K and N_W (=2*D_out) multiples of 8.
    // For SD1.5 FF1 these always hold (D_in ∈ {320,640,1280}; 2*D_out same).
    if ((D_in & 7) != 0 || (two_D & 7) != 0) {
        throw std::runtime_error("fused_linear_geglu: D_in and 2*D_out must be multiples of 8");
    }

    dim3 block(THREADS_PER_CTA);
    dim3 grid((D_out + BN - 1) / BN, (B + BM - 1) / BM);
    fused_linear_geglu_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(X.data_fp16()),
        reinterpret_cast<const __half*>(W.data_fp16()),
        reinterpret_cast<const __half*>(b.data_fp16()),
        reinterpret_cast<__half*>(Y.data_fp16()),
        B, D_out, D_in);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

void fused_linear_geglu(const bt::GpuTensor& X,
                        const bt::GpuTensor& W_int8,
                        const bt::GpuTensor& W_scales,
                        const bt::GpuTensor& b,
                        bt::GpuTensor& Y) {
    if (X.dtype != bt::Dtype::FP16 || b.dtype != bt::Dtype::FP16) {
        throw std::runtime_error("fused_linear_geglu(int8w): X and b must be FP16");
    }
    if (W_int8.dtype != bt::Dtype::INT8) {
        throw std::runtime_error("fused_linear_geglu(int8w): W_int8 must be INT8");
    }
    if (W_scales.dtype != bt::Dtype::FP32) {
        throw std::runtime_error("fused_linear_geglu(int8w): W_scales must be FP32");
    }
    const int B     = X.rows;
    const int D_in  = X.cols;
    const int two_D = W_int8.rows;
    if (W_int8.cols != D_in) {
        throw std::runtime_error("fused_linear_geglu(int8w): W.cols != X.cols");
    }
    if ((two_D & 1) != 0) {
        throw std::runtime_error("fused_linear_geglu(int8w): W.rows must be even");
    }
    if (static_cast<int>(b.size()) != two_D) {
        throw std::runtime_error("fused_linear_geglu(int8w): bias length != W.rows");
    }
    const int D_out = two_D / 2;
    if (Y.rows != B || Y.cols != D_out || Y.dtype != bt::Dtype::FP16) {
        Y.resize(B, D_out, bt::Dtype::FP16);
    }
    if (B == 0 || D_out == 0) return;

    // (B, in) @ dequant(W_int8)^T + b -> tmp (B, 2*D_out) FP16.
    thread_local static bt::GpuTensor ff1_tmp;
    bt::linear_forward_batched_int8w_fp16_gpu(W_int8, W_scales, &b, X, ff1_tmp);
    // GEGLU (exact GELU, matches SD1.5 / diffusers default).
    bt::geglu_exact_forward_gpu(ff1_tmp, Y);
}

void add_inplace_fp16_vec(bt::GpuTensor& Y, const bt::GpuTensor& X) {
    if (Y.dtype != bt::Dtype::FP16 || X.dtype != bt::Dtype::FP16) {
        throw std::runtime_error("add_inplace_fp16_vec: both tensors must be FP16");
    }
    if (Y.rows != X.rows || Y.cols != X.cols) {
        throw std::runtime_error("add_inplace_fp16_vec: shape mismatch");
    }
    const int n = Y.size();
    if (n == 0) return;
    // Need 16-byte alignment for the int4 path. GpuTensor allocations from
    // cudaMalloc are 256-byte aligned, so this always holds.
    constexpr int kThreads = 256;
    const int n_vec = n / 8;
    const int work = (n_vec > 0) ? n_vec : n;
    int blocks = (work + kThreads - 1) / kThreads;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;
    add_inplace_fp16_vec_kernel<<<blocks, kThreads>>>(
        reinterpret_cast<__half*>(Y.data_fp16()),
        reinterpret_cast<const __half*>(X.data_fp16()), n);
    BROTENSOR_CUDA_CHECK(cudaGetLastError());
}

} // namespace brodiffusion
