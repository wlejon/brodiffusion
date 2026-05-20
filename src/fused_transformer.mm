// Metal mirror of src/fused_transformer.cu.
//
// SD1.5-specific fused transformer ops, transliterated from the CUDA WMMA
// implementation to Metal simdgroup matrices.
//
// Fusions:
//   1. fused_linear_geglu:
//        Y(B, D) = value * gelu_exact(gate)
//        where T(B, 2D) = X @ W^T + b, value = T[:, :D], gate = T[:, D:].
//      The implicit-GEMM kernel carries two B-tiles per K-step (the value
//      half and gate half of W) with two accumulator-fragment sets, so the
//      (B, 2D) FF1 intermediate never lands in DRAM.
//
//   2. add_inplace_vec: vectorised FP16 elementwise add (Y[i] += X[i]).
//
//   3. add_inplace_row_bias: broadcast a per-column FP16 bias down every row.
//
// Tile shape mirrors the CUDA WMMA kernel: BM=BN=64, BK=32, 4 simdgroups in a
// 2x2 grid, FP32 accumulators. exact GELU = 0.5*x*(1+erf(x*0.70710678)).
//
// Only the FP16 path has a brodiffusion-tuned Metal kernel; INT8 is handled
// elsewhere (brotensor's INT8 linear + exact-GEGLU ops).

#include "brodiffusion/fused_transformer.h"
#include "brodiffusion/detail/fused_backend.h"
#include "brodiffusion/detail/device.h"

#include <brotensor/ops.h>
#include <brotensor/metal_interop.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include <string>

namespace brodiffusion {

namespace bt = ::brotensor;

namespace {

using bt::metal_impl::buffer_for;
using bt::metal_impl::buffer_offset_for;
using bt::metal_impl::compile_pipeline;
using bt::metal_impl::new_command_buffer;

// ─── MSL source ───────────────────────────────────────────────────────────

NSString* const kSrc = @R"msl(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// MSL has no built-in erf; Abramowitz & Stegun 7.1.26 (max abs err ~1.5e-7).
inline float erf_approx(float x) {
    const float a1 =  0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 =  1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 =  1.061405429f;
    const float pc =  0.3275911f;
    float sign_x = (x < 0.0f) ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t  = 1.0f / (1.0f + pc * ax);
    float y  = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
    return sign_x * y;
}

inline float gelu_exact_scalar(float x) {
    // y = 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5f * x * (1.0f + erf_approx(x * 0.70710678118654752440f));
}

// ─── Fused linear + GEGLU (implicit GEMM) ─────────────────────────────────
//
// Tile shape mirrors the CUDA WMMA kernel:
//   BM=64, BN=64, BK=32, 4 simdgroups arranged 2x2, 32 threads per simdgroup
//   -> 128 threads per threadgroup. Each simdgroup owns a 32x32 output region
//   covered by 4x4 simdgroup_matrix<half,8,8> tiles. FP32 accumulator.
//
// Per output-tile (BM rows of X, BN cols in D_out space) we compute TWO
// BM x BN tiles of T: the value half at W-row offset block_n, and the gate
// half at W-row offset block_n + D_out. Two accumulator-fragment sets, two
// B tiles per K-loop iteration.

constant int BM = 64;
constant int BN = 64;
constant int BK = 32;
constant int WARPS_M = 2;
constant int WARPS_N = 2;
constant int WARPS_PER_TG = WARPS_M * WARPS_N;       // 4
constant int THREADS_PER_TG = WARPS_PER_TG * 32;     // 128
constant int WM = BM / WARPS_M;                      // 32
constant int WN = BN / WARPS_N;                      // 32
constant int FRAGS_M = WM / 8;                       // 4
constant int FRAGS_N = WN / 8;                       // 4
constant int FRAGS_K = BK / 8;                       // 4

struct GegluParams {
    int B;
    int D_out;
    int D_in;
    uint has_bias;
};

kernel void k_fused_linear_geglu(
        device const half* X    [[buffer(0)]],
        device const half* W    [[buffer(1)]],
        device const half* bias [[buffer(2)]],
        device half*       Y    [[buffer(3)]],
        constant GegluParams& p [[buffer(4)]],
        uint3 tg_pos [[threadgroup_position_in_grid]],
        uint  tid    [[thread_index_in_threadgroup]],
        uint  sg_id  [[simdgroup_index_in_threadgroup]]) {
    const int two_D = 2 * p.D_out;
    const int K     = p.D_in;

    const int block_m = int(tg_pos.y) * BM;   // batch/row in (B,)
    const int block_n = int(tg_pos.x) * BN;   // output column in (D_out,)

    const int warp_m = int(sg_id) / WARPS_N;
    const int warp_n = int(sg_id) % WARPS_N;

    // Threadgroup memory is tight on Apple GPUs (32 KiB). The K-loop needs
    // As + Bs_v + Bs_g; the epilogue needs a float staging tile + a half copy
    // of the value half. The two phases never overlap, so they share storage
    // via a union: max(15360, 26624) = 26624 bytes.
    constexpr int LDA = BK + 8;
    constexpr int LDB = BK + 8;
    constexpr int LDC = BN + 8;
    struct LoopSmem {
        half As[BM * LDA];
        half Bs_v[BN * LDB];
        half Bs_g[BN * LDB];
    };
    struct EpiSmem {
        float Cs[BM * LDC];      // float staging for one half at a time
        half  Cs_v_h[BM * BN];   // half copy of the value half
    };
    union Smem {
        LoopSmem loop;
        EpiSmem  epi;
    };
    threadgroup Smem smem;
    threadgroup half* As   = smem.loop.As;
    threadgroup half* Bs_v = smem.loop.Bs_v;
    threadgroup half* Bs_g = smem.loop.Bs_g;

    // Two accumulator-fragment sets: value half and gate half.
    simdgroup_matrix<float, 8, 8> c_v[FRAGS_M][FRAGS_N];
    simdgroup_matrix<float, 8, 8> c_g[FRAGS_M][FRAGS_N];
    for (int i = 0; i < FRAGS_M; ++i) {
        for (int j = 0; j < FRAGS_N; ++j) {
            c_v[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
            c_g[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- A tile (BM x BK) from X(B, D_in) ----
        {
            constexpr int kElemsPerRow = BK;
            constexpr int kElemsTotal  = BM * BK;
            constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_TG;
            for (int li = 0; li < kElemsPerThr; ++li) {
                const int lin = int(tid) + li * THREADS_PER_TG;
                const int row = lin / kElemsPerRow;
                const int col = lin - row * kElemsPerRow;
                const int gk  = k0 + col;
                const int gr  = block_m + row;
                half v = half(0);
                if (gr < p.B && gk < K) {
                    v = X[gr * K + gk];
                }
                As[row * LDA + col] = v;
            }
        }

        // ---- Two B tiles (BN x BK) from W: value half at row offset
        //      block_n, gate half at block_n + D_out ----
        {
            constexpr int kElemsPerRow = BK;
            constexpr int kElemsTotal  = BN * BK;
            constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_TG;
            for (int li = 0; li < kElemsPerThr; ++li) {
                const int lin = int(tid) + li * THREADS_PER_TG;
                const int row = lin / kElemsPerRow;
                const int col = lin - row * kElemsPerRow;
                const int gk  = k0 + col;

                const int gr_v = block_n + row;
                half vv = half(0);
                if (gr_v < two_D && gk < K) {
                    vv = W[gr_v * K + gk];
                }
                Bs_v[row * LDB + col] = vv;

                const int gr_g = block_n + p.D_out + row;
                half vg = half(0);
                if (gr_g < two_D && gk < K) {
                    vg = W[gr_g * K + gk];
                }
                Bs_g[row * LDB + col] = vg;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- simdgroup-matrix compute ----
        for (int kk = 0; kk < FRAGS_K; ++kk) {
            simdgroup_matrix<half, 8, 8> a_frag[FRAGS_M];
            simdgroup_matrix<half, 8, 8> bv_frag[FRAGS_N];
            simdgroup_matrix<half, 8, 8> bg_frag[FRAGS_N];

            for (int i = 0; i < FRAGS_M; ++i) {
                const int a_row = warp_m * WM + i * 8;
                const int a_col = kk * 8;
                simdgroup_load(a_frag[i],
                               As + a_row * LDA + a_col,
                               LDA,
                               ulong2(0, 0),
                               false);
            }
            // B stored row-major (BN=oc, BK=k). Load with transpose=true.
            for (int j = 0; j < FRAGS_N; ++j) {
                const int b_row = warp_n * WN + j * 8;   // oc
                const int b_col = kk * 8;                 // k
                simdgroup_load(bv_frag[j],
                               Bs_v + b_row * LDB + b_col,
                               LDB,
                               ulong2(0, 0),
                               true);
                simdgroup_load(bg_frag[j],
                               Bs_g + b_row * LDB + b_col,
                               LDB,
                               ulong2(0, 0),
                               true);
            }
            for (int i = 0; i < FRAGS_M; ++i) {
                for (int j = 0; j < FRAGS_N; ++j) {
                    simdgroup_multiply_accumulate(c_v[i][j],
                                                  a_frag[i],
                                                  bv_frag[j],
                                                  c_v[i][j]);
                    simdgroup_multiply_accumulate(c_g[i][j],
                                                  a_frag[i],
                                                  bg_frag[j],
                                                  c_g[i][j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Epilogue: stage the value half to the shared float tile, copy it
    // down to a half buffer, then stage the gate half and combine. The CUDA
    // kernel downcasts both accumulators to FP16 before the GEGLU math; we
    // mirror that — the value half passes through the half copy and the gate
    // half is read straight from the (downcast) float store. ----
    threadgroup float* Cs     = smem.epi.Cs;
    threadgroup half*  Cs_v_h = smem.epi.Cs_v_h;

    constexpr int kElemsPerRow = BN;
    constexpr int kElemsTotal  = BM * BN;
    constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_TG;

    // Value half: store FP32 frags -> Cs, downcast -> Cs_v_h.
    for (int i = 0; i < FRAGS_M; ++i) {
        for (int j = 0; j < FRAGS_N; ++j) {
            const int c_row = warp_m * WM + i * 8;
            const int c_col = warp_n * WN + j * 8;
            simdgroup_store(c_v[i][j],
                            Cs + c_row * LDC + c_col,
                            LDC,
                            ulong2(0, 0),
                            false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int si = 0; si < kElemsPerThr; ++si) {
        const int lin = int(tid) + si * THREADS_PER_TG;
        const int row = lin / kElemsPerRow;
        const int col = lin - row * kElemsPerRow;
        Cs_v_h[row * BN + col] = half(Cs[row * LDC + col]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Gate half: store FP32 frags -> Cs.
    for (int i = 0; i < FRAGS_M; ++i) {
        for (int j = 0; j < FRAGS_N; ++j) {
            const int c_row = warp_m * WM + i * 8;
            const int c_col = warp_n * WN + j * 8;
            simdgroup_store(c_g[i][j],
                            Cs + c_row * LDC + c_col,
                            LDC,
                            ulong2(0, 0),
                            false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Combine + write Y(B, D_out) with bias + GEGLU fusion.
    for (int si = 0; si < kElemsPerThr; ++si) {
        const int lin = int(tid) + si * THREADS_PER_TG;
        const int row = lin / kElemsPerRow;
        const int col = lin - row * kElemsPerRow;
        const int grow = block_m + row;
        const int gcol = block_n + col;
        if (grow >= p.B) continue;
        if (gcol >= p.D_out) continue;

        float val  = float(Cs_v_h[row * BN + col]);
        float gate = float(half(Cs[row * LDC + col]));
        if (p.has_bias != 0u) {
            val  += float(bias[gcol]);
            gate += float(bias[gcol + p.D_out]);
        }
        const float y = val * gelu_exact_scalar(gate);
        Y[grow * p.D_out + gcol] = half(y);
    }
}

// ─── Vectorised FP16 add_inplace ──────────────────────────────────────────
// One thread per element (Metal half2 SIMD already serves the same purpose;
// numerically identical to the CUDA __hadd2 path).

kernel void k_add_inplace_fp16_vec(device half*       Y [[buffer(0)]],
                                   device const half* X [[buffer(1)]],
                                   constant uint& n     [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    Y[i] = Y[i] + X[i];
}

// ─── Per-column bias broadcast: Y[r,c] += bias[c] ─────────────────────────

struct RowBiasParams {
    uint rows;
    uint cols;
};

kernel void k_add_inplace_row_bias_fp16(device half*       Y    [[buffer(0)]],
                                        device const half* bias [[buffer(1)]],
                                        constant RowBiasParams& p [[buffer(2)]],
                                        uint j [[thread_position_in_grid]]) {
    if (j >= p.cols) return;
    const half b = bias[j];
    for (uint i = 0; i < p.rows; ++i) {
        const uint off = i * p.cols + j;
        Y[off] = Y[off] + b;
    }
}
)msl";

// ─── PSO accessors ────────────────────────────────────────────────────────

id<MTLComputePipelineState> pso_geglu() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{
        pso = compile_pipeline(kSrc, @"k_fused_linear_geglu");
    });
    return pso;
}

id<MTLComputePipelineState> pso_add_vec() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{
        pso = compile_pipeline(kSrc, @"k_add_inplace_fp16_vec");
    });
    return pso;
}

id<MTLComputePipelineState> pso_row_bias() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{
        pso = compile_pipeline(kSrc, @"k_add_inplace_row_bias_fp16");
    });
    return pso;
}

struct GegluParams {
    int32_t B;
    int32_t D_out;
    int32_t D_in;
    uint32_t has_bias;
};

struct RowBiasParams {
    uint32_t rows;
    uint32_t cols;
};

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int THREADS_PER_TG = 128;

} // anonymous namespace

void detail::fused_linear_geglu_metal(const bt::Tensor& X,
                                      const bt::Tensor& W,
                                      const bt::Tensor& b,
                                      bt::Tensor& Y) {
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
    detail::resize_like(Y, B, D_out, bt::Dtype::FP16, X.device);
    if (B == 0 || D_out == 0) return;

    // For SD1.5 FF1 D_in and 2*D_out are always multiples of 8.
    if ((D_in & 7) != 0 || (two_D & 7) != 0) {
        throw std::runtime_error("fused_linear_geglu: D_in and 2*D_out must be multiples of 8");
    }

    GegluParams p{};
    p.B = B; p.D_out = D_out; p.D_in = D_in; p.has_bias = 1u;

    NSUInteger grid_x = static_cast<NSUInteger>((D_out + BN - 1) / BN);
    NSUInteger grid_y = static_cast<NSUInteger>((B + BM - 1) / BM);

    id<MTLBuffer> bX = buffer_for(X);
    id<MTLBuffer> bW = buffer_for(W);
    id<MTLBuffer> bB = buffer_for(b);
    id<MTLBuffer> bY = buffer_for(Y);
    const NSUInteger oX = buffer_offset_for(X);
    const NSUInteger oW = buffer_offset_for(W);
    const NSUInteger oB = buffer_offset_for(b);
    const NSUInteger oY = buffer_offset_for(Y);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_geglu()];
        [enc setBuffer:bX offset:oX atIndex:0];
        [enc setBuffer:bW offset:oW atIndex:1];
        [enc setBuffer:bB offset:oB atIndex:2];
        [enc setBuffer:bY offset:oY atIndex:3];
        [enc setBytes:&p length:sizeof(GegluParams) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
             threadsPerThreadgroup:MTLSizeMake(THREADS_PER_TG, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void detail::add_inplace_vec_metal(bt::Tensor& Y, const bt::Tensor& X) {
    if (Y.dtype != bt::Dtype::FP16 || X.dtype != bt::Dtype::FP16) {
        throw std::runtime_error("add_inplace_vec: both tensors must be FP16");
    }
    if (Y.rows != X.rows || Y.cols != X.cols) {
        throw std::runtime_error("add_inplace_vec: shape mismatch");
    }
    const uint32_t n = static_cast<uint32_t>(Y.size());
    if (n == 0) return;

    id<MTLBuffer> bY = buffer_for(Y);
    id<MTLBuffer> bX = buffer_for(X);
    const NSUInteger oY = buffer_offset_for(Y);
    const NSUInteger oX = buffer_offset_for(X);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        id<MTLComputePipelineState> pso = pso_add_vec();
        [enc setComputePipelineState:pso];
        [enc setBuffer:bY offset:oY atIndex:0];
        [enc setBuffer:bX offset:oX atIndex:1];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:2];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void detail::add_inplace_row_bias_metal(bt::Tensor& Y, const bt::Tensor& bias) {
    if (Y.dtype != bt::Dtype::FP16 || bias.dtype != bt::Dtype::FP16) {
        throw std::runtime_error("add_inplace_row_bias: tensors must be FP16");
    }
    if (static_cast<int>(bias.size()) != Y.cols) {
        throw std::runtime_error("add_inplace_row_bias: bias.size() must equal Y.cols");
    }
    if (Y.cols == 0 || Y.rows == 0) return;

    RowBiasParams p{};
    p.rows = static_cast<uint32_t>(Y.rows);
    p.cols = static_cast<uint32_t>(Y.cols);

    id<MTLBuffer> bY = buffer_for(Y);
    id<MTLBuffer> bB = buffer_for(bias);
    const NSUInteger oY = buffer_offset_for(Y);
    const NSUInteger oB = buffer_offset_for(bias);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        id<MTLComputePipelineState> pso = pso_row_bias();
        [enc setComputePipelineState:pso];
        [enc setBuffer:bY offset:oY atIndex:0];
        [enc setBuffer:bB offset:oB atIndex:1];
        [enc setBytes:&p length:sizeof(RowBiasParams) atIndex:2];
        NSUInteger tg = [pso maxTotalThreadsPerThreadgroup];
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(p.cols, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

} // namespace brodiffusion
