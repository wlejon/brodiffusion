// Metal mirror of src/fused_resblock.cu.
//
// SD1.5-specific fused resblock forward, transliterated from the CUDA WMMA
// implementation to Metal simdgroup matrices. Folds:
//   * per-channel t_emb shift into conv1's epilogue
//   * residual (or skip) add into conv2's epilogue
//
// Hard-coded for the SD1.5 inference path:
//   N=1, kH=kW=3, pad=1, stride=1, dil=1, FP16 storage, FP32 accumulator.
//
// The implicit-GEMM conv kernel is a Metal copy of brotensor's
// src/metal/conv2d_wmma.mm (BM=BN=64, BK=32, 4 simdgroups in a 2x2 grid,
// FP32 accumulators, im2col-on-the-fly), specialised to N=1 + 3x3 s1 p1 d1,
// with an extended epilogue parameterised by two optional pointers:
//   shift_C : (C_out,) per-channel bias added to every output element
//   skip    : (C_out*H*W,) per-element residual add
//
// Only the FP16 path has a brodiffusion-tuned Metal kernel; INT8 is handled
// elsewhere (brotensor's INT8 resblock op).

#include "brodiffusion/fused_resblock.h"
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

// ─── conv3x3 implicit-GEMM with fused epilogue (N=1, 3x3 s1 p1 d1) ─────────
//
// Tile shape mirrors the CUDA WMMA kernel:
//   BM=64, BN=64, BK=32, 4 simdgroups arranged 2x2, 32 threads per simdgroup
//   -> 128 threads per threadgroup. Each simdgroup owns a 32x32 output region
//   covered by 4x4 simdgroup_matrix<half,8,8> tiles. FP32 accumulator.

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

struct ConvParams {
    int C_in;
    int H;
    int W;
    int C_out;
    uint has_bias;
    uint has_shift;
    uint has_skip;
};

kernel void k_conv3x3_fused_epilogue(
        device const half* X     [[buffer(0)]],
        device const half* Wt    [[buffer(1)]],
        device const half* bias  [[buffer(2)]],
        device const half* shift [[buffer(3)]],
        device const half* skip  [[buffer(4)]],
        device half*       Y     [[buffer(5)]],
        constant ConvParams& p   [[buffer(6)]],
        uint3 tg_pos [[threadgroup_position_in_grid]],
        uint  tid    [[thread_index_in_threadgroup]],
        uint  sg_id  [[simdgroup_index_in_threadgroup]]) {
    const int KH = 3;
    const int KW = 3;
    const int PAD_H = 1;
    const int PAD_W = 1;
    const int KHW = KH * KW;

    const int H_out = p.H;
    const int W_out = p.W;
    const int HW_out  = H_out * W_out;     // N=1, so M == HW_out
    const int K_total = p.C_in * KHW;

    const int block_m = int(tg_pos.y) * BM;   // output-pixel index (oh*W+ow)
    const int block_n = int(tg_pos.x) * BN;   // output channel

    const int warp_m = int(sg_id) / WARPS_N;
    const int warp_n = int(sg_id) % WARPS_N;

    constexpr int LDA = BK + 8;
    constexpr int LDB = BK + 8;
    constexpr int LDC = BN + 8;
    threadgroup half As[BM * LDA];
    threadgroup half Bs[BN * LDB];

    // FP32 accumulator fragments.
    simdgroup_matrix<float, 8, 8> c_frag[FRAGS_M][FRAGS_N];
    for (int i = 0; i < FRAGS_M; ++i) {
        for (int j = 0; j < FRAGS_N; ++j) {
            c_frag[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (int k0 = 0; k0 < K_total; k0 += BK) {
        // ---- A tile (BM x BK) gathered from X on the fly ----
        {
            constexpr int kElemsPerRow = BK;
            constexpr int kElemsTotal  = BM * BK;
            constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_TG;
            for (int li = 0; li < kElemsPerThr; ++li) {
                const int lin = int(tid) + li * THREADS_PER_TG;
                const int row = lin / kElemsPerRow;
                const int col = lin - row * kElemsPerRow;
                const int gk  = k0 + col;
                const int m_g = block_m + row;

                half v = half(0);
                if (m_g < HW_out && gk < K_total) {
                    // N=1, so m_g == oh*W_out + ow directly.
                    const int oh  = m_g / W_out;
                    const int ow  = m_g - oh * W_out;
                    const int ic  = gk / KHW;
                    const int khw = gk - ic * KHW;
                    const int kh  = khw / KW;
                    const int kw  = khw - kh * KW;

                    const int in_h = oh - PAD_H + kh;
                    const int in_w = ow - PAD_W + kw;
                    if (in_h >= 0 && in_h < p.H && in_w >= 0 && in_w < p.W) {
                        v = X[(ic * p.H + in_h) * p.W + in_w];
                    }
                }
                As[row * LDA + col] = v;
            }
        }

        // ---- B tile (BN x BK) copied from Wt[oc, k] ----
        {
            constexpr int kElemsPerRow = BK;
            constexpr int kElemsTotal  = BN * BK;
            constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_TG;
            for (int li = 0; li < kElemsPerThr; ++li) {
                const int lin = int(tid) + li * THREADS_PER_TG;
                const int row = lin / kElemsPerRow;
                const int col = lin - row * kElemsPerRow;
                const int gk  = k0 + col;
                const int oc  = block_n + row;
                half v = half(0);
                if (oc < p.C_out && gk < K_total) {
                    v = Wt[oc * K_total + gk];
                }
                Bs[row * LDB + col] = v;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- simdgroup-matrix compute ----
        for (int kk = 0; kk < FRAGS_K; ++kk) {
            simdgroup_matrix<half, 8, 8> a_frag[FRAGS_M];
            simdgroup_matrix<half, 8, 8> b_frag[FRAGS_N];

            for (int i = 0; i < FRAGS_M; ++i) {
                const int a_row = warp_m * WM + i * 8;
                const int a_col = kk * 8;
                simdgroup_load(a_frag[i],
                               As + a_row * LDA + a_col,
                               LDA,
                               ulong2(0, 0),
                               false);
            }
            // B is stored row-major (BN=oc, BK=k). Load with transpose=true
            // to feed K rows of 8 (k) x 8 (oc).
            for (int j = 0; j < FRAGS_N; ++j) {
                const int b_row = warp_n * WN + j * 8;   // oc
                const int b_col = kk * 8;                 // k
                simdgroup_load(b_frag[j],
                               Bs + b_row * LDB + b_col,
                               LDB,
                               ulong2(0, 0),
                               true);
            }
            for (int i = 0; i < FRAGS_M; ++i) {
                for (int j = 0; j < FRAGS_N; ++j) {
                    simdgroup_multiply_accumulate(c_frag[i][j],
                                                  a_frag[i],
                                                  b_frag[j],
                                                  c_frag[i][j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Epilogue: stage FP32 fragments to threadgroup mem, then fold
    // bias/shift/skip into FP32 and write Y (NCHW layout, N=1). ----
    threadgroup float Cs[BM * LDC];
    for (int i = 0; i < FRAGS_M; ++i) {
        for (int j = 0; j < FRAGS_N; ++j) {
            const int c_row = warp_m * WM + i * 8;
            const int c_col = warp_n * WN + j * 8;
            simdgroup_store(c_frag[i][j],
                            Cs + c_row * LDC + c_col,
                            LDC,
                            ulong2(0, 0),
                            false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        constexpr int kElemsPerCol = BN;
        constexpr int kElemsTotal  = BM * BN;
        constexpr int kElemsPerThr = kElemsTotal / THREADS_PER_TG;
        for (int si = 0; si < kElemsPerThr; ++si) {
            const int lin = int(tid) + si * THREADS_PER_TG;
            const int row = lin / kElemsPerCol;
            const int col = lin - row * kElemsPerCol;

            const int m_g = block_m + row;
            const int oc  = block_n + col;
            if (oc >= p.C_out) continue;
            if (m_g >= HW_out) continue;

            // Mirror the CUDA kernel: the GEMM accumulator is rounded to
            // FP16 before the bias/shift/skip adds are folded in FP32.
            float v = float(half(Cs[row * LDC + col]));
            if (p.has_bias != 0u)  v += float(bias[oc]);
            if (p.has_shift != 0u) v += float(shift[oc]);
            const int y_idx = oc * HW_out + m_g;
            if (p.has_skip != 0u)  v += float(skip[y_idx]);
            Y[y_idx] = half(v);
        }
    }
}

// ─── Standalone fused GroupNorm + SiLU (N=1) ──────────────────────────────
// One threadgroup per group; FP32 threadgroup reduction for mean/variance,
// then normalize + per-channel affine + SiLU. Matches gn_silu_n1_kernel.

constant int GN_BLOCK = 256;

struct GnParams {
    int C;
    int spatial;
    int channels_per_group;
    float eps;
};

kernel void k_gn_silu_n1(
        device const half* X     [[buffer(0)]],
        device const half* gamma [[buffer(1)]],
        device const half* beta  [[buffer(2)]],
        device half*       Y     [[buffer(3)]],
        constant GnParams& p     [[buffer(4)]],
        uint  g   [[threadgroup_position_in_grid]],
        uint  tid [[thread_index_in_threadgroup]],
        uint  ntg [[threads_per_threadgroup]]) {
    const int tile_size = p.channels_per_group * p.spatial;
    const int chan_base = int(g) * p.channels_per_group;
    device const half* x_tile = X + chan_base * p.spatial;
    device half*       y_tile = Y + chan_base * p.spatial;

    float sum = 0.0f, sumsq = 0.0f;
    for (int i = int(tid); i < tile_size; i += int(ntg)) {
        const float v = float(x_tile[i]);
        sum += v;
        sumsq += v * v;
    }
    threadgroup float s_sum[GN_BLOCK];
    threadgroup float s_sumsq[GN_BLOCK];
    s_sum[tid] = sum;
    s_sumsq[tid] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid]   += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float s_mean;
    threadgroup float s_rstd;
    if (tid == 0) {
        const float inv_n = 1.0f / float(tile_size);
        const float mean  = s_sum[0] * inv_n;
        const float var   = s_sumsq[0] * inv_n - mean * mean;
        s_mean = mean;
        s_rstd = rsqrt(var + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = s_mean;
    const float rstd = s_rstd;

    for (int i = int(tid); i < tile_size; i += int(ntg)) {
        const int local_c = i / p.spatial;
        const int channel = chan_base + local_c;
        const float gv = float(gamma[channel]);
        const float bv = float(beta[channel]);
        const float v  = float(x_tile[i]);
        const float yn = (v - mean) * rstd * gv + bv;
        const float silu = yn / (1.0f + exp(-yn));
        y_tile[i] = half(silu);
    }
}
)msl";

// ─── PSO accessors ────────────────────────────────────────────────────────

id<MTLComputePipelineState> pso_conv3x3() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{
        pso = compile_pipeline(kSrc, @"k_conv3x3_fused_epilogue");
    });
    return pso;
}

id<MTLComputePipelineState> pso_gn_silu() {
    static dispatch_once_t once;
    static id<MTLComputePipelineState> pso;
    dispatch_once(&once, ^{
        pso = compile_pipeline(kSrc, @"k_gn_silu_n1");
    });
    return pso;
}

struct ConvParams {
    int32_t C_in;
    int32_t H;
    int32_t W;
    int32_t C_out;
    uint32_t has_bias;
    uint32_t has_shift;
    uint32_t has_skip;
};

struct GnParams {
    int32_t C;
    int32_t spatial;
    int32_t channels_per_group;
    float eps;
};

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int THREADS_PER_TG = 128;
constexpr int GN_BLOCK = 256;

// conv3x3 + bias + optional shift + optional skip -> Y.
void launch_conv3x3_fused(const bt::Tensor& Xt, const bt::Tensor& Wtt,
                          const bt::Tensor& biast,
                          const bt::Tensor* shiftt,
                          const bt::Tensor* skipt,
                          bt::Tensor& Yt,
                          int C_in, int H, int W, int C_out) {
    const int M = H * W;          // N=1

    id<MTLBuffer> bX = buffer_for(Xt);
    id<MTLBuffer> bW = buffer_for(Wtt);
    id<MTLBuffer> bB = buffer_for(biast);
    id<MTLBuffer> bY = buffer_for(Yt);
    const NSUInteger oX = buffer_offset_for(Xt);
    const NSUInteger oW = buffer_offset_for(Wtt);
    const NSUInteger oB = buffer_offset_for(biast);
    const NSUInteger oY = buffer_offset_for(Yt);

    id<MTLBuffer> bShift = shiftt ? buffer_for(*shiftt) : bX;
    NSUInteger    oShift = shiftt ? buffer_offset_for(*shiftt) : oX;
    id<MTLBuffer> bSkip  = skipt ? buffer_for(*skipt) : bX;
    NSUInteger    oSkip  = skipt ? buffer_offset_for(*skipt) : oX;

    ConvParams p{};
    p.C_in = C_in; p.H = H; p.W = W; p.C_out = C_out;
    p.has_bias  = 1u;
    p.has_shift = shiftt ? 1u : 0u;
    p.has_skip  = skipt ? 1u : 0u;

    NSUInteger grid_x = static_cast<NSUInteger>((C_out + BN - 1) / BN);
    NSUInteger grid_y = static_cast<NSUInteger>((M + BM - 1) / BM);

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_conv3x3()];
        [enc setBuffer:bX     offset:oX     atIndex:0];
        [enc setBuffer:bW     offset:oW     atIndex:1];
        [enc setBuffer:bB     offset:oB     atIndex:2];
        [enc setBuffer:bShift offset:oShift atIndex:3];
        [enc setBuffer:bSkip  offset:oSkip  atIndex:4];
        [enc setBuffer:bY     offset:oY     atIndex:5];
        [enc setBytes:&p length:sizeof(ConvParams) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
             threadsPerThreadgroup:MTLSizeMake(THREADS_PER_TG, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

void launch_gn_silu(const bt::Tensor& Xt, const bt::Tensor& gammat,
                    const bt::Tensor& betat, bt::Tensor& Yt,
                    int C, int spatial, int channels_per_group, float eps,
                    int num_groups) {
    id<MTLBuffer> bX = buffer_for(Xt);
    id<MTLBuffer> bG = buffer_for(gammat);
    id<MTLBuffer> bB = buffer_for(betat);
    id<MTLBuffer> bY = buffer_for(Yt);
    const NSUInteger oX = buffer_offset_for(Xt);
    const NSUInteger oG = buffer_offset_for(gammat);
    const NSUInteger oB = buffer_offset_for(betat);
    const NSUInteger oY = buffer_offset_for(Yt);

    GnParams p{};
    p.C = C; p.spatial = spatial;
    p.channels_per_group = channels_per_group; p.eps = eps;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = new_command_buffer();
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_gn_silu()];
        [enc setBuffer:bX offset:oX atIndex:0];
        [enc setBuffer:bG offset:oG atIndex:1];
        [enc setBuffer:bB offset:oB atIndex:2];
        [enc setBuffer:bY offset:oY atIndex:3];
        [enc setBytes:&p length:sizeof(GnParams) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(GN_BLOCK, 1, 1)];
        [enc endEncoding];
        ::brotensor::metal_impl::submit(cmd);
    }
}

} // anonymous namespace

void detail::fused_resblock_forward_metal(
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
    launch_gn_silu(X, gn1_g, gn1_b, h1, C_in, spatial, C_in / num_groups,
                   eps, num_groups);

    // conv1 (3x3 s1 p1) + bias + t_emb_shift -> h2.
    launch_conv3x3_fused(h1, W1, b1, &t_emb_shift, /*skip=*/nullptr,
                         h2, C_in, H, W, C_out);

    // GN2 + SiLU on h2 -> h3.
    launch_gn_silu(h2, gn2_g, gn2_b, h3, C_out, spatial, C_out / num_groups,
                   eps, num_groups);

    // 1x1 skip if needed, via brotensor's conv2d (Metal path).
    const bt::Tensor* skip_ptr = nullptr;
    if (Wskip != nullptr) {
        bt::conv2d_forward(X, *Wskip, bskip,
                           N, C_in, H, W,
                           C_out, 1, 1,
                           /*stride*/1, 1,
                           /*pad*/0, 0,
                           /*dil*/1, 1,
                           skip_buf);
        skip_ptr = &skip_buf;
    } else {
        // skip == X directly (C_in == C_out).
        skip_ptr = &X;
    }

    // conv2 (3x3 s1 p1) + bias + skip -> Y (no shift on conv2).
    launch_conv3x3_fused(h3, W2, b2, /*shift=*/nullptr, skip_ptr,
                         Y, C_out, H, W, C_out);
}

} // namespace brodiffusion
