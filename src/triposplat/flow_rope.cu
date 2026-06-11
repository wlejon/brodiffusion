// TripoSplat flow DiT — content-conditioned axial RoPE table build, on device.
//
// FlowDiT::build_rope predicts a per-(token, head) 3-axis position `delta_pos`
// from the hidden state, then turns it into (L*num_heads, half) cos/sin tables:
// the half-vector is split into three axis groups, and entry j is
// cos/sin(delta_pos[axis(j)] * freq[j] * pi). The reference does this in FP32.
//
// On the CPU backend that is a cheap host loop, but on a GPU backend the host
// version (in flow_model.cpp) has to sync the stream, download delta_pos, run
// ~L*H*half trig calls on the CPU, and upload two L*H*half tables — every block,
// ~28 blocks per forward. This kernel does the whole build in one launch with no
// host round-trip and no stream sync, so the per-block stall disappears.

#include "brodiffusion/detail/flow_rope.h"
#include "brodiffusion/detail/cuda_check.cuh"
#include "brodiffusion/detail/device.h"

#include "brotensor/tensor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

// brotensor's CUDA-internal current-stream hook (src/cuda/runtime.cu). The
// launch below must target it so the kernel lands in the right place during
// CUDA-graph capture (a default-stream launch there escapes the graph).
namespace brotensor { void* cuda_current_stream(); }

namespace brodiffusion {

namespace bt = ::brotensor;

namespace {

__device__ inline float ld(const float* p, int i)  { return p[i]; }
__device__ inline float ld(const __half* p, int i) { return __half2float(p[i]); }

// One thread per (row, j) output element, row in [0, L*num_heads), j in [0,half).
// delta_pos is (L, 3*num_heads) == (L*num_heads, 3) contiguous, so row r's three
// axis values are dp[r*3 + {0,1,2}]. axis(j) follows the f0/f1 group boundaries.
// Inputs carry the pipeline compute dtype (FP16 on a GPU backend); the angle and
// the cos/sin tables are FP32 (rope_apply_perhead requires FP32 tables).
template <typename T>
__global__ void flow_rope_tables_kernel(const T* __restrict__ dp,
                                        const T* __restrict__ freqs_pi,
                                        int rows, int half, int f0, int f1,
                                        float* __restrict__ cos_out,
                                        float* __restrict__ sin_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * half;
    if (idx >= total) return;
    const int j   = idx % half;
    const int row = idx / half;
    const int axis = (j < f0) ? 0 : (j < f0 + f1) ? 1 : 2;
    const float a = ld(dp, row * 3 + axis) * ld(freqs_pi, j);
    cos_out[idx] = cosf(a);   // precise (matches the FP32 host reference)
    sin_out[idx] = sinf(a);
}

template <typename T>
void launch(const bt::Tensor& dp, const bt::Tensor& freqs_pi,
            int rows, int half, int f0, int f1,
            bt::Tensor& cos_out, bt::Tensor& sin_out) {
    const int total = rows * half;
    constexpr int kThreads = 256;
    int blocks = (total + kThreads - 1) / kThreads;
    if (blocks < 1) blocks = 1;
    flow_rope_tables_kernel<T><<<blocks, kThreads, 0,
        reinterpret_cast<cudaStream_t>(::brotensor::cuda_current_stream())>>>(
        static_cast<const T*>(dp.data),
        static_cast<const T*>(freqs_pi.data),
        rows, half, f0, f1,
        static_cast<float*>(cos_out.data),
        static_cast<float*>(sin_out.data));
    BRODIFFUSION_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void detail::flow_rope_tables_cuda(const bt::Tensor& delta_pos,
                                   const bt::Tensor& freqs_pi,
                                   int L, int num_heads, int half, int f0, int f1,
                                   bt::Tensor& cos_out, bt::Tensor& sin_out) {
    if (delta_pos.dtype != freqs_pi.dtype) {
        throw std::runtime_error("flow_rope_tables_cuda: delta_pos/freqs_pi dtype mismatch");
    }
    const int rows = L * num_heads;
    if (rows <= 0 || half <= 0) return;
    // Tables are FP32 regardless of the input (compute) dtype. resize_like
    // (rather than a fresh empty_on) keeps the table pointers stable across
    // calls — required when the caller's step is CUDA-graph captured.
    detail::resize_like(cos_out, rows, half, bt::Dtype::FP32, delta_pos.device);
    detail::resize_like(sin_out, rows, half, bt::Dtype::FP32, delta_pos.device);

    switch (delta_pos.dtype) {
        case bt::Dtype::FP32:
            launch<float>(delta_pos, freqs_pi, rows, half, f0, f1, cos_out, sin_out);
            break;
        case bt::Dtype::FP16:
            launch<__half>(delta_pos, freqs_pi, rows, half, f0, f1, cos_out, sin_out);
            break;
        default:
            throw std::runtime_error("flow_rope_tables_cuda: delta_pos must be FP32 or FP16");
    }
}

}  // namespace brodiffusion
