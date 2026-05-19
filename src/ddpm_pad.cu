// Asymmetric zero-pad helper for the DDPM UNet's stride-2 downsamplers.
//
// Pre-pads an NCHW FP16 activation by one row at the bottom and one column
// at the right (the (0, 1, 0, 1) PyTorch pad). The diffusers `Downsample2D`
// path that uses `padding=0` does exactly this before the stride-2 conv, so
// reproducing it bit-for-bit is necessary for the trained CIFAR weights to
// produce the right output.
//
// Why a custom kernel and not brotensor: brotensor's conv2d takes symmetric
// `(pad_h, pad_w)`; there's no asymmetric-pad path, and the only alternative
// (download → host pad → upload) would be much slower per step. The same
// shape gymnastics arise in other diffusers checkpoints, so this lives as a
// generic helper.

#include "brodiffusion/ddpm_unet.h"
#include "brotensor/tensor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace brodiffusion {

namespace {

__global__ void pad_right_bottom_zero_fp16_kernel(
    const __half* __restrict__ in,
    __half* __restrict__ out,
    int C, int H, int W) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z;
    const int Wo = W + 1;
    const int Ho = H + 1;
    if (w >= Wo || h >= Ho || c >= C) return;
    const int out_idx = (c * Ho + h) * Wo + w;
    if (h < H && w < W) {
        out[out_idx] = in[(c * H + h) * W + w];
    } else {
        out[out_idx] = __float2half(0.0f);
    }
}

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("pad_right_bottom_zero_fp16: " + msg);
}

}  // namespace

void pad_right_bottom_zero_fp16(const brotensor::GpuTensor& in,
                                int N, int C, int H, int W,
                                brotensor::GpuTensor& out) {
    if (N != 1) fail("only N=1 supported");
    if (in.dtype != brotensor::Dtype::FP16) fail("in must be FP16");
    if (in.size() != C * H * W) fail("in shape mismatch");

    const int Ho = H + 1;
    const int Wo = W + 1;
    if (out.rows != 1 || out.cols != C * Ho * Wo ||
        out.dtype != brotensor::Dtype::FP16) {
        out.resize(1, C * Ho * Wo, brotensor::Dtype::FP16);
    }

    const dim3 block(16, 16, 1);
    const dim3 grid((Wo + block.x - 1) / block.x,
                    (Ho + block.y - 1) / block.y,
                    static_cast<unsigned int>(C));
    pad_right_bottom_zero_fp16_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(in.data_fp16()),
        reinterpret_cast<__half*>(out.data_fp16()),
        C, H, W);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fail(std::string("kernel launch: ") + cudaGetErrorString(e));
    }
}

}  // namespace brodiffusion
