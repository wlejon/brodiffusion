#pragma once
//
// Unified compute-precision policy for brodiffusion.
//
// brodiffusion runs on whichever backend brotensor resolves at runtime —
// CPU by default, CUDA when a GPU is available. The compute precision
// follows the device: FP16 on a GPU backend (where the half-precision
// kernels pay for themselves) and FP32 on CPU (brotensor's CPU backend is
// FP32-only). `compute_dtype()` is the single decision point — every weight
// and activation tensor in the pipeline is allocated at this dtype.
//
// The helpers below paper over the two brotensor ops that ship a distinct
// FP16 entry point (batched linear and inference layernorm); every other
// brotensor op brodiffusion uses is already dtype-polymorphic.

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace brodiffusion {

// FP16 on a GPU backend, FP32 on CPU. Thin forwarder to brotensor's own
// compute-precision policy — kept as a brodiffusion-namespace name so the
// pipeline's existing call sites stay unchanged.
inline brotensor::Dtype compute_dtype() {
    return brotensor::compute_dtype();
}

namespace detail {

// Upload host FP32 values as a tensor at the pipeline compute dtype, on the
// current default device. FP16 builds convert through fp32_to_fp16_bits.
inline brotensor::Tensor upload_host(const float* src, int rows, int cols) {
    if (compute_dtype() == brotensor::Dtype::FP16) {
        std::vector<std::uint16_t> bits(
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
        for (std::size_t i = 0; i < bits.size(); ++i) {
            bits[i] = brotensor::fp32_to_fp16_bits(src[i]);
        }
        return brotensor::Tensor::from_host_fp16(bits.data(), rows, cols);
    }
    return brotensor::Tensor::from_host(src, rows, cols);
}

// Batched linear with optional bias. brotensor exposes the FP16 batched
// linear as a separate WMMA op; the plain op is the FP32 path. Dispatch on
// the weight dtype so one call site serves CPU and GPU.
inline void linear_batched(const brotensor::Tensor& W,
                           const brotensor::Tensor* bias,
                           const brotensor::Tensor& X,
                           brotensor::Tensor& Y) {
    if (W.dtype == brotensor::Dtype::FP16) {
        brotensor::linear_forward_batched_fp16(W, bias, X, Y);
        return;
    }
    if (bias != nullptr) {
        brotensor::linear_forward_batched(W, *bias, X, Y);
        return;
    }
    // The FP32 batched linear takes a required bias operand; synthesize a
    // zero one for the rare bias-free call (LCM cond_proj).
    brotensor::Tensor zero =
        brotensor::Tensor::zeros_on(W.device, W.rows, 1, W.dtype);
    brotensor::linear_forward_batched(W, zero, X, Y);
}

// Inference batched layernorm — same FP16-distinct-op situation.
inline void layernorm_batched(const brotensor::Tensor& X,
                              const brotensor::Tensor& gamma,
                              const brotensor::Tensor& beta,
                              brotensor::Tensor& Y, float eps) {
    if (X.dtype == brotensor::Dtype::FP16) {
        brotensor::layernorm_forward_inference_batched_fp16(X, gamma, beta, Y,
                                                            eps);
    } else {
        brotensor::layernorm_forward_inference_batched(X, gamma, beta, Y, eps);
    }
}

}  // namespace detail
}  // namespace brodiffusion
