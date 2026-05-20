#pragma once
//
// Device-agnostic test helpers for brodiffusion's smoke tests.
//
// brodiffusion runs CPU-by-default / GPU-when-available: compute_dtype() is
// FP32 on the CPU backend and FP16 on a GPU backend. These helpers let a test
// synthesize runtime input tensors and read back outputs without caring which
// dtype the active backend uses.

#include "brodiffusion/detail/compute.h"

#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <vector>

namespace bdtest {

// Upload host FP32 values as a runtime tensor at the pipeline compute dtype on
// the current default device.
inline brotensor::Tensor bd_upload(const std::vector<float>& v, int rows,
                                   int cols) {
    return brodiffusion::detail::upload_host(v.data(), rows, cols);
}

// Download a tensor to host as FP32 regardless of its dtype (FP32 verbatim,
// FP16 converted through fp16_bits_to_fp32). Syncs the device first.
inline std::vector<float> bd_download(const brotensor::Tensor& t) {
    brotensor::sync_all();
    if (t.dtype == brotensor::Dtype::FP16) {
        std::vector<uint16_t> bits = t.to_host_vector_fp16();
        std::vector<float> out(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            out[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return t.to_host_vector();
}

inline bool bd_finite(float v) { return std::isfinite(v); }

}  // namespace bdtest
