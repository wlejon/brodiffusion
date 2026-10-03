#pragma once
//
// Test-only GPU pick shared by the brodiffusion tests that compare a GPU run
// against the CPU or gate on a GPU-only path (INT8, fused kernels). It returns
// the default device when that is a GPU (so BROTENSOR_DEFAULT_DEVICE=vulkan
// runs the suite on Vulkan), else whichever GPU backend the binary registered —
// HIP on a ROCm build, CUDA on a CUDA build, Metal on a Metal build, Vulkan
// last — in brotensor's own default order, so one test covers every platform
// without naming a backend.
//
// Call brotensor::init() first (it performs the HIP / CUDA / Metal driver
// probe), then preferred_gpu(): Device::CPU means no GPU backend is registered
// (skip the GPU block).
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

namespace bdtest {

inline brotensor::Device preferred_gpu() {
    const brotensor::Device d = brotensor::default_device();
    if (d.is_gpu()) return d;
    if (brotensor::is_available(brotensor::Device::HIP))   return brotensor::Device::HIP;
    if (brotensor::is_available(brotensor::Device::CUDA))  return brotensor::Device::CUDA;
    if (brotensor::is_available(brotensor::Device::Metal)) return brotensor::Device::Metal;
    if (brotensor::is_available(brotensor::Device::VULKAN)) return brotensor::Device::VULKAN;
    return brotensor::Device::CPU;
}

}  // namespace bdtest
