#pragma once

// CUDA error-check macro for brodiffusion's own fused CUDA kernels.
//
// brotensor keeps its BROTENSOR_CUDA_CHECK in an internal (non-installed)
// header, so brodiffusion's .cu translation units carry their own.

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define BRODIFFUSION_CUDA_CHECK(expr)                                          \
    do {                                                                       \
        cudaError_t _bd_err = (expr);                                          \
        if (_bd_err != cudaSuccess) {                                          \
            throw std::runtime_error(                                          \
                std::string("brodiffusion CUDA error: ") +                     \
                cudaGetErrorString(_bd_err) + " (" #expr ")");                 \
        }                                                                      \
    } while (0)
