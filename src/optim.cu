// FP16-weight Adam optimizer plumbing. See include/brodiffusion/optim.h.

#include "brodiffusion/optim.h"
#include "brodiffusion/detail/cuda_check.cuh"
#include "brodiffusion/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <utility>

namespace brodiffusion::optim {

namespace bt = ::brotensor;

namespace {

__global__ void cast_h2f_k(const __half* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}
__global__ void cast_f2h_k(const float* src, __half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}
__global__ void acc_h2f_k(float* dst, const __half* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += __half2float(src[i]);
}
__global__ void acc_f2f_k(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

void cast_h_to_f(const bt::Tensor& src, bt::Tensor& dst) {
    detail::resize_like(dst, src.rows, src.cols, bt::Dtype::FP32, src.device);
    int n = src.size();
    if (n == 0) return;
    cast_h2f_k<<<(n + 255) / 256, 256>>>(
        reinterpret_cast<const __half*>(src.data),
        static_cast<float*>(dst.data), n);
    BRODIFFUSION_CUDA_CHECK(cudaGetLastError());
}

void cast_f_to_h(const bt::Tensor& src, bt::Tensor& dst) {
    // Compare full shape, not just element count: a same-size but
    // differently-shaped dst must still be reshaped, since callers rely on
    // dst's rows/cols metadata.
    if (dst.rows != src.rows || dst.cols != src.cols ||
        dst.dtype != bt::Dtype::FP16 || dst.device != src.device) {
        detail::resize_like(dst, src.rows, src.cols, bt::Dtype::FP16, src.device);
    }
    int n = src.size();
    if (n == 0) return;
    cast_f2h_k<<<(n + 255) / 256, 256>>>(
        static_cast<const float*>(src.data),
        reinterpret_cast<__half*>(dst.data), n);
    BRODIFFUSION_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

int AdamFP16::register_param(bt::Tensor& p, std::string name) {
    if (p.dtype != bt::Dtype::FP16) {
        throw std::runtime_error("AdamFP16: parameter must be FP16");
    }
    Slot s;
    s.p_fp16 = &p;
    s.name   = std::move(name);
    cast_h_to_f(p, s.p_fp32);
    detail::resize_like(s.g_fp32, p.rows, p.cols, bt::Dtype::FP32, p.device);
    s.g_fp32.zero();
    detail::resize_like(s.m_fp32, p.rows, p.cols, bt::Dtype::FP32, p.device);
    s.m_fp32.zero();
    detail::resize_like(s.v_fp32, p.rows, p.cols, bt::Dtype::FP32, p.device);
    s.v_fp32.zero();
    slots_.push_back(std::move(s));
    return static_cast<int>(slots_.size()) - 1;
}

int AdamFP16::find(const std::string& name) const {
    for (int i = 0; i < static_cast<int>(slots_.size()); ++i) {
        if (slots_[static_cast<std::size_t>(i)].name == name) return i;
    }
    return -1;
}

void AdamFP16::zero_grads() {
    for (auto& s : slots_) s.g_fp32.zero();
}

void AdamFP16::accumulate_fp16(int handle, const bt::Tensor& dW) {
    if (handle < 0 || handle >= static_cast<int>(slots_.size())) {
        throw std::runtime_error("AdamFP16: handle out of range");
    }
    auto& g = slots_[static_cast<std::size_t>(handle)].g_fp32;
    int n = g.size();
    if (dW.size() != n) {
        throw std::runtime_error("AdamFP16::accumulate_fp16: shape mismatch");
    }
    if (n == 0) return;
    acc_h2f_k<<<(n + 255) / 256, 256>>>(
        static_cast<float*>(g.data),
        reinterpret_cast<const __half*>(dW.data), n);
    BRODIFFUSION_CUDA_CHECK(cudaGetLastError());
}

void AdamFP16::accumulate_fp32(int handle, const bt::Tensor& dW) {
    if (handle < 0 || handle >= static_cast<int>(slots_.size())) {
        throw std::runtime_error("AdamFP16: handle out of range");
    }
    auto& g = slots_[static_cast<std::size_t>(handle)].g_fp32;
    int n = g.size();
    if (dW.size() != n) {
        throw std::runtime_error("AdamFP16::accumulate_fp32: shape mismatch");
    }
    if (n == 0) return;
    acc_f2f_k<<<(n + 255) / 256, 256>>>(
        static_cast<float*>(g.data),
        static_cast<const float*>(dW.data), n);
    BRODIFFUSION_CUDA_CHECK(cudaGetLastError());
}

void AdamFP16::step(const AdamHyperParams& hp) {
    ++step_n_;
    for (auto& s : slots_) {
        bt::adam_step(s.p_fp32, s.g_fp32, s.m_fp32, s.v_fp32,
                          hp.lr, hp.beta1, hp.beta2, hp.eps, step_n_);
        cast_f_to_h(s.p_fp32, *s.p_fp16);
    }
}

void AdamFP16::reset_state() {
    step_n_ = 0;
    for (auto& s : slots_) {
        cast_h_to_f(*s.p_fp16, s.p_fp32);
        s.g_fp32.zero();
        s.m_fp32.zero();
        s.v_fp32.zero();
    }
}

const bt::Tensor& AdamFP16::master(int handle) const {
    if (handle < 0 || handle >= static_cast<int>(slots_.size())) {
        throw std::runtime_error("AdamFP16: handle out of range");
    }
    return slots_[static_cast<std::size_t>(handle)].p_fp32;
}

}  // namespace brodiffusion::optim
