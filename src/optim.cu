// FP16-weight Adam optimizer plumbing. See include/brodiffusion/optim.h.

#include "brodiffusion/optim.h"

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

void cast_h_to_f(const bt::GpuTensor& src, bt::GpuTensor& dst) {
    dst.resize(src.rows, src.cols, bt::Dtype::FP32);
    int n = src.size();
    if (n == 0) return;
    cast_h2f_k<<<(n + 255) / 256, 256>>>(
        reinterpret_cast<const __half*>(src.data_fp16()), dst.data, n);
}

void cast_f_to_h(const bt::GpuTensor& src, bt::GpuTensor& dst) {
    if (dst.size() != src.size() || dst.dtype != bt::Dtype::FP16) {
        dst.resize(src.rows, src.cols, bt::Dtype::FP16);
    }
    int n = src.size();
    if (n == 0) return;
    cast_f2h_k<<<(n + 255) / 256, 256>>>(
        src.data, reinterpret_cast<__half*>(dst.data_fp16()), n);
}

}  // namespace

int AdamFP16::register_param(bt::GpuTensor& p, std::string name) {
    if (p.dtype != bt::Dtype::FP16) {
        throw std::runtime_error("AdamFP16: parameter must be FP16");
    }
    Slot s;
    s.p_fp16 = &p;
    s.name   = std::move(name);
    cast_h_to_f(p, s.p_fp32);
    s.g_fp32.resize(p.rows, p.cols, bt::Dtype::FP32); s.g_fp32.zero();
    s.m_fp32.resize(p.rows, p.cols, bt::Dtype::FP32); s.m_fp32.zero();
    s.v_fp32.resize(p.rows, p.cols, bt::Dtype::FP32); s.v_fp32.zero();
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

void AdamFP16::accumulate_fp16(int handle, const bt::GpuTensor& dW) {
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
        g.data, reinterpret_cast<const __half*>(dW.data_fp16()), n);
}

void AdamFP16::accumulate_fp32(int handle, const bt::GpuTensor& dW) {
    if (handle < 0 || handle >= static_cast<int>(slots_.size())) {
        throw std::runtime_error("AdamFP16: handle out of range");
    }
    auto& g = slots_[static_cast<std::size_t>(handle)].g_fp32;
    int n = g.size();
    if (dW.size() != n) {
        throw std::runtime_error("AdamFP16::accumulate_fp32: shape mismatch");
    }
    if (n == 0) return;
    acc_f2f_k<<<(n + 255) / 256, 256>>>(g.data, dW.data, n);
}

void AdamFP16::step(const AdamHyperParams& hp) {
    ++step_n_;
    for (auto& s : slots_) {
        bt::adam_step_gpu(s.p_fp32, s.g_fp32, s.m_fp32, s.v_fp32,
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

const bt::GpuTensor& AdamFP16::master(int handle) const {
    if (handle < 0 || handle >= static_cast<int>(slots_.size())) {
        throw std::runtime_error("AdamFP16: handle out of range");
    }
    return slots_[static_cast<std::size_t>(handle)].p_fp32;
}

}  // namespace brodiffusion::optim
