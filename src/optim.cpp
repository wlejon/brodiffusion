// FP16-weight Adam optimizer plumbing. See include/brodiffusion/optim.h.
//
// Backend-agnostic: composed entirely from device-polymorphic brotensor ops
// (cast / add_inplace / adam_step), so the same translation unit serves the
// CPU, CUDA, and Metal backends. The earlier implementation was a CUDA-only
// .cu with hand-written FP16<->FP32 cast kernels; brotensor::cast now supplies
// that primitive on every backend.
//
// AdamFP16 is a training-time component. Inference-only consumers never
// register a parameter, so this TU links cleanly into a CPU-only build even
// though brotensor's CPU backend is FP32-first.

#include "brodiffusion/optim.h"
#include "brodiffusion/detail/device.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <stdexcept>
#include <utility>

namespace brodiffusion::optim {

namespace bt = ::brotensor;

int AdamFP16::register_param(bt::Tensor& p, std::string name) {
    if (p.dtype != bt::Dtype::FP16) {
        throw std::runtime_error("AdamFP16: parameter must be FP16");
    }
    Slot s;
    s.p_fp16 = &p;
    s.name   = std::move(name);
    // FP32 master copy derived from the FP16 weight.
    bt::cast(p, s.p_fp32, bt::Dtype::FP32);
    // Grad accumulator + Adam moments, all FP32, zero-initialised on-device.
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
    if (dW.size() != g.size()) {
        throw std::runtime_error("AdamFP16::accumulate_fp16: shape mismatch");
    }
    if (g.size() == 0) return;
    // Up-cast the FP16 grad to FP32, then accumulate.
    thread_local bt::Tensor tmp;
    bt::cast(dW, tmp, bt::Dtype::FP32);
    bt::add_inplace(g, tmp);
}

void AdamFP16::accumulate_fp32(int handle, const bt::Tensor& dW) {
    if (handle < 0 || handle >= static_cast<int>(slots_.size())) {
        throw std::runtime_error("AdamFP16: handle out of range");
    }
    auto& g = slots_[static_cast<std::size_t>(handle)].g_fp32;
    if (dW.size() != g.size()) {
        throw std::runtime_error("AdamFP16::accumulate_fp32: shape mismatch");
    }
    if (g.size() == 0) return;
    bt::add_inplace(g, dW);
}

void AdamFP16::step(const AdamHyperParams& hp) {
    ++step_n_;
    for (auto& s : slots_) {
        bt::adam_step(s.p_fp32, s.g_fp32, s.m_fp32, s.v_fp32,
                      hp.lr, hp.beta1, hp.beta2, hp.eps, step_n_);
        // Write the updated FP32 master back into the live FP16 weight.
        bt::cast(s.p_fp32, *s.p_fp16, bt::Dtype::FP16);
    }
}

void AdamFP16::reset_state() {
    step_n_ = 0;
    for (auto& s : slots_) {
        bt::cast(*s.p_fp16, s.p_fp32, bt::Dtype::FP32);
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
