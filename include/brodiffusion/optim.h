#pragma once

// FP16-weight Adam: maintains an FP32 master copy + (m, v) state for each
// registered FP16 parameter, accumulates gradients, runs brotensor's
// adam_step_gpu, then casts the updated FP32 master back into the FP16 weight.
//
// Standard mixed-precision optimizer plumbing — not tied to any particular
// model. Callers register every trainable tensor once, then per training step:
//
//   opt.zero_grads();
//   // ... backward kernels produce dW tensors, FP16 or FP32 ...
//   opt.accumulate_fp16(handle, dW_fp16);    // or accumulate_fp32
//   opt.step(hp);

#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brodiffusion::optim {

struct AdamHyperParams {
    float lr    = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps   = 1e-8f;
};

class AdamFP16 {
public:
    // Register a parameter. `param_fp16` must outlive the optimizer; the
    // optimizer keeps a non-owning pointer and writes back to it on step().
    // Returns a handle (index in registration order) for grad accumulation.
    int register_param(brotensor::GpuTensor& param_fp16, std::string name = "");

    int size() const { return static_cast<int>(slots_.size()); }
    int find(const std::string& name) const;

    // Zero every grad accumulator.
    void zero_grads();

    // Accumulate a grad tensor into slot `handle`'s FP32 grad accumulator.
    void accumulate_fp16(int handle, const brotensor::GpuTensor& dW_fp16);
    void accumulate_fp32(int handle, const brotensor::GpuTensor& dW_fp32);

    // One Adam update over every registered parameter. Increments the internal
    // step counter (1-based, used by brotensor's bias-correction).
    void step(const AdamHyperParams& hp);

    // Reset (m, v, step counter) and re-derive the FP32 master from the
    // current FP16 weight. Call after loading weights from a checkpoint.
    void reset_state();

    // FP32 master copy of slot `handle` (read-only access for checkpointing).
    const brotensor::GpuTensor& master(int handle) const;

    int step_count() const { return step_n_; }

private:
    struct Slot {
        brotensor::GpuTensor* p_fp16;  // non-owning
        brotensor::GpuTensor  p_fp32;
        brotensor::GpuTensor  g_fp32;
        brotensor::GpuTensor  m_fp32, v_fp32;
        std::string           name;
    };
    std::vector<Slot> slots_;
    int step_n_ = 0;
};

}  // namespace brodiffusion::optim
