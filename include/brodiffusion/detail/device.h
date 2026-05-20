#pragma once

// Internal helper for the unified-brotensor device model.
//
// brotensor's op-dispatch layer pins an *unallocated* output tensor to the
// op's device (detail::adopt_output) before the backend resizes it. That
// covers every tensor brodiffusion hands to a brotensor op. It does NOT
// cover buffers brodiffusion allocates itself and then fills via a copy op
// (copy_d2d writes into pre-existing storage) or via brodiffusion's own
// fused CUDA kernels. For those, call resize_like so the first allocation
// lands on the intended backend instead of the host (a default-constructed
// Tensor is Device::CPU, and Tensor::resize preserves the current device).

#include "brotensor/tensor.h"

namespace brodiffusion::detail {

// Resize `t` to (r, c, dt). If `t` has no storage yet, first pin it to
// `dev` so the allocation lands on the right backend. Once allocated, a
// matching-shape resize reuses storage (brotensor::Tensor::resize fast path).
inline void resize_like(brotensor::Tensor& t, int r, int c,
                        brotensor::Dtype dt, brotensor::Device dev) {
    if (t.data == nullptr) t.device = dev;
    t.resize(r, c, dt);
}

}  // namespace brodiffusion::detail
