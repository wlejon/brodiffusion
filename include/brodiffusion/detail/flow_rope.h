#pragma once
//
// CUDA entry point for the TripoSplat flow DiT's content-conditioned axial RoPE
// table build. Defined in src/triposplat/flow_rope.cu — compiled only in a CUDA
// build. FlowDiT::build_rope dispatches a GPU-resident delta_pos here instead of
// the host round-trip (sync + download + host trig + upload) it runs per block
// on the CPU backend. This is flow-model domain math, so it lives in brodiffusion
// rather than brotensor (the generic library).

#include "brotensor/tensor.h"

namespace brodiffusion::detail {

// Build the (L*num_heads, half) FP32 cos/sin RoPE tables fully on-device.
//   delta_pos : (L, 3*num_heads) FP32, device — content-predicted positions,
//               laid out [head, axis] (so a free reshape to (L*num_heads, 3)).
//   freqs_pi  : (1, half) FP32, device — [freqs0|freqs1|freqs2] already ×π.
//   half      : head_dim/2 == f0 + f1 + f2.
//   f0, f1    : axis-group boundaries (axis 0 = [0,f0), 1 = [f0,f0+f1), 2 = rest).
// cos_out / sin_out are resized to (L*num_heads, half) FP32 on delta_pos's device.
void flow_rope_tables_cuda(const brotensor::Tensor& delta_pos,
                           const brotensor::Tensor& freqs_pi,
                           int L, int num_heads, int half, int f0, int f1,
                           brotensor::Tensor& cos_out,
                           brotensor::Tensor& sin_out);

}  // namespace brodiffusion::detail
