#pragma once
//
// Backend-neutral GPU entry point (Vulkan SPIR-V, CUDA PTX, Metal MSL) for the
// TripoSplat flow DiT's content-conditioned axial RoPE table build.
// Lowered once from Brass MIR in src/triposplat/flow_rope_mir.cpp and
// dispatched through brotensor::interop (no nvcc, glslc, or Objective-C++).

#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brodiffusion::detail {

// Build the (L*num_heads, half) FP32 cos/sin RoPE tables fully on-device.
//   delta_pos : (L, 3*num_heads) FP32 or FP16, GPU — content-predicted positions,
//               laid out [head, axis] (so a free reshape to (L*num_heads, 3)).
//   freqs_pi  : (1, half) matching dtype, GPU — [freqs0|freqs1|freqs2] already ×π.
//   half      : head_dim/2 == f0 + f1 + f2.
//   f0, f1    : axis-group boundaries (axis 0 = [0,f0), 1 = [f0,f0+f1), 2 = rest).
// cos_out / sin_out are resized to (L*num_heads, half) FP32 on delta_pos's device.
void flow_rope_tables_gpu(const brotensor::Tensor& delta_pos,
                          const brotensor::Tensor& freqs_pi,
                          int L, int num_heads, int half, int f0, int f1,
                          brotensor::Tensor& cos_out,
                          brotensor::Tensor& sin_out);

struct MirAuditShader {
    std::string name;
    std::vector<std::uint32_t> words;
};

/// Compiles both flow_rope Brass MIR kernels (FP32, FP16) to SPIR-V for static
/// formal verification in tests/test_vulkan_shader_audit.cpp.
std::vector<MirAuditShader> compile_flow_rope_spirv_for_audit();

}  // namespace brodiffusion::detail
