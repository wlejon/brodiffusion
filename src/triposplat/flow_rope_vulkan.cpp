// Vulkan host side of the TripoSplat flow DiT's RoPE table build: one dispatch
// of src/vulkan/flow_rope.comp (compiled and embedded by
// brotensor_vulkan_add_shaders in CMakeLists.txt) on the tensors' device
// stream, so it orders with brotensor's ops and is recorded by a step-graph
// capture like them.

#include "brodiffusion/detail/flow_rope.h"
#include "brodiffusion/detail/device.h"

#include "brodiffusion_vk_shaders.h"   // generated

#include <brotensor/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace brodiffusion::detail {

namespace bt = ::brotensor;

namespace {

struct FlowRopePush {
    std::uint64_t dp, freqs, cos_out, sin_out;
    std::uint32_t rows, half_dim, f0, f1;
};

}  // namespace

void flow_rope_tables_vulkan(const bt::Tensor& delta_pos, const bt::Tensor& freqs_pi,
                             int L, int num_heads, int half, int f0, int f1,
                             bt::Tensor& cos_out, bt::Tensor& sin_out) {
    if (delta_pos.dtype != freqs_pi.dtype) {
        throw std::runtime_error("flow_rope_tables_vulkan: delta_pos/freqs_pi dtype mismatch");
    }
    if (freqs_pi.device != delta_pos.device) {
        throw std::runtime_error("flow_rope_tables_vulkan: delta_pos/freqs_pi on different devices");
    }
    const int rows = L * num_heads;
    if (rows <= 0 || half <= 0) return;
    vk::Shader sh;
    switch (delta_pos.dtype) {
        case bt::Dtype::FP32: sh = vk::Shader::flow_rope_f32; break;
        case bt::Dtype::FP16: sh = vk::Shader::flow_rope_f16; break;
        default:
            throw std::runtime_error("flow_rope_tables_vulkan: delta_pos must be FP32 or FP16");
    }
    // FP32 tables; resize_like keeps their addresses stable across calls (the
    // step graph replays against them).
    resize_like(cos_out, rows, half, bt::Dtype::FP32, delta_pos.device);
    resize_like(sin_out, rows, half, bt::Dtype::FP32, delta_pos.device);
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * half;
    if (total > 0xffffffffull) throw std::runtime_error("flow_rope_tables_vulkan: too large");
    FlowRopePush pc{bt::vulkan::address(delta_pos), bt::vulkan::address(freqs_pi),
                    bt::vulkan::address(cos_out), bt::vulkan::address(sin_out),
                    static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(half),
                    static_cast<std::uint32_t>(f0), static_cast<std::uint32_t>(f1)};
    const auto groups = static_cast<std::uint32_t>(std::min<std::uint64_t>((total + 255) / 256, 65535));
    bt::vulkan::dispatch(delta_pos.device, vk::handle(sh), &pc, sizeof pc, groups);
}

}  // namespace brodiffusion::detail
