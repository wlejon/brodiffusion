// Static SPIR-V verification & memory-safety audit of brodiffusion Vulkan shaders.

#include "brodiffusion_vk_shaders.h"

#include <brass/target/spirv/spirv_binary_reader.hpp>
#include <brass/target/spirv/spirv_ir.hpp>
#include <brass/verify/gpu_kernel_prover.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>

static int g_failures = 0;

#define CHECK(cond) do {                                                    \
    if (!(cond)) {                                                          \
        std::printf("  FAIL  %s:%d  %s\n", __FILE__, __LINE__, #cond);      \
        ++g_failures;                                                       \
    }                                                                       \
} while (0)

int main() {
    namespace vk = brodiffusion::detail::vk;
    const std::size_t n = vk::shader_count();
    std::printf("Auditing %zu brodiffusion Vulkan shaders...\n", n);
    CHECK(n == 2);

    for (std::size_t i = 0; i < n; ++i) {
        const vk::EmbeddedShader* info = vk::shader_info(i);
        CHECK(info != nullptr && info->words != nullptr && info->word_count > 0);
        if (!info || !info->words || info->word_count == 0) continue;

        const std::string name = info->name;
        std::printf("  [%zu/%zu] %s (%zu words)\n", i + 1, n, name.c_str(), info->word_count);

        brass::spirv::Module mod;
        std::string parse_err;
        const bool parsed = brass::spirv::parse_binary(
            std::span<const std::uint32_t>(info->words, info->word_count), mod, &parse_err);
        if (!parsed) std::printf("    parse error: %s\n", parse_err.c_str());
        CHECK(parsed);
        if (!parsed) continue;

        std::string struct_err;
        const bool struct_ok = brass::spirv::verify_spirv_structural(mod, &struct_err);
        if (!struct_ok) std::printf("    structural error: %s\n", struct_err.c_str());
        CHECK(struct_ok);

        const brass::spirv::ComputeShaderAuditReport rep = brass::verify::prove_spirv_shader_safety(mod);
        CHECK(rep.structural_ok);
        CHECK(rep.dominance_ok);
        CHECK(rep.shared_bounds_ok);
        CHECK(rep.barriers_uniform);
        CHECK(rep.div_sqrt_ok);

        const std::uint32_t eb = (name == "flow_rope_f32") ? 4u : 2u;
        brass::verify::GpuProverOptions opts;
        opts.local_size_x = std::max(1u, rep.local_size[0]);
        opts.loop_unroll_bound = 2;
        opts.max_sat_conflicts = 4000;
        opts.buffers = {
            {0, 0, 4, eb, 16, -1, -1, 3},
            {1, 0, 5, eb, 16, -1, -1, 1},
            {2, 0, 4, 4, 16, 5, -1, 1},
            {3, 0, 4, 4, 16, 5, -1, 1},
        };
        opts.param_constraints = {
            {4, UINT32_MAX, 1, 0},
            {5, UINT32_MAX, 1, 0},
        };

        brass::verify::GpuMemorySafetyResult mem = brass::verify::verify_spirv_memory_safety(mod, opts);
        std::printf("    %-20s status=%d global=%u align=%u shared=%u diag=\"%s\"\n",
                    name.c_str(),
                    static_cast<int>(mem.status),
                    mem.global_accesses_proven,
                    mem.alignments_proven,
                    mem.shared_accesses_proven,
                    mem.diagnostic.c_str());
        const bool mem_ok =
            !mem.is_counterexample() &&
            mem.violations.empty() &&
            (mem.is_safe() || mem.status == brass::verify::EquivStatus::Unknown) &&
            mem.global_accesses_proven >= 16;
        CHECK(mem_ok);
    }

    if (g_failures != 0) {
        std::printf("FAILED (%d checks)\n", g_failures);
        return 1;
    }
    std::printf("ALL PASSED\n");
    return 0;
}
