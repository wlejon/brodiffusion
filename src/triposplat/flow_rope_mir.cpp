// Single-source Brass MIR GPU kernel for TripoSplat flow DiT axial RoPE tables.
// Lowered on demand to SPIR-V (Vulkan), PTX (CUDA), or MSL (Metal) and
// dispatched on brotensor's device stream via brotensor::interop.

#include "brodiffusion/detail/flow_rope.h"
#include "brodiffusion/detail/device.h"

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/msl_target.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv_target.hpp>

#include <brotensor/kernel_interop.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brodiffusion::detail {

namespace {

namespace bt = ::brotensor;
using brass::BasicBlock;
using brass::Function;
using brass::Builder;
using brass::Module;
using brass::Type;
using brass::Value;
using brass::codegen::KernelBuilder;

template <std::size_t N>
Value* horner_f32(Builder& b, Value* x, const float (&coeffs)[N]) {
    Value* acc = b.build_fconst_f32(coeffs[N - 1]);
    for (std::size_t i = N - 1; i > 0; --i) {
        acc = b.build_fma_f32(acc, x, b.build_fconst_f32(coeffs[i - 1]));
    }
    return acc;
}

// Accurate FP32 range-reduced sincos (~1 ulp at any argument, matching
// brotensor's emit_sincos_acc).
std::pair<Value*, Value*> emit_sincos_acc(KernelBuilder& kb, Builder& b, Value* x) {
    auto cf = [&](float v) { return b.build_fconst_f32(v); };
    auto ci = [&](std::int32_t v) { return b.build_iconst_i32(v); };
    Value* j = b.build_round_f32(b.build_mul(x, cf(0.636619772367581343f)));
    Value* nj = b.build_neg(j);
    Value* y = b.build_fma_f32(nj, cf(1.57079637050628662f), x);
    y = b.build_fma_f32(nj, cf(-4.37113882867379e-08f), y);
    y = b.build_fma_f32(nj, cf(-1.71512451e-15f), y);
    Value* z = b.build_mul(y, y);
    static constexpr float kSinPoly[] = {-1.6666654611e-1f, 8.3321608736e-3f, -1.9515295891e-4f};
    static constexpr float kCosPoly[] = {4.166664568298827e-2f, -1.388731625493765e-3f, 2.443315711809948e-5f};
    Value* sp = b.build_fma_f32(b.build_mul(y, z), horner_f32(b, z, kSinPoly), y);
    Value* cp = b.build_fma_f32(b.build_mul(z, z), horner_f32(b, z, kCosPoly),
                                b.build_fma_f32(cf(-0.5f), z, cf(1.0f)));
    Value* nsp = b.build_neg(sp);
    Value* ncp = b.build_neg(cp);
    Value* q = b.build_and(kb.f32_to_i32(j), ci(3));
    Value* q0 = b.build_eq(q, ci(0));
    Value* q1 = b.build_eq(q, ci(1));
    Value* q2 = b.build_eq(q, ci(2));
    Value* sin_v = b.build_select(q0, sp, b.build_select(q1, cp, b.build_select(q2, nsp, ncp)));
    Value* cos_v = b.build_select(q0, cp, b.build_select(q1, nsp, b.build_select(q2, ncp, sp)));
    return {sin_v, cos_v};
}

Value* load_elem(KernelBuilder& kb, Builder& b, Value* ptr, Value* idx64, bool is_fp16) {
    if (!is_fp16) return kb.load_f32_indexed(ptr, idx64);
    Value* byte_off = b.build_shl(idx64, b.build_iconst_i64(1));
    Value* p = b.build_add(ptr, byte_off);
    return kb.f16_to_f32(kb.load_u16(p, 0));
}

Function* build_flow_rope_mir(Module& mod, bool is_fp16) {
    const std::string name = is_fp16 ? "flow_rope_f16" : "flow_rope_f32";
    std::vector<Type> ptypes = {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(),
        Type::i32(), Type::i32(), Type::i32(), Type::i32(),
    };
    Function* fn = mod.create_function(name, Type::void_type(), brass::Span<const Type>(ptypes));
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* hdr = b.append_block("hdr");
    BasicBlock* body = b.append_block("body");
    BasicBlock* done = b.append_block("done");
    b.position_at_end(entry);

    Value* p_dp = b.add_param(Type::ptr());
    Value* p_freqs = b.add_param(Type::ptr());
    Value* p_cos = b.add_param(Type::ptr());
    Value* p_sin = b.add_param(Type::ptr());
    Value* p_rows = b.add_param(Type::i32());
    Value* p_half = b.add_param(Type::i32());
    Value* p_f0 = b.add_param(Type::i32());
    Value* p_f1 = b.add_param(Type::i32());

    auto ci = [&](std::int32_t v) { return b.build_iconst_i32(v); };
    Value* total = b.build_mul(p_rows, p_half);
    Value* f01 = b.build_add(p_f0, p_f1);
    Value* gid = kb.global_tid_x();
    Value* stride = b.build_mul(kb.nctaid_x(), ci(256));
    b.build_br(hdr, {gid});

    b.position_at_end(hdr);
    Value* phi_i = b.add_param(Type::i32());
    b.build_br_if(b.build_ult(phi_i, total), body, done);

    b.position_at_end(body);
    Value* row = b.build_udiv(phi_i, p_half);
    Value* j = b.build_sub(phi_i, b.build_mul(row, p_half));
    Value* axis = b.build_select(
        b.build_ult(j, p_f0), ci(0),
        b.build_select(b.build_ult(j, f01), ci(1), ci(2)));
    Value* dp_idx = b.build_zext_i64(b.build_add(b.build_mul(row, ci(3)), axis));
    Value* j64 = b.build_zext_i64(j);
    Value* i64 = b.build_zext_i64(phi_i);

    Value* dp_v = load_elem(kb, b, p_dp, dp_idx, is_fp16);
    Value* fr_v = load_elem(kb, b, p_freqs, j64, is_fp16);
    Value* angle = b.build_mul(dp_v, fr_v);
    auto [sin_v, cos_v] = emit_sincos_acc(kb, b, angle);
    kb.store_f32_indexed(p_cos, i64, cos_v);
    kb.store_f32_indexed(p_sin, i64, sin_v);

    Value* next_i = b.build_add(phi_i, stride);
    b.build_br(hdr, {next_i});

    b.position_at_end(done);
    b.build_ret_void();
    return fn;
}

enum class BackendKind : std::uint8_t { Vulkan = 0, Cuda = 1, Metal = 2 };

struct CompiledRope {
    brass::target::SpirvKernel spv;
    std::string ptx;
    std::string ptx_entry;
    brass::target::MslKernel msl;
};

const CompiledRope& get_rope_kernel(bool is_fp16, BackendKind bk) {
    struct Cache {
        std::mutex mu;
        std::unordered_map<std::uint32_t, CompiledRope> map;
    };
    static auto* cache = new Cache();
    const std::uint32_t key = (static_cast<std::uint32_t>(bk) << 1) | (is_fp16 ? 1u : 0u);
    {
        std::lock_guard<std::mutex> lk(cache->mu);
        if (auto it = cache->map.find(key); it != cache->map.end()) return it->second;
    }

    Module mod("brodiffusion_flow_rope");
    Function* fn = build_flow_rope_mir(mod, is_fp16);
    CompiledRope cr;
    if (bk == BackendKind::Vulkan) {
        brass::target::SpirvOptions opts;
        opts.stage = brass::target::SpirvStage::Compute;
        opts.local_size_x = 256;
        opts.local_size_spec_constants = false;
        cr.spv = brass::target::SpirvTarget::compile(*fn, opts);
    } else if (bk == BackendKind::Cuda) {
        brass::target::PtxOptions opts;
        opts.sm_arch = "sm_70";
        opts.ptx_version_major = 7;
        opts.ptx_version_minor = 0;
        cr.ptx = brass::target::PtxTarget::emit_function(*fn, opts);
        cr.ptx_entry = std::string(fn->name());
    } else {
        brass::target::MslOptions opts;
        opts.workgroup_x = 256;
        cr.msl = brass::target::MslTarget::compile(*fn, opts);
    }

    std::lock_guard<std::mutex> lk(cache->mu);
    auto [it, _] = cache->map.emplace(key, std::move(cr));
    return it->second;
}

}  // namespace

void flow_rope_tables_gpu(const bt::Tensor& delta_pos, const bt::Tensor& freqs_pi,
                          int L, int num_heads, int half, int f0, int f1,
                          bt::Tensor& cos_out, bt::Tensor& sin_out) {
    if (delta_pos.dtype != freqs_pi.dtype) {
        throw std::runtime_error("flow_rope_tables_gpu: delta_pos/freqs_pi dtype mismatch");
    }
    if (freqs_pi.device != delta_pos.device) {
        throw std::runtime_error("flow_rope_tables_gpu: delta_pos/freqs_pi on different devices");
    }
    if (!delta_pos.device.is_gpu()) {
        throw std::runtime_error("flow_rope_tables_gpu: requires a GPU device");
    }
    const int rows = L * num_heads;
    if (rows <= 0 || half <= 0) return;

    bool is_fp16 = false;
    if (delta_pos.dtype == bt::Dtype::FP32) {
        is_fp16 = false;
    } else if (delta_pos.dtype == bt::Dtype::FP16) {
        is_fp16 = true;
    } else {
        throw std::runtime_error("flow_rope_tables_gpu: delta_pos must be FP32 or FP16");
    }

    // FP32 tables; resize_like keeps their addresses stable across calls
    // (required by CUDA-graph / Vulkan step-graph replay).
    resize_like(cos_out, rows, half, bt::Dtype::FP32, delta_pos.device);
    resize_like(sin_out, rows, half, bt::Dtype::FP32, delta_pos.device);

    const std::uint64_t total = static_cast<std::uint64_t>(rows) * half;
    if (total > 0xffffffffull) {
        throw std::runtime_error("flow_rope_tables_gpu: table size exceeds uint32");
    }
    const auto groups = static_cast<std::uint32_t>(std::min<std::uint64_t>((total + 255u) / 256u, 65535u));

    const bt::Device dev = delta_pos.device;
    const std::uint64_t a_dp = bt::interop::view(delta_pos).device_address;
    const std::uint64_t a_fr = bt::interop::view(freqs_pi).device_address;
    const std::uint64_t a_cos = bt::interop::view(cos_out).device_address;
    const std::uint64_t a_sin = bt::interop::view(sin_out).device_address;
    const std::uint32_t u_rows = static_cast<std::uint32_t>(rows);
    const std::uint32_t u_half = static_cast<std::uint32_t>(half);
    const std::uint32_t u_f0 = static_cast<std::uint32_t>(f0);
    const std::uint32_t u_f1 = static_cast<std::uint32_t>(f1);

    std::string err;
    if (dev.is_vulkan()) {
        const CompiledRope& cr = get_rope_kernel(is_fp16, BackendKind::Vulkan);
        std::vector<std::uint8_t> push(cr.spv.push_constant_bytes, 0);
        std::memcpy(push.data() + cr.spv.params[0].offset, &a_dp, 8);
        std::memcpy(push.data() + cr.spv.params[1].offset, &a_fr, 8);
        std::memcpy(push.data() + cr.spv.params[2].offset, &a_cos, 8);
        std::memcpy(push.data() + cr.spv.params[3].offset, &a_sin, 8);
        std::memcpy(push.data() + cr.spv.params[4].offset, &u_rows, 4);
        std::memcpy(push.data() + cr.spv.params[5].offset, &u_half, 4);
        std::memcpy(push.data() + cr.spv.params[6].offset, &u_f0, 4);
        std::memcpy(push.data() + cr.spv.params[7].offset, &u_f1, 4);

        bt::interop::SpirvDispatchDesc desc;
        desc.words = cr.spv.words.data();
        desc.word_count = cr.spv.words.size();
        desc.entry_name = cr.spv.entry.c_str();
        desc.local_size[0] = 256;
        desc.groups[0] = groups;
        desc.push_data = push.data();
        desc.push_bytes = static_cast<std::uint32_t>(push.size());
        desc.local_size_spec_constants = false;
        if (!bt::interop::dispatch_spirv(dev, desc, &err)) {
            throw std::runtime_error("flow_rope_tables_gpu (Vulkan): " + err);
        }
        return;
    }

    if (dev.is_cuda()) {
        const CompiledRope& cr = get_rope_kernel(is_fp16, BackendKind::Cuda);
        std::uint64_t p0 = a_dp, p1 = a_fr, p2 = a_cos, p3 = a_sin;
        std::uint32_t s0 = u_rows, s1 = u_half, s2 = u_f0, s3 = u_f1;
        void* kparams[8] = {&p0, &p1, &p2, &p3, &s0, &s1, &s2, &s3};

        bt::interop::PtxDispatchDesc desc;
        desc.ptx = cr.ptx.data();
        desc.ptx_bytes = cr.ptx.size();
        desc.entry_name = cr.ptx_entry.c_str();
        desc.local_size[0] = 256;
        desc.groups[0] = groups;
        desc.kernel_params = kparams;
        if (!bt::interop::dispatch_ptx(dev, desc, &err)) {
            throw std::runtime_error("flow_rope_tables_gpu (CUDA): " + err);
        }
        return;
    }

    if (dev.is_metal()) {
        const CompiledRope& cr = get_rope_kernel(is_fp16, BackendKind::Metal);
        const void* bufs[4] = {delta_pos.data, freqs_pi.data, cos_out.data, sin_out.data};
        std::array<std::uint8_t, 64> scalars{};
        const std::uint32_t sc[4] = {u_rows, u_half, u_f0, u_f1};
        for (std::size_t i = 0; i < 4 && i < cr.msl.scalar_param_offsets.size(); ++i) {
            std::memcpy(scalars.data() + cr.msl.scalar_param_offsets[i], &sc[i], 4);
        }

        bt::interop::MslDispatchDesc desc;
        desc.msl = cr.msl.source.data();
        desc.msl_bytes = cr.msl.source.size();
        desc.entry_name = cr.msl.entry_name.c_str();
        desc.local_size[0] = 256;
        desc.groups[0] = groups;
        desc.buffer_ptrs = bufs;
        desc.buffer_count = 4;
        desc.scalar_data = scalars.data();
        desc.scalar_bytes = static_cast<std::uint32_t>(cr.msl.scalar_params_bytes);
        desc.scalar_buffer_index = static_cast<std::uint32_t>(cr.msl.num_buffers);
        if (!bt::interop::dispatch_msl(dev, desc, &err)) {
            throw std::runtime_error("flow_rope_tables_gpu (Metal): " + err);
        }
        return;
    }

    throw std::runtime_error("flow_rope_tables_gpu: unsupported GPU device");
}

std::vector<MirAuditShader> compile_flow_rope_spirv_for_audit() {
    std::vector<MirAuditShader> out;
    for (bool is_fp16 : {false, true}) {
        const char* name = is_fp16 ? "flow_rope_f16" : "flow_rope_f32";
        const CompiledRope& vk_cr = get_rope_kernel(is_fp16, BackendKind::Vulkan);
        const CompiledRope& cu_cr = get_rope_kernel(is_fp16, BackendKind::Cuda);
        const CompiledRope& mtl_cr = get_rope_kernel(is_fp16, BackendKind::Metal);
        if (cu_cr.ptx.empty()) throw std::runtime_error(std::string("PTX emit failed for ") + name);
        if (mtl_cr.msl.source.empty()) throw std::runtime_error(std::string("MSL emit failed for ") + name);
        out.push_back(MirAuditShader{name, vk_cr.spv.words});
    }
    return out;
}

}  // namespace brodiffusion::detail
