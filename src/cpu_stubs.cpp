// CPU-build stubs for brodiffusion's GPU-only translation units.
//
// brodiffusion can be configured and built without a GPU backend so it is not
// coupled to CUDA at the build-system level (matching brotensor / brogameagent).
// The actual diffusion work is GPU-only: brotensor's CPU backend implements no
// conv2d / group_norm / attention kernels, and brodiffusion's own fused CUDA
// kernels (fused_resblock.cu, fused_transformer.cu, optim.cu) are not compiled.
//
// This file is compiled *only* when no GPU backend is selected. It defines the
// symbols those .cu files would otherwise provide so the library and CLI still
// link; every entry throws. A CPU build therefore compiles, links, and runs
// non-diffusion code (tokenizer, safetensors, scheduler math) — anything that
// reaches a GPU kernel throws a clear std::runtime_error.

#include "brodiffusion/fused_resblock.h"
#include "brodiffusion/fused_transformer.h"
#include "brodiffusion/optim.h"

#include <stdexcept>

namespace {
[[noreturn]] void no_gpu(const char* what) {
    throw std::runtime_error(
        std::string("brodiffusion: ") + what +
        " requires a GPU backend — reconfigure with -DBROTENSOR_WITH_CUDA=ON "
        "(or -DBROTENSOR_WITH_METAL=ON).");
}
}  // namespace

namespace brodiffusion {

void fused_resblock_forward(
    const brotensor::Tensor&,
    const brotensor::Tensor&, const brotensor::Tensor&,
    const brotensor::Tensor&, const brotensor::Tensor&,
    const brotensor::Tensor&,
    const brotensor::Tensor&, const brotensor::Tensor&,
    const brotensor::Tensor&, const brotensor::Tensor&,
    const brotensor::Tensor*, const brotensor::Tensor*,
    int, int, int, int, int, float,
    brotensor::Tensor&) {
    no_gpu("fused_resblock_forward");
}

void fused_resblock_forward(
    const brotensor::Tensor&,
    const brotensor::Tensor&, const brotensor::Tensor&,
    const brotensor::Tensor&, const brotensor::Tensor&,
    const brotensor::Tensor&,
    const brotensor::Tensor&,
    const brotensor::Tensor&, const brotensor::Tensor&,
    const brotensor::Tensor&, const brotensor::Tensor&,
    const brotensor::Tensor&,
    const brotensor::Tensor*, const brotensor::Tensor*, const brotensor::Tensor*,
    int, int, int, int, int, float,
    brotensor::Tensor&) {
    no_gpu("fused_resblock_forward (W8A16)");
}

void fused_linear_geglu(const brotensor::Tensor&, const brotensor::Tensor&,
                        const brotensor::Tensor&, brotensor::Tensor&) {
    no_gpu("fused_linear_geglu");
}

void fused_linear_geglu(const brotensor::Tensor&, const brotensor::Tensor&,
                        const brotensor::Tensor&, const brotensor::Tensor&,
                        brotensor::Tensor&) {
    no_gpu("fused_linear_geglu (W8A16)");
}

void add_inplace_fp16_vec(brotensor::Tensor&, const brotensor::Tensor&) {
    no_gpu("add_inplace_fp16_vec");
}

void add_inplace_row_bias_fp16(brotensor::Tensor&, const brotensor::Tensor&) {
    no_gpu("add_inplace_row_bias_fp16");
}

}  // namespace brodiffusion

namespace brodiffusion::optim {

int  AdamFP16::register_param(brotensor::Tensor&, std::string) { no_gpu("AdamFP16"); }
int  AdamFP16::find(const std::string&) const { return -1; }
void AdamFP16::zero_grads() { no_gpu("AdamFP16::zero_grads"); }
void AdamFP16::accumulate_fp16(int, const brotensor::Tensor&) { no_gpu("AdamFP16::accumulate_fp16"); }
void AdamFP16::accumulate_fp32(int, const brotensor::Tensor&) { no_gpu("AdamFP16::accumulate_fp32"); }
void AdamFP16::step(const AdamHyperParams&) { no_gpu("AdamFP16::step"); }
void AdamFP16::reset_state() { no_gpu("AdamFP16::reset_state"); }
const brotensor::Tensor& AdamFP16::master(int) const { no_gpu("AdamFP16::master"); }

}  // namespace brodiffusion::optim
