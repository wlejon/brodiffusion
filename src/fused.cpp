// brodiffusion fused-op dispatch.
//
// All GPU backends (CUDA, Metal, Vulkan) route through brotensor's unified
// fused GPU ops (resblock_forward, resblock_forward_int8w_fp16,
// linear_forward_batched_ex + geglu_exact_forward,
// linear_forward_batched_int8w_fp16 + geglu_exact_forward, add_inplace,
// add_row_bias_inplace). Pure C++20 — no backend-specific CUDA/Metal kernels.

#include "brodiffusion/fused_resblock.h"
#include "brodiffusion/fused_transformer.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <stdexcept>
#include <string>

namespace brodiffusion {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void gpu_only(const char* what) {
    throw std::runtime_error(std::string("brodiffusion: ") + what +
                             " runs on a GPU (CUDA, Metal or Vulkan), not the CPU.");
}

// The geglu intermediate is a thread_local so its address stays stable across
// steps for CUDA-graph / Vulkan step-graph replay.
void gpu_linear_geglu(const bt::Tensor& X, const bt::Tensor& W,
                      const bt::Tensor& b, bt::Tensor& Y) {
    thread_local bt::Tensor T;
    bt::linear_forward_batched_ex(W, &b, X, bt::kLinearActNone, bt::kLinearEpiStore,
                                  nullptr, T);
    bt::geglu_exact_forward(T, Y);
}

void gpu_linear_geglu_int8(const bt::Tensor& X, const bt::Tensor& W_int8,
                           const bt::Tensor& W_scales, const bt::Tensor& b,
                           bt::Tensor& Y) {
    thread_local bt::Tensor T;
    bt::linear_forward_batched_int8w_fp16(W_int8, W_scales, &b, X, T);
    bt::geglu_exact_forward(T, Y);
}

}  // namespace

void fused_resblock_forward(
    const bt::Tensor& X,
    const bt::Tensor& gn1_g, const bt::Tensor& gn1_b,
    const bt::Tensor& W1,    const bt::Tensor& b1,
    const bt::Tensor& t_emb_shift,
    const bt::Tensor& gn2_g, const bt::Tensor& gn2_b,
    const bt::Tensor& W2,    const bt::Tensor& b2,
    const bt::Tensor* Wskip, const bt::Tensor* bskip,
    int C_in, int C_out, int H, int W,
    int num_groups, float eps,
    bt::Tensor& Y) {
    if (!X.device.is_gpu()) gpu_only("fused_resblock_forward");
    bt::resblock_forward(X, gn1_g, gn1_b, W1, &b1, &t_emb_shift,
                         gn2_g, gn2_b, W2, &b2, Wskip, bskip,
                         /*N=*/1, C_in, C_out, H, W, num_groups, eps, Y);
}

void fused_resblock_forward(  // W8A16
    const bt::Tensor& X,
    const bt::Tensor& gn1_g, const bt::Tensor& gn1_b,
    const bt::Tensor& W1_int8, const bt::Tensor& W1_scales,
    const bt::Tensor& b1,
    const bt::Tensor& t_emb_shift,
    const bt::Tensor& gn2_g, const bt::Tensor& gn2_b,
    const bt::Tensor& W2_int8, const bt::Tensor& W2_scales,
    const bt::Tensor& b2,
    const bt::Tensor* Wskip_int8, const bt::Tensor* Wskip_scales,
    const bt::Tensor* bskip,
    int C_in, int C_out, int H, int W,
    int num_groups, float eps,
    bt::Tensor& Y) {
    if (!X.device.is_gpu()) gpu_only("fused_resblock_forward (W8A16)");
    bt::resblock_forward_int8w_fp16(
        X, gn1_g, gn1_b, W1_int8, W1_scales, &b1, &t_emb_shift,
        gn2_g, gn2_b, W2_int8, W2_scales, &b2,
        Wskip_int8, Wskip_scales, bskip,
        /*N=*/1, C_in, C_out, H, W, num_groups, eps, Y);
}

void fused_linear_geglu(const bt::Tensor& X, const bt::Tensor& W,
                        const bt::Tensor& b, bt::Tensor& Y) {
    if (!X.device.is_gpu()) gpu_only("fused_linear_geglu");
    gpu_linear_geglu(X, W, b, Y);
}

void fused_linear_geglu(const bt::Tensor& X,       // W8A16
                        const bt::Tensor& W_int8,
                        const bt::Tensor& W_scales,
                        const bt::Tensor& b,
                        bt::Tensor& Y) {
    if (!X.device.is_gpu()) gpu_only("fused_linear_geglu (W8A16)");
    gpu_linear_geglu_int8(X, W_int8, W_scales, b, Y);
}

void add_inplace_vec(bt::Tensor& Y, const bt::Tensor& X) {
    if (!Y.device.is_gpu()) gpu_only("add_inplace_vec");
    bt::add_inplace(Y, X);
}

void add_inplace_row_bias(bt::Tensor& Y, const bt::Tensor& bias) {
    if (!Y.device.is_gpu()) gpu_only("add_inplace_row_bias");
    if (static_cast<int>(bias.size()) != Y.cols) {
        throw std::runtime_error(
            "brodiffusion: add_inplace_row_bias: bias.size() must equal Y.cols");
    }
    bt::add_row_bias_inplace(Y, bias);
}

}  // namespace brodiffusion
