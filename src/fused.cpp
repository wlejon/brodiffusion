// brodiffusion fused-op dispatch + CPU fallback.
//
// brodiffusion ships SD1.5-tuned fused CUDA kernels (fused_resblock.cu,
// fused_transformer.cu). This translation unit provides the public
// brodiffusion::fused_* entry points as runtime dispatchers: a GPU-resident
// input routes to the CUDA kernel (brodiffusion::detail::*_cuda); a
// CPU-resident input routes to an FP32 fallback composed from brotensor's
// CPU ops. The library therefore runs end-to-end on either backend — CPU by
// default, CUDA when available — mirroring brotensor's unified dispatch.
//
// Always compiled. The CUDA branch is gated on BROTENSOR_HAS_CUDA (set by
// the brotensor CUDA backend), so a CPU-only build needs no nvcc.

#include "brodiffusion/fused_resblock.h"
#include "brodiffusion/fused_transformer.h"
#include "brodiffusion/optim.h"
#include "brodiffusion/detail/fused_backend.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brodiffusion {

namespace bt = ::brotensor;

namespace {

bool on_gpu(const bt::Tensor& t) { return t.device != bt::Device::CPU; }

[[noreturn]] void int8_cpu_unsupported(const char* what) {
    throw std::runtime_error(
        std::string("brodiffusion: ") + what +
        ": INT8 (W8A16) quantization is GPU-only — run without "
        "--quantize-unet on the CPU backend.");
}

// ─── CPU fallbacks (FP32, composed from brotensor ops) ─────────────────────

void fused_resblock_forward_cpu(
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
    // The CUDA kernel folds the per-channel t_emb shift and the residual into
    // the conv epilogues; brotensor::resblock_forward is the unfused
    // reference with identical math. N = 1 (SD1.5 single-image path).
    bt::resblock_forward(X, gn1_g, gn1_b, W1, &b1, &t_emb_shift,
                         gn2_g, gn2_b, W2, &b2, Wskip, bskip,
                         /*N=*/1, C_in, C_out, H, W, num_groups, eps, Y);
}

void fused_linear_geglu_cpu(const bt::Tensor& X, const bt::Tensor& W,
                            const bt::Tensor& b, bt::Tensor& Y) {
    // T = X @ Wᵀ + b  (B, 2*D_out); exact-GEGLU splits T into (value, gate),
    // gates the gate through erf-GELU and multiplies → Y (B, D_out).
    bt::Tensor T;
    bt::linear_forward_batched(W, b, X, T);
    bt::geglu_exact_forward(T, Y);
}

void add_inplace_vec_cpu(bt::Tensor& Y, const bt::Tensor& X) {
    bt::add_inplace(Y, X);
}

void add_inplace_row_bias_cpu(bt::Tensor& Y, const bt::Tensor& bias) {
    // Y(rows, cols) += bias[col]. CPU tensors are FP32 host buffers.
    const int rows = Y.rows, cols = Y.cols;
    if (bias.size() != cols) {
        throw std::runtime_error(
            "brodiffusion: add_inplace_row_bias: bias.size() must equal Y.cols");
    }
    float* y = Y.host_f32_mut();
    const float* bvec = bias.host_f32();
    for (int r = 0; r < rows; ++r) {
        float* row = y + static_cast<std::size_t>(r) * cols;
        for (int c = 0; c < cols; ++c) row[c] += bvec[c];
    }
}

}  // namespace

// ─── public dispatchers ────────────────────────────────────────────────────

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
#ifdef BROTENSOR_HAS_CUDA
    if (on_gpu(X)) {
        detail::fused_resblock_forward_cuda(
            X, gn1_g, gn1_b, W1, b1, t_emb_shift, gn2_g, gn2_b, W2, b2,
            Wskip, bskip, C_in, C_out, H, W, num_groups, eps, Y);
        return;
    }
#endif
    fused_resblock_forward_cpu(X, gn1_g, gn1_b, W1, b1, t_emb_shift,
                               gn2_g, gn2_b, W2, b2, Wskip, bskip,
                               C_in, C_out, H, W, num_groups, eps, Y);
}

void fused_resblock_forward(  // W8A16 — GPU-only
    [[maybe_unused]] const bt::Tensor& X,
    [[maybe_unused]] const bt::Tensor& gn1_g,
    [[maybe_unused]] const bt::Tensor& gn1_b,
    [[maybe_unused]] const bt::Tensor& W1_int8,
    [[maybe_unused]] const bt::Tensor& W1_scales,
    [[maybe_unused]] const bt::Tensor& b1,
    [[maybe_unused]] const bt::Tensor& t_emb_shift,
    [[maybe_unused]] const bt::Tensor& gn2_g,
    [[maybe_unused]] const bt::Tensor& gn2_b,
    [[maybe_unused]] const bt::Tensor& W2_int8,
    [[maybe_unused]] const bt::Tensor& W2_scales,
    [[maybe_unused]] const bt::Tensor& b2,
    [[maybe_unused]] const bt::Tensor* Wskip_int8,
    [[maybe_unused]] const bt::Tensor* Wskip_scales,
    [[maybe_unused]] const bt::Tensor* bskip,
    [[maybe_unused]] int C_in, [[maybe_unused]] int C_out,
    [[maybe_unused]] int H, [[maybe_unused]] int W,
    [[maybe_unused]] int num_groups, [[maybe_unused]] float eps,
    [[maybe_unused]] bt::Tensor& Y) {
#ifdef BROTENSOR_HAS_CUDA
    if (on_gpu(X)) {
        detail::fused_resblock_forward_cuda(
            X, gn1_g, gn1_b, W1_int8, W1_scales, b1, t_emb_shift, gn2_g, gn2_b,
            W2_int8, W2_scales, b2, Wskip_int8, Wskip_scales, bskip,
            C_in, C_out, H, W, num_groups, eps, Y);
        return;
    }
#endif
    int8_cpu_unsupported("fused_resblock_forward (W8A16)");
}

void fused_linear_geglu(const bt::Tensor& X, const bt::Tensor& W,
                        const bt::Tensor& b, bt::Tensor& Y) {
#ifdef BROTENSOR_HAS_CUDA
    if (on_gpu(X)) {
        detail::fused_linear_geglu_cuda(X, W, b, Y);
        return;
    }
#endif
    fused_linear_geglu_cpu(X, W, b, Y);
}

void fused_linear_geglu([[maybe_unused]] const bt::Tensor& X,       // W8A16
                        [[maybe_unused]] const bt::Tensor& W_int8,
                        [[maybe_unused]] const bt::Tensor& W_scales,
                        [[maybe_unused]] const bt::Tensor& b,
                        [[maybe_unused]] bt::Tensor& Y) {
#ifdef BROTENSOR_HAS_CUDA
    if (on_gpu(X)) {
        detail::fused_linear_geglu_cuda(X, W_int8, W_scales, b, Y);
        return;
    }
#endif
    int8_cpu_unsupported("fused_linear_geglu (W8A16)");
}

void add_inplace_vec(bt::Tensor& Y, const bt::Tensor& X) {
#ifdef BROTENSOR_HAS_CUDA
    if (on_gpu(Y)) {
        detail::add_inplace_vec_cuda(Y, X);
        return;
    }
#endif
    add_inplace_vec_cpu(Y, X);
}

void add_inplace_row_bias(bt::Tensor& Y, const bt::Tensor& bias) {
#ifdef BROTENSOR_HAS_CUDA
    if (on_gpu(Y)) {
        detail::add_inplace_row_bias_cuda(Y, bias);
        return;
    }
#endif
    add_inplace_row_bias_cpu(Y, bias);
}

}  // namespace brodiffusion

// ─── AdamFP16: training-only, GPU-only ─────────────────────────────────────
//
// The mixed-precision Adam optimizer (src/optim.cu) is part of the training
// path, not inference, and has no CPU fallback. On a CPU-only build its
// methods throw; a CUDA build gets the real implementation from optim.cu.
#ifndef BROTENSOR_HAS_CUDA
namespace brodiffusion::optim {

namespace {
[[noreturn]] void no_cpu_adam(const char* what) {
    throw std::runtime_error(
        std::string("brodiffusion: ") + what +
        " is training-only and requires a CUDA build "
        "(-DBROTENSOR_WITH_CUDA=ON).");
}
}  // namespace

int  AdamFP16::register_param(brotensor::Tensor&, std::string) { no_cpu_adam("AdamFP16"); }
int  AdamFP16::find(const std::string&) const { return -1; }
void AdamFP16::zero_grads() { no_cpu_adam("AdamFP16::zero_grads"); }
void AdamFP16::accumulate_fp16(int, const brotensor::Tensor&) { no_cpu_adam("AdamFP16::accumulate_fp16"); }
void AdamFP16::accumulate_fp32(int, const brotensor::Tensor&) { no_cpu_adam("AdamFP16::accumulate_fp32"); }
void AdamFP16::step(const AdamHyperParams&) { no_cpu_adam("AdamFP16::step"); }
void AdamFP16::reset_state() { no_cpu_adam("AdamFP16::reset_state"); }
const brotensor::Tensor& AdamFP16::master(int) const { no_cpu_adam("AdamFP16::master"); }

}  // namespace brodiffusion::optim
#endif  // !BROTENSOR_HAS_CUDA
