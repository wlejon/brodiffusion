// brodiffusion fused-op dispatch + CPU fallback.
//
// brodiffusion ships SD1.5-tuned fused CUDA kernels (fused_resblock.cu,
// fused_transformer.cu, also built as HIP) and their Metal twins. This
// translation unit provides the public brodiffusion::fused_* entry points as
// runtime dispatchers on the input's device:
//   * CUDA / HIP -> the CUDA kernel (brodiffusion::detail::*_cuda);
//   * Metal      -> the Metal kernel (detail::*_metal);
//   * any other GPU (Vulkan) -> brotensor's own fused GPU ops, which is what
//     these kernels fuse: resblock_forward (GroupNorm+SiLU, conv with the
//     time-embedding shift folded into its bias, the skip, conv2 accumulated
//     onto it), a linear then geglu_exact_forward, add_inplace,
//     add_row_bias_inplace (gpu_* below);
//   * CPU        -> an FP32 fallback composed from brotensor's CPU ops.
// A tensor never reaches a kernel built for another backend: a HIP + Vulkan
// build sends a Vulkan tensor (a buffer device address) to the gpu_* path.
//
// Always compiled. The CUDA / HIP branch is gated on BROTENSOR_HAS_CUDA /
// BROTENSOR_HAS_HIP, so a build without them needs no nvcc / hipcc.

#include "brodiffusion/fused_resblock.h"
#include "brodiffusion/fused_transformer.h"
#include "brodiffusion/detail/fused_backend.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brodiffusion {

namespace bt = ::brotensor;

namespace {

// The tensor lives where brodiffusion's CUDA / HIP kernels can run on it.
[[maybe_unused]] bool on_cuda_kernel_device(const bt::Tensor& t) {
    return t.device.is_cuda() || t.device.is_hip();
}
[[maybe_unused]] bool on_metal(const bt::Tensor& t) { return t.device.is_metal(); }

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

// ─── other GPUs (Vulkan): brotensor's fused GPU ops ──────────────────────
//
// FP16 storage as the CUDA kernels take it (brotensor's ops accept the same
// tensors). The geglu intermediate is a thread_local so its address is stable
// across steps (step-graph replay).

void gpu_resblock(const bt::Tensor& X,
                  const bt::Tensor& gn1_g, const bt::Tensor& gn1_b,
                  const bt::Tensor& W1, const bt::Tensor& b1,
                  const bt::Tensor& t_emb_shift,
                  const bt::Tensor& gn2_g, const bt::Tensor& gn2_b,
                  const bt::Tensor& W2, const bt::Tensor& b2,
                  const bt::Tensor* Wskip, const bt::Tensor* bskip,
                  int C_in, int C_out, int H, int W, int num_groups, float eps,
                  bt::Tensor& Y) {
    bt::resblock_forward(X, gn1_g, gn1_b, W1, &b1, &t_emb_shift,
                         gn2_g, gn2_b, W2, &b2, Wskip, bskip,
                         /*N=*/1, C_in, C_out, H, W, num_groups, eps, Y);
}

void gpu_linear_geglu(const bt::Tensor& X, const bt::Tensor& W,
                      const bt::Tensor& b, bt::Tensor& Y) {
    // T = X W^T + b (B, 2 D_out), value half then gate half; Y = a * gelu(g).
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
#if defined(BROTENSOR_HAS_CUDA) || defined(BROTENSOR_HAS_HIP)
    if (on_cuda_kernel_device(X)) {
        detail::fused_resblock_forward_cuda(
            X, gn1_g, gn1_b, W1, b1, t_emb_shift, gn2_g, gn2_b, W2, b2,
            Wskip, bskip, C_in, C_out, H, W, num_groups, eps, Y);
        return;
    }
#endif
#if defined(BROTENSOR_HAS_METAL)
    if (on_metal(X)) {
        detail::fused_resblock_forward_metal(
            X, gn1_g, gn1_b, W1, b1, t_emb_shift, gn2_g, gn2_b, W2, b2,
            Wskip, bskip, C_in, C_out, H, W, num_groups, eps, Y);
        return;
    }
#endif
    if (X.device.is_gpu()) {
        gpu_resblock(X, gn1_g, gn1_b, W1, b1, t_emb_shift, gn2_g, gn2_b, W2, b2,
                     Wskip, bskip, C_in, C_out, H, W, num_groups, eps, Y);
        return;
    }
    fused_resblock_forward_cpu(X, gn1_g, gn1_b, W1, b1, t_emb_shift,
                               gn2_g, gn2_b, W2, b2, Wskip, bskip,
                               C_in, C_out, H, W, num_groups, eps, Y);
}

void fused_resblock_forward(  // W8A16 — GPU-only
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
#if defined(BROTENSOR_HAS_CUDA) || defined(BROTENSOR_HAS_HIP)
    if (on_cuda_kernel_device(X)) {
        detail::fused_resblock_forward_cuda(
            X, gn1_g, gn1_b, W1_int8, W1_scales, b1, t_emb_shift, gn2_g, gn2_b,
            W2_int8, W2_scales, b2, Wskip_int8, Wskip_scales, bskip,
            C_in, C_out, H, W, num_groups, eps, Y);
        return;
    }
#endif
    if (X.device.is_gpu()) {
        // Metal and Vulkan: brotensor's INT8 resblock op (no brodiffusion-tuned
        // kernel for the W8A16 resblock there).
        bt::resblock_forward_int8w_fp16(
            X, gn1_g, gn1_b, W1_int8, W1_scales, &b1, &t_emb_shift,
            gn2_g, gn2_b, W2_int8, W2_scales, &b2,
            Wskip_int8, Wskip_scales, bskip,
            /*N=*/1, C_in, C_out, H, W, num_groups, eps, Y);
        return;
    }
    int8_cpu_unsupported("fused_resblock_forward (W8A16)");
}

void fused_linear_geglu(const bt::Tensor& X, const bt::Tensor& W,
                        const bt::Tensor& b, bt::Tensor& Y) {
#if defined(BROTENSOR_HAS_CUDA) || defined(BROTENSOR_HAS_HIP)
    if (on_cuda_kernel_device(X)) {
        detail::fused_linear_geglu_cuda(X, W, b, Y);
        return;
    }
#endif
#if defined(BROTENSOR_HAS_METAL)
    if (on_metal(X)) {
        detail::fused_linear_geglu_metal(X, W, b, Y);
        return;
    }
#endif
    if (X.device.is_gpu()) {
        gpu_linear_geglu(X, W, b, Y);
        return;
    }
    fused_linear_geglu_cpu(X, W, b, Y);
}

void fused_linear_geglu(const bt::Tensor& X,       // W8A16
                        const bt::Tensor& W_int8,
                        const bt::Tensor& W_scales,
                        const bt::Tensor& b,
                        bt::Tensor& Y) {
#if defined(BROTENSOR_HAS_CUDA) || defined(BROTENSOR_HAS_HIP)
    if (on_cuda_kernel_device(X)) {
        detail::fused_linear_geglu_cuda(X, W_int8, W_scales, b, Y);
        return;
    }
#endif
    if (X.device.is_gpu()) {
        // (B, in) @ dequant(W_int8)^T + b -> FP16 (B, 2*D_out), then GEGLU;
        // brotensor's INT8 linear + exact-GEGLU on Metal and Vulkan.
        gpu_linear_geglu_int8(X, W_int8, W_scales, b, Y);
        return;
    }
    int8_cpu_unsupported("fused_linear_geglu (W8A16)");
}

void add_inplace_vec(bt::Tensor& Y, const bt::Tensor& X) {
#if defined(BROTENSOR_HAS_CUDA) || defined(BROTENSOR_HAS_HIP)
    if (on_cuda_kernel_device(Y)) {
        detail::add_inplace_vec_cuda(Y, X);
        return;
    }
#endif
#if defined(BROTENSOR_HAS_METAL)
    if (on_metal(Y)) {
        detail::add_inplace_vec_metal(Y, X);
        return;
    }
#endif
    // Vulkan's add_inplace is the vectorised FP16 pass already; CPU likewise.
    add_inplace_vec_cpu(Y, X);
}

void add_inplace_row_bias(bt::Tensor& Y, const bt::Tensor& bias) {
#if defined(BROTENSOR_HAS_CUDA) || defined(BROTENSOR_HAS_HIP)
    if (on_cuda_kernel_device(Y)) {
        detail::add_inplace_row_bias_cuda(Y, bias);
        return;
    }
#endif
#if defined(BROTENSOR_HAS_METAL)
    if (on_metal(Y)) {
        detail::add_inplace_row_bias_metal(Y, bias);
        return;
    }
#endif
    if (Y.device.is_gpu()) {
        if (static_cast<int>(bias.size()) != Y.cols) {
            throw std::runtime_error(
                "brodiffusion: add_inplace_row_bias: bias.size() must equal Y.cols");
        }
        bt::add_row_bias_inplace(Y, bias);
        return;
    }
    add_inplace_row_bias_cpu(Y, bias);
}

}  // namespace brodiffusion
