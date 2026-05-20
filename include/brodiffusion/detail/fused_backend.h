#pragma once
//
// CUDA entry points for brodiffusion's SD1.5-tuned fused kernels.
//
// Defined in src/fused_resblock.cu / src/fused_transformer.cu — compiled
// only in a CUDA build. The public brodiffusion::fused_* functions
// (src/fused.cpp) are runtime dispatchers: GPU-resident tensors route here;
// CPU-resident tensors route to an FP32 fallback composed from brotensor's
// CPU ops. This mirrors brotensor's own unified-dispatch model so the whole
// library runs on either backend from a single build.

#include "brotensor/tensor.h"

namespace brodiffusion::detail {

void fused_resblock_forward_cuda(
    const brotensor::Tensor& X,
    const brotensor::Tensor& gn1_g, const brotensor::Tensor& gn1_b,
    const brotensor::Tensor& W1,    const brotensor::Tensor& b1,
    const brotensor::Tensor& t_emb_shift,
    const brotensor::Tensor& gn2_g, const brotensor::Tensor& gn2_b,
    const brotensor::Tensor& W2,    const brotensor::Tensor& b2,
    const brotensor::Tensor* Wskip, const brotensor::Tensor* bskip,
    int C_in, int C_out, int H, int W,
    int num_groups, float eps,
    brotensor::Tensor& Y);

void fused_resblock_forward_cuda(  // W8A16 (INT8 weight-only) variant
    const brotensor::Tensor& X,
    const brotensor::Tensor& gn1_g, const brotensor::Tensor& gn1_b,
    const brotensor::Tensor& W1_int8, const brotensor::Tensor& W1_scales,
    const brotensor::Tensor& b1,
    const brotensor::Tensor& t_emb_shift,
    const brotensor::Tensor& gn2_g, const brotensor::Tensor& gn2_b,
    const brotensor::Tensor& W2_int8, const brotensor::Tensor& W2_scales,
    const brotensor::Tensor& b2,
    const brotensor::Tensor* Wskip_int8,
    const brotensor::Tensor* Wskip_scales,
    const brotensor::Tensor* bskip,
    int C_in, int C_out, int H, int W,
    int num_groups, float eps,
    brotensor::Tensor& Y);

void fused_linear_geglu_cuda(const brotensor::Tensor& X,
                             const brotensor::Tensor& W,
                             const brotensor::Tensor& b,
                             brotensor::Tensor& Y);

void fused_linear_geglu_cuda(const brotensor::Tensor& X,        // W8A16
                             const brotensor::Tensor& W_int8,
                             const brotensor::Tensor& W_scales,
                             const brotensor::Tensor& b,
                             brotensor::Tensor& Y);

void add_inplace_vec_cuda(brotensor::Tensor& Y, const brotensor::Tensor& X);

void add_inplace_row_bias_cuda(brotensor::Tensor& Y,
                               const brotensor::Tensor& bias);

// ─── Metal entry points ────────────────────────────────────────────────────
//
// Defined in src/fused_resblock.mm / src/fused_transformer.mm — compiled only
// in a Metal build. simdgroup-matrix transliterations of the CUDA kernels
// above; the public brodiffusion::fused_* dispatchers (src/fused.cpp) route
// Metal-resident tensors here. Only the FP16 path has a brodiffusion-tuned
// Metal kernel — the W8A16 path dispatches straight to brotensor's INT8 ops
// (which already have Metal kernels), so there is no INT8 *_metal overload.

void fused_resblock_forward_metal(
    const brotensor::Tensor& X,
    const brotensor::Tensor& gn1_g, const brotensor::Tensor& gn1_b,
    const brotensor::Tensor& W1,    const brotensor::Tensor& b1,
    const brotensor::Tensor& t_emb_shift,
    const brotensor::Tensor& gn2_g, const brotensor::Tensor& gn2_b,
    const brotensor::Tensor& W2,    const brotensor::Tensor& b2,
    const brotensor::Tensor* Wskip, const brotensor::Tensor* bskip,
    int C_in, int C_out, int H, int W,
    int num_groups, float eps,
    brotensor::Tensor& Y);

void fused_linear_geglu_metal(const brotensor::Tensor& X,
                              const brotensor::Tensor& W,
                              const brotensor::Tensor& b,
                              brotensor::Tensor& Y);

void add_inplace_vec_metal(brotensor::Tensor& Y, const brotensor::Tensor& X);

void add_inplace_row_bias_metal(brotensor::Tensor& Y,
                                const brotensor::Tensor& bias);

}  // namespace brodiffusion::detail
