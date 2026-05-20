#pragma once
//
// SD1.5-specific fused resblock forward.
//
// Mirrors the semantics of brotensor::resblock_forward but folds the
// per-channel t_emb shift into conv1's epilogue and folds the residual
// (or 1x1-skip) add into conv2's epilogue, cutting kernel launches per
// resblock from ~8 to ~3-5.
//
// Runtime dispatcher (see src/fused.cpp): a GPU-resident input runs the
// fused CUDA kernel below; a CPU-resident input delegates to the unfused
// brotensor::resblock_forward in FP32. Same math either way.
//
// Restrictions (SD1.5-only):
//   * N = 1 (single-image inference path)
//   * conv1/conv2 are 3x3 stride 1 pad 1 dilation 1
//   * conv = 1x1 skip only when C_in != C_out (handled by brotensor's path)
//   * FP16 on a GPU backend; FP32 on CPU.

#include "brotensor/tensor.h"

namespace brodiffusion {

void fused_resblock_forward(
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

// W8A16 (INT8 weight-only) variant of fused_resblock_forward. Semantics are
// identical to the FP16 overload except that W1, W2 and (optional) Wskip are
// supplied as INT8 weights + per-output-row FP32 scales. The internal conv
// kernel routes through brotensor::conv2d_int8w_fp16_forward while the
// GroupNorm / SiLU / residual paths stay FP16. Biases stay FP16.
void fused_resblock_forward(
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

} // namespace brodiffusion
