#pragma once
//
// SD1.5-specific fused resblock forward.
//
// Mirrors the semantics of brotensor::resblock_forward_gpu but folds the
// per-channel t_emb shift into conv1's epilogue and folds the residual
// (or 1x1-skip) add into conv2's epilogue, cutting kernel launches per
// resblock from ~8 to ~3-5.
//
// Restrictions (SD1.5-only):
//   * N = 1 (single-image inference path)
//   * conv1/conv2 are 3x3 stride 1 pad 1 dilation 1
//   * conv = 1x1 skip only when C_in != C_out (handled by brotensor's path)
//   * All tensors FP16.

#include "brotensor/tensor.h"

namespace brodiffusion {

void fused_resblock_forward(
    const brotensor::GpuTensor& X,
    const brotensor::GpuTensor& gn1_g, const brotensor::GpuTensor& gn1_b,
    const brotensor::GpuTensor& W1,    const brotensor::GpuTensor& b1,
    const brotensor::GpuTensor& t_emb_shift,
    const brotensor::GpuTensor& gn2_g, const brotensor::GpuTensor& gn2_b,
    const brotensor::GpuTensor& W2,    const brotensor::GpuTensor& b2,
    const brotensor::GpuTensor* Wskip, const brotensor::GpuTensor* bskip,
    int C_in, int C_out, int H, int W,
    int num_groups, float eps,
    brotensor::GpuTensor& Y);

} // namespace brodiffusion
