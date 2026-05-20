#pragma once
//
// SD1.5-specific fused transformer ops.
//
// Hard-coded inference assumptions:
//   * N=1 outer batch (B in these APIs is the flattened (1, L) sequence length).
//   * FP16 storage, FP32 accumulators.
//   * Shapes used by SD1.5 BasicTransformerBlock: D in {320, 640, 1280},
//     L in {4096, 1024, 256, 64}, FF inner = 4 * D.
//
// Generic versions of these ops live in brotensor; SD1.5-tuned fusions
// belong here.

#include "brotensor/tensor.h"

namespace brodiffusion {

// Fused FF1 + GEGLU.
//
// Computes Y(B, D_out) = a * GELU_exact(g), where
//   T(B, 2*D_out) = X @ W^T + b
//   a = T[:, :D_out]
//   g = T[:, D_out:]
// i.e. T is split column-wise into (value, gate); the gate is passed through
// exact (erf-based) GELU; the value is multiplied by the gated gate. This
// matches brotensor::geglu_exact_forward_fp16_kernel byte-for-byte modulo
// FP32 accumulation differences (we never materialise the 2*D_out tile).
//
// W is (2*D_out, D_in), b is (2*D_out,). All tensors FP16.
void fused_linear_geglu(const brotensor::Tensor& X,
                        const brotensor::Tensor& W,
                        const brotensor::Tensor& b,
                        brotensor::Tensor& Y);

// W8A16 variant: ff1 weight is INT8 with per-output-row FP32 scales. The
// linear part is delegated to brotensor::linear_forward_batched_int8w_fp16
// into an FP16 (B, 2*D_out) buffer (not fused), then geglu_forward turns
// that into Y(B, D_out). Bias stays FP16 (2*D_out, 1).
//   W_int8: (2*D_out, D_in) Dtype::INT8
//   scales: (2*D_out, 1)    FP32
//   b:      FP16 (2*D_out, 1)
void fused_linear_geglu(const brotensor::Tensor& X,
                        const brotensor::Tensor& W_int8,
                        const brotensor::Tensor& W_scales,
                        const brotensor::Tensor& b,
                        brotensor::Tensor& Y);

// Vectorised FP16 elementwise add: Y[i] += X[i].
// brotensor::add_inplace's FP16 kernel goes through __half2float per
// element; this one uses __half2 vector adds (and int4 vector loads when
// alignment allows). Same shape semantics: Y and X must match.
void add_inplace_fp16_vec(brotensor::Tensor& Y, const brotensor::Tensor& X);

// Per-column broadcast bias add: Y[i, j] += bias[j], where Y is (rows, cols)
// FP16 row-major and bias is FP16 length cols (shape (cols,1) or (1,cols)
// both accepted — we only look at size()). Used by trace-mode cross-attention
// to fold the attn2.to_out bias that
// brotensor::cross_attention_forward_with_attn does not accept as an input.
void add_inplace_row_bias_fp16(brotensor::Tensor& Y,
                               const brotensor::Tensor& bias);

} // namespace brodiffusion
