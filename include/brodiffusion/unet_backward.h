#pragma once

// dX-only backward through the FROZEN teacher UNet's mid_block + up_blocks +
// conv_out path. Used by end-to-end inlet distillation: the gradient on
// eps_pred is back-propagated all the way to the 12 skip tensors and the
// bottleneck (the inputs the inlet produces), which then feed the inlet's
// own backward (already implemented in distill.cu).
//
// Teacher weights are FROZEN throughout — every conv/linear/norm/attention
// backward call discards weight grads into thread_local scratch buffers
// (sinks) so the caller never sees a dW/dB tensor.
//
// To avoid having to plumb forward intermediates from the inference path
// (which uses fused kernels that don't expose them), the implementation
// re-runs the teacher up path inside itself, caching the input to every
// op. The composite backwards (resblock_backward_gpu, flash_attention_qkvo
// _backward_gpu) recompute their own intermediates from those inputs.
//
// Cost: roughly 2× a single forward of the up path (one forward pass to
// build the cache, plus the backward which re-recomputes some intermediates
// inside each composite). Peak activation cache size dominated by the
// 64×64×320 caches near conv_out (~2.5 MB per tap × O(20) taps ≈ 50 MB).

#include "brodiffusion/unet.h"
#include "brotensor/tensor.h"

#include <array>

namespace brodiffusion::unet {

// See top of file. Inputs:
//   net            : teacher UNet with weights loaded (frozen).
//   bottleneck_in  : (1, 1280, 8, 8) FP16 — same tensor the inlet produced.
//   skips_in       : 12 FP16 skip tensors in teacher push order
//                    (s0..s11; s11 has the same shape as bottleneck).
//   ctx            : (77, 768) FP16 text-encoder context.
//   t_emb_raw      : (1, 1280) FP16 — pre-silu time embedding (caller hasn't
//                    silu'd; we silu internally before feeding resblocks).
//   d_eps_pred     : (1, 4, 64, 64) FP16 — gradient on eps_pred.
//
// Outputs (caller-allocated; resized in place):
//   d_skips_out    : 12 dSkip tensors, same shapes as skips_in.
//   d_bottleneck_out: dBottleneck, same shape as bottleneck_in.
void unet_up_path_backward(const UNet& net,
                           const brotensor::GpuTensor& bottleneck_in,
                           const std::array<const brotensor::GpuTensor*, 12>& skips_in,
                           const brotensor::GpuTensor& ctx,
                           const brotensor::GpuTensor& t_emb_raw,
                           const brotensor::GpuTensor& d_eps_pred,
                           std::array<brotensor::GpuTensor, 12>& d_skips_out,
                           brotensor::GpuTensor& d_bottleneck_out);

}  // namespace brodiffusion::unet
