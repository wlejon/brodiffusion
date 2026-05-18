#pragma once

// Self-attention student module for SD1.5 distillation (Phase 1).
//
// Drop-in replacement for the *self-attention sub-layer* of a Transformer2D
// block at L = 4096 (i.e. the most expensive attention call in the U-Net:
// 64x64 spatial, C = 320, 8 heads, ~10.5 GFLOPs per call). The cross-attention
// and feed-forward sub-layers of the host block are unchanged.
//
// Architecture (fixed — do not redesign):
//   Operates on NCHW (N=1, C, H, W) directly. The host block already does a
//   sequence<->NCHW transpose around the attention call; when the student is
//   active those surrounding transposes are skipped (student consumes/produces
//   NCHW directly).
//
//   y = x
//   for i in 0..3:
//       y = y + pw1x1_i(silu(dwconv3x3_i(y)))
//   return y
//
//   - dwconv3x3_i: 3x3 depthwise conv (groups = C in spirit; brotensor's
//     conv2d_forward_gpu has no `groups` arg so we fall back to a *full*
//     3x3 C->C conv per layer until brotensor grows depthwise support).
//   - pw1x1_i:     1x1 pointwise (C, C), no bias.
//   - silu:        in-place via brotensor::silu_forward_gpu.
//   - Residual add per inner layer.
//
// Init: weights zero, biases zero — block starts as identity so the residual
// passes through cleanly and an untrained student doesn't destroy outputs.
//
// All tensors FP16.

#include "brotensor/tensor.h"

#include <string>

namespace brodiffusion::safetensors { class File; }

namespace brodiffusion::student {

struct SelfAttnStudent {
    int channels = 0;            // C  (e.g. 320 for SD1.5 L=4096 blocks)
    int height   = 0;            // H  (e.g. 64)
    int width    = 0;            // W  (e.g. 64)

    // NOTE on layout: brotensor's conv2d expects weights as
    //   (C_out, C_in * kH * kW)  i.e. OIHW row-major flattened to 2D.
    // We use full conv (no `groups`) since brotensor has no depthwise
    // dispatch yet (see header notes). Depthwise behaviour is recovered at
    // train time by structurally zeroing the off-diagonal channels of dw_w
    // before each forward, but for the swap-in scaffolding (zero init) the
    // distinction is moot.
    static constexpr int kNumLayers = 3;
    brotensor::GpuTensor dw_w[kNumLayers];   // (C, C*3*3) FP16
    brotensor::GpuTensor dw_b[kNumLayers];   // (C, 1)     FP16
    brotensor::GpuTensor pw_w[kNumLayers];   // (C, C)     FP16  (1x1, no bias)

    // Resize the parameter tensors for the given shape and leave contents
    // undefined. Call zero_init() afterwards if you want identity behaviour.
    void allocate(int channels_, int height_, int width_);

    // Zero every weight + bias. With this state the inner layer
    //     y += pw(silu(dwconv(y) + 0)) = pw(silu(0)) = pw(0) = 0
    // so the whole student forward is identity (y == x), and the host
    // transformer block's residual passes through unchanged on the self-attn
    // sub-step (skipping self-attention entirely, which is the desired
    // pre-training behaviour — it leaves the cross-attn + FF intact and
    // gives us a clean signal that the swap-in is wired up).
    void zero_init();

    // Load layer weights from safetensors under
    //     <prefix>dw.<i>.weight   (C, 1, 3, 3) or (C, C, 3, 3)  FP16/FP32
    //     <prefix>dw.<i>.bias     (C,)                          FP16/FP32
    //     <prefix>pw.<i>.weight   (C, C, 1, 1)                  FP16/FP32
    // If any tensor is missing the call no-ops on that tensor (leaves the
    // current value, which is zero after zero_init) — this lets us run with
    // the flag on before training has produced weights.
    void load_from_safetensors(const brodiffusion::safetensors::File& f,
                               const std::string& prefix);

    // Forward. x_nchw, y_nchw, scratch are all (1, C*H*W) FP16 in NCHW order.
    // `scratch` is used as the post-dwconv buffer to avoid per-call allocs.
    // y_nchw is resized as needed. x_nchw is *not* modified.
    void forward(const brotensor::GpuTensor& x_nchw,
                 brotensor::GpuTensor& y_nchw,
                 brotensor::GpuTensor& scratch) const;
};

}  // namespace brodiffusion::student
