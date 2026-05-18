#pragma once

// SD1.5 UNet "inlet" — distillation-friendly drop-in replacement for the
// teacher down path. Produces the 12 skip tensors consumed (LIFO) by the
// teacher up path, plus the mid-block input. Inference scaffolding only;
// weights are loaded from a separately trained safetensors file (or
// zero-initialised for ceiling benches).
//
// Architecture (Candidate B): conv_in (4->320, 3x3) + a chain of
// depthwise-separable convs with FiLM(t_emb) conditioning, one cross-attn
// block at 8x8, and 12 tap heads (1x1 conv) projecting to each skip width.
//
// All FP16. N=1.

#include "brotensor/tensor.h"

#include <array>
#include <string>

namespace brodiffusion::safetensors { class File; }

namespace brodiffusion::inlet {

struct Inlet {
    // ─── Weights ───────────────────────────────────────────────────────────
    // Stem: conv 3x3, in_channels=4 -> 320.
    brotensor::GpuTensor stem_w, stem_b;

    // Depthwise-separable conv pair:
    //   dw_w : (C_in,  3*3)   — depthwise 3x3, groups=C_in, padding=1
    //   dw_b : (C_in, 1)
    //   pw_w : (C_out, C_in*1*1) — pointwise 1x1
    //   pw_b : (C_out, 1)
    struct DWSWeights {
        brotensor::GpuTensor dw_w, dw_b, pw_w, pw_b;
    };

    // 8 FiLM-conditioned stride-1 DWS layers:
    // [0] 320->320 @64, [1] 320->320 @64, [2] 320->640 @32, [3] 640->640 @32,
    // [4] 640->1280 @16, [5] 1280->1280 @16, [6] 1280->1280 @8, [7] 1280->1280 @8.
    std::array<DWSWeights, 8> dws_film;

    // 3 stride-2 DWS layers (no FiLM): downto32 (320->320), downto16 (640->640),
    // downto8 (1280->1280).
    std::array<DWSWeights, 3> dws_down;

    // FiLM Linear (1280 -> 2*C_out), one per FiLM-conditioned stage.
    // film_w[i] : (2*C_out_i, 1280), film_b[i] : (2*C_out_i, 1).
    std::array<brotensor::GpuTensor, 8> film_w;
    std::array<brotensor::GpuTensor, 8> film_b;

    // Cross-attention at 8x8: pre-LN(1280), Q/K/V/O projections (+biases).
    // Q layout: (Lq=64, D=1280). K/V from ctx (Lk=77, D_ctx=768) projected
    // to D=1280; 8 heads, head_dim=160.
    brotensor::GpuTensor xattn_ln_g, xattn_ln_b;     // (1280, 1)
    brotensor::GpuTensor xattn_wq,   xattn_bq;       // Wq: (1280, 1280)
    brotensor::GpuTensor xattn_wk,   xattn_bk;       // Wk: (1280, 768)
    brotensor::GpuTensor xattn_wv,   xattn_bv;       // Wv: (1280, 768)
    brotensor::GpuTensor xattn_wo,   xattn_bo;       // Wo: (1280, 1280)

    // 12 tap heads (1x1 conv, C->C with bias). Channel widths in tap order:
    //   320,320,320,320, 640,640,640, 1280,1280,1280,1280,1280.
    std::array<brotensor::GpuTensor, 12> tap_w;      // (C_i, C_i*1*1)
    std::array<brotensor::GpuTensor, 12> tap_b;      // (C_i, 1)

    // ─── API ───────────────────────────────────────────────────────────────

    // Allocate all weight tensors at their fixed shapes (FP16). Contents
    // undefined — call zero_init() or load_from_safetensors() before use.
    void allocate();

    // Zero every weight tensor in-place. Produces a forward pass that emits
    // zeros at every tap (broken images, but useful for ceiling benches).
    void zero_init();

    // Load weights from a safetensors file. Names follow:
    //   <prefix>stem.weight, <prefix>stem.bias,
    //   <prefix>dws_film.{0..7}.{dw_w,dw_b,pw_w,pw_b},
    //   <prefix>dws_down.{0..2}.{dw_w,dw_b,pw_w,pw_b},
    //   <prefix>film.{0..7}.{weight,bias},
    //   <prefix>xattn.{ln_g,ln_b,wq,bq,wk,bk,wv,bv,wo,bo},
    //   <prefix>tap.{0..11}.{weight,bias}.
    void load_from_safetensors(const safetensors::File& f,
                               const std::string& prefix);

    // Forward pass. Produces `skips[0..11]` in teacher push order:
    //   s0  (320, 64, 64), s1  (320, 64, 64), s2  (320, 64, 64),
    //   s3  (320, 32, 32), s4  (640, 32, 32), s5  (640, 32, 32),
    //   s6  (640, 16, 16), s7  (1280, 16, 16), s8  (1280, 16, 16),
    //   s9  (1280, 8, 8),  s10 (1280, 8, 8),   s11 (1280, 8, 8).
    // `bottleneck` is the mid-block input (== skips[11] semantically; produced
    // as a separate buffer so the up-path interface stays clean).
    //   sample : (1, 4*64*64)   FP16   (4, 64, 64)
    //   t_emb  : (1, 1280)      FP16
    //   ctx    : (77, 768)      FP16
    void forward(const brotensor::GpuTensor& sample,
                 const brotensor::GpuTensor& t_emb,
                 const brotensor::GpuTensor& ctx,
                 std::array<brotensor::GpuTensor, 12>& skips,
                 brotensor::GpuTensor& bottleneck) const;
};

}  // namespace brodiffusion::inlet
