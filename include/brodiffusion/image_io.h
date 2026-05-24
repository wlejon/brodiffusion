#pragma once

// Phase A.3: image-in preprocessor.
//
// Decode an image file (PNG / JPEG / ...) and produce a (1, 3 * H * W) NCHW
// tensor in [-1, 1] on the active brotensor device at the pipeline compute
// dtype. The pixel pipeline is:
//   broimage::decode_file        (RGBA8 host buffer)
//   broimage::rgba_to_rgb_u8     (drop alpha)
//   broimage::resize_hwc_u8      (bilinear) to (dst_w, dst_h)
//   broimage::u8_nhwc_to_f32_nchw (scale = 2/255, bias = -1)
//   upload to compute dtype on the default device.
//
// This is a Phase A building block — not yet wired into Pipeline. img2img /
// inpaint / ControlNet in later phases will call into this seam.

#include "brotensor/tensor.h"

#include <string>

namespace brodiffusion {

// dst_h and dst_w must be positive multiples of 8 (matches the VAE 8x stride).
// Throws std::runtime_error on decode failure or invalid dims.
brotensor::Tensor load_image_as_latent_input(const std::string& path,
                                             int dst_w, int dst_h);

}  // namespace brodiffusion
