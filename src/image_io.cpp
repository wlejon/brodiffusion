#include "brodiffusion/image_io.h"
#include "brodiffusion/detail/compute.h"

#include "broimage/buffer.h"
#include "broimage/color.h"
#include "broimage/decode.h"
#include "broimage/geometric.h"
#include "broimage/preproc.h"

#include "brotensor/tensor.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion {

brotensor::Tensor load_image_as_latent_input(const std::string& path,
                                             int dst_w, int dst_h,
                                             PixelRange range) {
    if (dst_w <= 0 || dst_h <= 0) {
        throw std::runtime_error("load_image_as_latent_input: dst_w/dst_h must be positive");
    }
    if (dst_w % 8 != 0 || dst_h % 8 != 0) {
        throw std::runtime_error("load_image_as_latent_input: dst_w/dst_h must be multiples of 8");
    }

    broimage::Image img;
    std::string err;
    if (!broimage::decode_file(path, img, &err)) {
        throw std::runtime_error("load_image_as_latent_input: decode failed for '" +
                                 path + "': " + err);
    }
    if (img.width <= 0 || img.height <= 0) {
        throw std::runtime_error("load_image_as_latent_input: empty image '" + path + "'");
    }

    // 1. RGBA8 -> RGB8 (drop alpha). decode_file always emits 4 channels.
    const int src_w = img.width;
    const int src_h = img.height;
    const int pixel_count = src_w * src_h;
    std::vector<uint8_t> rgb(static_cast<std::size_t>(pixel_count) * 3);
    broimage::rgba_to_rgb_u8(img.pixels.data(), rgb.data(), pixel_count);

    // 2. Resize RGB HWC u8 -> (dst_w, dst_h, 3) HWC u8, bilinear.
    std::vector<uint8_t> rgb_resized(
        static_cast<std::size_t>(dst_w) * static_cast<std::size_t>(dst_h) * 3);
    broimage::resize_hwc_u8(rgb.data(), src_w, src_h, /*channels=*/3,
                            rgb_resized.data(), dst_w, dst_h,
                            broimage::Filter::Bilinear,
                            /*src_stride_bytes=*/0, /*dst_stride_bytes=*/0);

    // 3. u8 NHWC -> f32 NCHW, scaled to either [-1, 1] (VAE input) or [0, 1]
    //    (ControlNet conditioning input — diffusers does just /255, no bias).
    const float scale = (range == PixelRange::UnsignedUnit) ? 1.0f / 255.0f
                                                            : 2.0f / 255.0f;
    const float bias  = (range == PixelRange::UnsignedUnit) ? 0.0f : -1.0f;
    std::vector<float> f32(
        static_cast<std::size_t>(dst_w) * static_cast<std::size_t>(dst_h) * 3);
    broimage::u8_nhwc_to_f32_nchw(rgb_resized.data(),
                                  /*N=*/1, /*H=*/dst_h, /*W=*/dst_w, /*C=*/3,
                                  scale, bias, f32.data());

    // 4. Upload at the pipeline compute dtype.
    return detail::upload_host(f32.data(),
                               /*rows=*/1,
                               /*cols=*/3 * dst_h * dst_w);
}

brotensor::Tensor load_mask_as_latent(const std::string& path,
                                      int H_lat, int W_lat,
                                      uint8_t threshold) {
    if (H_lat <= 0 || W_lat <= 0) {
        throw std::runtime_error("load_mask_as_latent: H_lat/W_lat must be positive");
    }

    broimage::Image img;
    std::string err;
    if (!broimage::decode_file(path, img, &err)) {
        throw std::runtime_error("load_mask_as_latent: decode failed for '" +
                                 path + "': " + err);
    }
    if (img.width <= 0 || img.height <= 0) {
        throw std::runtime_error("load_mask_as_latent: empty image '" + path + "'");
    }

    const int src_w = img.width;
    const int src_h = img.height;
    const int pixel_count = src_w * src_h;

    // broimage::decode_file always emits RGBA8. Strip alpha, then take
    // luminance to get a single channel — robust to colored masks (a
    // grayscale white-on-black mask hits the same answer either way).
    std::vector<uint8_t> rgb(static_cast<std::size_t>(pixel_count) * 3);
    broimage::rgba_to_rgb_u8(img.pixels.data(), rgb.data(), pixel_count);
    std::vector<uint8_t> gray(static_cast<std::size_t>(pixel_count));
    broimage::rgb_to_gray_u8(rgb.data(), gray.data(), pixel_count);

    // Resize to LATENT dims using NEAREST so the binary mask boundary stays
    // sharp; bilinear would smear it into an in-between gradient.
    std::vector<uint8_t> gray_resized(
        static_cast<std::size_t>(W_lat) * static_cast<std::size_t>(H_lat));
    broimage::resize_hwc_u8(gray.data(), src_w, src_h, /*channels=*/1,
                            gray_resized.data(), W_lat, H_lat,
                            broimage::Filter::Nearest,
                            /*src_stride_bytes=*/0, /*dst_stride_bytes=*/0);

    // Threshold to {0.0, 1.0} FP32.
    const std::size_t n_lat =
        static_cast<std::size_t>(H_lat) * static_cast<std::size_t>(W_lat);
    std::vector<float> mask_f32(n_lat);
    for (std::size_t i = 0; i < n_lat; ++i) {
        mask_f32[i] = (gray_resized[i] >= threshold) ? 1.0f : 0.0f;
    }

    return detail::upload_host(mask_f32.data(),
                               /*rows=*/1,
                               /*cols=*/H_lat * W_lat);
}

}  // namespace brodiffusion
