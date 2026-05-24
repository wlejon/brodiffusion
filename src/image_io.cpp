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
                                             int dst_w, int dst_h) {
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

    // 3. u8 NHWC -> f32 NCHW, scale 2/255, bias -1  (maps [0,255] -> [-1,1]).
    std::vector<float> f32(
        static_cast<std::size_t>(dst_w) * static_cast<std::size_t>(dst_h) * 3);
    broimage::u8_nhwc_to_f32_nchw(rgb_resized.data(),
                                  /*N=*/1, /*H=*/dst_h, /*W=*/dst_w, /*C=*/3,
                                  /*scale=*/2.0f / 255.0f, /*bias=*/-1.0f,
                                  f32.data());

    // 4. Upload at the pipeline compute dtype.
    return detail::upload_host(f32.data(),
                               /*rows=*/1,
                               /*cols=*/3 * dst_h * dst_w);
}

}  // namespace brodiffusion
