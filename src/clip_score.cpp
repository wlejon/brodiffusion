#include "brodiffusion/clip_score.h"
#include "brodiffusion/safetensors.h"
#include "brodiffusion/detail/device.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::clip_score {

namespace bt = ::brotensor;
namespace st = ::brodiffusion::safetensors;

// Validate-and-upload a safetensors weight view at the compute dtype.
using st::upload_compute_checked;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("clip_score::CLIPScorer: " + msg);
}

// Download a compute-dtype tensor's contents into a host FP32 vector,
// converting from FP16 bits on a GPU backend.
std::vector<float> download_compute(const bt::Tensor& t) {
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(t.size()));
        t.copy_to_host_fp16(bits.data());
        std::vector<float> out(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            out[i] = bt::fp16_bits_to_fp32(bits[i]);
        }
        return out;
    }
    return t.to_host_vector();
}

const st::TensorView& need(const st::File& f, const std::string& key) {
    const auto* v = f.find(key);
    if (!v) throw std::runtime_error("clip_score::CLIPScorer: missing tensor '" + key + "'");
    return *v;
}

float l2_normalise_in_place(std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * static_cast<double>(x);
    const float n = static_cast<float>(std::sqrt(s));
    if (n > 0.0f) {
        const float inv = 1.0f / n;
        for (float& x : v) x *= inv;
    }
    return n;
}

}  // namespace

CLIPScorer::CLIPScorer(const clip::Tokenizer& tokenizer,
                       clip::TextEncoder& text_encoder,
                       clip_image::ImageEncoder& image_encoder,
                       Config cfg)
    : tok_(tokenizer), text_enc_(text_encoder), image_enc_(image_encoder),
      cfg_(cfg) {}

void CLIPScorer::load_projections(const st::File& f, const std::string& prefix) {
    const int P  = cfg_.projection_dim;
    const int Dv = image_enc_.config().hidden_dim;
    const int Dt = text_enc_.config().hidden_dim;

    upload_compute_checked(need(f, prefix + "visual_projection.weight"),
                        P, Dv, visual_proj_W_, "visual_projection.weight");
    upload_compute_checked(need(f, prefix + "text_projection.weight"),
                        P, Dt, text_proj_W_, "text_projection.weight");
}

void CLIPScorer::set_prompt(std::string_view prompt) {
    if (visual_proj_W_.size() == 0 || text_proj_W_.size() == 0) {
        fail("set_prompt: projections not loaded — call load_projections first");
    }

    auto ids = tok_.encode(prompt);
    const int L  = static_cast<int>(ids.size());
    const int Dt = text_enc_.config().hidden_dim;
    if (L != text_enc_.config().max_position) {
        fail("set_prompt: tokenizer produced unexpected sequence length");
    }

    // CLIP pooling rule: take the EOS token's hidden state. CLIP pads with
    // EOS (49407), and BOS is 49406; argmax over IDs therefore lands on
    // the first EOS-padded position — i.e. the token *just after* the last
    // real content token. This matches the official CLIP pooling logic.
    int eos_idx = 0;
    int32_t best = -1;
    for (int i = 0; i < L; ++i) {
        if (ids[static_cast<std::size_t>(i)] > best) {
            best = ids[static_cast<std::size_t>(i)];
            eos_idx = i;
        }
    }

    text_enc_.forward(ids.data(), text_hidden_);  // (L, Dt) at compute dtype

    // Pool: copy row eos_idx into text_pooled_ (1, Dt).
    detail::resize_like(text_pooled_, 1, Dt, compute_dtype(), text_hidden_.device);
    bt::copy_d2d(text_hidden_, /*src_off=*/eos_idx * Dt,
                     text_pooled_,  /*dst_off=*/0,
                     /*count=*/Dt);

    // Project to shared space: (1, P) = text_pooled_ @ text_proj_W_.T
    detail::linear_batched(text_proj_W_, /*bias=*/nullptr,
                           text_pooled_, text_proj_);

    bt::sync_all();
    text_feat_ = download_compute(text_proj_);
    l2_normalise_in_place(text_feat_);
}

float CLIPScorer::score(const std::vector<float>& image, int H, int W) {
    if (text_feat_.empty()) {
        fail("score: set_prompt was not called");
    }

    auto pixel_vals = preprocess_(image, H, W);
    const int S  = image_enc_.config().image_size;
    const int C  = image_enc_.config().in_channels;
    const int P  = cfg_.projection_dim;

    pixels_dev_ = detail::upload_host(pixel_vals.data(), 1, C * S * S);
    image_enc_.forward(pixels_dev_, img_cls_);

    // (1, P) = img_cls_ @ visual_proj_W_.T
    detail::linear_batched(visual_proj_W_, /*bias=*/nullptr,
                           img_cls_, img_proj_);

    bt::sync_all();
    std::vector<float> img_feat = download_compute(img_proj_);
    l2_normalise_in_place(img_feat);

    double dot = 0.0;
    for (int i = 0; i < P; ++i) {
        dot += static_cast<double>(img_feat[static_cast<std::size_t>(i)]) *
               static_cast<double>(text_feat_[static_cast<std::size_t>(i)]);
    }
    return static_cast<float>(dot);
}

std::vector<float> CLIPScorer::preprocess_(
    const std::vector<float>& image, int H, int W) const {

    const int S = image_enc_.config().image_size;       // 224
    const int C = image_enc_.config().in_channels;      // 3
    if (static_cast<int>(image.size()) != C * H * W) {
        fail("preprocess: image size mismatch (expected " +
             std::to_string(C * H * W) + ", got " +
             std::to_string(image.size()) + ")");
    }

    // Bilinear resize per channel. We center each output pixel: src_x =
    // (x + 0.5) * (W / S) - 0.5. Standard "align_corners=False" rule.
    const float sx = static_cast<float>(W) / static_cast<float>(S);
    const float sy = static_cast<float>(H) / static_cast<float>(S);
    const std::size_t plane_in  = static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
    const std::size_t plane_out = static_cast<std::size_t>(S) * static_cast<std::size_t>(S);

    std::vector<float> out(static_cast<std::size_t>(C) * plane_out);

    for (int c = 0; c < C; ++c) {
        const float* in_plane = image.data() + c * plane_in;
        const float mean = cfg_.mean[c];
        const float std_ = cfg_.std_[c];
        const float inv_std = 1.0f / std_;
        for (int y = 0; y < S; ++y) {
            const float fy = (y + 0.5f) * sy - 0.5f;
            int y0 = static_cast<int>(std::floor(fy));
            int y1 = y0 + 1;
            float wy = fy - static_cast<float>(y0);
            y0 = std::clamp(y0, 0, H - 1);
            y1 = std::clamp(y1, 0, H - 1);
            for (int x = 0; x < S; ++x) {
                const float fx = (x + 0.5f) * sx - 0.5f;
                int x0 = static_cast<int>(std::floor(fx));
                int x1 = x0 + 1;
                float wx = fx - static_cast<float>(x0);
                x0 = std::clamp(x0, 0, W - 1);
                x1 = std::clamp(x1, 0, W - 1);

                const float v00 = in_plane[y0 * W + x0];
                const float v01 = in_plane[y0 * W + x1];
                const float v10 = in_plane[y1 * W + x0];
                const float v11 = in_plane[y1 * W + x1];
                const float v0  = v00 + wx * (v01 - v00);
                const float v1  = v10 + wx * (v11 - v10);
                float v         = v0  + wy * (v1  - v0 );

                // [-1, 1] -> [0, 1] -> CLIP-normalised.
                v = (v + 1.0f) * 0.5f;
                v = (v - mean) * inv_std;

                out[c * plane_out + y * S + x] = v;
            }
        }
    }
    return out;
}

}  // namespace brodiffusion::clip_score
