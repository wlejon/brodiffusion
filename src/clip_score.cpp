#include "brodiffusion/clip_score.h"
#include "brodiffusion/safetensors.h"
#include "brodiffusion/detail/device.h"

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

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("clip_score::CLIPScorer: " + msg);
}

void upload_fp16_checked(const st::TensorView& v, int rows, int cols,
                         bt::Tensor& dst, const char* name) {
    if (v.dtype != st::Dtype::F16 && v.dtype != st::Dtype::F32) {
        fail(std::string(name) + ": expected F16 or F32, got " +
             st::dtype_name(v.dtype));
    }
    int64_t expected = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    if (v.numel() != expected) {
        fail(std::string(name) + ": shape mismatch (expected " +
             std::to_string(rows) + "x" + std::to_string(cols) + ", got " +
             std::to_string(v.numel()) + " elements)");
    }
    st::upload_fp16(v, rows, cols, dst);
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

    upload_fp16_checked(need(f, prefix + "visual_projection.weight"),
                        P, Dv, visual_proj_W_, "visual_projection.weight");
    upload_fp16_checked(need(f, prefix + "text_projection.weight"),
                        P, Dt, text_proj_W_, "text_projection.weight");
}

void CLIPScorer::set_prompt(std::string_view prompt) {
    if (visual_proj_W_.size() == 0 || text_proj_W_.size() == 0) {
        fail("set_prompt: projections not loaded — call load_projections first");
    }

    auto ids = tok_.encode(prompt);
    const int L  = static_cast<int>(ids.size());
    const int Dt = text_enc_.config().hidden_dim;
    const int P  = cfg_.projection_dim;
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

    text_enc_.forward(ids.data(), text_hidden_);  // (L, Dt) FP16

    // Pool: copy row eos_idx into text_pooled_ (1, Dt).
    detail::resize_like(text_pooled_, 1, Dt, bt::Dtype::FP16, text_hidden_.device);
    bt::copy_d2d(text_hidden_, /*src_off=*/eos_idx * Dt,
                     text_pooled_,  /*dst_off=*/0,
                     /*count=*/Dt);

    // Project to shared space: (1, P) = text_pooled_ @ text_proj_W_.T
    bt::linear_forward_batched_fp16(text_proj_W_, /*bias=*/nullptr,
                                        text_pooled_, text_proj_);

    bt::sync_all();
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(P));
    text_proj_.copy_to_host_fp16(bits.data());

    text_feat_.assign(static_cast<std::size_t>(P), 0.0f);
    for (int i = 0; i < P; ++i) {
        text_feat_[static_cast<std::size_t>(i)] =
            bt::fp16_bits_to_fp32(bits[static_cast<std::size_t>(i)]);
    }
    l2_normalise_in_place(text_feat_);
}

float CLIPScorer::score(const std::vector<float>& image, int H, int W) {
    if (text_feat_.empty()) {
        fail("score: set_prompt was not called");
    }

    auto pixel_bits = preprocess_(image, H, W);
    const int S  = image_enc_.config().image_size;
    const int C  = image_enc_.config().in_channels;
    const int P  = cfg_.projection_dim;

    pixels_dev_ = brotensor::Tensor::from_host_fp16(pixel_bits.data(), 1, C * S * S);
    image_enc_.forward(pixels_dev_, img_cls_);

    // (1, P) = img_cls_ @ visual_proj_W_.T
    bt::linear_forward_batched_fp16(visual_proj_W_, /*bias=*/nullptr,
                                        img_cls_, img_proj_);

    bt::sync_all();
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(P));
    img_proj_.copy_to_host_fp16(bits.data());

    std::vector<float> img_feat(static_cast<std::size_t>(P), 0.0f);
    for (int i = 0; i < P; ++i) {
        img_feat[static_cast<std::size_t>(i)] =
            bt::fp16_bits_to_fp32(bits[static_cast<std::size_t>(i)]);
    }
    l2_normalise_in_place(img_feat);

    double dot = 0.0;
    for (int i = 0; i < P; ++i) {
        dot += static_cast<double>(img_feat[static_cast<std::size_t>(i)]) *
               static_cast<double>(text_feat_[static_cast<std::size_t>(i)]);
    }
    return static_cast<float>(dot);
}

std::vector<std::uint16_t> CLIPScorer::preprocess_(
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

    std::vector<std::uint16_t> out(static_cast<std::size_t>(C) * plane_out);

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

                out[c * plane_out + y * S + x] = bt::fp32_to_fp16_bits(v);
            }
        }
    }
    return out;
}

}  // namespace brodiffusion::clip_score
