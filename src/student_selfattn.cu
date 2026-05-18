// Self-attention student module — see student_selfattn.h for design.
//
// Implementation notes:
//   * True depthwise via brotensor conv2d_forward_gpu(..., groups=C). Filter
//     shape is (C, kH*kW) = (C, 9), one set of taps per channel.
//   * All weights and activations are FP16. scratch is the post-dwconv buffer
//     reused across the three inner layers.
//   * y_nchw is resized to match x_nchw (FP16, same numel) on entry.

#include "brodiffusion/student_selfattn.h"
#include "brodiffusion/safetensors.h"
#include "brodiffusion/fused_transformer.h"

#include "brotensor/ops.h"
#include "brotensor/tensor.h"

#include <stdexcept>
#include <string>

namespace brodiffusion::student {

namespace bt = ::brotensor;
namespace st = ::brodiffusion::safetensors;

namespace {

void try_upload_fp16(const st::File& f, const std::string& key,
                     int rows, int cols, bt::GpuTensor& dst) {
    const st::TensorView* v = f.find(key);
    if (!v) return;  // missing → leave the existing (zero-initialised) value
    if (v->dtype != st::Dtype::F16 && v->dtype != st::Dtype::F32) {
        throw std::runtime_error("SelfAttnStudent: '" + key +
                                 "' has unsupported dtype " + st::dtype_name(v->dtype));
    }
    const int64_t expected = static_cast<int64_t>(rows) *
                             static_cast<int64_t>(cols);
    if (v->numel() != expected) {
        throw std::runtime_error("SelfAttnStudent: '" + key +
                                 "' shape mismatch (expected " +
                                 std::to_string(rows) + "x" + std::to_string(cols) +
                                 ", got " + std::to_string(v->numel()) + " elements)");
    }
    st::upload_fp16(*v, rows, cols, dst);
}

}  // namespace

void SelfAttnStudent::allocate(int channels_, int height_, int width_) {
    channels = channels_;
    height   = height_;
    width    = width_;
    const int C = channels;
    for (int i = 0; i < kNumLayers; ++i) {
        // Depthwise 3x3 (groups=C): one (kH*kW) tap-set per channel.
        dw_w[i].resize(C, 3 * 3, bt::Dtype::FP16);
        dw_b[i].resize(C, 1,     bt::Dtype::FP16);
        pw_w[i].resize(C, C,     bt::Dtype::FP16);
    }
}

void SelfAttnStudent::zero_init() {
    for (int i = 0; i < kNumLayers; ++i) {
        dw_w[i].zero();
        dw_b[i].zero();
        pw_w[i].zero();
    }
}

void SelfAttnStudent::load_from_safetensors(const st::File& f,
                                            const std::string& prefix) {
    const int C = channels;
    for (int i = 0; i < kNumLayers; ++i) {
        const std::string idx = std::to_string(i);
        try_upload_fp16(f, prefix + "dw." + idx + ".weight",
                        C, 3 * 3, dw_w[i]);
        try_upload_fp16(f, prefix + "dw." + idx + ".bias",
                        C, 1,         dw_b[i]);
        try_upload_fp16(f, prefix + "pw." + idx + ".weight",
                        C, C,         pw_w[i]);
    }
}

void SelfAttnStudent::forward(const bt::GpuTensor& x_nchw,
                              bt::GpuTensor& y_nchw,
                              bt::GpuTensor& scratch) const {
    const int C = channels;
    const int H = height;
    const int W = width;
    const int N_elems = C * H * W;
    if (x_nchw.size() != N_elems) {
        throw std::runtime_error(
            "SelfAttnStudent::forward: x_nchw size mismatch (expected " +
            std::to_string(N_elems) + ", got " + std::to_string(x_nchw.size()) + ")");
    }

    // y = x  (deep copy; we mutate y in-place across the 3 inner residuals).
    y_nchw = x_nchw.clone();

    // Persistent scratch buffer for the pw1x1 output (between dwconv→silu and
    // the residual add). Hidden inside the forward to mirror the thread_local
    // pattern used by other brodiffusion fused kernels.
    thread_local bt::GpuTensor pw_out;
    pw_out.resize(1, N_elems, bt::Dtype::FP16);

    for (int i = 0; i < kNumLayers; ++i) {
        // scratch = dwconv3x3(y) + dw_b   (depthwise via groups=C).
        bt::conv2d_forward_gpu(y_nchw, dw_w[i], &dw_b[i],
                               /*N=*/1, C, H, W,
                               /*C_out=*/C, /*kH=*/3, /*kW=*/3,
                               /*stride_h=*/1, /*stride_w=*/1,
                               /*pad_h=*/1, /*pad_w=*/1,
                               /*dil_h=*/1, /*dil_w=*/1,
                               /*groups=*/C,
                               scratch);
        // scratch = silu(scratch)  (in-place).
        bt::silu_forward_gpu(scratch, scratch);
        // pw_out = pw1x1(scratch)  — 1x1 conv, no bias; OIHW (C,C,1,1) → (C,C).
        bt::conv2d_forward_gpu(scratch, pw_w[i], /*bias=*/nullptr,
                               /*N=*/1, C, H, W,
                               /*C_out=*/C, /*kH=*/1, /*kW=*/1,
                               /*stride_h=*/1, /*stride_w=*/1,
                               /*pad_h=*/0, /*pad_w=*/0,
                               /*dil_h=*/1, /*dil_w=*/1,
                               pw_out);
        // y += pw_out  (vectorised FP16 elementwise add).
        brodiffusion::add_inplace_fp16_vec(y_nchw, pw_out);
    }
}

}  // namespace brodiffusion::student
