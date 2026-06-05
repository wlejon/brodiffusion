#include "brodiffusion/triposplat/sampler.h"

#include "brodiffusion/triposplat/flow_model.h"
#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace brodiffusion::triposplat {

namespace bt = ::brotensor;

namespace {

std::vector<float> download_f32(const bt::Tensor& t) {
    if (t.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits = t.to_host_vector_fp16();
        std::vector<float> out(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) out[i] = bt::fp16_bits_to_fp32(bits[i]);
        return out;
    }
    return t.to_host_vector();
}

bt::Tensor upload_like(const float* src, int rows, int cols, const bt::Tensor& ref) {
    if (ref.dtype == bt::Dtype::FP16) {
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(rows) * cols);
        for (std::size_t i = 0; i < bits.size(); ++i) bits[i] = bt::fp32_to_fp16_bits(src[i]);
        return bt::Tensor::from_host_fp16_on(ref.device, bits.data(), rows, cols);
    }
    return bt::Tensor::from_host_on(ref.device, src, rows, cols);
}

// CFG blend: v_cond <- s*v_cond - (s-1)*v_unc, computed in FP32 then returned at
// the compute dtype. On the FP16 path the two velocities are both large
// (~guidance * |v|), so the combine catastrophically cancels in FP16 — doing
// the tiny (L, C) elementwise blend in FP32 keeps the velocity precise before
// the latent step (matches the all-FP32 reference). A device fp32-combine
// kernel could replace this host round-trip later.
void cfg_blend(bt::Tensor& v_cond, const bt::Tensor& v_unc, float s) {
    bt::sync_all();
    std::vector<float> c = download_f32(v_cond);
    std::vector<float> u = download_f32(v_unc);
    const float a = s, b = -(s - 1.0f);
    for (std::size_t i = 0; i < c.size(); ++i) c[i] = a * c[i] + b * u[i];
    v_cond = upload_like(c.data(), v_cond.rows, v_cond.cols, v_cond);
}

}  // namespace

void sample_latent(FlowDiT& flow,
                   const bt::Tensor& feature1,
                   const bt::Tensor& feature2,
                   const bt::Tensor& noise_latent,
                   const bt::Tensor& noise_camera,
                   const FlowSampleOptions& opts,
                   bt::Tensor& out_latent) {
    if (opts.steps <= 0) throw std::runtime_error("triposplat::sample_latent: steps must be positive");

    const bool cfg = opts.guidance_scale > 1.0f;

    // Zeroed image features for the unconditional pass.
    bt::Tensor zero1, zero2;
    if (cfg) {
        zero1 = bt::Tensor::zeros_on(feature1.device, feature1.rows, feature1.cols, feature1.dtype);
        zero2 = bt::Tensor::zeros_on(feature2.device, feature2.rows, feature2.cols, feature2.dtype);
    }

    // Rectified-flow schedule (identical to TripoSplat's t_seq).
    scheduler::FlowMatchConfig sc;
    sc.shift = opts.shift;
    scheduler::FlowMatch sched(sc);
    sched.set_timesteps(opts.steps);

    bt::Tensor latent = noise_latent.clone();
    bt::Tensor camera = noise_camera.clone();

    bt::Tensor v_lat, v_cam, v_lat_u, v_cam_u, scratch_l, scratch_c;
    for (int i = 0; i < opts.steps; ++i) {
        const float t = sched.timesteps()[static_cast<std::size_t>(i)];   // sigma*1000
        flow.forward(latent, camera, feature1, feature2, t, v_lat, v_cam);
        if (cfg) {
            flow.forward(latent, camera, zero1, zero2, t, v_lat_u, v_cam_u);
            cfg_blend(v_lat, v_lat_u, opts.guidance_scale);
            cfg_blend(v_cam, v_cam_u, opts.guidance_scale);
        }
        sched.step(v_lat, i, latent, scratch_l);
        sched.step(v_cam, i, camera, scratch_c);
    }

    out_latent = std::move(latent);
}

}  // namespace brodiffusion::triposplat
