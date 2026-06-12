#include "brodiffusion/triposplat/sampler.h"

#include "brodiffusion/triposplat/flow_model.h"
#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/detail/compute.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#ifdef BROTENSOR_HAS_CUDA
#include "brotensor/cuda_graph.h"
#endif

#include <cstdlib>
#include <stdexcept>

namespace brodiffusion::triposplat {

namespace bt = ::brotensor;

namespace {

// Escape hatch: set BRODIFFUSION_DISABLE_STEP_GRAPH=1 to force eager stepping
// even when the CUDA-graph session would be eligible. Same env var as the
// Pipeline's denoising-step graph — one hatch disables every step graph. Read
// per call so tests can flip it between runs in-process.
bool step_graph_disabled() {
    const char* e = std::getenv("BRODIFFUSION_DISABLE_STEP_GRAPH");
    if (e != nullptr && e[0] != '\0' && e[0] != '0') return true;
    // The flow per-op profiler syncs inside forward_body, which is illegal
    // mid-capture — profiling implies eager steps.
    const char* p = std::getenv("BRODIFFUSION_FLOW_PROFILE");
    return p != nullptr && p[0] != '\0' && p[0] != '0';
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
    const float s = opts.guidance_scale;

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

    // All step buffers are function-locals declared before the loop, so their
    // storage persists across steps; capacity-aware resize keeps the pointers
    // stable after warm-up. The captured graph reads latent/camera and writes
    // v_lat/v_cam in place; sched.step updates latent/camera in place —
    // pointer-stable end to end. The CFG blend runs at FP32 internal precision
    // on-device: v = s*v_cond + (1-s)*v_uncond ≡ s*v_cond - (s-1)*v_uncond.
    bt::Tensor v_lat, v_cam, v_lat_u, v_cam_u, scratch_l, scratch_c;

#ifdef BROTENSOR_HAS_CUDA
    // Step-capture session, local to this call: the latent/camera/feature
    // buffers (and the model's scratch members) are fixed for the whole call,
    // so the session never needs a cross-call identity key.
    const bool capture_enabled =
        bt::default_device() == bt::Device::CUDA && !step_graph_disabled();
    bt::CudaGraph graph;
    int eager_steps = 0;
#endif

    for (int i = 0; i < opts.steps; ++i) {
        // Cooperative cancellation: bail out between steps if asked. Checked
        // before any of this step's work so an abort is honoured promptly.
        if (opts.should_cancel && opts.should_cancel()) throw SampleCancelled();

        const float t = sched.timesteps()[static_cast<std::size_t>(i)];   // sigma*1000

        // Host-dependent per-step head (time-embedding chain) — always eager,
        // writes the persistent t_emb_/t_mod_ buffers the captured body reads.
        flow.prepare_step(t);

#ifdef BROTENSOR_HAS_CUDA
        if (graph.valid()) {
            graph.launch();
        } else
#endif
        {
            // Eager warm-up step through the capture seam: computes this
            // step's real outputs and settles every body buffer at its
            // high-water capacity.
            flow.forward_body(latent, camera, feature1, feature2, v_lat, v_cam);
            if (cfg) {
                flow.forward_body(latent, camera, zero1, zero2, v_lat_u, v_cam_u);
                bt::axpby_inplace(v_lat, v_lat_u, s, 1.0f - s);
                bt::axpby_inplace(v_cam, v_cam_u, s, 1.0f - s);
            }
#ifdef BROTENSOR_HAS_CUDA
            ++eager_steps;
            // Capture only once every buffer-role assignment the captured
            // calls can start from has already been warmed (an alloc/free of
            // non-graph memory mid-capture is illegal and poisons the graph).
            // Here axpby reads and writes v_lat in place — there is no buffer-
            // role permutation anywhere in this loop (no std::swap), so the
            // established >= 3 body-call threshold from the SD1.5 step graph
            // (period-3 buffer ping-pong) is comfortably sufficient.
            const int body_calls = eager_steps * (cfg ? 2 : 1);
            if (capture_enabled && body_calls >= 3 && i + 1 < opts.steps) {
                // Capture at the end of an eager step: re-issue the identical
                // body sequence (incl. the in-place CFG blends) on the capture
                // stream. Capture records the op sequence without executing it
                // (the eager outputs above stand for this step); every later
                // step replays the whole sequence with one cudaGraphLaunch.
                bt::sync_all();
                bt::CudaGraphCapture cap;
                flow.forward_body(latent, camera, feature1, feature2, v_lat, v_cam);
                if (cfg) {
                    flow.forward_body(latent, camera, zero1, zero2, v_lat_u, v_cam_u);
                    bt::axpby_inplace(v_lat, v_lat_u, s, 1.0f - s);
                    bt::axpby_inplace(v_cam, v_cam_u, s, 1.0f - s);
                }
                graph = cap.finish();
            }
#endif
        }

        // Euler step — eager: the scheduler bakes the per-step d_sigma scalar
        // into kernel args, different each step, so it stays outside the graph.
        sched.step(v_lat, i, latent, scratch_l);
        sched.step(v_cam, i, camera, scratch_c);
    }

    out_latent = std::move(latent);
}

}  // namespace brodiffusion::triposplat
