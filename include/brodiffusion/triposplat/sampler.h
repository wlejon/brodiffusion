#pragma once
//
// TripoSplat — rectified-flow Euler CFG sampler for the flow DiT.
//
// Integrates the flow DiT (LatentSeqMMFlowModel) from noise to a clean latent
// the octree decoder consumes. The schedule is brodiffusion's existing
// FlowMatch scheduler — TripoSplat's `t_seq = shift*l/(1+(shift-1)*l)` over
// l = linspace(1, 0, steps+1) is bit-identical to FlowMatch's static-shift
// sigma schedule, and the model is evaluated at t = sigma*1000 (FlowMatch's
// `timesteps()`), so no bespoke schedule math is needed here.
//
// The flow DiT jointly attends the latent and a camera token, so both are
// integrated together (the decoder only reads the latent, but the latent's
// evolution depends on the camera through joint attention). Classifier-free
// guidance follows the reference (diffusers convention): guidance_scale <= 1
// runs only the conditional pass; > 1 blends
//   v = s*v_cond - (s-1)*v_uncond
// with the unconditional pass using zeroed image features.
//
// Noise is an input (not generated here) so the sampler is deterministic and
// golden-testable; the bro composition layer draws the seeded noise.

#include "brotensor/tensor.h"

#include <functional>

namespace brodiffusion::triposplat {

class FlowDiT;

struct FlowSampleOptions {
    int   steps          = 20;
    float guidance_scale = 3.0f;   // <= 1 disables CFG
    float shift          = 3.0f;   // FlowMatch static shift

    // Optional cooperative cancellation. Checked once per Euler step (before the
    // step's work). If it returns true, sample_latent throws a SampleCancelled
    // exception so a long reconstruction can be aborted between steps. Left
    // empty by default (no cancellation).
    std::function<bool()> should_cancel;
};

// Thrown by sample_latent when should_cancel() returns true. Distinct type so
// callers can tell a user-requested abort apart from a genuine failure.
struct SampleCancelled : std::exception {
    const char* what() const noexcept override { return "triposplat: sample cancelled"; }
};

// Run the sampler. feature1 (K, cond_channels=1280) and feature2
// (K, cond2_channels=128) are the image conditioning; noise_latent
// (L=q_token_length, in_channels=16) and noise_camera (1, cam_channels=5) are
// the seeded initial noise. All at the pipeline compute dtype. Writes the clean
// latent (L, 16) into out_latent. Caller syncs before reading.
void sample_latent(FlowDiT& flow,
                   const brotensor::Tensor& feature1,
                   const brotensor::Tensor& feature2,
                   const brotensor::Tensor& noise_latent,
                   const brotensor::Tensor& noise_camera,
                   const FlowSampleOptions& opts,
                   brotensor::Tensor& out_latent);

}  // namespace brodiffusion::triposplat
