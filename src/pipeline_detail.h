#pragma once

// Internals shared by the Pipeline translation units.
//
// pipeline.cpp holds the model-agnostic generation machinery — prime(),
// step_once(), the scheduler loop. Everything family-specific lives beside it:
//
//   pipeline_load.cpp        from_model_dir + the load_weights overloads + LoRA
//   pipeline_controlnet.cpp  the ControlNet registry
//   pipeline_sd.cpp          SD1.5's img2img / inpaint priming
//   pipeline_sana.cpp        Sana priming, the SCM step, the identity anchor
//   pipeline_step_graph.cpp  the CUDA-graph denoising-step session
//   pipeline_decode.cpp      VAE routing and the generate() convenience loop
//   pipeline_krea2.cpp       Krea 2 loading + research hooks
//   pipeline_qwenimage21.cpp          Qwen-Image 2.1 loading
//   pipeline_qwenimage21_edit.cpp     Qwen-Image 2.1 image-conditioned priming
//   pipeline_qwenimage21_hooks.cpp    Qwen-Image 2.1 research hooks
//   pipeline_qwenimage21_prefix.cpp   Qwen-Image 2.1 prefix cache + prompt memo
//
// These were the anonymous-namespace helpers of a single 2000-line
// pipeline.cpp; splitting that file is what made them need a home of their
// own. Nothing here is public API — the header is not installed.

#include "brodiffusion/denoiser.h"
#include "brodiffusion/dit/qwenimage21.h"
#include "brodiffusion/dpm_solver.h"
#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/lcm_scheduler.h"
#include "brodiffusion/pipeline.h"
#include "brodiffusion/scheduler.h"
#include "brodiffusion/scm_scheduler.h"
#include "brodiffusion/vae.h"

#include "brotensor/tensor.h"

#ifdef BROTENSOR_HAS_CUDA
#include "brotensor/cuda_graph.h"
#endif

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

namespace brodiffusion::pipeline {

namespace detail_pipe {

[[noreturn]] inline void fail(const std::string& msg) {
    throw std::runtime_error("pipeline::Pipeline: " + msg);
}

// The Qwen-Image 2.1 denoiser / transformer behind the Denoiser interface, or
// a throw naming the caller. Shared by the two qi21 translation units: the
// research hooks in pipeline_qwenimage21_hooks.cpp and the prefix-cache
// surface in pipeline_qwenimage21_prefix.cpp.
inline dit::QwenImage21Denoiser& qi21_denoiser(
    ModelClass model_class, const std::unique_ptr<Denoiser>& d,
    const char* who) {
    if (model_class != ModelClass::QwenImage21) {
        fail(std::string(who) + ": Qwen-Image 2.1 only");
    }
    auto* den = dynamic_cast<dit::QwenImage21Denoiser*>(d.get());
    if (!den) fail(std::string(who) + ": no Qwen-Image 2.1 denoiser");
    return *den;
}

inline dit::QwenImage21Transformer2DModel& qi21_model(
    ModelClass model_class, const std::unique_ptr<Denoiser>& d,
    const char* who) {
    return qi21_denoiser(model_class, d, who).model();
}

using SchedulerVariant =
    std::variant<scheduler::DDIM, scheduler::LCM, scheduler::FlowMatch,
                 scheduler::SCM, scheduler::DPMSolverMultistep>;

// Construct the scheduler variant from the matching config variant.
SchedulerVariant
make_scheduler(const std::variant<scheduler::DDIMConfig, scheduler::LCMConfig,
                                  scheduler::FlowMatchConfig,
                                  scheduler::SCMConfig,
                                  scheduler::DPMSolverConfig>& v);

// Fetch the timestep at step `i` as a float, uniformly across schedulers
// (DDIM / LCM return int timesteps; FlowMatch returns continuous floats).
float timestep_at(const SchedulerVariant& v, int i);

// Fill `out` (allocated to (1, n) at compute_dtype() on the default device)
// with n N(0,1) draws from the Philox stream keyed by (key, counter). When
// compute_dtype() is FP16 the draws are made into an FP32 scratch tensor and
// cast — brotensor::randn is FP32-only by design. `out` is always
// reassigned via Tensor::empty so a default-constructed (= CPU) input lands
// on the active default device.
void randn_compute(std::uint64_t key, std::uint64_t counter, int n,
                   brotensor::Tensor& out);

// Run one scheduler step on `state.latent`. LCM resamples per-step Gaussian
// noise from the state's Philox stream (key = state.rng_key, counter offset
// past the initial latent + all earlier steps); FlowMatch / DDIM are
// deterministic.
void scheduler_step(SchedulerVariant& sched, const brotensor::Tensor& pred,
                    int step_index, PipelineState& state, int n_lat,
                    brotensor::Tensor& scratch,
                    brotensor::Tensor& noise_step);

// Build an EncoderConfig from a DecoderConfig. The two structs share fields
// by design — both describe the same SD1.5 AutoencoderKL with mirrored
// topology. Used by Pipeline's encoder member so callers don't have to
// duplicate channel counts in PipelineConfig.
vae::EncoderConfig encoder_config_from_decoder(const vae::DecoderConfig& d);

// Derive the encoder-weight prefix from a decoder-weight prefix by stripping
// a trailing "decoder." and appending "encoder.". This keeps the existing
// (text, unet, vae) 3-prefix load_weights signature working for img2img:
// callers don't pass an explicit encoder prefix because in every diffusers /
// CompVis layout we've seen, the encoder is a sibling subtree of the decoder
// under the same VAE parent.
std::string encoder_prefix_from_decoder(const std::string& vae_prefix);

// img2img t_start: how many steps to skip at the front of the schedule.
//   init_timestep = max(1, floor(n_steps * strength))   (clamped to 1..n_steps)
//   t_start       = n_steps - init_timestep             (clamped to 0..n_steps-1)
// strength=1.0 -> t_start=0 (full schedule); strength near 0 -> t_start near
// n_steps - 1 (almost no denoising). Mirrors diffusers'
// StableDiffusionImg2ImgPipeline.get_timesteps.
int img2img_t_start(int n_steps, float strength);

// Escape hatch: set BRODIFFUSION_DISABLE_STEP_GRAPH=1 to force eager
// denoiser stepping even when the CUDA-graph session would be eligible.
// Read per step so tests can flip it between generations in-process.
bool step_graph_disabled();

// Construct the denoiser matching the model class.
std::unique_ptr<Denoiser> make_denoiser(const PipelineConfig& cfg);

// Is the classifier-free-guidance branch live for this generation?
//
// Every model but Qwen-Image 2.1 reads `guidance_scale` as diffusers'
// `guidance_scale`: 1.0 is the no-op value and anything else (including the
// below-1 "negative guidance" some callers use) runs both branches.
// Qwen-Image 2.1 follows the reference pipeline's `true_cfg_scale`, whose
// gate is `> 1` — its default is 1.0, meaning a plain single-branch sample,
// and a value below 1 is not a request for an uncond pass. prime() and
// step_once() must agree here or a step would combine against an uncond
// prediction prime() never encoded.
bool cfg_branch_active(ModelClass model_class, const Denoiser& denoiser,
                       bool is_lcm, float guidance_scale);

}  // namespace detail_pipe

// CUDA-graph denoising-step session. One per (state identity, H, W, CFG mode).
//
// The identity is PipelineState::id — a counter, NOT the address of the latent
// or of the prepared payload. Both of those are freed and re-allocated at the
// same size by the next generate(), so the allocator hands back the same address
// and an address-keyed session silently matches ACROSS generations: the graph
// captured for one image is replayed for the next, with the previous
// generation's buffer pointers baked into it. It works right up until the
// allocator state shifts (another model allocating between generations is
// enough), one buffer lands somewhere new, and the replay reads a freed pointer.
// The loud version of that is CUDA error 700; the quiet version is an image
// denoised against a stale conditioning buffer, with no error at all.
// `eager_steps` counts the warm-up steps run through the capture seam for
// this key; the second one settles every U-Net scratch buffer at its
// high-water capacity (the x_/y_ ping-pong roles permute per body call, so
// one step is not enough), after which the body's pointer set is stable and
// the capture at the end of that step records a replayable graph.
//
// Defined here rather than in pipeline_step_graph.cpp because ~Pipeline
// destroys the unique_ptr that holds it and lives in pipeline.cpp.
struct Pipeline::StepGraphSession {
#ifdef BROTENSOR_HAS_CUDA
    brotensor::CudaGraph graph;          // cond branch (also the single graph when no CFG)
    brotensor::CudaGraph graph_uncond;   // uncond branch — a SEPARATE graph under CFG
#endif
    std::uint64_t state_id = 0;   // PipelineState::id — never a recycled address
    int  H = 0, W = 0;
    bool do_cfg = false;
    int  eager_steps = 0;
    bool captured = false;   // warm-up done, graph(s) instantiated
};

}  // namespace brodiffusion::pipeline
