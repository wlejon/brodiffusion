// The CUDA-graph denoising-step seam: warm-up, capture, and replay of the
// denoiser body. Moved verbatim out of pipeline.cpp — see pipeline_detail.h
// for the file map and for StepGraphSession's keying rules.

#include "brodiffusion/pipeline.h"

#include "pipeline_detail.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"

#ifdef BROTENSOR_HAS_CUDA
#include "brotensor/cuda_graph.h"
#endif

#include <memory>

namespace brodiffusion::pipeline {

namespace bt = ::brotensor;

void Pipeline::step_denoise_captured_(PipelineState& state, float t,
                                      bool do_cfg) {
#ifdef BROTENSOR_HAS_CUDA
    StepGraphSession* s = step_graph_.get();
    const bool key_match =
        s != nullptr &&
        s->state_id == state.id &&
        s->H == state.H_lat && s->W == state.W_lat &&
        s->do_cfg == do_cfg;
    if (!key_match) {
        step_graph_ = std::make_unique<StepGraphSession>();
        s = step_graph_.get();
        s->state_id = state.id;
        s->H        = state.H_lat;
        s->W        = state.W_lat;
        s->do_cfg   = do_cfg;
    }

    // Host-dependent per-step inputs (time-embedding chain) — always eager,
    // writes the persistent temb buffers the captured body reads.
    denoiser_->prepare_step(t, *state.prepared);

    if (s->captured) {
        s->graph.launch();                       // cond branch
        if (do_cfg) s->graph_uncond.launch();    // uncond branch
        return;
    }

    // Eager warm-up step through the capture seam: computes this step's real
    // outputs and settles every body buffer at its high-water capacity.
    denoiser_->forward_body(state.latent, state.H_lat, state.W_lat,
                            *state.prepared, Branch::Cond, noise_pred_cond_);
    if (do_cfg) {
        denoiser_->forward_body(state.latent, state.H_lat, state.W_lat,
                                *state.prepared, Branch::Uncond,
                                noise_pred_uncond_);
    }
    ++s->eager_steps;

    // Capture only once every buffer-role assignment the captured calls can
    // start from has already been warmed. The UNet body ping-pongs a pool of
    // three buffers (x_/y_/cat_buf_) via std::swap, so the per-call role
    // permutation has period at most 3 — after 3 eager body calls, any
    // subsequent call repeats an already-warmed assignment and performs no
    // reallocation (an alloc/free of non-graph memory mid-capture is illegal
    // and poisons the graph).
    const int body_calls = s->eager_steps * (do_cfg ? 2 : 1);
    if (body_calls < 3) return;

    // Capture at the end of the second eager step: re-issue each body on the
    // capture stream into its OWN graph. Capture records the op sequence without
    // executing it (the eager outputs above stand for this step); every later
    // step replays each branch with one cudaGraphLaunch. Each CFG branch gets its
    // own graph (the bodies share all scratch buffers).
    {
        bt::sync_all();
        bt::CudaGraphCapture cap;
        denoiser_->forward_body(state.latent, state.H_lat, state.W_lat,
                                *state.prepared, Branch::Cond,
                                noise_pred_cond_);
        s->graph = cap.finish();
    }
    if (do_cfg) {
        bt::sync_all();
        bt::CudaGraphCapture cap;
        denoiser_->forward_body(state.latent, state.H_lat, state.W_lat,
                                *state.prepared, Branch::Uncond,
                                noise_pred_uncond_);
        s->graph_uncond = cap.finish();
    }
    s->captured = true;
#else
    // No CUDA backend in this build: plain eager forwards.
    denoiser_->forward(state.latent, state.H_lat, state.W_lat, t,
                       *state.prepared, Branch::Cond, noise_pred_cond_);
    if (do_cfg) {
        denoiser_->forward(state.latent, state.H_lat, state.W_lat, t,
                           *state.prepared, Branch::Uncond,
                           noise_pred_uncond_);
    }
#endif
}

}  // namespace brodiffusion::pipeline
