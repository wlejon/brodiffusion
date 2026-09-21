// Sana-specific Pipeline bodies: DC-AE priming, the sCM (Sana-Sprint) step,
// and the reference-attention identity anchor. Moved verbatim out of
// pipeline.cpp — see pipeline_detail.h for the file map.

#include "brodiffusion/pipeline.h"

#include "pipeline_detail.h"

#include "brodiffusion/detail/device.h"
#include "brodiffusion/detail/torch_rng.h"
#include "brodiffusion/dit/sana.h"
#include "brodiffusion/sana_text.h"
#include "brodiffusion/scm_scheduler.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace brodiffusion::pipeline {

namespace bt = ::brotensor;

using namespace detail_pipe;

PipelineState Pipeline::prime_sana_(std::string_view prompt,
                                    const GenerateOptions& opts) {
    // Sana downsamples 32x (DC-AE f32c32), not the 8x of SD / Flux.
    if (opts.height <= 0 || opts.width <= 0 ||
        opts.height % 32 != 0 || opts.width % 32 != 0) {
        fail("Sana: height and width must be positive multiples of 32");
    }
    if (opts.num_inference_steps <= 0) {
        fail("num_inference_steps must be positive");
    }
    // img2img / inpaint / ControlNet are not wired for Sana yet (init_noise
    // and the txt2img RNG paths below are).
    if (!opts.init_image_path.empty() || !opts.mask_image_path.empty() ||
        !opts.controls.empty()) {
        fail("Sana: img2img / inpaint / ControlNet are not supported");
    }
    if (!gemma_model_ || !gemma_tokenizer_ || !dcae_) {
        fail("Sana: pipeline missing Gemma encoder / DC-AE decoder");
    }

    // Reset the SD-only run state so a previous non-Sana generation can't leak.
    inpaint_active_    = false;
    controlnet_active_ = false;
    control_inputs_.clear();
    control_images_.clear();

    const int H_lat = opts.height / 32;
    const int W_lat = opts.width  / 32;
    const int C_lat = denoiser_->latent_channels();      // 32
    const int n_lat = C_lat * H_lat * W_lat;
    // Base Sana is not guidance-distilled: a separate uncond branch + CFG
    // combine runs whenever guidance != 1.0 (uses_cfg() is true). Sana-Sprint
    // (SCMScheduler) is guidance-distilled — uses_cfg() is false, so do_cfg
    // stays false and the guidance scale is fed in as an embedding instead.
    const bool is_scm = std::holds_alternative<scheduler::SCM>(scheduler_);
    const bool do_cfg = denoiser_->uses_cfg() && (opts.guidance_scale != 1.0f);

    denoiser_->finalize_weights();

    // 1. Gemma-encode the positive prompt (and, under CFG, the negative). The
    //    Sana DiT cross-attends over exactly the returned valid token rows.
    //    The positive prompt is wrapped in Sana's Complex Human Instruction
    //    (diffusers' default) — the DiT was trained on CHI-prefixed conditioning,
    //    so omitting it yields soft, under-detailed images. The negative branch
    //    uses the plain prompt (no CHI), matching diffusers.
    conditioning_.text_embeddings = brodiffusion::sana::encode_prompt(
        *gemma_model_, *gemma_tokenizer_, std::string(prompt),
        cfg_.sana_max_seq_len,
        brodiffusion::sana::default_complex_human_instruction());
    // Conditioning-space control seam: add the weighted control axes to the
    // positive caption embeddings (rows [1, L), BOS untouched) before the DiT
    // projects them. No-op unless a dictionary is loaded with a nonzero weight.
    // The negative branch is left clean (steer the prompt, not the baseline).
    cond_control_.apply(conditioning_.text_embeddings);
    if (do_cfg) {
        conditioning_.uncond_embeddings = brodiffusion::sana::encode_prompt(
            *gemma_model_, *gemma_tokenizer_,
            std::string(opts.negative_prompt), cfg_.sana_max_seq_len,
            /*chi=*/{});
        conditioning_.has_uncond = true;
    } else {
        conditioning_.has_uncond = false;
        conditioning_.uncond_embeddings = bt::Tensor{};
    }
    // Sana-Sprint embeds the raw guidance scale (the denoiser applies
    // guidance_embeds_scale); base Sana uses CFG instead, so leaves it at 0.
    conditioning_.guidance = is_scm ? opts.guidance_scale : 0.0f;

    // 1b. Project the caption context once per generation (caption_projection
    //     + caption_norm), shared across all states branched from this prime.
    auto prepared = std::make_shared<PreparedConditioning>(
        denoiser_->prepare(conditioning_));

    // Reference-attention identity seam (Sana linear-attention only). When
    // capture_identity_anchor() is driving this run the denoiser is already in
    // Capture mode (left as-is). Otherwise, inject the cached anchor's summaries
    // when a positive weight is set, or disable the seam. Done once per
    // generation, before the stepping loop advances the per-branch step counter.
    {
        auto* sana = static_cast<dit::SanaDenoiser*>(denoiser_.get());
        if (capturing_anchor_) {
            // leave Capture mode as set by capture_identity_anchor()
        } else if (sana->ref_has_anchor() && identity_weight_ > 0.0f) {
            sana->ref_begin_inject(identity_weight_);
        } else {
            sana->ref_set_off();
        }
    }

    // 2. State shell + timestep schedule (rectified-flow FlowMatch, shift=3).
    PipelineState state;
    state.H_lat    = H_lat;
    state.W_lat    = W_lat;
    state.rng_key  = opts.seed;
    state.prepared = std::move(prepared);
    std::visit([&](auto& s) { s.set_timesteps(opts.num_inference_steps); },
               scheduler_);
    state.n_steps = std::visit(
        [](const auto& s) { return s.num_inference_steps(); }, scheduler_);
    const float sigma = std::visit(
        [](const auto& s) { return s.init_noise_sigma(); }, scheduler_);

    const bt::Dtype ldt = denoiser_->compute_dtype();
    if (!opts.init_noise.empty()) {
        if (static_cast<int>(opts.init_noise.size()) != n_lat) {
            fail("prime: init_noise has " +
                 std::to_string(opts.init_noise.size()) +
                 " values, expected " + std::to_string(n_lat));
        }
        std::vector<float> noise(static_cast<std::size_t>(n_lat));
        for (int i = 0; i < n_lat; ++i) noise[i] = sigma * opts.init_noise[i];
        bt::Tensor host_t =
            bt::Tensor::from_host(noise.data(), 1, n_lat).to(bt::default_device());
        if (host_t.dtype != ldt) {
            bt::cast(host_t, state.latent, ldt);
        } else {
            state.latent = std::move(host_t);
        }
    } else if (opts.noise_source == NoiseSource::Torch) {
        std::vector<float> noise =
            detail::torch_randn_f32(opts.seed, static_cast<std::size_t>(n_lat));
        if (sigma != 1.0f) for (int i = 0; i < n_lat; ++i) noise[i] *= sigma;
        bt::Tensor host_t =
            bt::Tensor::from_host(noise.data(), 1, n_lat).to(bt::default_device());
        if (host_t.dtype != ldt) {
            bt::cast(host_t, state.latent, ldt);
        } else {
            state.latent = std::move(host_t);
        }
    } else {
        if (ldt == bt::Dtype::FP32) {
            state.latent = bt::Tensor::empty(1, n_lat, bt::Dtype::FP32);
            bt::randn(opts.seed, 0, state.latent);
        } else {
            bt::Tensor f32 = bt::Tensor::empty(1, n_lat, bt::Dtype::FP32);
            bt::randn(opts.seed, 0, f32);
            bt::cast(f32, state.latent, ldt);
        }
        if (sigma != 1.0f) bt::scale_inplace(state.latent, sigma);
    }

    // Sana-Sprint works in sCM coordinates: the initial latent is scaled by
    // sigma_data (diffusers: `latents = latents * sigma_data`). The matching
    // 1/sigma_data is applied to the denoised latent after the final step.
    if (is_scm) {
        bt::scale_inplace(state.latent,
                          std::get<scheduler::SCM>(scheduler_).sigma_data());
    }

    state.step_index = 0;
    return state;
}

void Pipeline::step_once_scm_(PipelineState& state,
                              const GenerateOptions& opts) {
    (void)opts;  // guidance is baked into the prepared conditioning
    auto& sched = std::get<scheduler::SCM>(scheduler_);
    const int i      = state.step_index;
    const int n_lat  = denoiser_->latent_channels() *
                       state.H_lat * state.W_lat;
    const float sigma_data = sched.sigma_data();

    // sCM time transform. The schedule angle s maps to the network's input
    // timestep scm_t = sin(s)/(cos(s)+sin(s)); the latent is rescaled into the
    // network's input parameterisation by `scale`. (diffusers
    // SanaSprintPipeline.__call__ denoising loop.)
    const float s     = sched.timesteps()[i];
    const float scm_t = std::sin(s) / (std::cos(s) + std::sin(s));
    const float scale =
        std::sqrt(scm_t * scm_t + (1.0f - scm_t) * (1.0f - scm_t));

    // latent_model_input = (latent / sigma_data) * scale, into scratch_. This
    // tensor is both the DiT input and the `lmi` term in the output
    // reconstruction below, so it must survive the forward (which reads, never
    // writes, its latent argument).
    detail::resize_like(scratch_, 1, n_lat, state.latent.dtype,
                        state.latent.device);
    bt::copy_d2d(state.latent, 0, scratch_, 0, n_lat);
    bt::scale_inplace(scratch_, scale / sigma_data);

    // One DiT forward at the sCM input timestep. Guidance (already embedded via
    // the prepared conditioning) makes this a single, CFG-free pass.
    denoiser_->forward(scratch_, state.H_lat, state.W_lat, scm_t,
                       *state.prepared, Branch::Cond, noise_pred_cond_);

    // Reconstruct the scheduler's model_output from the network output:
    //   np = ((1 - 2t)·lmi + (1 - 2t + 2t²)·np) / scale · sigma_data
    const float a = 1.0f - 2.0f * scm_t;
    const float b = 1.0f - 2.0f * scm_t + 2.0f * scm_t * scm_t;
    bt::axpby_inplace(noise_pred_cond_, scratch_, /*a=*/b, /*b=*/a);
    bt::scale_inplace(noise_pred_cond_, sigma_data / scale);

    // Fresh unit-variance noise for the inter-step injection (the scheduler
    // applies sigma_data). FP32 to match Sana's FP32 latent. Counter offset
    // past the initial-latent draw (counters 0..n_lat), mirroring LCM.
    const std::uint64_t counter =
        static_cast<std::uint64_t>(1 + i) * static_cast<std::uint64_t>(n_lat);
    noise_step_ = bt::Tensor::empty(1, n_lat, bt::Dtype::FP32);
    bt::randn(state.rng_key, counter, noise_step_);

    sched.step(noise_pred_cond_, i, state.latent, noise_step_);
    ++state.step_index;

    // The final step returns the denoised latent in sCM coordinates; diffusers
    // decodes `denoised / sigma_data`. Undo the sigma_data scaling applied in
    // prime_sana_ so decode() sees a plain DC-AE latent.
    if (state.step_index >= state.n_steps) {
        bt::scale_inplace(state.latent, 1.0f / sigma_data);
    }
}

std::vector<float> Pipeline::capture_identity_anchor(std::string_view prompt,
                                                     const GenerateOptions& opts) {
    if (model_class_ != ModelClass::Sana) {
        fail("capture_identity_anchor: the identity anchor is Sana-only");
    }
    auto* sana = static_cast<dit::SanaDenoiser*>(denoiser_.get());
    // Record the anchor's per-(branch, step, block) linear-attention summaries
    // across one full denoise. generate() → prime_sana_() sees capturing_anchor_
    // and leaves the denoiser in Capture mode; the returned image is the anchor
    // itself (e.g. a neutral portrait), handed back so the caller can show it.
    sana->ref_begin_capture(opts.num_inference_steps);
    capturing_anchor_ = true;
    std::vector<float> img;
    try {
        img = generate(prompt, opts);
    } catch (...) {
        capturing_anchor_ = false;
        sana->ref_clear();
        throw;
    }
    capturing_anchor_ = false;
    sana->ref_mark_anchor();
    identity_anchor_ = true;
    return img;
}

void Pipeline::set_identity_weight(float weight) { identity_weight_ = weight; }

float Pipeline::identity_weight() const { return identity_weight_; }

bool Pipeline::has_identity_anchor() const { return identity_anchor_; }

void Pipeline::clear_identity_anchor() {
    identity_anchor_ = false;
    identity_weight_ = 0.0f;
    if (model_class_ == ModelClass::Sana) {
        static_cast<dit::SanaDenoiser*>(denoiser_.get())->ref_clear();
    }
}

}  // namespace brodiffusion::pipeline
