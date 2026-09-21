#include "brodiffusion/pipeline.h"

#include "pipeline_detail.h"

#include "brolm/clip.h"
#include "brodiffusion/controlnet.h"
#include "brodiffusion/denoiser.h"
#include "brodiffusion/dit/flux.h"
#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/scm_scheduler.h"
#include "brodiffusion/lcm_scheduler.h"
#include "brodiffusion/dpm_solver.h"
#include "brodiffusion/dit/pixart.h"
#include "brodiffusion/detail/device.h"
#include "brodiffusion/lora.h"
#include "brodiffusion/model_config.h"
#include "brotensor/gguf.h"
#include "brotensor/safetensors.h"
#include "brodiffusion/scheduler.h"
#include "brolm/t5.h"
#include "brolm/tokenizer.h"
#include "brolm/tokenizer_t5.h"
#include "brodiffusion/unet.h"
#include "brodiffusion/vae.h"
#include "brodiffusion/detail/compute.h"
#include "brodiffusion/detail/safetensors_dir.h"
#include "brodiffusion/detail/torch_rng.h"
#include "brodiffusion/dit/sana.h"
#include "brodiffusion/dit/krea2.h"
#include "brodiffusion/dit/qwenimage21.h"
#include "brodiffusion/sana_text.h"
#include "brodiffusion/krea2_text.h"
#include "brodiffusion/qwenimage21_text.h"
#include "brodiffusion/vae_dcae.h"
#include "brodiffusion/vae_qwenimage.h"
#include "brodiffusion/vae_qwenimage21.h"
#include "brodiffusion/image_io.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#ifdef BROTENSOR_HAS_CUDA
#include "brotensor/cuda_graph.h"
#endif

#include <algorithm>
#include <cmath>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace brodiffusion::pipeline {

namespace bt = ::brotensor;

// The helpers this file used to carry in an anonymous namespace, plus
// Pipeline::StepGraphSession, now live in pipeline_detail.h so the
// family-specific translation units beside this one can share them.
using namespace detail_pipe;

Pipeline::~Pipeline() = default;
Pipeline::Pipeline(Pipeline&&) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;

Pipeline::Pipeline(const PipelineConfig& cfg, brolm::clip::Tokenizer tokenizer)
    : cfg_(cfg),
      model_class_(cfg.model_class),
      tokenizer_(std::move(tokenizer)),
      text_encoder_(cfg.text_encoder),
      denoiser_(make_denoiser(cfg)),
      vae_(cfg.vae),
      vae_encoder_(encoder_config_from_decoder(cfg.vae)),
      scheduler_(make_scheduler(cfg.scheduler)) {
    if (cfg.model_class == ModelClass::Flux) {
        fail("Pipeline: Flux model_class requires the (cfg, clip_tok, t5_tok) "
             "constructor");
    }
}

Pipeline::Pipeline(const PipelineConfig& cfg, brolm::clip::Tokenizer clip_tok,
                   brolm::t5::Tokenizer t5_tok)
    : cfg_(cfg),
      model_class_(cfg.model_class),
      tokenizer_(std::move(clip_tok)),
      text_encoder_(cfg.text_encoder),
      t5_tokenizer_(std::move(t5_tok)),
      t5_encoder_(std::in_place, cfg.t5),
      denoiser_(make_denoiser(cfg)),
      vae_(cfg.vae),
      vae_encoder_(encoder_config_from_decoder(cfg.vae)),
      scheduler_(make_scheduler(cfg.scheduler)) {
    if (cfg.model_class != ModelClass::Flux) {
        fail("Pipeline: the (cfg, clip_tok, t5_tok) constructor requires "
             "model_class == Flux");
    }
}

Pipeline::Pipeline(const PipelineConfig& cfg, brolm::gemma::Tokenizer gemma_tok)
    : cfg_(cfg),
      model_class_(cfg.model_class),
      tokenizer_(std::nullopt),           // CLIP unused for Sana
      text_encoder_(cfg.text_encoder),    // CLIP unused for Sana
      denoiser_(make_denoiser(cfg)),
      vae_(cfg.vae),                       // KL-VAE unused for Sana
      vae_encoder_(encoder_config_from_decoder(cfg.vae)),
      scheduler_(make_scheduler(cfg.scheduler)),
      dcae_(std::in_place, cfg.dcae),
      gemma_model_(std::in_place, cfg.gemma),
      gemma_tokenizer_(std::move(gemma_tok)) {
    if (cfg.model_class != ModelClass::Sana) {
        fail("Pipeline: the (cfg, gemma_tok) constructor requires "
             "model_class == Sana");
    }
}

Pipeline::Pipeline(const PipelineConfig& cfg, brolm::t5::Tokenizer t5_tok)
    : cfg_(cfg),
      model_class_(cfg.model_class),
      tokenizer_(std::nullopt),           // no CLIP frontend for PixArt
      text_encoder_(cfg.text_encoder),    // CLIP unused for PixArt
      t5_tokenizer_(std::move(t5_tok)),
      t5_encoder_(std::in_place, cfg.t5),
      denoiser_(make_denoiser(cfg)),
      vae_(cfg.vae),
      vae_encoder_(encoder_config_from_decoder(cfg.vae)),
      scheduler_(make_scheduler(cfg.scheduler)) {
    if (cfg.model_class != ModelClass::PixArt) {
        fail("Pipeline: the (cfg, t5_tok) constructor requires "
             "model_class == PixArt");
    }
}

Pipeline::Pipeline(const PipelineConfig& cfg,
                   brolm::qwen3vl::Tokenizer qwen3vl_tok)
    : cfg_(cfg),
      model_class_(cfg.model_class),
      tokenizer_(std::nullopt),           // no CLIP frontend for Krea 2
      text_encoder_(cfg.text_encoder),    // CLIP unused for Krea 2
      denoiser_(make_denoiser(cfg)),
      vae_(cfg.vae),                       // KL-VAE unused for Krea 2
      vae_encoder_(encoder_config_from_decoder(cfg.vae)),
      scheduler_(make_scheduler(cfg.scheduler)),
      qwen3vl_tokenizer_(std::move(qwen3vl_tok)),
      qwen3vl_pp_() {
    // Both families share the Qwen3-VL text frontend but nothing else: Krea 2
    // pairs it with the 8x Qwen-Image VAE and a vision tower, Qwen-Image 2.1
    // with the 16x RGBA VAE (both halves) and — for now — no tower. Build only
    // the half that will be loaded; the other stays nullopt, so a wrong-class
    // access fails on the optional instead of silently using empty weights.
    if (cfg.model_class == ModelClass::Krea2) {
        vae_qwen_.emplace(cfg.krea2.vae);
        qwen3vl_model_.emplace(cfg.krea2.text.text);
        qwen3vl_vision_.emplace(cfg.krea2.text.vision,
                                cfg.krea2.text.text.hidden_size);
    } else if (cfg.model_class == ModelClass::QwenImage21) {
        vae_qi21_.emplace(cfg.qwenimage21.vae);
        vae_qi21_encoder_.emplace(cfg.qwenimage21.vae);
        qwen3vl_model_.emplace(cfg.qwenimage21.text.text);
        qwen3vl_vision_.emplace(cfg.qwenimage21.text.vision,
                                cfg.qwenimage21.text.text.hidden_size);
    } else {
        fail("Pipeline: the (cfg, qwen3vl_tok) constructor requires "
             "model_class == Krea2 or QwenImage21");
    }
}

const unet::UNet& Pipeline::unet() const {
    const unet::UNet* u = denoiser_->as_unet();
    if (u == nullptr) fail("unet(): the active denoiser is not a UNet");
    return *u;
}

// from_model_dir, the load_weights overloads and apply_lora live in
// pipeline_load.cpp.

// The ControlNet registry, encode_prompt_ and img2img/inpaint priming live
// in pipeline_sd.cpp.

bt::Tensor Pipeline::encode_conditioning(std::string_view prompt) {
    if (model_class_ == ModelClass::Sana) {
        if (!gemma_model_ || !gemma_tokenizer_) {
            fail("encode_conditioning: Gemma model/tokenizer not loaded");
        }
        return brodiffusion::sana::encode_prompt(
            *gemma_model_, *gemma_tokenizer_, std::string(prompt),
            cfg_.sana_max_seq_len,
            brodiffusion::sana::default_complex_human_instruction());
    }
    if (model_class_ == ModelClass::Krea2) {
        // Krea 2: the FUSED (n_valid, hidden_size) conditioning — the same
        // space cond_control() axes are minted in and applied to (see
        // prime()'s Krea2 branch), not the raw pre-fusion taps.
        krea2::TextConditioning tc = krea_encode_prompt_taps(prompt);
        return krea_encode_text(tc.prompt_embeds, tc.prompt_embeds_mask);
    }
    if (model_class_ == ModelClass::QwenImage21) {
        // Qwen-Image 2.1: the raw (n, 4096) Qwen3-VL-8B rows — the space
        // prime() applies control axes in, and what txt_in consumes.
        return qi21_encode_prompt(prompt).embeds;
    }
    // CLIP-based models (SD / Flux): reuse the fixed-length CLIP encode path.
    bt::Tensor out;
    encode_prompt_(prompt, out);
    return out;
}

std::uint64_t next_state_id() {
    static std::atomic<std::uint64_t> counter{1};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

PipelineState PipelineState::clone() const {
    PipelineState out;
    out.latent     = latent.clone();
    out.rng_key    = rng_key;
    out.step_index = step_index;
    out.n_steps    = n_steps;
    out.H_lat      = H_lat;
    out.W_lat      = W_lat;
    out.prepared   = prepared;   // shared, not copied — one encode per prime()
    return out;
}

// prime_sana_, step_once_scm_ and the identity anchor live in
// pipeline_sana.cpp.

PipelineState Pipeline::prime(std::string_view prompt,
                              const GenerateOptions& opts) {
    // Sana has its own (32x downsample, FP32 latent, Gemma-encoded) priming
    // path; the SD / Flux machinery below (8x latent, CLIP / T5, img2img /
    // inpaint / ControlNet) does not apply.
    if (model_class_ == ModelClass::Sana) {
        return prime_sana_(prompt, opts);
    }
    // Qwen-Image 2.1 with condition images accepts height/width 0, meaning
    // "take the canvas from the last condition image's aspect". Resolve it
    // once here and re-enter with a concrete size, so every check and every
    // latent-shape computation below sees real dimensions.
    if (!opts.condition_images.empty() && (opts.height <= 0 || opts.width <= 0)) {
        GenerateOptions sized = opts;
        qi21_resolve_size(opts, sized.width, sized.height);
        return prime(prompt, sized);
    }
    if (!opts.condition_images.empty() &&
        model_class_ != ModelClass::QwenImage21) {
        fail("prime: condition_images is Qwen-Image 2.1 only — img2img "
             "(init_image_path) is the seam for the other model classes");
    }
    // Resolution granularity is the autoencoder's stride — 8 for the KL-VAEs,
    // 16 for Qwen-Image 2.1's residual VAE. The reference pipeline silently
    // rounds a request down to a multiple of 2*vae_scale_factor; rounding
    // behind the caller's back would make decode()'s buffer a different size
    // than the width/height they wrote the PNG with, so demand it instead.
    const int vsf = vae_scale_factor();
    if (opts.height <= 0 || opts.width <= 0 ||
        opts.height % vsf != 0 || opts.width % vsf != 0) {
        fail("height and width must be positive multiples of " +
             std::to_string(vsf));
    }
    if (opts.num_inference_steps <= 0) fail("num_inference_steps must be positive");
    if (!opts.init_image_path.empty() && !opts.init_noise.empty()) {
        fail("prime: init_image_path and init_noise cannot both be set "
             "(img2img and explicit-noise priming are mutually exclusive)");
    }
    if (!opts.init_image_path.empty() &&
        model_class_ != ModelClass::StableDiffusion) {
        fail("prime: img2img (init_image_path) is currently SD1.5 only; "
             "Flux img2img is not yet supported");
    }
    if (!opts.mask_image_path.empty() && opts.init_image_path.empty()) {
        fail("inpaint: --mask requires --init (mask_image_path is set but "
             "init_image_path is empty)");
    }
    if (!opts.mask_image_path.empty() &&
        model_class_ != ModelClass::StableDiffusion) {
        fail("inpaint: Flux is not yet supported (SD1.5 only)");
    }

    // Reset inpaint state so a previous inpaint generation doesn't leak into
    // the next call. Re-armed below in the img2img branch when mask_image_path
    // is set.
    inpaint_active_ = false;

    // Reset ControlNet activation; armed below when controls are supplied.
    controlnet_active_ = false;
    control_inputs_.clear();
    control_images_.clear();

    if (!opts.controls.empty()) {
        if (model_class_ != ModelClass::StableDiffusion) {
            fail("prime: ControlNet is currently SD1.5 only; "
                 "Flux ControlNet is not supported");
        }
        if (controlnets_.empty()) {
            fail("prime: GenerateOptions.controls is non-empty but no "
                 "ControlNet has been registered — call add_controlnet() "
                 "first");
        }
        if (opts.controls.size() != controlnets_.size()) {
            fail("prime: GenerateOptions has " +
                 std::to_string(opts.controls.size()) +
                 " ControlNet input(s) but " +
                 std::to_string(controlnets_.size()) +
                 " ControlNet(s) are registered — counts must match");
        }
        // Control images live at FULL image resolution; each ControlNet's
        // conditioning_embedding does the 8x downsample to latent space.
        // Pixel range is [0, 1] (UnsignedUnit) to match diffusers — the HF
        // ControlNetPipeline's preprocessor does pixel/255 with no recentering.
        control_images_.reserve(opts.controls.size());
        for (const auto& ci : opts.controls) {
            if (ci.image_path.empty()) {
                fail("prime: ControlNetInput.image_path must be non-empty");
            }
            if (!(ci.start_step >= 0.0f && ci.end_step <= 1.0f &&
                  ci.start_step <= ci.end_step)) {
                fail("prime: ControlNetInput.start_step/end_step must "
                     "satisfy 0 <= start <= end <= 1");
            }
            control_images_.push_back(brodiffusion::load_image_as_latent_input(
                ci.image_path, opts.width, opts.height,
                brodiffusion::PixelRange::UnsignedUnit));
        }
        control_inputs_   = opts.controls;
        controlnet_active_ = true;
    }

    const int H_lat = opts.height / vsf;
    const int W_lat = opts.width  / vsf;
    // Latent channel count is denoiser-defined (4 for SD1.5, 16 for Flux,
    // 64 for Qwen-Image 2.1).
    const int C_lat = denoiser_->latent_channels();
    const int n_lat = C_lat * H_lat * W_lat;
    const bool is_lcm = std::holds_alternative<scheduler::LCM>(scheduler_);
    const bool do_cfg = cfg_branch_active(model_class_, *denoiser_, is_lcm,
                                          opts.guidance_scale);

    // 0. Finalize denoiser weights (W8A16 quantisation happens here if enabled).
    denoiser_->finalize_weights();

    // 1. Encode prompt(s) into the model-agnostic Conditioning struct.
    if (model_class_ == ModelClass::Flux) {
        // Flux: T5 token sequence is the cross-attention context; the CLIP
        // pooled vector feeds the AdaLN time-text embedding. No CFG branch.
        if (!t5_tokenizer_ || !t5_encoder_) {
            fail("prime: Flux pipeline missing T5 tokenizer / encoder");
        }
        std::vector<std::int32_t> t5_ids =
            t5_tokenizer_->encode(prompt, cfg_.t5_max_length);
        // encode() pads to t5_max_length with <pad>; pass pad_id so the T5
        // encoder masks those positions out of self-attention (HF parity).
        t5_encoder_->forward(t5_ids.data(),
                             static_cast<int>(t5_ids.size()),
                             conditioning_.text_embeddings,
                             t5_tokenizer_->pad_id());

        // CLIP pooled vector — discard the CLIP sequence output for Flux.
        std::vector<std::int32_t> clip_ids = tokenizer_->encode(prompt);
        if (static_cast<int>(clip_ids.size()) !=
            cfg_.text_encoder.max_position) {
            fail("prime: CLIP tokenizer returned " +
                 std::to_string(clip_ids.size()) + " ids, expected " +
                 std::to_string(cfg_.text_encoder.max_position));
        }
        text_encoder_.forward(clip_ids.data(), scratch_,
                              &conditioning_.pooled);

        conditioning_.has_uncond = false;
        conditioning_.uncond_embeddings = brotensor::Tensor{};
        conditioning_.guidance =
            cfg_.flux.guidance_embeds ? opts.guidance_scale : 0.0f;
    } else if (model_class_ == ModelClass::PixArt) {
        // PixArt-Sigma: the T5-XXL token sequence is the cross-attention
        // context; there is no CLIP pooled vector. Runs true classifier-free
        // guidance, so the negative prompt (empty by default) gets its own T5
        // pass for the uncond branch.
        if (!t5_tokenizer_ || !t5_encoder_) {
            fail("prime: PixArt pipeline missing T5 tokenizer / encoder");
        }
        // Encode, then TRIM the (max_length, 4096) sequence to its valid
        // (non-pad) leading rows. T5 pads contiguously at the end, so trimming
        // is exactly equivalent to diffusers' cross-attention padding mask: the
        // DiT then cross-attends over precisely the real tokens (the same
        // mask-free convention Sana uses for its Gemma caption).
        const int t5_pad = t5_tokenizer_->pad_id();
        auto encode_trim = [&](std::string_view text, bt::Tensor& emb) {
            std::vector<std::int32_t> ids =
                t5_tokenizer_->encode(text, cfg_.t5_max_length);
            t5_encoder_->forward(ids.data(), static_cast<int>(ids.size()),
                                 emb, t5_pad);
            int valid = 0;
            for (std::int32_t id : ids) if (id != t5_pad) ++valid;
            if (valid < 1) valid = 1;
            if (valid < emb.rows) {
                bt::Tensor v = bt::Tensor::view(emb.device, emb.data, valid,
                                                emb.cols, emb.dtype);
                emb = v.clone();
            }
        };
        const bool t5_time = std::getenv("BRODIFFUSION_TIME") != nullptr;
        if (t5_time) brotensor::sync_all();
        const auto t5_t0 = std::chrono::steady_clock::now();
        encode_trim(prompt, conditioning_.text_embeddings);
        if (do_cfg) {
            encode_trim(opts.negative_prompt, conditioning_.uncond_embeddings);
            conditioning_.has_uncond = true;
        } else {
            conditioning_.has_uncond = false;
            conditioning_.uncond_embeddings = brotensor::Tensor{};
        }
        if (t5_time) {
            brotensor::sync_all();
            std::fprintf(stderr, "[time] T5 encode (%d pass%s): %.3f s\n",
                         do_cfg ? 2 : 1, do_cfg ? "es" : "",
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t5_t0).count());
        }
        conditioning_.guidance = 0.0f;
    } else if (model_class_ == ModelClass::Krea2) {
        // Krea 2: Qwen3-VL tapped-hidden-states conditioning + validity mask.
        // Runs true classifier-free guidance, so the negative prompt (empty by
        // default, matching the reference pipeline) gets its own encode for the
        // uncond branch. No CLIP pooled vector / distilled guidance embedding.
        if (!qwen3vl_model_ || !qwen3vl_tokenizer_) {
            fail("prime: Krea2 pipeline missing Qwen3-VL model / tokenizer");
        }
        const bool enc_time = std::getenv("BRODIFFUSION_TIME") != nullptr;
        const auto enc_t0 = std::chrono::steady_clock::now();
        // krea_prime_from_taps() stashes caller-supplied raw taps here just
        // before calling prime(); consume (move out) them instead of
        // encoding `prompt` when present. This is the shared entry point for
        // the band dial (scaled taps) and image-as-prompt (taps from an
        // image instead of text) — everything below (fusion, cond_control,
        // prepare(), latent alloc, scheduler, CUDA graph keying) runs
        // identically either way.
        krea2::TextConditioning pos;
        if (krea_taps_override_) {
            pos = std::move(*krea_taps_override_);
            krea_taps_override_.reset();
        } else {
            pos = krea2::encode_prompt(
                *qwen3vl_tokenizer_, *qwen3vl_model_, std::string(prompt));
        }
        conditioning_.text_embeddings      = pos.prompt_embeds;
        conditioning_.text_embeddings_mask = pos.prompt_embeds_mask;
        if (do_cfg) {
            krea2::TextConditioning neg;
            if (krea_uncond_taps_override_) {
                neg = std::move(*krea_uncond_taps_override_);
                krea_uncond_taps_override_.reset();
            } else {
                neg = krea2::encode_prompt(
                    *qwen3vl_tokenizer_, *qwen3vl_model_,
                    std::string(opts.negative_prompt));
            }
            conditioning_.uncond_embeddings      = neg.prompt_embeds;
            conditioning_.uncond_embeddings_mask = neg.prompt_embeds_mask;
            conditioning_.has_uncond = true;
        } else {
            krea_uncond_taps_override_.reset();
            conditioning_.has_uncond = false;
            conditioning_.uncond_embeddings      = bt::Tensor{};
            conditioning_.uncond_embeddings_mask = bt::Tensor{};
        }
        if (enc_time) {
            brotensor::sync_all();
            std::fprintf(stderr,
                         "[time] Qwen3-VL encode (%d pass%s): %.3f s\n",
                         do_cfg ? 2 : 1, do_cfg ? "es" : "",
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - enc_t0).count());
        }
        conditioning_.guidance = 0.0f;
    } else if (model_class_ == ModelClass::QwenImage21) {
        // Qwen-Image 2.1: the DiT cross-reads a Qwen3-VL-8B hidden-state run
        // (the chat template's post-system rows), with an all-ones validity
        // mask at batch 1. Guidance is `true_cfg_scale`: at the reference
        // default of 1.0 there is no uncond branch at all, so the negative
        // prompt is only encoded when the caller asks for > 1.
        //
        // A generation carries the image prefix only when this prime built
        // one, so clear it first: a text-only run after an edit run must not
        // inherit the previous call's condition latents.
        qi21_edit_active_ = false;
        qi21_edit_prefix_ = dit::QwenImage21EditPrefix{};
        qi21_edit_uncond_prefix_ = dit::QwenImage21EditPrefix{};
        //
        // The image-conditioned path fills conditioning_ (both branches) and
        // the prefix descriptions; everything after it — prepare, latent
        // allocation, the schedule, the denoise loop — is the same code
        // text-to-image runs.
        if (!opts.condition_images.empty()) {
        qi21_prime_edit_(prompt, opts, do_cfg);
        } else {
        // The backbone is only needed for the branches this prime actually
        // encodes — qi21_prime_from_text() supplies rows for one or both, and
        // qi21_release_text_encoder() may have freed the 8.5 GiB model in
        // between. Check per branch rather than up front.
        const bool need_pos_encode = !qi21_text_override_.has_value();
        const bool need_neg_encode = do_cfg && !qi21_uncond_text_override_;
        if ((need_pos_encode || need_neg_encode) &&
            (!qwen3vl_model_ || !qwen3vl_tokenizer_)) {
            fail("prime: QwenImage21 pipeline has no Qwen3-VL text encoder "
                 "(it was released — reload it, or prime from caller-supplied "
                 "rows with qi21_prime_from_text)");
        }
        const bool enc_time = std::getenv("BRODIFFUSION_TIME") != nullptr;
        const auto enc_t0 = std::chrono::steady_clock::now();
        // qi21_prime_from_text() parks caller-edited rows here; consume them
        // instead of encoding, so both prime paths share every later step.
        qwenimage21::TextConditioning pos =
            qi21_text_override_
                ? std::move(*qi21_text_override_)
                : qwenimage21::encode_prompt(*qwen3vl_tokenizer_,
                                             *qwen3vl_model_,
                                             std::string(prompt));
        qi21_text_override_.reset();
        conditioning_.text_embeddings      = std::move(pos.embeds);
        conditioning_.text_embeddings_mask = std::move(pos.mask);
        // Conditioning-space control seam. 2.1's conditioning is ONE
        // (n, 4096) run of Qwen3-VL hidden states, so the axes apply to it
        // directly here — before prepare() runs txt_in over it — exactly as
        // they do for Sana, rather than after fusion the way Krea 2 needs.
        // There is no BOS row to protect (row 0 is the template's
        // <|im_start|>), so every row is steered.
        cond_control_.apply(conditioning_.text_embeddings, /*row_end=*/-1,
                            /*row_start=*/0);
        if (do_cfg) {
            qwenimage21::TextConditioning neg =
                qi21_uncond_text_override_
                    ? std::move(*qi21_uncond_text_override_)
                    : qwenimage21::encode_prompt(
                          *qwen3vl_tokenizer_, *qwen3vl_model_,
                          std::string(opts.negative_prompt));
            conditioning_.uncond_embeddings      = std::move(neg.embeds);
            conditioning_.uncond_embeddings_mask = std::move(neg.mask);
            conditioning_.has_uncond = true;
        } else {
            conditioning_.has_uncond = false;
            conditioning_.uncond_embeddings      = bt::Tensor{};
            conditioning_.uncond_embeddings_mask = bt::Tensor{};
        }
        qi21_uncond_text_override_.reset();
        if (enc_time) {
            brotensor::sync_all();
            std::fprintf(stderr,
                         "[time] Qwen3-VL encode (%d pass%s): %.3f s\n",
                         do_cfg ? 2 : 1, do_cfg ? "es" : "",
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - enc_t0).count());
        }
        conditioning_.guidance = 0.0f;
        }
    } else {
        int content_end = -1;
        encode_prompt_(prompt, conditioning_.text_embeddings, &content_end);
        // Conditioning-space control seam (CLIP): add the weighted control axes to
        // the positive prompt's CONTENT rows [1, eos) — BOS and the EOS/padding
        // tail untouched (content-rows policy; clip-research found steering the
        // live padding rows over-drives generation). No-op unless an axis carries
        // a nonzero weight. The negative branch is left clean (steer the prompt,
        // not the baseline) — mirrors the Sana path.
        cond_control_.apply(conditioning_.text_embeddings, content_end);
        if (do_cfg) {
            encode_prompt_(opts.negative_prompt,
                           conditioning_.uncond_embeddings);
            conditioning_.has_uncond = true;
        } else {
            conditioning_.has_uncond = false;
            conditioning_.uncond_embeddings = brotensor::Tensor{};
        }
        conditioning_.guidance = is_lcm ? opts.guidance_scale : 0.0f;
    }

    // 1b. Pre-process conditioning once per generation (cross-attention K/V
    // projection for the UNet; pre-projected T5 context for Flux). Rides the
    // returned state, shared across all states branched from this prime.
    // Qwen-Image 2.1's image-conditioned path needs prepare() to be told the
    // joint prefix the condition images occupy; the model-agnostic
    // Conditioning has no room for that, so it goes in through the denoiser's
    // own entry point.
    auto prepared = std::make_shared<PreparedConditioning>(
        qi21_edit_active_
            ? static_cast<dit::QwenImage21Denoiser*>(denoiser_.get())
                  ->prepare_edit(conditioning_, qi21_edit_prefix_,
                                 conditioning_.has_uncond
                                     ? &qi21_edit_uncond_prefix_
                                     : nullptr)
            : denoiser_->prepare(conditioning_));
    // Non-owning handle for the Qwen-Image 2.1 prefix-cache hooks, which have
    // to reach into the payload the caller's PipelineState owns.
    last_prepared_ = prepared;

    // Conditioning-space control seam for Krea 2: unlike Sana/CLIP (steered
    // BEFORE prepare(), on the raw per-token conditioning), Krea 2's raw
    // conditioning is pre-fusion taps (token-major/layer-minor — wrong row
    // semantics for CondControl's per-token model). Its axis vectors live in
    // the FUSED (n_valid, hidden_size) space prepare()'s text_fusion just
    // produced, so steer that instead, with no BOS row to protect
    // (row_start=0) — see cond_control()'s doc comment and
    // Krea2Denoiser::fused_text().
    if (model_class_ == ModelClass::Krea2 && cond_control_.active()) {
        auto* krea2d = static_cast<dit::Krea2Denoiser*>(denoiser_.get());
        cond_control_.apply(krea2d->fused_text(*prepared, /*uncond=*/false),
                            /*row_end=*/-1, /*row_start=*/0);
    }

    // Text encoding is over: return its cached allocator blocks to the
    // driver before the denoise loop. On Windows (WDDM) a near-full commit
    // makes the OS demote large resident allocations — the DiT's biggest
    // weight tensors — to shared memory, silently turning their per-step
    // reads into PCIe traffic (measured: the 100 MB FF GEMMs run 2.6x
    // slower once that starts).
    brotensor::device_mem_trim(brotensor::default_device());
    if (std::getenv("BRODIFFUSION_TIME") != nullptr) {
        std::size_t fb = 0, tb = 0;
        if (brotensor::device_mem_info(brotensor::default_device(), fb, tb)) {
            std::fprintf(stderr, "[mem] after prime (pool trimmed): "
                         "%.2f / %.2f GiB used\n",
                         double(tb - fb) / (1024.0 * 1024.0 * 1024.0),
                         double(tb) / (1024.0 * 1024.0 * 1024.0));
        }
    }

    // 2. Build the initial state shell (latent assigned below).
    PipelineState state;
    state.H_lat = H_lat;
    state.W_lat = W_lat;
    state.prepared = std::move(prepared);
    // Stash the seed: LCM resamples per-step noise from this same Philox key,
    // and branched states diverge by mutating rng_key on the clone.
    state.rng_key = opts.seed;
    const float sigma = std::visit(
        [](const auto& s) { return s.init_noise_sigma(); }, scheduler_);

    // 3. Set the timestep schedule (lives on the scheduler — shared across
    // all branches). FlowMatch with dynamic shifting (flux-dev) needs the
    // image token count; DDIM / LCM ignore the second argument. We set the
    // schedule *before* building the initial latent so img2img can pick a
    // t_start off the populated timesteps_ vector.
    // Dynamic shifting keys on the DiT's TOKEN count, which is the packed
    // sequence length — one token per 2x2 latent group for Flux / Krea 2,
    // but one per latent pixel for Qwen-Image 2.1 (patch_size 1). At 1024²
    // that is 4096 tokens there against 1024 here, and mu is a function of
    // it, so getting this wrong shifts the whole sigma schedule.
    const int image_seq_len = (model_class_ == ModelClass::QwenImage21)
                                  ? H_lat * W_lat
                                  : (H_lat / 2) * (W_lat / 2);
    std::visit([&](auto& s) {
        using S = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<S, scheduler::FlowMatch>) {
            s.set_timesteps(opts.num_inference_steps, image_seq_len);
        } else {
            s.set_timesteps(opts.num_inference_steps);
        }
    }, scheduler_);
    state.n_steps = std::visit(
        [](const auto& s) { return s.num_inference_steps(); }, scheduler_);

    if (!opts.init_image_path.empty()) {
        // img2img / inpaint priming lives in pipeline_sd.cpp.
        prime_img2img_(opts, state, n_lat, H_lat, W_lat, C_lat);
        return state;
    }

    // ── txt2img: initial latent is pure Gaussian noise * init_noise_sigma ──
    if (!opts.init_noise.empty()) {
        // Caller-supplied initial noise (raw N(0,1)); used for exact
        // cross-implementation comparison. Still scaled by init_noise_sigma.
        if (static_cast<int>(opts.init_noise.size()) != n_lat) {
            fail("prime: init_noise has " +
                 std::to_string(opts.init_noise.size()) +
                 " values, expected " + std::to_string(n_lat));
        }
        std::vector<float> noise(n_lat);
        for (int i = 0; i < n_lat; ++i) {
            noise[i] = sigma * opts.init_noise[i];
        }
        state.latent = detail::upload_host(noise.data(), 1, n_lat);
    } else if (opts.noise_source == NoiseSource::Torch) {
        // torch.randn-compatible noise: makes `seed` reproduce a PyTorch
        // reference run's starting latent (see detail/torch_rng.h). This
        // path stays host-side by construction — the bit-exact Box-Muller
        // is the whole point.
        std::vector<float> noise =
            detail::torch_randn_f32(opts.seed, static_cast<std::size_t>(n_lat));
        if (sigma != 1.0f) {
            for (int i = 0; i < n_lat; ++i) noise[i] *= sigma;
        }
        state.latent = detail::upload_host(noise.data(), 1, n_lat);
    } else {
        // Internal: device-side Philox stream. Counter 0..n_lat is the
        // initial latent; per-step LCM noise uses (1+step)*n_lat onwards.
        randn_compute(opts.seed, 0, n_lat, state.latent);
        if (sigma != 1.0f) bt::scale_inplace(state.latent, sigma);
    }

    state.step_index = 0;
    return state;
}

// step_denoise_captured_ — the CUDA-graph capture/replay seam — lives in
// pipeline_step_graph.cpp.

void Pipeline::step_once(PipelineState& state, const GenerateOptions& opts,
                         AttentionTrace* trace_out,
                         const std::vector<const bt::Tensor*>*
                             attn_logit_biases) {
    if (state.step_index >= state.n_steps) {
        fail("step_once: step_index (" + std::to_string(state.step_index) +
             ") >= n_steps (" + std::to_string(state.n_steps) + ")");
    }
    if (!state.prepared) {
        fail("step_once: state has no prepared conditioning — prime() builds "
             "it; a default-constructed PipelineState cannot be stepped");
    }
    // Sana-Sprint (SCMScheduler / TrigFlow): a dedicated few-step, no-CFG,
    // no-trace path with model-specific input/output parameterisation.
    if (std::holds_alternative<scheduler::SCM>(scheduler_)) {
        if (trace_out != nullptr || attn_logit_biases != nullptr) {
            fail("step_once: attention trace / steering is not supported for "
                 "Sana-Sprint (SCMScheduler)");
        }
        step_once_scm_(state, opts);
        return;
    }
    const bool is_lcm = std::holds_alternative<scheduler::LCM>(scheduler_);
    const bool do_cfg = cfg_branch_active(model_class_, *denoiser_, is_lcm,
                                          opts.guidance_scale);
    const int i = state.step_index;
    const float t = timestep_at(scheduler_, i);
    const int n_lat = denoiser_->latent_channels() *
                      state.H_lat * state.W_lat;

    // Trace mode is forced if either the caller wants a trace OR is injecting
    // attention biases (forward_trace is the only path that accepts biases).
    const bool trace_mode = (trace_out != nullptr) || (attn_logit_biases != nullptr);

    // ── ControlNet-augmented forward ──────────────────────────────────────
    // When a ControlNet is active, run it once per step against the cond
    // branch's RAW text context (HF default guess_mode=false reuses the same
    // residuals for both CFG branches). The residual-aware UNet overload is
    // reached through Denoiser::as_unet() — bypassing the Denoiser virtual
    // dispatch, which (intentionally) has no residual-carrying forward.
    if (controlnet_active_) {
        unet::UNet* u = denoiser_->as_unet();
        if (u == nullptr) {
            fail("step_once: ControlNet requires a UNet denoiser");
        }
        const bt::Tensor& ctx_cond = conditioning_.text_embeddings;

        // Active-window membership: a net contributes at step i iff
        // start_step <= i/n_steps < end_step. Avoids running the forward
        // for inactive nets; the residual contribution is simply skipped.
        const float frac = (state.n_steps > 0)
            ? (static_cast<float>(i) / static_cast<float>(state.n_steps))
            : 0.0f;
        auto in_window = [&](const ControlNetInput& ci) {
            return frac >= ci.start_step && frac < ci.end_step;
        };

        // Sum residuals across all registered ControlNets. First active net
        // writes directly into the summed buffers; subsequent active nets
        // run into per-net scratch and accumulate via bt::add_inplace.
        // When no nets are in-window this step the summed buffers retain
        // the previous step's contents — we zero them by writing the first
        // net's contribution. If literally no net is active we feed
        // nullptrs to the UNet (== plain non-CN forward).
        bool any_active = false;
        for (std::size_t k = 0; k < controlnets_.size(); ++k) {
            const auto& ci = control_inputs_[k];
            if (!in_window(ci)) continue;
            if (!any_active) {
                controlnets_[k]->forward(
                    state.latent, state.H_lat, state.W_lat, t,
                    ctx_cond, control_images_[k], ci.scale,
                    cn_down_residuals_, cn_mid_residual_);
                any_active = true;
            } else {
                controlnets_[k]->forward(
                    state.latent, state.H_lat, state.W_lat, t,
                    ctx_cond, control_images_[k], ci.scale,
                    cn_down_residuals_scratch_, cn_mid_residual_scratch_);
                for (std::size_t r = 0; r < cn_down_residuals_.size(); ++r) {
                    bt::add_inplace(cn_down_residuals_[r],
                                    cn_down_residuals_scratch_[r]);
                }
                bt::add_inplace(cn_mid_residual_, cn_mid_residual_scratch_);
            }
        }

        std::vector<const bt::Tensor*> down_ptrs;
        const bt::Tensor* mid_ptr = nullptr;
        if (any_active) {
            down_ptrs.reserve(cn_down_residuals_.size());
            for (const auto& rt : cn_down_residuals_) down_ptrs.push_back(&rt);
            mid_ptr = &cn_mid_residual_;
        }

        if (trace_mode) {
            // Trace mode bypasses the K/V cache (same as the non-CN trace
            // path) and captures the cond pass only. For LCM the guidance
            // scale is passed via the cond_proj path; for vanilla SD1.5 the
            // pointer is null. A CFG uncond branch (DDIM only) still uses
            // the fast cached + residual-aware path.
            if (u->num_xattn_blocks() == 0) {
                fail("step_once: trace mode requires a denoiser with "
                     "traceable attention blocks");
            }
            AttentionTrace scratch_trace;
            AttentionTrace* trace_dst =
                trace_out ? trace_out : &scratch_trace;
            const float gs   = conditioning_.guidance;
            const float* gsp = is_lcm ? &gs : nullptr;
            u->forward_trace(state.latent, state.H_lat, state.W_lat, t,
                             gsp, ctx_cond, down_ptrs, mid_ptr,
                             attn_logit_biases, trace_dst,
                             noise_pred_cond_);
            if (do_cfg) {
                const auto& cache_uncond =
                    u->kv_cache_for(*state.prepared, Branch::Uncond);
                const bt::Tensor& ctx_uncond =
                    conditioning_.uncond_embeddings;
                u->forward(state.latent, state.H_lat, state.W_lat, t,
                           ctx_uncond, cache_uncond, down_ptrs, mid_ptr,
                           noise_pred_uncond_);
            }
        } else {
            const auto& cache_cond =
                u->kv_cache_for(*state.prepared, Branch::Cond);
            if (is_lcm) {
                // LCM-distilled UNet: cond_proj path adds the guidance
                // embedding to the time embedding. No CFG branch under LCM.
                u->forward(state.latent, state.H_lat, state.W_lat, t,
                           conditioning_.guidance, ctx_cond, cache_cond,
                           down_ptrs, mid_ptr, noise_pred_cond_);
            } else {
                u->forward(state.latent, state.H_lat, state.W_lat, t,
                           ctx_cond, cache_cond, down_ptrs, mid_ptr,
                           noise_pred_cond_);
                if (do_cfg) {
                    const auto& cache_uncond =
                        u->kv_cache_for(*state.prepared, Branch::Uncond);
                    const bt::Tensor& ctx_uncond =
                        conditioning_.uncond_embeddings;
                    u->forward(state.latent, state.H_lat, state.W_lat, t,
                               ctx_uncond, cache_uncond, down_ptrs, mid_ptr,
                               noise_pred_uncond_);
                }
            }
        }
    } else if (trace_mode) {
        // Trace mode bypasses the K/V cache + INT8 inside the denoiser and
        // captures the conditional pass only; a CFG uncond branch (if any)
        // still uses the fast prepared path. A scratch trace absorbs the maps
        // when the caller wants biases but not the trace itself. The denoiser
        // pulls the raw context (and any LCM guidance) from the state's prepared.
        if (denoiser_->num_xattn_blocks() == 0) {
            fail("step_once: trace mode requires a denoiser with traceable "
                 "attention blocks");
        }
        AttentionTrace scratch_trace;
        AttentionTrace* trace_dst = trace_out ? trace_out : &scratch_trace;
        denoiser_->forward_traced(state.latent, state.H_lat, state.W_lat, t,
                                  *state.prepared, Branch::Cond,
                                  attn_logit_biases, trace_dst,
                                  noise_pred_cond_);
        if (do_cfg) {
            denoiser_->forward(state.latent, state.H_lat, state.W_lat, t,
                               *state.prepared, Branch::Uncond,
                               noise_pred_uncond_);
        }
    } else {
        // Plain fast path. When the denoiser exposes the step-capture seam
        // and the latent is CUDA-resident, run it through the CUDA-graph
        // session (warm-up → capture → single-launch replay); otherwise the
        // classic eager forwards.
        const bool graph_eligible =
            denoiser_->supports_step_capture() &&
            state.latent.device == bt::Device::CUDA &&
            !step_graph_disabled();
        if (graph_eligible) {
            step_denoise_captured_(state, t, do_cfg);
        } else {
            denoiser_->forward(state.latent, state.H_lat, state.W_lat, t,
                               *state.prepared, Branch::Cond,
                               noise_pred_cond_);
            if (do_cfg) {
                denoiser_->forward(state.latent, state.H_lat, state.W_lat, t,
                                   *state.prepared, Branch::Uncond,
                                   noise_pred_uncond_);
            }
        }
    }
    // CFG combine (DDIM only; LCM has no uncond branch).
    if (do_cfg) {
        bt::scale_inplace(noise_pred_cond_, opts.guidance_scale);
        bt::scale_inplace(noise_pred_uncond_, 1.0f - opts.guidance_scale);
        bt::add_inplace(noise_pred_cond_, noise_pred_uncond_);
    }

    // ── Scheduler step ────────────────────────────────────────────────────
    scheduler_step(scheduler_, noise_pred_cond_, i, state, n_lat,
                   scratch_, noise_step_);
    ++state.step_index;

    // ── Inpaint latent blend ──────────────────────────────────────────────
    // After the scheduler step, the latent represents the state at the next
    // timestep (i+1). Re-noise x0 to that same timestep and replace the
    // unmasked region (mask=0) with it — keeping the inpaint region (mask=1)
    // on the model's own trajectory. Skip on the final step: the latent is at
    // t=0 (clean) and re-noising would just re-add noise to a clean image.
    if (inpaint_active_ && state.step_index < state.n_steps) {
        // Per-step renoise noise: separate Philox slot from the LCM per-step
        // stream. We XOR the state's rng_key with the golden-ratio constant
        // to decorrelate from both the initial-noise (counter 0..n_lat) and
        // LCM per-step (counter (1+step)*n_lat..) draws on the same key.
        const std::uint64_t inpaint_key =
            state.rng_key ^ 0x9E3779B97F4A7C15ULL;
        const std::uint64_t counter =
            static_cast<std::uint64_t>(state.step_index) *
            static_cast<std::uint64_t>(n_lat);
        randn_compute(inpaint_key, counter, n_lat, inpaint_noise_step_);

        // x0_renoised = add_noise(x0, noise_step, state.step_index).
        // Only DDIM / LCM have add_noise on the SD1.5 path; FlowMatch is
        // unreachable here (the Flux guard in prime() rejected it).
        std::visit([&](auto& s) {
            using S = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<S, scheduler::DDIM> ||
                          std::is_same_v<S, scheduler::LCM>) {
                s.add_noise(inpaint_x0_, inpaint_noise_step_,
                            state.step_index, inpaint_renoise_buf_, scratch_);
            } else {
                fail("step_once: inpaint blend reached non-SD1.5 scheduler "
                     "(this should be impossible — Flux guard in prime() "
                     "should have rejected it)");
            }
        }, scheduler_);

        // Blend: latent = mask_b * latent + (1 - mask_b) * x0_renoised.
        bt::mul_inplace(state.latent, inpaint_mask_b_);
        bt::mul_inplace(inpaint_renoise_buf_, inpaint_one_minus_b_);
        bt::add_inplace(state.latent, inpaint_renoise_buf_);
    }
}

// schedule_sigmas, decode, generate and expand_init_noise live in
// pipeline_decode.cpp.
//
// The Krea 2 research hooks (krea_*, the runtime-adapter LoRA controls,
// reload_krea2_text_encoder and from_model_dir's Krea 2 branch) live in
// pipeline_krea2.cpp.

}  // namespace brodiffusion::pipeline
