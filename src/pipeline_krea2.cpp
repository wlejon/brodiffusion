// Krea 2-specific Pipeline members.
//
// The Krea 2 research seam (krea2_capi / krea-research) hangs a couple of
// dozen methods off Pipeline that only ever touch dit::Krea2Transformer2DModel
// or the Qwen3-VL text backbone: the AdaLN mod-delta / gate dials, the raw-tap
// entry points, the runtime-adapter LoRA controls, and the in-place text
// encoder swap. None of them interact with the model-agnostic generate /
// prime / step / decode machinery in pipeline.cpp, so they live here, along
// with from_model_dir()'s Krea 2 component-loading branch.

#include "brodiffusion/pipeline.h"

#include "brodiffusion/denoiser.h"
#include "brodiffusion/dit/krea2.h"
#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/krea2_text.h"
#include "brodiffusion/detail/safetensors_dir.h"

#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"
#include "brolm/qwen3vl_vl.h"

#include "brotensor/gguf.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace brodiffusion::pipeline {

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("pipeline::Pipeline: " + msg);
}

dit::Krea2Transformer2DModel& krea_model(ModelClass model_class,
                                         const std::unique_ptr<Denoiser>& d,
                                         const char* who) {
    if (model_class != ModelClass::Krea2) {
        fail(std::string(who) + ": Krea 2 only");
    }
    return static_cast<dit::Krea2Denoiser*>(d.get())->model();
}

// Load ONLY the Qwen3-VL text backbone's language-model weights into `model`
// from an override path (a .gguf, or a diffusers safetensors file/dir), or —
// when the override is empty — from `<model_dir>/text_encoder`. The vision
// tower is left to the caller (from_model_dir loads it inline; the in-place
// reload keeps the resident one). Mirrors from_model_dir's Krea2 text branch.
void load_qwen3vl_text_weights(brolm::qwen3vl::TextModel& model,
                               const std::string& model_dir,
                               const std::string& override_path) {
    namespace fs = std::filesystem;
    auto lm_prefix = [](const std::vector<const brotensor::safetensors::File*>& sh)
            -> std::string {
        for (const auto* f : sh) {
            if (f->find("language_model.embed_tokens.weight")) return "language_model.";
            if (f->find("model.language_model.embed_tokens.weight"))
                return "model.language_model.";
        }
        return "language_model.";
    };

    if (override_path.empty()) {
        auto te = detail::open_component_files(
            (fs::path(model_dir) / "text_encoder").string());
        std::vector<const brotensor::safetensors::File*> ptrs;
        for (const auto& f : te) ptrs.push_back(&f);
        model.load_weights(ptrs, "language_model.");
        return;
    }

    const fs::path ovr(override_path);
    const bool is_gguf =
        ovr.has_extension() &&
        (ovr.extension() == ".gguf" || ovr.extension() == ".GGUF");
    if (is_gguf) {
        brotensor::gguf::File gf = brotensor::gguf::File::open(ovr.string());
        model.load_weights(gf);
        return;
    }

    std::vector<brotensor::safetensors::File> files;
    if (fs::is_directory(ovr)) {
        files = detail::open_component_files(ovr.string());
    } else {
        files.push_back(brotensor::safetensors::File::open(ovr.string()));
    }
    std::vector<const brotensor::safetensors::File*> ptrs;
    for (const auto& f : files) ptrs.push_back(&f);
    model.load_weights(ptrs, lm_prefix(ptrs));
}

}  // namespace

Pipeline Pipeline::from_model_dir_krea2_(const std::string& model_dir,
                                         const PipelineConfig& cfg,
                                         const ModelDirOptions& dir_opts) {
    namespace fs = std::filesystem;
    const fs::path root(model_dir);

    // Krea 2: Qwen3-VL-4B text encoder (single-file), Qwen-Image VAE
    // decoder, and the sharded (3-shard) single-stream flow DiT. The
    // tokenizer's vocab.json / merges.txt live under tokenizer/.
    brolm::qwen3vl::Tokenizer qwen_tok = brolm::qwen3vl::Tokenizer::load(
        (root / "tokenizer" / "vocab.json").string(),
        (root / "tokenizer" / "merges.txt").string());

    Pipeline p(cfg, std::move(qwen_tok));

    const bool time_load = std::getenv("BRODIFFUSION_TIME") != nullptr;
    auto stamp = [&](const char* what, auto t0) {
        if (time_load) {
            std::fprintf(stderr, "[time]   %s: %.2f s\n", what,
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count());
        }
        return std::chrono::steady_clock::now();
    };
    auto t = std::chrono::steady_clock::now();
    auto tf_files  = detail::open_component_files(
        (root / "transformer").string());
    auto vae_files = detail::open_component_files(
        (root / "vae").string());
    auto te_files  = detail::open_component_files(
        (root / "text_encoder").string());
    t = stamp("open files (mmap)", t);

    const auto& cancel = dir_opts.should_cancel;
    auto check_cancel = [&]() { if (cancel && cancel()) throw LoadCancelled{}; };

    std::vector<const brotensor::safetensors::File*> tf_ptrs;
    for (const auto& f : tf_files) tf_ptrs.push_back(&f);
    auto* krea2 = dynamic_cast<dit::Krea2Denoiser*>(p.denoiser_.get());
    if (!krea2) fail("from_model_dir: Krea2 denoiser construction failed");
    check_cancel();
    // The DiT is ~25 of the ~26 GB, so it polls `cancel` per block itself.
    krea2->load_weights(tf_ptrs, "", cancel);
    t = stamp("DiT weights", t);

    check_cancel();
    p.vae_qwen_->load_weights(vae_files.front(), "");
    t = stamp("Qwen-Image VAE weights", t);

    // The Qwen3-VL-4B text encoder ships unsharded (text_encoder/
    // model.safetensors); load its language-model subtree. The dir's own
    // shards, as pointers.
    std::vector<const brotensor::safetensors::File*> te_ptrs;
    for (const auto& f : te_files) te_ptrs.push_back(&f);

    // The checkpoint's Qwen3-VL-4B text encoder ships its vision tower too
    // (unused by plain text prompting) — the diffusers-packaged
    // text_encoder strips the "model.*" wrapper a standalone Qwen3-VL
    // checkpoint uses, so its tensors are top-level "visual.*" /
    // "language_model.*". Probe for the right language-model prefix in a
    // shard set (so a raw "model.language_model.*" checkpoint also loads).
    auto lm_prefix = [](const std::vector<const brotensor::safetensors::File*>& sh)
            -> std::string {
        for (const auto* f : sh) {
            if (f->find("language_model.embed_tokens.weight")) return "language_model.";
            if (f->find("model.language_model.embed_tokens.weight"))
                return "model.language_model.";
        }
        return "language_model.";
    };
    auto vis_prefix = [](const std::vector<const brotensor::safetensors::File*>& sh)
            -> const char* {
        for (const auto* f : sh) {
            if (f->find("visual.patch_embed.proj.weight")) return "visual.";
            if (f->find("model.visual.patch_embed.proj.weight")) return "model.visual.";
        }
        return nullptr;
    };

    check_cancel();
    const std::string& te_override = dir_opts.text_encoder_path;
    if (te_override.empty()) {
        // Default: text + vision from the bundled text_encoder/ (top-level
        // "language_model.*" / "visual.*", per the diffusers packaging).
        p.qwen3vl_model_->load_weights(te_ptrs, "language_model.");
        t = stamp("Qwen3-VL weights", t);
        check_cancel();
        p.qwen3vl_vision_->load_weights(te_ptrs, "visual.");
        t = stamp("Qwen3-VL vision tower weights", t);
    } else {
        const fs::path ovr(te_override);
        const bool is_gguf =
            ovr.has_extension() &&
            (ovr.extension() == ".gguf" || ovr.extension() == ".GGUF");
        if (is_gguf) {
            // Text-only gguf: language-model weights from the gguf, vision
            // tower still from the dir's bundled text_encoder.
            brotensor::gguf::File gf = brotensor::gguf::File::open(ovr.string());
            p.qwen3vl_model_->load_weights(gf);
            t = stamp("Qwen3-VL weights (gguf override)", t);
            check_cancel();
            if (const char* vp = vis_prefix(te_ptrs))
                p.qwen3vl_vision_->load_weights(te_ptrs, vp);
            t = stamp("Qwen3-VL vision tower weights", t);
        } else {
            // safetensors file or directory of shards.
            std::vector<brotensor::safetensors::File> ovr_files;
            if (fs::is_directory(ovr)) {
                ovr_files = detail::open_component_files(ovr.string());
            } else {
                ovr_files.push_back(brotensor::safetensors::File::open(ovr.string()));
            }
            std::vector<const brotensor::safetensors::File*> ovr_ptrs;
            for (const auto& f : ovr_files) ovr_ptrs.push_back(&f);
            p.qwen3vl_model_->load_weights(ovr_ptrs, lm_prefix(ovr_ptrs));
            t = stamp("Qwen3-VL weights (safetensors override)", t);
            check_cancel();
            // Prefer the override's own vision tower; fall back to the dir's.
            if (const char* vp = vis_prefix(ovr_ptrs)) {
                p.qwen3vl_vision_->load_weights(ovr_ptrs, vp);
            } else if (const char* vp_dir = vis_prefix(te_ptrs)) {
                p.qwen3vl_vision_->load_weights(te_ptrs, vp_dir);
            }
            t = stamp("Qwen3-VL vision tower weights", t);
        }
    }

    return p;
}

void Pipeline::reload_krea2_text_encoder(const std::string& model_dir,
                                         const std::string& text_encoder_path,
                                         bool quantize) {
    if (cfg_.model_class != ModelClass::Krea2) {
        fail("reload_krea2_text_encoder: not a Krea 2 pipeline");
    }
    // Reconstruct the backbone from the stored config so no weight slot carries
    // over from the previous encoder (the INT8 vs dense/quant paths populate
    // different slots). emplace() frees the old backbone's VRAM before the new
    // one is built — the DiT/VAE/vision tower are untouched.
    auto tcfg = cfg_.krea2.text.text;
    tcfg.quantize_weights = quantize;
    qwen3vl_model_.emplace(tcfg);
    load_qwen3vl_text_weights(*qwen3vl_model_, model_dir, text_encoder_path);
}

// ── Krea 2 research hooks ───────────────────────────────────────────────────

void Pipeline::krea_set_mod_delta(const brotensor::Tensor& delta,
                                  int block_lo, int block_hi) {
    krea_model(model_class_, denoiser_, "krea_set_mod_delta")
        .set_mod_delta(delta, block_lo, block_hi);
}

void Pipeline::krea_time_mod(float timestep, brotensor::Tensor& temb_out,
                             brotensor::Tensor& mod_out) {
    // Callers work in the same 0..1000-scale timestep krea_step_timestep()
    // returns / step_once() consumes; Krea2Denoiser::forward() divides by
    // 1000 before reaching the model's own flow-time convention, so mirror
    // that here rather than leaking the internal scale to JS.
    krea_model(model_class_, denoiser_, "krea_time_mod")
        .compute_time_mod(timestep / 1000.0f, temb_out, mod_out);
}

float Pipeline::krea_step_timestep(const PipelineState& state) const {
    if (model_class_ != ModelClass::Krea2) {
        fail("krea_step_timestep: Krea 2 only");
    }
    return std::get<scheduler::FlowMatch>(scheduler_).timesteps()
        .at(static_cast<std::size_t>(state.step_index));
}

void Pipeline::krea_set_gate_scale(float txt_scale, float img_scale,
                                   int block_lo, int block_hi) {
    krea_model(model_class_, denoiser_, "krea_set_gate_scale")
        .set_gate_scale(txt_scale, img_scale, block_lo, block_hi);
}

void Pipeline::krea_set_gate_mask(const brotensor::Tensor& mask,
                                  int block_lo, int block_hi) {
    krea_model(model_class_, denoiser_, "krea_set_gate_mask")
        .set_gate_mask(mask, block_lo, block_hi);
}

void Pipeline::krea_capture_gates(bool enable) {
    auto& model = krea_model(model_class_, denoiser_, "krea_capture_gates");
    if (enable) {
        model.capture_gates(&krea_gate_sink_);
    } else {
        model.capture_gates(nullptr);
        krea_gate_sink_.clear();
    }
}

std::vector<float> Pipeline::krea_gates() const {
    if (model_class_ != ModelClass::Krea2) fail("krea_gates: Krea 2 only");
    return krea_gate_sink_;
}

void Pipeline::set_lora_scale(int index, float scale) {
    // Only runtime-adapter LoRAs (Krea 2) can rescale after apply_lora();
    // the SD1.5 path merges deltas into the base weights irreversibly.
    krea_model(model_class_, denoiser_, "set_lora_scale")
        .set_lora_scale(index, scale);
}

void Pipeline::clear_loras() {
    krea_model(model_class_, denoiser_, "clear_loras").clear_loras();
}

int Pipeline::num_loras() const {
    return krea_model(model_class_, denoiser_, "num_loras").num_loras();
}

int Pipeline::krea_hidden_size() const {
    return krea_model(model_class_, denoiser_, "krea_hidden_size")
        .config().hidden_size();
}

int Pipeline::krea_num_layers() const {
    return krea_model(model_class_, denoiser_, "krea_num_layers")
        .config().num_layers;
}

krea2::TextConditioning Pipeline::krea_encode_prompt_taps(
    std::string_view prompt) {
    if (model_class_ != ModelClass::Krea2) {
        fail("krea_encode_prompt_taps: Krea 2 only");
    }
    if (!qwen3vl_model_ || !qwen3vl_tokenizer_) {
        fail("krea_encode_prompt_taps: missing Qwen3-VL model / tokenizer");
    }
    return krea2::encode_prompt(*qwen3vl_tokenizer_, *qwen3vl_model_,
                               std::string(prompt));
}

brotensor::Tensor Pipeline::krea_encode_text(
    const brotensor::Tensor& prompt_embeds,
    const brotensor::Tensor& prompt_embeds_mask) {
    auto& model = krea_model(model_class_, denoiser_, "krea_encode_text");
    brotensor::Tensor out;
    model.encode_text(prompt_embeds, prompt_embeds_mask, out);
    return out;
}

krea2::TextConditioning Pipeline::krea_encode_image_prompt(const float* pixels,
                                                           int H, int W) {
    if (model_class_ != ModelClass::Krea2) {
        fail("krea_encode_image_prompt: Krea 2 only");
    }
    if (!qwen3vl_model_ || !qwen3vl_tokenizer_ || !qwen3vl_vision_) {
        fail("krea_encode_image_prompt: missing Qwen3-VL model / tokenizer / "
             "vision tower");
    }
    brolm::qwen3vl::ImageInput img;
    img.pixels = pixels;
    img.H = H;
    img.W = W;
    return krea2::encode_image_prompt(*qwen3vl_tokenizer_, *qwen3vl_model_,
                                      *qwen3vl_vision_, qwen3vl_pp_, img);
}

PipelineState Pipeline::krea_prime_from_taps(
    const brotensor::Tensor& embeds, const brotensor::Tensor& mask,
    const brotensor::Tensor* uncond_embeds,
    const brotensor::Tensor* uncond_mask, const GenerateOptions& opts) {
    if (model_class_ != ModelClass::Krea2) {
        fail("krea_prime_from_taps: Krea 2 only");
    }
    krea_taps_override_ = krea2::TextConditioning{embeds, mask};
    if (uncond_embeds != nullptr && uncond_mask != nullptr) {
        krea_uncond_taps_override_ =
            krea2::TextConditioning{*uncond_embeds, *uncond_mask};
    } else {
        krea_uncond_taps_override_.reset();
    }
    // prime()'s `prompt` argument is only used by the fallback (non-override)
    // path, which the override above bypasses for the positive branch; a
    // missing uncond override still falls back to encoding
    // opts.negative_prompt normally when do_cfg is true.
    return prime(std::string_view{}, opts);
}

}  // namespace brodiffusion::pipeline
