// Weight loading for Pipeline: the model-directory entry point, the three
// load_weights overloads, and LoRA application. Moved verbatim out of
// pipeline.cpp — see pipeline_detail.h for the file map.
//
// Krea 2 and Qwen-Image 2.1 have their own from_model_dir_*_ bodies in
// pipeline_krea2.cpp / pipeline_qwenimage21.cpp; this file dispatches to them.

#include "brodiffusion/pipeline.h"

#include "pipeline_detail.h"

#include "brodiffusion/detail/safetensors_dir.h"
#include "brodiffusion/dit/flux.h"
#include "brodiffusion/dit/krea2.h"
#include "brodiffusion/dit/pixart.h"
#include "brodiffusion/lora.h"
#include "brodiffusion/model_config.h"
#include "brodiffusion/unet.h"

#include "brolm/clip.h"
#include "brolm/t5.h"
#include "brolm/tokenizer.h"
#include "brolm/tokenizer_t5.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::pipeline {

using namespace detail_pipe;

Pipeline Pipeline::from_model_dir(const std::string& model_dir,
                                  const ModelDirOptions& dir_opts) {
    namespace fs = std::filesystem;
    const ModelConfig mc = load_model_config(model_dir);

    PipelineConfig cfg;
    cfg.model_class   = mc.model_class;
    cfg.unet          = mc.unet;
    cfg.flux          = mc.flux;
    cfg.sana          = mc.sana;
    cfg.vae           = mc.vae;
    cfg.dcae          = mc.dcae;
    cfg.text_encoder  = mc.text_encoder;
    cfg.t5            = mc.t5;
    cfg.t5_max_length = mc.t5_max_length;
    cfg.gemma         = mc.gemma;
    cfg.sana_max_seq_len = mc.sana_max_seq_len;
    cfg.krea2         = mc.krea2;
    cfg.qwenimage21   = mc.qwenimage21;
    cfg.scheduler     = mc.scheduler;
    if (dir_opts.quantize) {
        cfg.unet.quantize_weights = true;
        cfg.flux.quantize_weights = true;
        cfg.t5.quantize_weights   = true;
        cfg.krea2.transformer.quantize_weights = true;
        cfg.krea2.text.text.quantize_weights   = true;
        cfg.qwenimage21.transformer.quantize_weights = true;
    }
    if (mc.model_class == ModelClass::QwenImage21) {
        // The 2.1 text backbone is Qwen3-VL *8B* — ~17 GB at BF16, which
        // cannot share a 24 GB card with a 14 GB DiT under any arrangement.
        // INT8 (W8A16, ~8.5 GB) is how this model class runs on one GPU at
        // all, so it is the default here rather than a --quantize opt-in; the
        // parity script measures the cost (cosine ~0.997 against an fp32
        // reference, tighter than the bf16 reference's own 0.996).
        cfg.qwenimage21.text.text.quantize_weights =
            (brotensor::default_device() != brotensor::Device::CPU);
        // Nothing in the diffusion path ever reads logits from this backbone —
        // only hidden states — so the untied lm_head (151936x4096, 1.2 GiB at
        // BF16) is dead weight. Telling brolm the embeddings are tied makes it
        // skip that tensor entirely.
        cfg.qwenimage21.text.text.tie_word_embeddings = true;
    }

    const fs::path root(model_dir);

    if (mc.model_class == ModelClass::Sana) {
        // Sana's text frontend is Gemma-2, not CLIP/T5 — load that tokenizer
        // (the CLIP vocab.json / merges.txt below don't exist in a Sana dir).
        brolm::gemma::Tokenizer gemma_tok = brolm::gemma::Tokenizer::load(
            (root / "tokenizer" / "tokenizer.json").string());

        Pipeline p(cfg, std::move(gemma_tok));

        // transformer (single-file) → SanaDenoiser; DC-AE decoder under the
        // "decoder." subtree of the VAE file; Gemma-2 text encoder may be
        // sharded (2 fp16 shards) — search every shard by name.
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

        p.denoiser_->load_weights(tf_files.front(), "");
        t = stamp("DiT weights", t);
        p.dcae_->load_weights(vae_files.front(), "decoder.");
        t = stamp("DC-AE weights", t);

        std::vector<const brotensor::safetensors::File*> te_ptrs;
        for (const auto& f : te_files) te_ptrs.push_back(&f);
        p.gemma_model_->load_weights(te_ptrs, "");
        t = stamp("Gemma weights", t);

        return p;
    }

    if (mc.model_class == ModelClass::Krea2) {
        return from_model_dir_krea2_(model_dir, cfg, dir_opts);
    }

    if (mc.model_class == ModelClass::QwenImage21) {
        return from_model_dir_qwenimage21_(model_dir, cfg, dir_opts);
    }

    if (mc.model_class == ModelClass::PixArt) {
        // PixArt-Sigma uses the SDXL KL-VAE, which overflows FP16 internally
        // (its config ships force_upcast=false, but diffusers always runs this
        // VAE upcast). Force the upcast so the decoder runs in BF16's FP32-range
        // — without it every decoded pixel saturates to black.
        cfg.vae.force_upcast = true;
        // PixArt-Sigma: T5-XXL frontend (no CLIP), KL-VAE, PixArt DiT. The
        // T5-XXL encoder is byte-identical to Flux's and is typically NOT
        // bundled in a PixArt model dir (it dominates the download). Resolve it
        // from, in priority order: $BRODIFFUSION_T5_DIR, a bundled
        // <dir>/text_encoder, or a sibling <dir>/../t5-xxl.
        fs::path t5_weights_dir;
        fs::path t5_tok_path;
        if (const char* e = std::getenv("BRODIFFUSION_T5_DIR"); e && *e) {
            t5_weights_dir = fs::path(e);
            t5_tok_path    = t5_weights_dir / "tokenizer.json";
        } else {
            bool bundled = false;
            const fs::path te_dir = root / "text_encoder";
            if (fs::exists(te_dir)) {
                for (const auto& de : fs::directory_iterator(te_dir)) {
                    if (de.path().extension() == ".safetensors") {
                        bundled = true;
                        break;
                    }
                }
            }
            if (bundled) {
                t5_weights_dir = te_dir;
                t5_tok_path    = root / "tokenizer" / "tokenizer.json";
            } else {
                t5_weights_dir = root.parent_path() / "t5-xxl";
                t5_tok_path    = t5_weights_dir / "tokenizer.json";
            }
        }
        // Tokenizer fallback to a sibling t5-xxl if the resolved path is absent
        // (e.g. a bundled dir that ships spiece.model but no tokenizer.json).
        if (!fs::exists(t5_tok_path)) {
            const fs::path alt = root.parent_path() / "t5-xxl" / "tokenizer.json";
            if (fs::exists(alt)) t5_tok_path = alt;
        }
        if (!fs::exists(t5_tok_path)) {
            fail("from_model_dir: PixArt T5 tokenizer.json not found (looked at '" +
                 t5_tok_path.string() + "'). Set BRODIFFUSION_T5_DIR to a "
                 "t5-xxl directory or place one alongside the model dir.");
        }

        brolm::t5::Tokenizer t5_tok =
            brolm::t5::Tokenizer::load(t5_tok_path.string());
        Pipeline p(cfg, std::move(t5_tok));

        auto vae_files = detail::open_component_files((root / "vae").string());
        auto tf_files  = detail::open_component_files(
            (root / "transformer").string());
        auto t5_files  = detail::open_component_files(t5_weights_dir.string());
        if (t5_files.empty()) {
            fail("from_model_dir: PixArt T5 weights not found in '" +
                 t5_weights_dir.string() + "'. Set BRODIFFUSION_T5_DIR or "
                 "run scripts/download-weights.sh t5-xxl.");
        }

        auto* pix = dynamic_cast<dit::PixArtDenoiser*>(p.denoiser_.get());
        if (!pix) fail("from_model_dir: PixArt denoiser construction failed");
        pix->load_weights(tf_files.front(), "");
        p.vae_.load_weights(vae_files.front(), "decoder.");
        p.vae_encoder_.load_weights(vae_files.front(), "encoder.");

        std::vector<const brotensor::safetensors::File*> t5_ptrs;
        for (const auto& f : t5_files) t5_ptrs.push_back(&f);
        p.t5_encoder_->load_weights(t5_ptrs, "");

        return p;
    }

    // CLIP tokenizer (SD / Flux).
    brolm::clip::Tokenizer clip_tok = brolm::clip::Tokenizer::load(
        (root / "tokenizer" / "vocab.json").string(),
        (root / "tokenizer" / "merges.txt").string());

    if (mc.model_class == ModelClass::Flux) {
        // T5 tokenizer for the second text encoder.
        brolm::t5::Tokenizer t5_tok = brolm::t5::Tokenizer::load(
            (root / "tokenizer_2" / "tokenizer.json").string());

        Pipeline p(cfg, std::move(clip_tok), std::move(t5_tok));

        // Load component weights. CLIP + VAE are single-file; the Flux
        // transformer and the T5-XXL encoder may be sharded — search every
        // shard by name (no .index.json parse needed).
        auto te_files  = detail::open_component_files(
            (root / "text_encoder").string());
        auto vae_files = detail::open_component_files(
            (root / "vae").string());
        auto tf_files  = detail::open_component_files(
            (root / "transformer").string());
        auto t52_files = detail::open_component_files(
            (root / "text_encoder_2").string());

        p.text_encoder_.load_weights(te_files.front(), "text_model.");
        p.vae_.load_weights(vae_files.front(), "decoder.");
        // The diffusers Flux VAE ships an encoder; img2img priming uses it
        // (Flux has no quant_conv, which the encoder load detects).
        p.vae_encoder_.load_weights(vae_files.front(), "encoder.");

        std::vector<const brotensor::safetensors::File*> tf_ptrs;
        for (const auto& f : tf_files) tf_ptrs.push_back(&f);
        auto* flux = dynamic_cast<dit::FluxDenoiser*>(p.denoiser_.get());
        if (!flux) fail("from_model_dir: Flux denoiser construction failed");
        flux->load_weights(tf_ptrs, "");

        std::vector<const brotensor::safetensors::File*> t52_ptrs;
        for (const auto& f : t52_files) t52_ptrs.push_back(&f);
        p.t5_encoder_->load_weights(t52_ptrs, "");

        return p;
    }

    // StableDiffusion: single-file CLIP / UNet / VAE.
    Pipeline p(cfg, std::move(clip_tok));
    auto te_files   = detail::open_component_files(
        (root / "text_encoder").string());
    auto unet_files = detail::open_component_files(
        (root / "unet").string());
    auto vae_files  = detail::open_component_files(
        (root / "vae").string());
    p.load_weights(te_files.front(), unet_files.front(), vae_files.front());
    return p;
}

void Pipeline::load_weights(const brotensor::safetensors::File& f) {
    load_weights(f,
                 "cond_stage_model.transformer.text_model.",
                 "model.diffusion_model.",
                 "first_stage_model.decoder.");
}

void Pipeline::load_weights(const brotensor::safetensors::File& f,
                            const std::string& text_prefix,
                            const std::string& unet_prefix,
                            const std::string& vae_prefix) {
    text_encoder_.load_weights(f, text_prefix);
    denoiser_->load_weights(f, unet_prefix);
    vae_.load_weights(f, vae_prefix);
    // The encoder prefix is derived from the decoder prefix: strip trailing
    // "decoder." and append "encoder." (so "first_stage_model.decoder." ->
    // "first_stage_model.encoder."). Encoder's load_weights handles the
    // sibling quant_conv lookup off its parent automatically.
    vae_encoder_.load_weights(f, encoder_prefix_from_decoder(vae_prefix));
}

void Pipeline::load_weights(const brotensor::safetensors::File& text_file,
                            const brotensor::safetensors::File& unet_file,
                            const brotensor::safetensors::File& vae_file) {
    text_encoder_.load_weights(text_file, "text_model.");
    denoiser_->load_weights(unet_file, "");
    vae_.load_weights(vae_file, "decoder.");
    // Encoder lives in the same diffusers VAE safetensors as the decoder.
    vae_encoder_.load_weights(vae_file, "encoder.");
}

int Pipeline::apply_lora(const brotensor::safetensors::File& f, float scale) {
    const std::vector<lora::Triple> triples = lora::enumerate(f);
    if (triples.empty()) {
        fail("apply_lora: no LoRA triples found in file");
    }
    // Krea 2: transformer-domain triples become ONE runtime-adapter group on
    // the DiT (never merged — the base linears may be INT8; see
    // Krea2Transformer2DModel::add_lora). set_lora_scale()/clear_loras()
    // address the group afterwards.
    if (model_class_ == ModelClass::Krea2) {
        std::vector<dit::Krea2Transformer2DModel::LoraTarget> targets;
        targets.reserve(triples.size());
        for (const lora::Triple& t : triples) {
            if (t.domain != "transformer") {
                fail("apply_lora: '" + t.domain + "' LoRA target in a Krea 2 "
                     "pipeline (need transformer-domain keys)");
            }
            targets.push_back({t.target_path, &f.get(t.down_key),
                               &f.get(t.up_key),
                               t.alpha / static_cast<float>(t.rank)});
        }
        return static_cast<dit::Krea2Denoiser*>(denoiser_.get())
            ->model()
            .add_lora(targets, scale);
    }
    for (const lora::Triple& t : triples) {
        if (t.domain == "transformer") {
            fail("apply_lora: transformer-domain (DiT) LoRA but the active "
                 "model is not Krea 2");
        }
    }
    for (const lora::Triple& t : triples) {
        const float scale_total = (static_cast<float>(t.alpha) /
                                   static_cast<float>(t.rank)) * scale;
        const brotensor::safetensors::TensorView& down = f.get(t.down_key);
        const brotensor::safetensors::TensorView& up   = f.get(t.up_key);
        if (t.domain == "unet") {
            unet::UNet* u = denoiser_->as_unet();
            if (u == nullptr) {
                fail("apply_lora: UNet LoRA target but the active denoiser "
                     "is not a UNet");
            }
            u->apply_lora_delta(t.target_path, down, up, scale_total);
        } else if (t.domain == "text_encoder") {
            text_encoder_.apply_lora_delta(t.target_path, down, up, scale_total);
        } else {
            fail("apply_lora: unknown domain '" + t.domain + "'");
        }
    }
    return -1;   // merged in place — no adapter group to address
}

}  // namespace brodiffusion::pipeline
