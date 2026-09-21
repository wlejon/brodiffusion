// Qwen-Image 2.1-specific Pipeline members.
//
// Only the component loading lives here — everything the generation loop does
// (prime / step_once / decode / generate) is the model-agnostic machinery in
// pipeline.cpp, which branches on ModelClass::QwenImage21 in a handful of
// places (text encode, latent grid, dynamic-shift token count, VAE route).
// This file is the counterpart to pipeline_krea2.cpp: it reads the four
// components out of a diffusers model directory and hands back a loaded
// Pipeline.
//
// Directory layout this expects (HF `Qwen/Qwen-Image-2.1`):
//
//   transformer/    2 BF16 shards of the 7.1B block-causal DiT
//   vae/            one file, AutoencoderKLQwenImage21 (16x, RGBA, z_dim 64)
//   text_encoder/   4 shards of Qwen3-VL-8B (language_model.* + visual.*)
//   processor/      vocab.json + merges.txt for the Qwen3-VL tokenizer
//
// Note the tokenizer's home: Krea 2 keeps it under tokenizer/, but a 2.1 dir
// ships a full `Qwen3VLProcessor` (the image preprocessor rides along for the
// edit path) and the BPE files live under processor/.

#include "brodiffusion/pipeline.h"

#include "brodiffusion/denoiser.h"
#include "brodiffusion/dit/qwenimage21.h"
#include "brodiffusion/detail/safetensors_dir.h"

#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"

#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::pipeline {

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("pipeline::Pipeline: " + msg);
}

// The diffusers packaging strips the "model." wrapper a standalone Qwen3-VL
// checkpoint uses, so its tensors are top-level "language_model.*". Qwen-Image
// 2.1 ships them the other way round ("model.language_model.*"), and both
// spellings appear in the wild — probe rather than assume.
std::string language_model_prefix(
        const std::vector<const brotensor::safetensors::File*>& shards) {
    for (const auto* f : shards) {
        if (f->find("model.language_model.embed_tokens.weight")) {
            return "model.language_model.";
        }
        if (f->find("language_model.embed_tokens.weight")) {
            return "language_model.";
        }
    }
    return "model.language_model.";
}

// Same probe for the vision tower. Returns nullptr when the shard set carries
// no tower at all (a text-only text_encoder override), which is not fatal:
// text-to-image never touches it.
const char* vision_prefix(
        const std::vector<const brotensor::safetensors::File*>& shards) {
    for (const auto* f : shards) {
        if (f->find("visual.patch_embed.proj.weight")) return "visual.";
        if (f->find("model.visual.patch_embed.proj.weight")) {
            return "model.visual.";
        }
    }
    return nullptr;
}

}  // namespace

Pipeline Pipeline::from_model_dir_qwenimage21_(const std::string& model_dir,
                                               const PipelineConfig& cfg,
                                               const ModelDirOptions& dir_opts) {
    namespace fs = std::filesystem;
    const fs::path root(model_dir);

    brolm::qwen3vl::Tokenizer qwen_tok = brolm::qwen3vl::Tokenizer::load(
        (root / "processor" / "vocab.json").string(),
        (root / "processor" / "merges.txt").string());

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
    auto tf_files  = detail::open_component_files((root / "transformer").string());
    auto vae_files = detail::open_component_files((root / "vae").string());
    auto te_files  = detail::open_component_files((root / "text_encoder").string());
    t = stamp("open files (mmap)", t);

    const auto& cancel = dir_opts.should_cancel;
    auto check_cancel = [&]() { if (cancel && cancel()) throw LoadCancelled{}; };

    std::vector<const brotensor::safetensors::File*> tf_ptrs;
    for (const auto& f : tf_files) tf_ptrs.push_back(&f);
    auto* qi21 = dynamic_cast<dit::QwenImage21Denoiser*>(p.denoiser_.get());
    if (!qi21) fail("from_model_dir: QwenImage21 denoiser construction failed");
    check_cancel();
    // 14 GB at BF16 (7 quantized), so the loader polls `cancel` per block.
    qi21->load_weights(tf_ptrs, "", cancel);
    t = stamp("DiT weights", t);

    check_cancel();
    // One file holds both halves of the autoencoder. The encoder is loaded
    // even though text-to-image never runs it: it is ~200 MB against the DiT's
    // 7-14 GB, and having it resident is what makes the image-conditioned
    // paths a pipeline call rather than a reload.
    p.vae_qi21_->load_weights(vae_files.front(), "");
    p.vae_qi21_encoder_->load_weights(vae_files.front(), "");
    t = stamp("Qwen-Image 2.1 VAE weights", t);

    std::vector<const brotensor::safetensors::File*> te_ptrs;
    for (const auto& f : te_files) te_ptrs.push_back(&f);
    check_cancel();
    p.qwen3vl_model_->load_weights(te_ptrs, language_model_prefix(te_ptrs));
    t = stamp("Qwen3-VL 8B weights", t);

    check_cancel();
    // The vision tower rides along in the same shards. It is what turns a
    // condition image into rows of the prompt stream, so the edit path needs
    // it resident; at ~1 GB against the backbone's 8.5 it is not what decides
    // whether a card fits the model. A checkpoint without one still loads —
    // text-to-image never reads it, and the edit entry points say so.
    if (const char* vp = vision_prefix(te_ptrs)) {
        p.qwen3vl_vision_->load_weights(te_ptrs, vp);
        t = stamp("Qwen3-VL vision tower weights", t);
    } else {
        p.qwen3vl_vision_.reset();
    }

    return p;
}

}  // namespace brodiffusion::pipeline
