// Qwen-Image 2.1 research hooks on Pipeline — the qi21_* block declared in
// pipeline.h.
//
// Three groups, and the reason each needs Pipeline rather than the bare DiT:
//
//   * DiT forwarding (mod delta, gate dials, time-mod readout). Mostly a
//     straight delegation, except that Pipeline knows about the prefix KV
//     cache living on the prepared conditioning and can invalidate it when a
//     hook touches the prefix side — which the DiT, holding no cache, cannot.
//   * conditioning entry points (encode_prompt / prime_from_text / the prefix
//     cache surface / the text rows). These need the Qwen3-VL backbone and
//     the prepared payload, both of which Pipeline owns.
//   * the image seam and text-encoder residency. The 16x RGBA VAE encoder is
//     already resident for the image-conditioned paths, and the 8.5 GiB
//     Qwen3-VL-8B backbone is dead weight during the denoise loop — releasing
//     it is what makes a BF16 DiT or a bigger canvas fit a 24 GB card.
//
// Component loading for the model class lives in pipeline_qwenimage21.cpp;
// the generation loop itself is the model-agnostic machinery in pipeline.cpp.

#include "brodiffusion/pipeline.h"

#include "brodiffusion/denoiser.h"
#include "brodiffusion/dit/qwenimage21.h"
#include "brodiffusion/flow_match_scheduler.h"
#include "brodiffusion/qwenimage21_text.h"
#include "brodiffusion/detail/safetensors_dir.h"

#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"

#include "brotensor/gguf.h"
#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace brodiffusion::pipeline {

namespace bt = ::brotensor;

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("pipeline::Pipeline: " + msg);
}

dit::QwenImage21Denoiser& qi21_denoiser(ModelClass model_class,
                                        const std::unique_ptr<Denoiser>& d,
                                        const char* who) {
    if (model_class != ModelClass::QwenImage21) {
        fail(std::string(who) + ": Qwen-Image 2.1 only");
    }
    auto* den = dynamic_cast<dit::QwenImage21Denoiser*>(d.get());
    if (!den) fail(std::string(who) + ": no Qwen-Image 2.1 denoiser");
    return *den;
}

dit::QwenImage21Transformer2DModel& qi21_model(
    ModelClass model_class, const std::unique_ptr<Denoiser>& d,
    const char* who) {
    return qi21_denoiser(model_class, d, who).model();
}

// Load the Qwen3-VL language-model subtree into `model` from an override path
// (a .gguf, or a diffusers safetensors file/dir) or, when empty, from
// `<model_dir>/text_encoder`. Qwen-Image 2.1 ships the backbone as
// "model.language_model.*"; a standalone checkpoint may use the bare
// "language_model.*" spelling, so probe rather than assume.
void load_qwen3vl_text_weights(brolm::qwen3vl::TextModel& model,
                               const std::string& model_dir,
                               const std::string& override_path) {
    namespace fs = std::filesystem;
    auto lm_prefix =
        [](const std::vector<const bt::safetensors::File*>& sh) -> std::string {
        for (const auto* f : sh) {
            if (f->find("model.language_model.embed_tokens.weight")) {
                return "model.language_model.";
            }
            if (f->find("language_model.embed_tokens.weight")) {
                return "language_model.";
            }
        }
        return "model.language_model.";
    };

    std::vector<bt::safetensors::File> files;
    if (override_path.empty()) {
        files = detail::open_component_files(
            (fs::path(model_dir) / "text_encoder").string());
    } else {
        const fs::path ovr(override_path);
        if (ovr.has_extension() &&
            (ovr.extension() == ".gguf" || ovr.extension() == ".GGUF")) {
            bt::gguf::File gf = bt::gguf::File::open(ovr.string());
            model.load_weights(gf);
            return;
        }
        if (fs::is_directory(ovr)) {
            files = detail::open_component_files(ovr.string());
        } else {
            files.push_back(bt::safetensors::File::open(ovr.string()));
        }
    }
    std::vector<const bt::safetensors::File*> ptrs;
    for (const auto& f : files) ptrs.push_back(&f);
    model.load_weights(ptrs, lm_prefix(ptrs));
}

}  // namespace

// ── DiT hook forwarding ────────────────────────────────────────────────────

void Pipeline::qi21_set_mod_delta(const bt::Tensor& delta, int block_lo,
                                  int block_hi,
                                  dit::QwenImage21ModTarget target) {
    qi21_model(model_class_, denoiser_, "qi21_set_mod_delta")
        .set_mod_delta(delta, block_lo, block_hi, target);
    // The prefix (t = 0) rows are only computed on an extract step, so a
    // prefix-side delta is invisible until the cache is dropped. Do it for
    // the caller — the failure mode otherwise is a hook that silently does
    // nothing for the rest of the generation. Clearing one needs the same
    // treatment, hence the remembered flag.
    const bool hits_prefix = delta.size() > 0 && block_hi > block_lo &&
                             target != dit::QwenImage21ModTarget::Target;
    if (hits_prefix || qi21_mod_delta_hits_prefix_) qi21_reset_cache();
    qi21_mod_delta_hits_prefix_ = hits_prefix;
}

void Pipeline::qi21_time_mod(float timestep, bt::Tensor& temb_out,
                             bt::Tensor& mod_out) {
    // Callers work in the 0..1000-scale timestep qi21_step_timestep() returns
    // and step_once() consumes; QwenImage21Denoiser::forward() divides by
    // 1000 before the model's own flow-time convention, so mirror that here
    // rather than leaking the internal scale.
    qi21_model(model_class_, denoiser_, "qi21_time_mod")
        .compute_time_mod(timestep / 1000.0f, temb_out, mod_out);
}

float Pipeline::qi21_step_timestep(const PipelineState& state) const {
    if (model_class_ != ModelClass::QwenImage21) {
        fail("qi21_step_timestep: Qwen-Image 2.1 only");
    }
    return std::get<scheduler::FlowMatch>(scheduler_).timesteps()
        .at(static_cast<std::size_t>(state.step_index));
}

void Pipeline::qi21_set_gate_scale(float attn_scale, float mlp_scale,
                                   float txt_scale, float img_scale,
                                   int block_lo, int block_hi) {
    qi21_model(model_class_, denoiser_, "qi21_set_gate_scale")
        .set_gate_scale(attn_scale, mlp_scale, txt_scale, img_scale, block_lo,
                        block_hi);
    // The prefix gate only ever fires on an extract step, so a change to the
    // factors the PREFIX rows see needs the cache re-extracted to land —
    // while a pure img_scale sweep (the common per-step dial) must not pay
    // for a re-extract.
    const bool ranged = block_hi > block_lo;
    const float pa = ranged ? attn_scale * txt_scale : 1.0f;
    const float pm = ranged ? mlp_scale  * txt_scale : 1.0f;
    if (pa != qi21_prefix_gate_attn_ || pm != qi21_prefix_gate_mlp_) {
        qi21_reset_cache();
        qi21_prefix_gate_attn_ = pa;
        qi21_prefix_gate_mlp_  = pm;
    }
}

void Pipeline::qi21_set_gate_delta(const bt::Tensor& delta, int block_lo,
                                   int block_hi,
                                   dit::QwenImage21ModTarget target) {
    qi21_model(model_class_, denoiser_, "qi21_set_gate_delta")
        .set_gate_delta(delta, block_lo, block_hi, target);
    // Same prefix-cache rule as qi21_set_mod_delta(): the t = 0 gate is only
    // ever applied on an extract step, so arming OR clearing a prefix-side
    // delta needs the cache dropped for the change to land.
    const bool hits_prefix = delta.size() > 0 && block_hi > block_lo &&
                             target != dit::QwenImage21ModTarget::Target;
    if (hits_prefix || qi21_gate_delta_hits_prefix_) qi21_reset_cache();
    qi21_gate_delta_hits_prefix_ = hits_prefix;
}

void Pipeline::qi21_set_gate_mask(const bt::Tensor& mask, int block_lo,
                                  int block_hi) {
    qi21_model(model_class_, denoiser_, "qi21_set_gate_mask")
        .set_gate_mask(mask, block_lo, block_hi);
    // A mask spans the joint sequence, so its prefix half is in the same
    // position as a prefix-side delta: only an extract step applies it.
    const bool armed = mask.size() > 0 && block_hi > block_lo;
    if (armed || qi21_gate_mask_armed_) qi21_reset_cache();
    qi21_gate_mask_armed_ = armed;
}

void Pipeline::qi21_capture_gates(bool enable) {
    auto& model = qi21_model(model_class_, denoiser_, "qi21_capture_gates");
    if (enable) {
        model.capture_gates(&qi21_gate_sink_);
    } else {
        model.capture_gates(nullptr);
        qi21_gate_sink_.clear();
    }
}

std::vector<float> Pipeline::qi21_gates() const {
    if (model_class_ != ModelClass::QwenImage21) {
        fail("qi21_gates: Qwen-Image 2.1 only");
    }
    return qi21_gate_sink_;
}

void Pipeline::qi21_set_norm_out_scale_delta(const bt::Tensor& delta) {
    qi21_model(model_class_, denoiser_, "qi21_set_norm_out_scale_delta")
        .set_norm_out_scale_delta(delta);
}

int Pipeline::qi21_hidden_size() const {
    return qi21_model(model_class_, denoiser_, "qi21_hidden_size")
        .config().hidden_size();
}

int Pipeline::qi21_num_layers() const {
    return qi21_model(model_class_, denoiser_, "qi21_num_layers")
        .config().num_layers;
}

int Pipeline::qi21_text_hidden_dim() const {
    return qi21_model(model_class_, denoiser_, "qi21_text_hidden_dim")
        .config().context_in_dim;
}

int Pipeline::qi21_latent_channels() const {
    return qi21_model(model_class_, denoiser_, "qi21_latent_channels")
        .config().latent_channels();
}

// ── prefix cache surface ───────────────────────────────────────────────────

void Pipeline::qi21_reset_cache() {
    if (model_class_ != ModelClass::QwenImage21) return;
    auto prepared = last_prepared_.lock();
    if (!prepared) return;   // nothing primed, or the state was dropped
    auto& den = qi21_denoiser(model_class_, denoiser_, "qi21_reset_cache");
    den.reset_cache(*prepared);
}

void Pipeline::qi21_scale_prefix_kv(int layer_lo, int layer_hi, float k_scale,
                                    float v_scale) {
    auto& den = qi21_denoiser(model_class_, denoiser_, "qi21_scale_prefix_kv");
    auto prepared = last_prepared_.lock();
    if (!prepared) {
        fail("qi21_scale_prefix_kv: nothing primed — call prime() (and run at "
             "least one step, so the prefix has been extracted) first");
    }
    auto& cache = den.prefix_cache(*prepared, /*uncond=*/false);
    if (!cache.valid()) {
        fail("qi21_scale_prefix_kv: the prefix has not been extracted yet — "
             "run one step first");
    }
    cache.scale_kv(layer_lo, layer_hi, k_scale, v_scale);
    if (den.has_uncond(*prepared)) {
        auto& ucache = den.prefix_cache(*prepared, /*uncond=*/true);
        if (ucache.valid()) ucache.scale_kv(layer_lo, layer_hi, k_scale, v_scale);
    }
}

bt::Tensor Pipeline::qi21_text_rows(bool uncond) const {
    // text_rows() is a mutable accessor on the denoiser; this const overload
    // hands back a copy, which is what a reader wants anyway.
    auto& den = qi21_denoiser(model_class_, denoiser_, "qi21_text_rows");
    auto prepared = last_prepared_.lock();
    if (!prepared) fail("qi21_text_rows: nothing primed — call prime() first");
    return den.text_rows(*prepared, uncond);
}

void Pipeline::qi21_set_text_rows(const bt::Tensor& rows, bool uncond) {
    auto& den = qi21_denoiser(model_class_, denoiser_, "qi21_set_text_rows");
    auto prepared = last_prepared_.lock();
    if (!prepared) {
        fail("qi21_set_text_rows: nothing primed — call prime() first");
    }
    bt::Tensor& dst = den.text_rows(*prepared, uncond);
    const int H = den.config().hidden_size();
    if (rows.cols != H || rows.rows <= 0) {
        fail("qi21_set_text_rows: rows must be (n, qi21_hidden_size())");
    }
    bt::Tensor src = rows.to(bt::default_device());
    if (src.dtype != den.compute_dtype()) {
        bt::Tensor t;
        bt::cast(src, t, den.compute_dtype());
        src = std::move(t);
    }
    dst = std::move(src);
    // The joint sequence's text half just changed; the cached prefix K/V
    // describe the old one.
    den.reset_cache(*prepared);
}

// ── conditioning entry points ──────────────────────────────────────────────

qwenimage21::TextConditioning Pipeline::qi21_encode_prompt(
    std::string_view prompt) {
    if (model_class_ != ModelClass::QwenImage21) {
        fail("qi21_encode_prompt: Qwen-Image 2.1 only");
    }
    if (!qwen3vl_model_ || !qwen3vl_tokenizer_) {
        fail("qi21_encode_prompt: no Qwen3-VL text encoder (released?)");
    }
    return qwenimage21::encode_prompt(*qwen3vl_tokenizer_, *qwen3vl_model_,
                                      std::string(prompt));
}

PipelineState Pipeline::qi21_prime_from_text(const bt::Tensor& embeds,
                                             const bt::Tensor& mask,
                                             const bt::Tensor* uncond_embeds,
                                             const bt::Tensor* uncond_mask,
                                             const GenerateOptions& opts) {
    if (model_class_ != ModelClass::QwenImage21) {
        fail("qi21_prime_from_text: Qwen-Image 2.1 only");
    }
    const int TH = qi21_text_hidden_dim();
    if (embeds.cols != TH || embeds.rows <= 0) {
        fail("qi21_prime_from_text: embeds must be (n, qi21_text_hidden_dim())");
    }
    qwenimage21::TextConditioning tc;
    tc.embeds = embeds;
    tc.mask = mask;
    qi21_text_override_ = std::move(tc);
    if (uncond_embeds != nullptr) {
        qwenimage21::TextConditioning utc;
        utc.embeds = *uncond_embeds;
        utc.mask = uncond_mask != nullptr ? *uncond_mask : bt::Tensor{};
        qi21_uncond_text_override_ = std::move(utc);
    } else {
        qi21_uncond_text_override_.reset();
    }
    // prime()'s `prompt` argument only feeds the paths the overrides bypass;
    // a missing uncond override still falls back to encoding
    // opts.negative_prompt when guidance_scale > 1.
    return prime(std::string_view{}, opts);
}

// ── image seam ─────────────────────────────────────────────────────────────

bt::Tensor Pipeline::qi21_encode_image(const float* pixels, int H, int W) {
    if (model_class_ != ModelClass::QwenImage21) {
        fail("qi21_encode_image: Qwen-Image 2.1 only");
    }
    if (!vae_qi21_encoder_) fail("qi21_encode_image: VAE encoder not loaded");
    if (pixels == nullptr || H <= 0 || W <= 0) {
        fail("qi21_encode_image: pixels/H/W required");
    }
    const int S = vae_qi21_encoder_->config().spatial_scale();
    if (H % S != 0 || W % S != 0) {
        fail("qi21_encode_image: H and W must be multiples of " +
             std::to_string(S));
    }
    const int in_ch = vae_qi21_encoder_->config().in_channels;
    if (in_ch != 3 && in_ch != 4) {
        fail("qi21_encode_image: unexpected VAE input channel count");
    }
    // The autoencoder is RGBA; the pipeline's contract is RGB. Append an
    // opaque alpha plane (the value the decoder emits for a generation), and
    // map [0,1] to the [-1,1] the encoder expects.
    const std::size_t plane = static_cast<std::size_t>(H) * W;
    std::vector<float> nchw(static_cast<std::size_t>(in_ch) * plane);
    for (std::size_t i = 0; i < static_cast<std::size_t>(3) * plane; ++i) {
        nchw[i] = pixels[i] * 2.0f - 1.0f;
    }
    if (in_ch == 4) {
        for (std::size_t i = 0; i < plane; ++i) {
            nchw[static_cast<std::size_t>(3) * plane + i] = 1.0f;
        }
    }
    bt::Tensor img =
        bt::Tensor::from_host(nchw.data(), 1, static_cast<int>(nchw.size()))
            .to(bt::default_device());
    bt::Tensor latent;
    vae_qi21_encoder_->encode(img, H, W, nullptr, latent);
    bt::sync_all();
    return latent;
}

std::vector<float> Pipeline::qi21_decode(const bt::Tensor& latent, int h_lat,
                                         int w_lat) {
    if (model_class_ != ModelClass::QwenImage21) {
        fail("qi21_decode: Qwen-Image 2.1 only");
    }
    if (!vae_qi21_) fail("qi21_decode: VAE decoder not loaded");
    if (h_lat <= 0 || w_lat <= 0) fail("qi21_decode: h_lat/w_lat required");
    const int zc = cfg_.qwenimage21.vae.z_dim;
    const std::size_t expect = static_cast<std::size_t>(zc) * h_lat * w_lat;
    if (static_cast<std::size_t>(latent.size()) != expect) {
        fail("qi21_decode: latent must hold z_dim*h_lat*w_lat elements");
    }
    bt::Tensor z = latent.to(bt::default_device());
    z.rows = 1;
    z.cols = static_cast<int>(expect);
    bt::Tensor img;
    vae_qi21_->decode(z, h_lat, w_lat, img);
    bt::sync_all();

    const int S = cfg_.qwenimage21.vae.spatial_scale();
    const int out_ch = cfg_.qwenimage21.vae.out_channels;
    const std::size_t plane =
        static_cast<std::size_t>(h_lat) * S * static_cast<std::size_t>(w_lat) * S;
    std::vector<float> host;
    if (img.dtype == bt::Dtype::FP16 || img.dtype == bt::Dtype::BF16) {
        bt::Tensor f32;
        bt::cast(img, f32, bt::Dtype::FP32);
        bt::sync_all();
        host = f32.to(bt::Device::CPU).to_host_vector();
    } else {
        host = img.to(bt::Device::CPU).to_host_vector();
    }
    // Planar NCHW, so dropping the alpha plane is a truncation — the same
    // rule decode() applies to a generation.
    const std::size_t rgb = static_cast<std::size_t>(3) * plane;
    if (out_ch == 4 && host.size() >= rgb) host.resize(rgb);
    return host;
}

// ── text-encoder residency ─────────────────────────────────────────────────

void Pipeline::qi21_release_text_encoder() {
    if (model_class_ != ModelClass::QwenImage21) {
        fail("qi21_release_text_encoder: Qwen-Image 2.1 only");
    }
    if (!qwen3vl_model_) return;   // already released
    qwen3vl_model_.reset();
    // Free the allocator's now-unowned blocks rather than leaving 8.5 GiB
    // cached: the whole point is to make the freed VRAM available to the DiT
    // (and on Windows/WDDM to stay under the ~21 GiB residency cliff).
    brotensor::device_mem_trim(brotensor::default_device());
}

bool Pipeline::qi21_text_encoder_resident() const {
    if (model_class_ != ModelClass::QwenImage21) {
        fail("qi21_text_encoder_resident: Qwen-Image 2.1 only");
    }
    return qwen3vl_model_.has_value();
}

void Pipeline::qi21_reload_text_encoder(const std::string& model_dir,
                                        const std::string& text_encoder_path,
                                        bool quantize) {
    if (model_class_ != ModelClass::QwenImage21) {
        fail("qi21_reload_text_encoder: not a Qwen-Image 2.1 pipeline");
    }
    // Reconstruct from the stored config so no weight slot carries over from
    // a previous encoder (the INT8 and dense paths populate different ones).
    auto tcfg = cfg_.qwenimage21.text.text;
    tcfg.quantize_weights = quantize;
    qwen3vl_model_.emplace(tcfg);
    load_qwen3vl_text_weights(*qwen3vl_model_, model_dir, text_encoder_path);
}

}  // namespace brodiffusion::pipeline
