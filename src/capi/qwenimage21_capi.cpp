// qwenimage21_capi — see include/brodiffusion/qwenimage21_capi.h.
//
// A deliberately thin layer, exactly as krea2_capi is: every entry point
// converts caller FP32 buffers to brotensor tensors, calls the same component
// methods the Pipeline drives, and copies results back as FP32. All state
// lives in qi_ctx; errors cross as a per-thread message string and a -1/NULL
// return (no exception escapes).
//
// The one piece of state that is NOT just a wrapped component is the prefix
// KV cache: qi_forward owns it, re-extracting whenever the joint layout
// changes (a different text row count or target grid) and decoding from it
// otherwise. That mirrors what QwenImage21Denoiser does per CFG branch, so a
// ctypes caller driving the step loop by hand gets the same 1.1x-ish per-step
// saving the Pipeline does without managing anything.

#include "brodiffusion/qwenimage21_capi.h"

#include "brodiffusion/detail/safetensors_dir.h"
#include "brodiffusion/dit/qwenimage21.h"
#include "brodiffusion/model_config.h"
#include "brodiffusion/qwenimage21_text.h"
#include "brodiffusion/vae_qwenimage21.h"

#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <array>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace bt = brotensor;
namespace bd = brodiffusion;

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& m) { g_last_error = m; }

// Run `fn` with the exception wall: 0 on success, -1 with the message stored.
template <typename Fn>
int guarded(Fn&& fn) {
    try {
        fn();
        return 0;
    } catch (const std::exception& e) {
        set_error(e.what());
        return -1;
    } catch (...) {
        set_error("unknown error");
        return -1;
    }
}

// Download any-dtype tensor as FP32 into a caller buffer.
void download_fp32(const bt::Tensor& t, float* dst) {
    bt::Tensor f32;
    if (t.dtype != bt::Dtype::FP32) bt::cast(t, f32, bt::Dtype::FP32);
    else f32 = t;
    bt::sync_all();
    bt::Tensor host = f32.to(bt::Device::CPU);
    std::memcpy(dst, host.data,
                sizeof(float) * static_cast<std::size_t>(host.size()));
}

// The diffusers packaging of Qwen-Image 2.1 ships the backbone as
// "model.language_model.*"; a standalone Qwen3-VL checkpoint drops the
// wrapper. Probe rather than assume.
std::string lm_prefix(const std::vector<const bt::safetensors::File*>& shards) {
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

}  // namespace

struct qi_ctx {
    bd::ModelConfig mc;
    std::optional<brolm::qwen3vl::Tokenizer> tokenizer;
    std::optional<brolm::qwen3vl::TextModel> te;
    std::optional<bd::dit::QwenImage21Transformer2DModel> dit;
    std::optional<bd::vae_qwenimage21::Decoder> vae;
    std::optional<bd::vae_qwenimage21::Encoder> vae_enc;

    // Most recent qi_encode_prompt result.
    bd::qwenimage21::TextConditioning prompt;
    bool have_prompt = false;

    // The live prefix KV cache and its saved snapshots.
    bd::dit::QwenImage21PrefixCache cache;
    std::array<bd::dit::QwenImage21PrefixCache, QI_PREFIX_SLOTS> slots;

    std::vector<float> gates;   // capture sink for qi_capture_gates
};

extern "C" {

const char* qi_last_error(void) { return g_last_error.c_str(); }

qi_ctx* qi_open(const char* model_dir, int components, int quantize) {
    qi_ctx* c = nullptr;
    const int rc = guarded([&] {
        bt::init();
        namespace fs = std::filesystem;
        const fs::path root(model_dir ? model_dir : "");

        auto ctx = std::make_unique<qi_ctx>();
        ctx->mc = bd::load_model_config(root.string());
        if (ctx->mc.model_class != bd::ModelClass::QwenImage21) {
            throw std::runtime_error(
                "qi_open: not a Qwen-Image 2.1 model dir: " + root.string());
        }
        if (quantize) {
            ctx->mc.qwenimage21.transformer.quantize_weights = true;
            ctx->mc.qwenimage21.text.text.quantize_weights   = true;
        }

        if (components & QI_LOAD_TE) {
            // 2.1 ships a full Qwen3VLProcessor, so the BPE files live under
            // processor/ rather than Krea 2's tokenizer/.
            ctx->tokenizer = brolm::qwen3vl::Tokenizer::load(
                (root / "processor" / "vocab.json").string(),
                (root / "processor" / "merges.txt").string());
            ctx->te.emplace(ctx->mc.qwenimage21.text.text);
            auto files = bd::detail::open_component_files(
                (root / "text_encoder").string());
            std::vector<const bt::safetensors::File*> ptrs;
            for (const auto& f : files) ptrs.push_back(&f);
            ctx->te->load_weights(ptrs, lm_prefix(ptrs));
        }
        if (components & QI_LOAD_DIT) {
            ctx->dit.emplace(ctx->mc.qwenimage21.transformer);
            auto files = bd::detail::open_component_files(
                (root / "transformer").string());
            std::vector<const bt::safetensors::File*> ptrs;
            for (const auto& f : files) ptrs.push_back(&f);
            ctx->dit->load_weights(ptrs, "");
        }
        if (components & QI_LOAD_VAE) {
            auto files = bd::detail::open_component_files(
                (root / "vae").string());
            ctx->vae.emplace(ctx->mc.qwenimage21.vae);
            ctx->vae->load_weights(files.front(), "");
            ctx->vae_enc.emplace(ctx->mc.qwenimage21.vae);
            ctx->vae_enc->load_weights(files.front(), "");
        }
        // Weight loading churns the allocator; return the slack before the
        // caller starts committing activations (the WDDM ~21 GiB cliff).
        bt::device_mem_trim(bt::default_device());
        c = ctx.release();
    });
    return rc == 0 ? c : nullptr;
}

void qi_close(qi_ctx* c) { delete c; }

int qi_hidden_size(const qi_ctx* c) {
    return c->mc.qwenimage21.transformer.hidden_size();
}
int qi_num_layers(const qi_ctx* c) {
    return c->mc.qwenimage21.transformer.num_layers;
}
int qi_text_hidden_dim(const qi_ctx* c) {
    return c->mc.qwenimage21.transformer.context_in_dim;
}
int qi_latent_channels(const qi_ctx* c) {
    return c->mc.qwenimage21.transformer.latent_channels();
}
int qi_vae_scale(const qi_ctx* c) {
    return c->mc.qwenimage21.vae.spatial_scale();
}

// ── text encoder ───────────────────────────────────────────────────────────

int qi_encode_prompt(qi_ctx* c, const char* prompt) {
    int n = -1;
    const int rc = guarded([&] {
        if (!c->te || !c->tokenizer) {
            throw std::runtime_error("qi_encode_prompt: TE not loaded "
                                     "(open with QI_LOAD_TE)");
        }
        c->prompt = bd::qwenimage21::encode_prompt(*c->tokenizer, *c->te,
                                                   prompt ? prompt : "");
        c->have_prompt = true;
        n = c->prompt.n_valid();
    });
    return rc == 0 ? n : -1;
}

int qi_get_prompt_embeds(qi_ctx* c, float* out) {
    return guarded([&] {
        if (!c->have_prompt) {
            throw std::runtime_error("qi_get_prompt_embeds: call "
                                     "qi_encode_prompt first");
        }
        if (!out) throw std::runtime_error("qi_get_prompt_embeds: out is NULL");
        download_fp32(c->prompt.embeds, out);
    });
}

int qi_prompt_num_ids(qi_ctx* c) {
    return c->have_prompt ? static_cast<int>(c->prompt.token_ids.size()) : -1;
}

int qi_get_prompt_ids(qi_ctx* c, int32_t* out) {
    return guarded([&] {
        if (!c->have_prompt) {
            throw std::runtime_error("qi_get_prompt_ids: call "
                                     "qi_encode_prompt first");
        }
        if (!out) throw std::runtime_error("qi_get_prompt_ids: out is NULL");
        for (std::size_t i = 0; i < c->prompt.token_ids.size(); ++i) {
            out[i] = static_cast<int32_t>(c->prompt.token_ids[i]);
        }
    });
}

// ── DiT ────────────────────────────────────────────────────────────────────

int qi_encode_text(qi_ctx* c, const float* embeds, int n, float* txt_out) {
    int rows = -1;
    const int rc = guarded([&] {
        if (!c->dit) {
            throw std::runtime_error("qi_encode_text: DiT not loaded "
                                     "(open with QI_LOAD_DIT)");
        }
        if (!embeds || !txt_out || n <= 0) {
            throw std::runtime_error("qi_encode_text: embeds/txt_out/n required");
        }
        const int th = c->mc.qwenimage21.transformer.context_in_dim;
        bt::Tensor e = bt::Tensor::from_host(embeds, n, th)
                           .to(bt::default_device());
        bt::Tensor txt;
        c->dit->encode_text(e, txt);
        download_fp32(txt, txt_out);
        rows = txt.rows;
    });
    return rc == 0 ? rows : -1;
}

int qi_forward(qi_ctx* c, const float* latent, int h_lat, int w_lat,
               const float* txt, int n_txt, float timestep, float* out) {
    return guarded([&] {
        if (!c->dit) {
            throw std::runtime_error("qi_forward: DiT not loaded "
                                     "(open with QI_LOAD_DIT)");
        }
        if (!latent || !txt || !out || h_lat <= 0 || w_lat <= 0 || n_txt <= 0) {
            throw std::runtime_error("qi_forward: null buffer or bad shape");
        }
        const int ic = c->mc.qwenimage21.transformer.in_channels;
        const int h  = c->mc.qwenimage21.transformer.hidden_size();
        bt::Tensor lat = bt::Tensor::from_host(latent, h_lat * w_lat, ic)
                             .to(bt::default_device());
        // forward() wants txt at the compute dtype on device.
        bt::Tensor txt_f32 = bt::Tensor::from_host(txt, n_txt, h)
                                 .to(bt::default_device());
        bt::Tensor txt_dev = txt_f32;
        const bt::Dtype dt = c->dit->compute_dtype();
        if (dt != bt::Dtype::FP32) bt::cast(txt_f32, txt_dev, dt);

        // A layout change invalidates the cache. Anything else — including an
        // edit to the text rows at the same length — is the caller's to
        // declare with qi_reset_cache().
        if (c->cache.valid() && !c->cache.matches(n_txt, h_lat, w_lat)) {
            c->cache.reset();
        }
        bt::Tensor v;
        c->dit->forward(lat, h_lat, w_lat, txt_dev, timestep, &c->cache, v);
        download_fp32(v, out);
    });
}

int qi_reset_cache(qi_ctx* c) {
    return guarded([&] { c->cache.reset(); });
}

int qi_set_mod_delta(qi_ctx* c, const float* delta, int block_lo,
                     int block_hi, int target) {
    return guarded([&] {
        if (!c->dit) {
            throw std::runtime_error("qi_set_mod_delta: DiT not loaded "
                                     "(open with QI_LOAD_DIT)");
        }
        using MT = bd::dit::QwenImage21ModTarget;
        MT mt = MT::Target;
        if (target == QI_MOD_PREFIX) mt = MT::Prefix;
        else if (target == QI_MOD_BOTH) mt = MT::Both;
        else if (target != QI_MOD_TARGET) {
            throw std::runtime_error("qi_set_mod_delta: target must be one of "
                                     "QI_MOD_TARGET/PREFIX/BOTH");
        }
        if (!delta) {
            c->dit->set_mod_delta(bt::Tensor(), 0, 0, mt);
            return;
        }
        const int h = c->mc.qwenimage21.transformer.hidden_size();
        bt::Tensor d = bt::Tensor::from_host(delta, 1, 4 * h)
                           .to(bt::default_device());
        c->dit->set_mod_delta(d, block_lo, block_hi, mt);
    });
}

int qi_time_mod(qi_ctx* c, float timestep, float* temb_out, float* mod_out) {
    return guarded([&] {
        if (!c->dit) {
            throw std::runtime_error("qi_time_mod: DiT not loaded "
                                     "(open with QI_LOAD_DIT)");
        }
        bt::Tensor temb, mod;
        c->dit->compute_time_mod(timestep, temb, mod);
        if (temb_out) download_fp32(temb, temb_out);
        if (mod_out)  download_fp32(mod, mod_out);
    });
}

int qi_set_gate_scale(qi_ctx* c, float attn_scale, float mlp_scale,
                      float txt_scale, float img_scale, int block_lo,
                      int block_hi) {
    return guarded([&] {
        if (!c->dit) {
            throw std::runtime_error("qi_set_gate_scale: DiT not loaded "
                                     "(open with QI_LOAD_DIT)");
        }
        c->dit->set_gate_scale(attn_scale, mlp_scale, txt_scale, img_scale,
                               block_lo, block_hi);
    });
}

int qi_set_gate_delta(qi_ctx* c, const float* delta, int block_lo,
                      int block_hi, int target) {
    return guarded([&] {
        if (!c->dit) {
            throw std::runtime_error("qi_set_gate_delta: DiT not loaded "
                                     "(open with QI_LOAD_DIT)");
        }
        using MT = bd::dit::QwenImage21ModTarget;
        MT mt = MT::Target;
        if (target == QI_MOD_PREFIX) mt = MT::Prefix;
        else if (target == QI_MOD_BOTH) mt = MT::Both;
        else if (target != QI_MOD_TARGET) {
            throw std::runtime_error("qi_set_gate_delta: target must be one of "
                                     "QI_MOD_TARGET/PREFIX/BOTH");
        }
        if (!delta) {
            c->dit->set_gate_delta(bt::Tensor(), 0, 0, mt);
            return;
        }
        const int h = c->mc.qwenimage21.transformer.hidden_size();
        bt::Tensor d = bt::Tensor::from_host(delta, 1, 2 * h)
                           .to(bt::default_device());
        c->dit->set_gate_delta(d, block_lo, block_hi, mt);
    });
}

int qi_set_gate_mask(qi_ctx* c, const float* mask, int64_t n, int block_lo,
                     int block_hi) {
    return guarded([&] {
        if (!c->dit) {
            throw std::runtime_error("qi_set_gate_mask: DiT not loaded "
                                     "(open with QI_LOAD_DIT)");
        }
        if (!mask) {
            c->dit->set_gate_mask(bt::Tensor(), 0, 0);
            return;
        }
        bt::Tensor m = bt::Tensor::from_host(mask, static_cast<int>(n), 1)
                           .to(bt::default_device());
        c->dit->set_gate_mask(m, block_lo, block_hi);
    });
}

int qi_set_norm_out_scale_delta(qi_ctx* c, const float* delta) {
    return guarded([&] {
        if (!c->dit) {
            throw std::runtime_error("qi_set_norm_out_scale_delta: DiT not "
                                     "loaded (open with QI_LOAD_DIT)");
        }
        if (!delta) {
            c->dit->set_norm_out_scale_delta(bt::Tensor());
            return;
        }
        const int h = c->mc.qwenimage21.transformer.hidden_size();
        bt::Tensor d = bt::Tensor::from_host(delta, 1, h)
                           .to(bt::default_device());
        c->dit->set_norm_out_scale_delta(d);
    });
}

int qi_capture_gates(qi_ctx* c, int enable) {
    return guarded([&] {
        if (!c->dit) {
            throw std::runtime_error("qi_capture_gates: DiT not loaded "
                                     "(open with QI_LOAD_DIT)");
        }
        c->dit->capture_gates(enable ? &c->gates : nullptr);
        if (!enable) c->gates.clear();
    });
}

int64_t qi_gates_size(qi_ctx* c) {
    return static_cast<int64_t>(c->gates.size());
}

int qi_get_gates(qi_ctx* c, float* out) {
    return guarded([&] {
        if (c->gates.empty()) {
            throw std::runtime_error("qi_get_gates: nothing captured");
        }
        if (!out) throw std::runtime_error("qi_get_gates: out is NULL");
        std::memcpy(out, c->gates.data(), c->gates.size() * sizeof(float));
    });
}

// ── prefix KV cache surface ────────────────────────────────────────────────

int qi_scale_prefix_kv(qi_ctx* c, int layer_lo, int layer_hi, float k_scale,
                       float v_scale) {
    return guarded([&] {
        if (!c->cache.valid()) {
            throw std::runtime_error("qi_scale_prefix_kv: nothing extracted "
                                     "yet — run one qi_forward first");
        }
        c->cache.scale_kv(layer_lo, layer_hi, k_scale, v_scale);
    });
}

int qi_save_prefix(qi_ctx* c, int slot) {
    return guarded([&] {
        if (slot < 0 || slot >= QI_PREFIX_SLOTS) {
            throw std::runtime_error("qi_save_prefix: slot out of range");
        }
        if (!c->cache.valid()) {
            throw std::runtime_error("qi_save_prefix: nothing extracted yet — "
                                     "run one qi_forward first");
        }
        c->slots[static_cast<std::size_t>(slot)] = c->cache;
    });
}

int qi_blend_prefix(qi_ctx* c, int slot, float alpha) {
    return guarded([&] {
        if (slot < 0 || slot >= QI_PREFIX_SLOTS) {
            throw std::runtime_error("qi_blend_prefix: slot out of range");
        }
        if (!c->cache.valid()) {
            throw std::runtime_error("qi_blend_prefix: nothing extracted yet — "
                                     "run one qi_forward first");
        }
        c->cache.blend_from(c->slots[static_cast<std::size_t>(slot)], alpha);
    });
}

// ── VAE ────────────────────────────────────────────────────────────────────

int qi_encode_image(qi_ctx* c, const float* pixels, int H, int W, float* out) {
    return guarded([&] {
        if (!c->vae_enc) {
            throw std::runtime_error("qi_encode_image: VAE not loaded "
                                     "(open with QI_LOAD_VAE)");
        }
        if (!pixels || !out || H <= 0 || W <= 0) {
            throw std::runtime_error("qi_encode_image: null buffer or bad shape");
        }
        const int S = c->mc.qwenimage21.vae.spatial_scale();
        if (H % S != 0 || W % S != 0) {
            throw std::runtime_error("qi_encode_image: H and W must be "
                                     "multiples of " + std::to_string(S));
        }
        const int in_ch = c->mc.qwenimage21.vae.in_channels;
        const std::size_t plane = static_cast<std::size_t>(H) * W;
        std::vector<float> nchw(static_cast<std::size_t>(in_ch) * plane);
        for (std::size_t i = 0; i < static_cast<std::size_t>(3) * plane; ++i) {
            nchw[i] = pixels[i] * 2.0f - 1.0f;   // [0,1] -> [-1,1]
        }
        if (in_ch == 4) {
            for (std::size_t i = 0; i < plane; ++i) {
                nchw[static_cast<std::size_t>(3) * plane + i] = 1.0f;  // opaque
            }
        }
        bt::Tensor img = bt::Tensor::from_host(nchw.data(), 1,
                                               static_cast<int>(nchw.size()))
                             .to(bt::default_device());
        bt::Tensor z;
        c->vae_enc->encode(img, H, W, nullptr, z);
        bt::sync_all();
        download_fp32(z, out);
    });
}

int qi_decode(qi_ctx* c, const float* latent, int h_lat, int w_lat,
              float* out) {
    return guarded([&] {
        if (!c->vae) {
            throw std::runtime_error("qi_decode: VAE not loaded "
                                     "(open with QI_LOAD_VAE)");
        }
        if (!latent || !out || h_lat <= 0 || w_lat <= 0) {
            throw std::runtime_error("qi_decode: null buffer or bad shape");
        }
        const int zc = c->mc.qwenimage21.vae.z_dim;
        bt::Tensor z = bt::Tensor::from_host(latent, 1, zc * h_lat * w_lat)
                           .to(bt::default_device());
        bt::Tensor img;
        c->vae->decode(z, h_lat, w_lat, img);
        bt::sync_all();
        // RGBA in, RGB out: planar NCHW, so the caller's three planes are the
        // leading 3/4 of the decode and dropping alpha is a truncation.
        bt::Tensor f32 = img;
        if (img.dtype != bt::Dtype::FP32) bt::cast(img, f32, bt::Dtype::FP32);
        bt::sync_all();
        std::vector<float> host = f32.to(bt::Device::CPU).to_host_vector();
        const int S = c->mc.qwenimage21.vae.spatial_scale();
        const std::size_t rgb = static_cast<std::size_t>(3) *
                                static_cast<std::size_t>(h_lat) * S *
                                static_cast<std::size_t>(w_lat) * S;
        if (host.size() < rgb) {
            throw std::runtime_error("qi_decode: decoder produced fewer "
                                     "planes than expected");
        }
        std::memcpy(out, host.data(), rgb * sizeof(float));
    });
}

// ── utilities ──────────────────────────────────────────────────────────────

int qi_randn(uint64_t key, uint64_t counter, int64_t n, float* out) {
    return guarded([&] {
        bt::init();
        bt::Tensor y = bt::Tensor::zeros_on(bt::default_device(),
                                            static_cast<int>(n), 1);
        bt::randn(key, counter, y);
        download_fp32(y, out);
    });
}

int qi_mem_info(uint64_t* free_bytes, uint64_t* total_bytes) {
    return guarded([&] {
        bt::init();
        std::size_t f = 0, t = 0;
        if (!bt::device_mem_info(bt::default_device(), f, t)) {
            throw std::runtime_error("qi_mem_info: backend has no mem_info");
        }
        if (free_bytes)  *free_bytes = f;
        if (total_bytes) *total_bytes = t;
    });
}

int qi_mem_trim(void) {
    return guarded([&] {
        bt::init();
        bt::device_mem_trim(bt::default_device());
    });
}

}  // extern "C"
