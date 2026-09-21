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
//
// This file holds the context, the components, encoding, qi_forward, the VAE
// and the utilities; the research hooks and the prefix-cache surface are in
// qwenimage21_capi_hooks.cpp, over the shared qwenimage21_capi_detail.h.

#include "qwenimage21_capi_detail.h"

#include "brodiffusion/detail/safetensors_dir.h"
#include "brodiffusion/image_io.h"

#include "brotensor/safetensors.h"

#include <filesystem>
#include <memory>

namespace bt = brotensor;
namespace bd = brodiffusion;

using qi_capi::download_fp32;
using qi_capi::guarded;

namespace {

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

extern "C" {

const char* qi_last_error(void) { return qi_capi::last_error().c_str(); }

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
            // The vision tower rides in the same shards. Optional: a
            // text-only checkpoint still opens, and qi_encode_prompt_images
            // is the only entry point that misses it.
            const char* vp = nullptr;
            for (const auto* f : ptrs) {
                if (f->find("visual.patch_embed.proj.weight")) {
                    vp = "visual."; break;
                }
                if (f->find("model.visual.patch_embed.proj.weight")) {
                    vp = "model.visual."; break;
                }
            }
            if (vp != nullptr) {
                ctx->vision.emplace(ctx->mc.qwenimage21.text.vision,
                                    ctx->mc.qwenimage21.text.text.hidden_size);
                ctx->vision->load_weights(ptrs, vp);
            }
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
        // A new prompt invalidates any armed image prefix: its segments were
        // cut to the previous prompt's row counts.
        c->edit = bd::dit::QwenImage21EditPrefix{};
        c->cache.reset();
        n = c->prompt.n_valid();
    });
    return rc == 0 ? n : -1;
}

int qi_encode_prompt_images(qi_ctx* c, const char* prompt,
                            const char* const* image_paths, int n_images,
                            int output_resolution) {
    int n = -1;
    const int rc = guarded([&] {
        if (!c->te || !c->tokenizer) {
            throw std::runtime_error("qi_encode_prompt_images: TE not loaded "
                                     "(open with QI_LOAD_TE)");
        }
        if (!c->vision) {
            throw std::runtime_error("qi_encode_prompt_images: this "
                                     "checkpoint's text_encoder carries no "
                                     "vision tower");
        }
        if (!image_paths || n_images <= 0) {
            throw std::runtime_error("qi_encode_prompt_images: pass at least "
                                     "one image path");
        }
        const int res = output_resolution > 0 ? output_resolution : 1024;
        const double area = static_cast<double>(res) * static_cast<double>(res);

        // One resize per image, to the geometry that makes the vision grid and
        // the latent grid agree; the vision tower gets the alpha composited
        // over white, as the checkpoint was trained.
        std::vector<bd::HostImage> rgba;
        std::vector<std::vector<float>> rgb;
        rgba.reserve(static_cast<std::size_t>(n_images));
        rgb.reserve(static_cast<std::size_t>(n_images));
        for (int i = 0; i < n_images; ++i) {
            if (image_paths[i] == nullptr) {
                throw std::runtime_error("qi_encode_prompt_images: image path " +
                                         std::to_string(i) + " is NULL");
            }
            bd::HostImage native = bd::load_image_rgba(image_paths[i]);
            int w = 0, h = 0;
            bd::qwenimage21::calculate_dimensions(
                area,
                static_cast<double>(native.W) / static_cast<double>(native.H),
                w, h);
            rgba.push_back(bd::resize_rgba(native, w, h));
            rgb.push_back(bd::composite_over_white(rgba.back()));
        }
        std::vector<brolm::qwen3vl::ImageInput> inputs(rgb.size());
        for (std::size_t i = 0; i < rgb.size(); ++i) {
            inputs[i].pixels = rgb[i].data();
            inputs[i].H = rgba[i].H;
            inputs[i].W = rgba[i].W;
        }
        brolm::qwen3vl::PreprocessConfig pp;
        c->prompt = bd::qwenimage21::encode_prompt_with_images(
            *c->tokenizer, *c->te, *c->vision, pp, prompt ? prompt : "",
            inputs);
        c->have_prompt = true;
        c->edit = bd::dit::QwenImage21EditPrefix{};
        c->cache.reset();
        n = c->prompt.n_valid();
    });
    return rc == 0 ? n : -1;
}

int qi_get_prompt_pad_mask(qi_ctx* c, int32_t* out) {
    return guarded([&] {
        if (!c->have_prompt) {
            throw std::runtime_error("qi_get_prompt_pad_mask: encode a prompt "
                                     "first");
        }
        if (!out) throw std::runtime_error("qi_get_prompt_pad_mask: out is NULL");
        const std::size_t n = static_cast<std::size_t>(c->prompt.n_valid());
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = (i < c->prompt.image_pad_mask.size() &&
                      c->prompt.image_pad_mask[i]) ? 1 : 0;
        }
    });
}

int qi_prompt_num_images(qi_ctx* c) {
    return c->have_prompt ? static_cast<int>(c->prompt.image_runs.size()) : -1;
}

int qi_get_prompt_image_grid(qi_ctx* c, int index, int* h_lat, int* w_lat) {
    return guarded([&] {
        if (!c->have_prompt) {
            throw std::runtime_error("qi_get_prompt_image_grid: encode a "
                                     "prompt first");
        }
        if (index < 0 ||
            static_cast<std::size_t>(index) >= c->prompt.image_runs.size()) {
            throw std::runtime_error("qi_get_prompt_image_grid: index out of "
                                     "range");
        }
        const auto& r = c->prompt.image_runs[static_cast<std::size_t>(index)];
        if (h_lat) *h_lat = r.h_lat;
        if (w_lat) *w_lat = r.w_lat;
    });
}

int qi_set_condition_latents(qi_ctx* c, const float* latents, int n_tokens) {
    return guarded([&] {
        c->cache.reset();
        c->edit = bd::dit::QwenImage21EditPrefix{};
        if (latents == nullptr) return;
        if (!c->have_prompt) {
            throw std::runtime_error("qi_set_condition_latents: encode a "
                                     "prompt with images first — the prefix "
                                     "layout comes from its image runs");
        }
        if (c->prompt.image_runs.empty()) {
            throw std::runtime_error("qi_set_condition_latents: the parked "
                                     "prompt carries no condition images");
        }
        int expect = 0;
        for (const auto& r : c->prompt.image_runs) expect += r.h_lat * r.w_lat;
        if (n_tokens != expect) {
            throw std::runtime_error(
                "qi_set_condition_latents: got " + std::to_string(n_tokens) +
                " latent tokens, the prompt's image runs need " +
                std::to_string(expect));
        }
        const int ic = c->mc.qwenimage21.transformer.in_channels;
        c->edit.cond_latents = bt::Tensor::from_host(latents, n_tokens, ic)
                                   .to(bt::default_device());

        // Segments: text runs between the image runs, in row order. The image
        // runs' rows are encoder SLOTS (one per four latents); the segment
        // carries the latent grid instead.
        const int n_valid = c->prompt.n_valid();
        std::size_t next = 0;
        int row = 0;
        while (row < n_valid) {
            if (next < c->prompt.image_runs.size() &&
                c->prompt.image_runs[next].row == row) {
                const auto& r = c->prompt.image_runs[next];
                bd::dit::QwenImage21Segment s;
                s.kind     = bd::dit::QwenImage21Segment::Kind::Image;
                s.h        = r.h_lat;
                s.w        = r.w_lat;
                s.n_tokens = r.h_lat * r.w_lat;
                c->edit.segments.push_back(s);
                row += r.n_slots;
                ++next;
                continue;
            }
            int end = n_valid;
            if (next < c->prompt.image_runs.size()) {
                end = c->prompt.image_runs[next].row;
            }
            bd::dit::QwenImage21Segment s;
            s.kind     = bd::dit::QwenImage21Segment::Kind::Text;
            s.n_tokens = end - row;
            c->edit.segments.push_back(s);
            row = end;
        }
    });
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
        // Park the input as the between-step schedule's base. A schedule
        // rebuilds from the conditioning the run is actually using, and on
        // this API that is whatever the caller last projected — not a prompt
        // the context encoded, which may have been edited since.
        c->ctl_base.assign(embeds, embeds + static_cast<std::size_t>(n) * th);
        c->ctl_base_rows = n;
        c->ctl_sched.reset_applied();
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

        // The joint prefix: the armed image layout, or one text run.
        std::vector<bd::dit::QwenImage21Segment> segments;
        int prefix_len = n_txt;
        if (!c->edit.empty()) {
            segments = c->edit.segments;
            prefix_len = 0;
            int n_text = 0;
            for (const auto& s : segments) {
                if (s.kind == bd::dit::QwenImage21Segment::Kind::Text) {
                    n_text += s.n_tokens;
                    prefix_len += s.n_tokens;
                } else {
                    prefix_len += s.h * s.w;
                }
            }
            if (n_text != n_txt) {
                throw std::runtime_error(
                    "qi_forward: the armed condition prefix expects " +
                    std::to_string(n_text) + " text rows but got " +
                    std::to_string(n_txt) + " — pass qi_encode_text's output "
                    "over the NON-image rows only (see qi_get_prompt_pad_mask)");
            }
        } else {
            bd::dit::QwenImage21Segment s;
            s.kind     = bd::dit::QwenImage21Segment::Kind::Text;
            s.n_tokens = n_txt;
            segments.push_back(s);
        }

        // A layout change invalidates the cache. Anything else — including an
        // edit to the text rows at the same length — is the caller's to
        // declare with qi_reset_cache().
        if (c->cache.valid() && !c->cache.matches(prefix_len, h_lat, w_lat)) {
            c->cache.reset();
        }
        bt::Tensor v;
        c->dit->forward_joint(lat, h_lat, w_lat, txt_dev,
                              c->edit.empty() ? nullptr : &c->edit.cond_latents,
                              segments, timestep, &c->cache, v);
        download_fp32(v, out);
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
