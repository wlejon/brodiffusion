// krea2_capi — see include/brodiffusion/krea2_capi.h.
//
// A deliberately thin layer: every entry point converts caller FP32 buffers
// to brotensor tensors, calls the same component methods the Pipeline drives,
// and copies results back as FP32. All state lives in k2_ctx; errors cross
// as a per-thread message string + a -1/NULL return (no exceptions escape).

#include "brodiffusion/krea2_capi.h"

#include "brodiffusion/detail/safetensors_dir.h"
#include "brodiffusion/dit/krea2.h"
#include "brodiffusion/krea2_text.h"
#include "brodiffusion/model_config.h"
#include "brodiffusion/vae_qwenimage.h"

#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <cstring>
#include <exception>
#include <filesystem>
#include <optional>
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
    std::memcpy(dst, host.data, sizeof(float) * static_cast<std::size_t>(host.size()));
}

}  // namespace

struct k2_ctx {
    bd::ModelConfig mc;
    std::optional<brolm::qwen3vl::Tokenizer> tokenizer;
    std::optional<brolm::qwen3vl::TextModel> te;
    std::optional<bd::dit::Krea2Transformer2DModel> dit;
    std::optional<bd::vae_qwenimage::Decoder> vae;
};

extern "C" {

const char* k2_last_error(void) { return g_last_error.c_str(); }

k2_ctx* k2_open(const char* model_dir, int components, int quantize) {
    k2_ctx* c = nullptr;
    const int rc = guarded([&] {
        bt::init();
        namespace fs = std::filesystem;
        const fs::path root(model_dir);

        auto ctx = std::make_unique<k2_ctx>();
        ctx->mc = bd::load_model_config(model_dir);
        if (ctx->mc.model_class != bd::ModelClass::Krea2) {
            throw std::runtime_error("k2_open: not a Krea 2 model dir: " +
                                     std::string(model_dir));
        }
        if (quantize) {
            ctx->mc.krea2.transformer.quantize_weights = true;
            ctx->mc.krea2.text.text.quantize_weights   = true;
        }

        if (components & K2_LOAD_TE) {
            ctx->tokenizer = brolm::qwen3vl::Tokenizer::load(
                (root / "tokenizer" / "vocab.json").string(),
                (root / "tokenizer" / "merges.txt").string());
            ctx->te.emplace(ctx->mc.krea2.text.text);
            auto files = bd::detail::open_component_files(
                (root / "text_encoder").string());
            std::vector<const bt::safetensors::File*> ptrs;
            for (const auto& f : files) ptrs.push_back(&f);
            ctx->te->load_weights(ptrs, "language_model.");
        }
        if (components & K2_LOAD_DIT) {
            ctx->dit.emplace(ctx->mc.krea2.transformer);
            auto files = bd::detail::open_component_files(
                (root / "transformer").string());
            std::vector<const bt::safetensors::File*> ptrs;
            for (const auto& f : files) ptrs.push_back(&f);
            ctx->dit->load_weights(ptrs, "");
        }
        if (components & K2_LOAD_VAE) {
            ctx->vae.emplace(ctx->mc.krea2.vae);
            auto files = bd::detail::open_component_files(
                (root / "vae").string());
            ctx->vae->load_weights(files.front(), "");
        }
        // Weight loading churns the allocator; return the slack before the
        // caller starts committing activations (the WDDM ~21 GiB cliff).
        bt::device_mem_trim(bt::default_device());
        c = ctx.release();
    });
    return rc == 0 ? c : nullptr;
}

void k2_close(k2_ctx* c) { delete c; }

int k2_hidden_size(const k2_ctx* c)     { return c->mc.krea2.transformer.hidden_size(); }
int k2_text_hidden_dim(const k2_ctx* c) { return c->mc.krea2.transformer.text_hidden_dim; }
int k2_num_text_layers(const k2_ctx* c) { return c->mc.krea2.transformer.num_text_layers; }
int k2_max_text_seq(const k2_ctx*)      { return bd::krea2::kMaxSequenceLength; }
int k2_in_channels(const k2_ctx* c)     { return c->mc.krea2.transformer.in_channels; }
int k2_latent_channels(const k2_ctx* c) { return c->mc.krea2.transformer.latent_channels(); }

int k2_encode_prompt(k2_ctx* c, const char* prompt,
                     float* embeds_out, float* mask_out) {
    return guarded([&] {
        if (!c->te || !c->tokenizer) {
            throw std::runtime_error("k2_encode_prompt: TE not loaded "
                                     "(open with K2_LOAD_TE)");
        }
        bd::krea2::TextConditioning tc = bd::krea2::encode_prompt(
            *c->tokenizer, *c->te, prompt ? prompt : "");
        download_fp32(tc.prompt_embeds, embeds_out);
        download_fp32(tc.prompt_embeds_mask, mask_out);
    });
}

int k2_encode_text(k2_ctx* c, const float* embeds, const float* mask,
                   float* txt_out) {
    int n_valid = -1;
    const int rc = guarded([&] {
        if (!c->dit) {
            throw std::runtime_error("k2_encode_text: DiT not loaded "
                                     "(open with K2_LOAD_DIT)");
        }
        const int seq = bd::krea2::kMaxSequenceLength;
        const int nl  = c->mc.krea2.transformer.num_text_layers;
        const int th  = c->mc.krea2.transformer.text_hidden_dim;
        bt::Tensor e = bt::Tensor::from_host(embeds, seq * nl, th)
                           .to(bt::default_device());
        bt::Tensor m = bt::Tensor::from_host(mask, seq, 1)
                           .to(bt::default_device());
        bt::Tensor txt;
        c->dit->encode_text(e, m, txt);
        download_fp32(txt, txt_out);
        n_valid = txt.rows;
    });
    return rc == 0 ? n_valid : -1;
}

int k2_forward(k2_ctx* c, const float* packed, int hp, int wp,
               const float* txt, int n_txt, float timestep, float* out) {
    return guarded([&] {
        if (!c->dit) {
            throw std::runtime_error("k2_forward: DiT not loaded "
                                     "(open with K2_LOAD_DIT)");
        }
        const int ic = c->mc.krea2.transformer.in_channels;
        const int h  = c->mc.krea2.transformer.hidden_size();
        bt::Tensor lat = bt::Tensor::from_host(packed, hp * wp, ic)
                             .to(bt::default_device());
        // forward_with_text requires txt at the compute dtype on device.
        bt::Tensor txt_f32 = bt::Tensor::from_host(txt, n_txt, h)
                                 .to(bt::default_device());
        bt::Tensor txt_dev = txt_f32;
        const bt::Dtype dt = c->dit->compute_dtype();
        if (dt != bt::Dtype::FP32) bt::cast(txt_f32, txt_dev, dt);
        bt::Tensor v;
        c->dit->forward_with_text(lat, hp, wp, txt_dev, timestep, v);
        download_fp32(v, out);
    });
}

int k2_decode(k2_ctx* c, const float* latent, int h_lat, int w_lat,
              float* out) {
    return guarded([&] {
        if (!c->vae) {
            throw std::runtime_error("k2_decode: VAE not loaded "
                                     "(open with K2_LOAD_VAE)");
        }
        const int zc = c->mc.krea2.vae.z_dim;
        bt::Tensor z = bt::Tensor::from_host(latent, 1, zc * h_lat * w_lat)
                           .to(bt::default_device());
        bt::Tensor img;
        c->vae->decode(z, h_lat, w_lat, img);
        bt::sync_all();
        download_fp32(img, out);
    });
}

int k2_randn(uint64_t key, uint64_t counter, int64_t n, float* out) {
    return guarded([&] {
        bt::init();
        bt::Tensor y = bt::Tensor::zeros_on(bt::default_device(),
                                            static_cast<int>(n), 1);
        bt::randn(key, counter, y);
        download_fp32(y, out);
    });
}

int k2_mem_info(uint64_t* free_bytes, uint64_t* total_bytes) {
    return guarded([&] {
        bt::init();
        std::size_t f = 0, t = 0;
        if (!bt::device_mem_info(bt::default_device(), f, t)) {
            throw std::runtime_error("k2_mem_info: backend has no mem_info");
        }
        if (free_bytes)  *free_bytes = f;
        if (total_bytes) *total_bytes = t;
    });
}

int k2_mem_trim(void) {
    return guarded([&] {
        bt::init();
        bt::device_mem_trim(bt::default_device());
    });
}

}  // extern "C"
