#pragma once

// Internals shared by the qwenimage21_capi translation units:
//
//   qwenimage21_capi.cpp        the context, the components, encoding,
//                               qi_forward, the VAE and the utilities
//   qwenimage21_capi_hooks.cpp  the research hooks and the prefix KV cache
//                               surface
//
// Not a public header — the installed surface is
// include/brodiffusion/qwenimage21_capi.h.
//
// Note on the error string: it is a function-local thread_local inside an
// inline function, so the two translation units share ONE instance per
// thread. A file-static would give each its own, and qi_last_error() would
// report whichever TU the last call happened to live in.

#include "brodiffusion/qwenimage21_capi.h"

#include "brodiffusion/dit/qwenimage21.h"
#include "brodiffusion/model_config.h"
#include "brodiffusion/qwenimage21_text.h"
#include "brodiffusion/vae_qwenimage21.h"

#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace qi_capi {

namespace bt = ::brotensor;
namespace bd = ::brodiffusion;

inline std::string& last_error() {
    thread_local std::string e;
    return e;
}

inline void set_error(const std::string& m) { last_error() = m; }

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
inline void download_fp32(const bt::Tensor& t, float* dst) {
    bt::Tensor f32;
    if (t.dtype != bt::Dtype::FP32) bt::cast(t, f32, bt::Dtype::FP32);
    else f32 = t;
    bt::sync_all();
    bt::Tensor host = f32.to(bt::Device::CPU);
    std::memcpy(dst, host.data,
                sizeof(float) * static_cast<std::size_t>(host.size()));
}

// QI_MOD_* -> the enum, or a throw naming the caller.
inline bd::dit::QwenImage21ModTarget mod_target(int target, const char* who) {
    using MT = bd::dit::QwenImage21ModTarget;
    if (target == QI_MOD_TARGET) return MT::Target;
    if (target == QI_MOD_PREFIX) return MT::Prefix;
    if (target == QI_MOD_BOTH)   return MT::Both;
    throw std::runtime_error(std::string(who) +
                             ": target must be one of "
                             "QI_MOD_TARGET/PREFIX/BOTH");
}

}  // namespace qi_capi

struct qi_ctx {
    ::brodiffusion::ModelConfig mc;
    std::optional<brolm::qwen3vl::Tokenizer> tokenizer;
    std::optional<brolm::qwen3vl::TextModel> te;
    std::optional<::brodiffusion::dit::QwenImage21Transformer2DModel> dit;
    std::optional<::brodiffusion::vae_qwenimage21::Decoder> vae;
    std::optional<::brodiffusion::vae_qwenimage21::Encoder> vae_enc;

    std::optional<brolm::qwen3vl::VisionTower> vision;

    // Most recent qi_encode_prompt / qi_encode_prompt_images result.
    ::brodiffusion::qwenimage21::TextConditioning prompt;
    bool have_prompt = false;

    // Condition latents armed by qi_set_condition_latents, plus the prefix
    // segments the parked prompt's image runs imply. Empty = text-to-image.
    ::brodiffusion::dit::QwenImage21EditPrefix edit;

    // The live prefix KV cache and its saved snapshots.
    ::brodiffusion::dit::QwenImage21PrefixCache cache;
    std::array<::brodiffusion::dit::QwenImage21PrefixCache, QI_PREFIX_SLOTS>
        slots;

    std::vector<float> gates;   // capture sink for qi_capture_gates

    // The DiT, or a throw naming the caller. Every hook needs this line.
    ::brodiffusion::dit::QwenImage21Transformer2DModel& need_dit(
        const char* who) {
        if (!dit) {
            throw std::runtime_error(std::string(who) +
                                     ": DiT not loaded (open with "
                                     "QI_LOAD_DIT)");
        }
        return *dit;
    }
    int hidden() const { return mc.qwenimage21.transformer.hidden_size(); }
};
