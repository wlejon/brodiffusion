#pragma once

// Helpers shared by the Qwen-Image 2.1 JS binding translation units:
//
//   native_diffusion_qwenimage21.cpp        the in-network hooks, the
//                                           conditioning entry points, the
//                                           VAE seam and text-encoder
//                                           residency
//   native_diffusion_qwenimage21_slots.cpp  the multi-slot Add/Clear/Count
//                                           forms, the prefix KV dial and
//                                           its saved slots, the text rows
//                                           and the prompt memo
//
// Not a public header.

#include "host_diffusion_internal.h"

#include <brodiffusion/dit/qwenimage21.h>

#include <span>
#include <string>

namespace brodiffusion::api {

// Every qwenImage21* method is QwenImage21-only. Returns null and leaves the
// caller to raise a TypeError (notQi21) when `thisVal` is not a loaded
// Qwen-Image 2.1 Pipeline.
inline PipelineWrapper* qi21Pipeline(Value thisVal) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return nullptr;
    if (w->pipeline->config().model_class !=
        brodiffusion::ModelClass::QwenImage21) {
        return nullptr;
    }
    return w;
}

inline Value notQi21(const char* method) {
    return ev::throwTypeError(
        std::string("Pipeline.") + method +
        ": not a loaded Qwen-Image 2.1 Pipeline");
}

// "target" (default) / "prefix" / "both", or the equivalent 0 / 1 / 2.
inline bool readModTarget(Value v,
                          brodiffusion::dit::QwenImage21ModTarget& out) {
    using MT = brodiffusion::dit::QwenImage21ModTarget;
    out = MT::Target;
    if (ev::isUndefined(v) || ev::isNull(v)) return true;
    if (ev::isNumber(v)) {
        const int n = static_cast<int>(ev::toDouble(v));
        if (n == 0) { out = MT::Target; return true; }
        if (n == 1) { out = MT::Prefix; return true; }
        if (n == 2) { out = MT::Both;   return true; }
        return false;
    }
    if (!ev::isString(v)) return false;
    const std::string s = ev::toUtf8(v);
    if (s == "target") { out = MT::Target; return true; }
    if (s == "prefix") { out = MT::Prefix; return true; }
    if (s == "both")   { out = MT::Both;   return true; }
    return false;
}

// "both" (default) / "attn" / "mlp", or the equivalent 0 / 1 / 2. Names the
// sublayer a gate mask scales.
inline bool readGateSublayer(Value v,
                             brodiffusion::dit::QwenImage21GateSublayer& out) {
    using GS = brodiffusion::dit::QwenImage21GateSublayer;
    out = GS::Both;
    if (ev::isUndefined(v) || ev::isNull(v)) return true;
    if (ev::isNumber(v)) {
        const int n = static_cast<int>(ev::toDouble(v));
        if (n == 0) { out = GS::Both; return true; }
        if (n == 1) { out = GS::Attn; return true; }
        if (n == 2) { out = GS::Mlp;  return true; }
        return false;
    }
    if (!ev::isString(v)) return false;
    const std::string s = ev::toUtf8(v);
    if (s == "both") { out = GS::Both; return true; }
    if (s == "attn") { out = GS::Attn; return true; }
    if (s == "mlp")  { out = GS::Mlp;  return true; }
    return false;
}

// The multi-slot half of the surface, defined in the _slots translation unit.
void decoratePipelineQwenImage21SlotsProto(ObjectBuilder& proto);

}  // namespace brodiffusion::api
