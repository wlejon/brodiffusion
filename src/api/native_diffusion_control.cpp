// Pipeline conditioning-control surface: the loaded control dictionary, the
// per-axis weights that steer generate()/prime(), axis introspection, the
// encoder-space primitives axes are built from, and the Sana identity anchor.
//
// Ported from the QuickJS-era src/js/diffusion_bindings.cpp (the
// js_pipeline_loadControlDictionary .. js_pipeline_clearIdentityAnchor block),
// which the bronze rewrite dropped wholesale.

#include "host_diffusion_internal.h"

#include <stdexcept>

namespace brodiffusion::api {

namespace {

// loadControlDictionary(path, { merge = false }) — load a conditioning-space
// control dictionary (a BCD1 file of named direction axes built offline). By
// default this replaces the loaded axes and resets weights; { merge: true }
// ADDS the file's axes to those already loaded (same-named axes overwritten),
// so banks of different provenance can be stacked without concatenating them
// offline. The dictionary's dim must match the model's text encoder.
Value pipelineLoadControlDictionary(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.loadControlDictionary: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.loadControlDictionary(path): path string required");
    }
    std::string path = resolveDiffusionPath(ev::toUtf8(args[0]));
    bool merge = false;
    if (args.size() >= 2 && ev::isObject(args[1])) merge = propBool(args[1], "merge");
    try {
        w->pipeline->cond_control().load(path, merge);
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.loadControlDictionary failed: ") + e.what());
    }
}

// setControl(name, alpha) or setControl({name: alpha, ...}) — per-axis control
// weights in natural units (the applied vector is alpha * scale * dir).
// Unknown axis names throw. Weights persist until changed or clearControl().
Value pipelineSetControl(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.setControl: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    if (args.empty()) {
        return ev::throwTypeError("Pipeline.setControl(name, alpha) or setControl(map) required");
    }

    // Form 1: (string name, number alpha).
    if (ev::isString(args[0])) {
        std::string name = ev::toUtf8(args[0]);
        if (args.size() < 2 || !ev::isNumber(args[1])) {
            return ev::throwTypeError("Pipeline.setControl(name, alpha): numeric alpha required");
        }
        float alpha = static_cast<float>(ev::toDouble(args[1]));
        try {
            w->pipeline->cond_control().set(name, alpha);
        } catch (const std::exception& e) {
            return ev::throwTypeError(std::string("Pipeline.setControl: ") + e.what());
        }
        return ev::undefined();
    }

    // Form 2: ({name: alpha, ...}) object map. Every entry is attempted; the
    // first failure is reported once the whole map has been walked, so a typo
    // in one axis does not silently skip the rest.
    if (!ev::isObject(args[0])) {
        return ev::throwTypeError("Pipeline.setControl: expected a name string or an object map");
    }
    ev::Persistent map(args[0]);
    std::string err;
    for (const std::string& key : objectKeys(map.get())) {
        Value v = ev::getProperty(map.get(), key);
        if (!ev::isNumber(v)) continue;
        float alpha = static_cast<float>(ev::toDouble(v));
        try {
            w->pipeline->cond_control().set(key, alpha);
        } catch (const std::exception& e) {
            if (err.empty()) err = e.what();
        }
    }
    if (!err.empty()) return ev::throwTypeError("Pipeline.setControl: " + err);
    return ev::undefined();
}

// clearControl() — reset every control-axis weight to zero. The loaded
// dictionary is kept; only the weights are cleared.
Value pipelineClearControl(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.clearControl: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    w->pipeline->cond_control().clear();
    return ev::undefined();
}

// setControlBudget(alpha) — cap how much a STACK of axes may inject, in the
// same alpha units the weights use. Over budget, every active axis is scaled
// by one common factor at apply() time. 0 = off.
Value pipelineSetControlBudget(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.setControlBudget: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    if (args.empty() || !ev::isNumber(args[0])) {
        return ev::throwTypeError("Pipeline.setControlBudget(alpha): numeric alpha required");
    }
    w->pipeline->cond_control().set_budget(static_cast<float>(ev::toDouble(args[0])));
    return ev::undefined();
}

// controlNorm() -> {norm, budget, clamped, scale} — the current stack's length
// in alpha units, what it may spend, and whether apply() will hold it back.
// The numbers a stack meter is drawn from; no render needed.
Value pipelineControlNorm(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.controlNorm: not a loaded Pipeline");
    const auto& cc = w->pipeline->cond_control();
    const float norm = cc.active_norm();
    const float budget = cc.budget();
    const bool clamped = budget > 0.0f && norm > budget;

    ObjectBuilder o;
    o.set("norm", static_cast<double>(norm));
    o.set("budget", static_cast<double>(budget));
    o.set("clamped", clamped);
    // The factor apply() will multiply every active axis by (1.0 when in budget).
    o.set("scale", clamped ? static_cast<double>(budget) / static_cast<double>(norm) : 1.0);
    return o.build();
}

// controlAxes() -> [name, ...] — the loaded dictionary's axis names (plus any
// runtime axes), empty when nothing is loaded. For building UI.
Value pipelineControlAxes(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.controlAxes: not a loaded Pipeline");
    const std::vector<std::string>& names = w->pipeline->cond_control().names();
    // Copy first: hostArrayOf allocates, and `names` is owned by the pipeline
    // (stable), but the lambda must not capture a dangling reference.
    std::vector<std::string> copy = names;
    return hostArrayOf(copy.size(), [&copy](size_t i) { return ev::fromUtf8(copy[i]); });
}

// controlVector(name) -> { dir: Float32Array, scale } — the stored direction
// and baked scale of a dictionary or runtime axis. Throws on unknown name.
Value pipelineControlVector(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.controlVector: not a loaded Pipeline");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.controlVector(name): name string required");
    }
    std::string name = ev::toUtf8(args[0]);
    try {
        std::vector<float> dir = w->pipeline->cond_control().direction(name);
        float scale = w->pipeline->cond_control().axis_scale(name);
        ObjectBuilder o;
        {
            ev::Persistent d(makeFloat32Array(dir.data(), dir.size()));
            o.set("dir", d.get());
        }
        o.set("scale", static_cast<double>(scale));
        return o.build();
    } catch (const std::exception& e) {
        return ev::throwTypeError(std::string("Pipeline.controlVector: ") + e.what());
    }
}

// encodeConditioning(prompt) -> { rows, cols, data: Float32Array } — the
// (L, hidden) embeddings the denoiser cross-attends to (row 0 = BOS),
// downloaded to host FP32. The primitive for building control directions in
// the encoder's own space (e.g. a diff-of-means axis from two phrase sets).
Value pipelineEncodeConditioning(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.encodeConditioning: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    if (!w->weights_loaded) {
        return ev::throwError("Pipeline.encodeConditioning: call loadWeights() first");
    }
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.encodeConditioning(prompt): prompt string required");
    }
    std::string prompt = ev::toUtf8(args[0]);
    try {
        brotensor::Tensor emb = w->pipeline->encode_conditioning(prompt);
        return tensorToJs(emb);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.encodeConditioning failed: ") + e.what());
    }
}

// setControlVector(name, dir, alpha, scale=1) — register/replace a runtime axis
// from an explicit direction (Float32Array of width = encoder hidden dim) and
// set its weight. `dir` is taken as-is; the injected vector is alpha*scale*dir.
Value pipelineSetControlVector(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.setControlVector: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.setControlVector(name, dir, alpha, scale?): name required");
    }
    std::string name = ev::toUtf8(args[0]);
    if (args.size() < 3 || !ev::isNumber(args[2])) {
        return ev::throwTypeError("Pipeline.setControlVector: numeric alpha required");
    }
    float alpha = static_cast<float>(ev::toDouble(args[2]));
    float scale = 1.0f;
    if (args.size() >= 4 && ev::isNumber(args[3])) scale = static_cast<float>(ev::toDouble(args[3]));

    const float* dir = nullptr;
    size_t count = 0;
    if (args.size() < 2 || !readFloat32Array(args[1], dir, count) || count == 0) {
        return ev::throwTypeError("Pipeline.setControlVector: dir must be a non-empty Float32Array");
    }
    std::vector<float> v(dir, dir + count);
    try {
        w->pipeline->cond_control().set_vector(name, alpha, v, scale);
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwTypeError(std::string("Pipeline.setControlVector: ") + e.what());
    }
}

// removeControl(name) — remove a single axis (runtime or dictionary). No-op if
// unknown; lets a lab drop a built search axis without reloading.
Value pipelineRemoveControl(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.removeControl: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.removeControl(name): name string required");
    }
    w->pipeline->cond_control().remove(ev::toUtf8(args[0]));
    return ev::undefined();
}

// setIdentityAnchor(prompt, opts?) -> image — capture a reference identity from
// one full Sana generation and arm the reference-attention seam. The returned
// image IS the anchor. Later generate()/prime() calls inject the anchor's
// appearance (scaled by setIdentityWeight) while each prompt still sets pose /
// expression. Match opts (steps, size, seed) for tight alignment. Sana only.
Value pipelineSetIdentityAnchor(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.setIdentityAnchor: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    if (!w->weights_loaded) {
        return ev::throwError("Pipeline.setIdentityAnchor: call loadWeights() first");
    }
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.setIdentityAnchor(prompt, opts?): prompt string required");
    }
    std::string prompt = ev::toUtf8(args[0]);
    ev::Persistent opt(args.size() > 1 ? args[1] : ev::undefined());
    const bool includeFp32 = propBool(opt.get(), "includeFp32");
    auto opts = parseGenerateOptions(opt.get());

    try {
        w->cancel_requested.store(false, std::memory_order_relaxed);
        opts.should_cancel = [w]() {
            return w->cancel_requested.load(std::memory_order_relaxed);
        };
        std::vector<float> img = w->pipeline->capture_identity_anchor(prompt, opts);
        return makeImageResult(img, opts.height, opts.width, includeFp32);
    } catch (const brodiffusion::pipeline::GenerateCancelled&) {
        ObjectBuilder b;
        b.set("cancelled", true);
        return b.build();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.setIdentityAnchor failed: ") + e.what());
    }
}

// setIdentityWeight(weight) — injection strength for the armed anchor. 0
// disables (even with an anchor set); ~1 holds identity faithfully. Takes
// effect on the next generate()/prime(). Sana only.
Value pipelineSetIdentityWeight(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.setIdentityWeight: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    if (args.empty() || !ev::isNumber(args[0])) {
        return ev::throwTypeError("Pipeline.setIdentityWeight(weight): numeric weight required");
    }
    w->pipeline->set_identity_weight(static_cast<float>(ev::toDouble(args[0])));
    return ev::undefined();
}

// hasIdentityAnchor() -> bool — whether an anchor has been captured + armed.
Value pipelineHasIdentityAnchor(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.hasIdentityAnchor: not a loaded Pipeline");
    return ev::fromBool(w->pipeline->has_identity_anchor());
}

// clearIdentityAnchor() — drop the cached anchor and zero the weight.
Value pipelineClearIdentityAnchor(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.clearIdentityAnchor: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    w->pipeline->clear_identity_anchor();
    return ev::undefined();
}

} // namespace

void decoratePipelineControlProto(ObjectBuilder& proto) {
    proto.def("loadControlDictionary", 1, pipelineLoadControlDictionary);
    proto.def("setControl", 2, pipelineSetControl);
    proto.def("clearControl", 0, pipelineClearControl);
    proto.def("setControlBudget", 1, pipelineSetControlBudget);
    proto.def("controlNorm", 0, pipelineControlNorm);
    proto.def("controlAxes", 0, pipelineControlAxes);
    proto.def("controlVector", 1, pipelineControlVector);
    proto.def("encodeConditioning", 1, pipelineEncodeConditioning);
    proto.def("setControlVector", 4, pipelineSetControlVector);
    proto.def("removeControl", 1, pipelineRemoveControl);
    proto.def("setIdentityAnchor", 2, pipelineSetIdentityAnchor);
    proto.def("setIdentityWeight", 1, pipelineSetIdentityWeight);
    proto.def("hasIdentityAnchor", 0, pipelineHasIdentityAnchor);
    proto.def("clearIdentityAnchor", 0, pipelineClearIdentityAnchor);
}

} // namespace brodiffusion::api
