// Krea 2 research hooks on Pipeline — AdaLN / gate dials, raw-taps
// conditioning, image-as-prompt. Every method here is Krea2-only; the native
// Pipeline methods they wrap already throw a clear error for any other
// model_class, so no separate guard is added (the same way the old
// diffusion_bindings.cpp relied on the native side's own checks).
//
// Ported from the QuickJS-era src/js/diffusion_bindings.cpp krea2 block.

#include "host_diffusion_internal.h"

#include <stdexcept>

namespace brodiffusion::api {

namespace {

// krea2SetModDelta(delta: {rows,cols,data} | null, blockLo, blockHi) — delta is
// (1, 6*krea2HiddenSize()); null/undefined clears.
Value krea2SetModDelta(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.krea2SetModDelta: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    brotensor::Tensor delta;
    if (!args.empty() && ev::isObject(args[0])) {
        if (!tensorFromJs(args[0], delta)) {
            return ev::throwTypeError("Pipeline.krea2SetModDelta: delta must be {rows,cols,data}");
        }
    }
    if (args.size() < 3 || !ev::isNumber(args[1]) || !ev::isNumber(args[2])) {
        return ev::throwTypeError("Pipeline.krea2SetModDelta(delta, blockLo, blockHi): integer range required");
    }
    try {
        w->pipeline->krea_set_mod_delta(delta, i32At(args, 1), i32At(args, 2));
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.krea2SetModDelta failed: ") + e.what());
    }
}

// krea2TimeMod(timestep) -> { temb: {rows,cols,data}, mod: {rows,cols,data} }
Value krea2TimeMod(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.krea2TimeMod: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    if (args.empty() || !ev::isNumber(args[0])) {
        return ev::throwTypeError("Pipeline.krea2TimeMod(timestep): numeric timestep required");
    }
    try {
        brotensor::Tensor temb, mod;
        w->pipeline->krea_time_mod(static_cast<float>(ev::toDouble(args[0])), temb, mod);
        ObjectBuilder o;
        {
            ev::Persistent t(tensorToJs(temb));
            o.set("temb", t.get());
        }
        {
            ev::Persistent m(tensorToJs(mod));
            o.set("mod", m.get());
        }
        return o.build();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.krea2TimeMod failed: ") + e.what());
    }
}

// krea2SetGateScale(txtScale, imgScale, blockLo, blockHi)
Value krea2SetGateScale(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.krea2SetGateScale: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    if (args.size() < 4 || !ev::isNumber(args[0]) || !ev::isNumber(args[1]) ||
        !ev::isNumber(args[2]) || !ev::isNumber(args[3])) {
        return ev::throwTypeError(
            "Pipeline.krea2SetGateScale(txtScale, imgScale, blockLo, blockHi): numeric args required");
    }
    try {
        w->pipeline->krea_set_gate_scale(static_cast<float>(ev::toDouble(args[0])),
                                         static_cast<float>(ev::toDouble(args[1])),
                                         i32At(args, 2), i32At(args, 3));
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.krea2SetGateScale failed: ") + e.what());
    }
}

// krea2SetGateMask(mask: {rows,cols,data} | null, blockLo, blockHi) — mask holds
// (text_seq + img_len) values; null/undefined clears.
Value krea2SetGateMask(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.krea2SetGateMask: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    brotensor::Tensor mask;
    if (!args.empty() && ev::isObject(args[0])) {
        if (!tensorFromJs(args[0], mask)) {
            return ev::throwTypeError("Pipeline.krea2SetGateMask: mask must be {rows,cols,data}");
        }
    }
    if (args.size() < 3 || !ev::isNumber(args[1]) || !ev::isNumber(args[2])) {
        return ev::throwTypeError("Pipeline.krea2SetGateMask(mask, blockLo, blockHi): integer range required");
    }
    try {
        w->pipeline->krea_set_gate_mask(mask, i32At(args, 1), i32At(args, 2));
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.krea2SetGateMask failed: ") + e.what());
    }
}

// krea2CaptureGates(enable) — when on, every step_once() overwrites the sink.
Value krea2CaptureGates(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.krea2CaptureGates: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    const bool enable = !args.empty() && ev::toBool(args[0]);
    try {
        w->pipeline->krea_capture_gates(enable);
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.krea2CaptureGates failed: ") + e.what());
    }
}

// krea2Gates() -> { rows, cols, data } — rows = krea2NumLayers(), cols =
// text_seq + img_len (inferred from the flat buffer length).
Value krea2Gates(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.krea2Gates: not a loaded Pipeline");
    // The sink is rewritten by every step a background generate takes.
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    try {
        std::vector<float> flat = w->pipeline->krea_gates();
        const int rows = w->pipeline->krea_num_layers();
        const int cols = rows > 0 ? static_cast<int>(flat.size() / static_cast<size_t>(rows)) : 0;
        ObjectBuilder o;
        o.set("rows", static_cast<double>(rows));
        o.set("cols", static_cast<double>(cols));
        ev::Persistent d(makeFloat32Array(flat.data(), flat.size()));
        o.set("data", d.get());
        return o.build();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.krea2Gates failed: ") + e.what());
    }
}

// krea2HiddenSize() -> number (6144)
Value krea2HiddenSize(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.krea2HiddenSize: not a loaded Pipeline");
    try {
        return ev::fromDouble(w->pipeline->krea_hidden_size());
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.krea2HiddenSize failed: ") + e.what());
    }
}

// krea2NumLayers() -> number (28)
Value krea2NumLayers(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.krea2NumLayers: not a loaded Pipeline");
    try {
        return ev::fromDouble(w->pipeline->krea_num_layers());
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.krea2NumLayers failed: ") + e.what());
    }
}

// krea2EncodePromptTaps(prompt) -> { embeds, mask } — raw per-layer Qwen3-VL
// taps, pre-fusion (token-major/layer-minor). Edit rows, then feed to
// krea2EncodeText()/krea2PrimeFromTaps().
Value krea2EncodePromptTaps(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.krea2EncodePromptTaps: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    if (!w->weights_loaded) {
        return ev::throwError("Pipeline.krea2EncodePromptTaps: call loadWeights() first");
    }
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.krea2EncodePromptTaps(prompt): prompt string required");
    }
    std::string prompt = ev::toUtf8(args[0]);
    try {
        return textConditioningToJs(w->pipeline->krea_encode_prompt_taps(prompt));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.krea2EncodePromptTaps failed: ") + e.what());
    }
}

// krea2EncodeText(embeds, mask) -> {rows,cols,data} — fuse raw taps into the
// (n_valid, krea2HiddenSize()) conditioning the DiT cross-attends to. The same
// space cond_control axes (setControl/setControlVector) apply in.
Value krea2EncodeText(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.krea2EncodeText: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    brotensor::Tensor embeds, mask;
    if (args.size() < 2 || !tensorFromJs(args[0], embeds) || !tensorFromJs(args[1], mask)) {
        return ev::throwTypeError("Pipeline.krea2EncodeText(embeds, mask): {rows,cols,data} tensors required");
    }
    try {
        return tensorToJs(w->pipeline->krea_encode_text(embeds, mask));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.krea2EncodeText failed: ") + e.what());
    }
}

// krea2EncodeImagePrompt(pixels: Float32Array, H, W) -> { embeds, mask } — the
// raw-taps shape krea2EncodePromptTaps() produces for text, from an image
// through Krea 2's own Qwen3-VL vision tower. `pixels` is FP32 CHW in [0,1],
// length 3*H*W.
Value krea2EncodeImagePrompt(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.krea2EncodeImagePrompt: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    if (!w->weights_loaded) {
        return ev::throwError("Pipeline.krea2EncodeImagePrompt: call loadWeights() first");
    }
    if (args.size() < 3 || !ev::isNumber(args[1]) || !ev::isNumber(args[2])) {
        return ev::throwTypeError(
            "Pipeline.krea2EncodeImagePrompt(pixels, H, W): pixels must be a Float32Array of length 3*H*W");
    }
    const int H = i32At(args, 1);
    const int W = i32At(args, 2);
    const float* px = nullptr;
    size_t cnt = 0;
    if (!readFloat32Array(args[0], px, cnt) || H <= 0 || W <= 0 ||
        cnt != static_cast<size_t>(3) * static_cast<size_t>(H) * static_cast<size_t>(W)) {
        return ev::throwTypeError(
            "Pipeline.krea2EncodeImagePrompt(pixels, H, W): pixels must be a Float32Array of length 3*H*W");
    }
    try {
        return textConditioningToJs(w->pipeline->krea_encode_image_prompt(px, H, W));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.krea2EncodeImagePrompt failed: ") + e.what());
    }
}

// krea2PrimeFromTaps(embeds, mask, opts?, uncondEmbeds?, uncondMask?)
//   -> PipelineState. Prime a step-wise generation from caller-supplied raw
// taps instead of a prompt string. Omit the uncond pair to fall back to
// encoding opts.negativePrompt normally.
Value krea2PrimeFromTaps(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.krea2PrimeFromTaps: not a loaded Pipeline");
    if (w->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    if (!w->weights_loaded) {
        return ev::throwError("Pipeline.krea2PrimeFromTaps: call loadWeights() first");
    }

    ev::Persistent self(thisVal);
    brotensor::Tensor embeds, mask;
    if (args.size() < 2 || !tensorFromJs(args[0], embeds) || !tensorFromJs(args[1], mask)) {
        return ev::throwTypeError("Pipeline.krea2PrimeFromTaps(embeds, mask, opts?, uncondEmbeds?, uncondMask?): "
                                  "{rows,cols,data} tensors required");
    }
    auto opts = parseGenerateOptions(args.size() >= 3 ? args[2] : ev::undefined());
    w->cancel_requested.store(false, std::memory_order_relaxed);
    opts.should_cancel = [w]() {
        return w->cancel_requested.load(std::memory_order_relaxed);
    };

    brotensor::Tensor uembeds, umask;
    bool haveUncond = false;
    if (args.size() >= 5 && ev::isObject(args[3]) && ev::isObject(args[4])) {
        if (!tensorFromJs(args[3], uembeds) || !tensorFromJs(args[4], umask)) {
            return ev::throwTypeError(
                "Pipeline.krea2PrimeFromTaps: uncondEmbeds/uncondMask must be {rows,cols,data}");
        }
        haveUncond = true;
    }

    try {
        auto sw = std::make_unique<PipelineStateWrapper>();
        sw->opts = opts;
        sw->state = w->pipeline->krea_prime_from_taps(
            embeds, mask, haveUncond ? &uembeds : nullptr, haveUncond ? &umask : nullptr, opts);
        Value st = g_pipelineStateClass.createInstance(std::move(sw));
        return attachPipelineToState(st, self.get());
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.krea2PrimeFromTaps failed: ") + e.what());
    }
}

} // namespace

void decoratePipelineKrea2Proto(ObjectBuilder& proto) {
    proto.def("krea2SetModDelta", 3, krea2SetModDelta);
    proto.def("krea2TimeMod", 1, krea2TimeMod);
    proto.def("krea2SetGateScale", 4, krea2SetGateScale);
    proto.def("krea2SetGateMask", 3, krea2SetGateMask);
    proto.def("krea2CaptureGates", 1, krea2CaptureGates);
    proto.def("krea2Gates", 0, krea2Gates);
    proto.def("krea2HiddenSize", 0, krea2HiddenSize);
    proto.def("krea2NumLayers", 0, krea2NumLayers);
    proto.def("krea2EncodePromptTaps", 1, krea2EncodePromptTaps);
    proto.def("krea2EncodeText", 2, krea2EncodeText);
    proto.def("krea2EncodeImagePrompt", 3, krea2EncodeImagePrompt);
    proto.def("krea2PrimeFromTaps", 5, krea2PrimeFromTaps);
}

} // namespace brodiffusion::api
