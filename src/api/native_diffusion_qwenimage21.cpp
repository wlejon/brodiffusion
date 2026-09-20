// Qwen-Image 2.1 research hooks on Pipeline — the qwenImage21* block.
//
// The sibling of native_diffusion_krea2.cpp, with one deliberate difference:
// every method here checks the model class itself and throws a TypeError
// rather than letting the native side's std::runtime_error surface as a plain
// Error. A JS caller reaching for qwenImage21SetModDelta on a Krea 2 pipeline
// has made a type mistake, not hit a runtime failure, and the hook block is
// large enough that the distinction is worth the four lines.
//
// The prefix-cache discipline that pipeline_qwenimage21_hooks.cpp documents
// applies here unchanged: the Pipeline resets the live prefix KV cache when a
// prefix-side hook is armed or cleared, so nothing in JS has to think about
// it — except qwenImage21ResetCache(), which is exposed for the case where a
// caller edits conditioning out of band.

#include "host_diffusion_internal.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::api {

namespace {

// Every qwenImage21* method is QwenImage21-only. Returns null and leaves a
// TypeError pending (via the caller's return) when `thisVal` is not a loaded
// Qwen-Image 2.1 Pipeline.
PipelineWrapper* qi21Pipeline(Value thisVal) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return nullptr;
    if (w->pipeline->config().model_class !=
        brodiffusion::ModelClass::QwenImage21) {
        return nullptr;
    }
    return w;
}

Value notQi21(const char* method) {
    return ev::throwTypeError(
        std::string("Pipeline.") + method +
        ": not a loaded Qwen-Image 2.1 Pipeline");
}

// A (rows, cols) JS tensor object over a host FP32 slice.
Value tensorFromHost(const float* data, int rows, int cols) {
    ObjectBuilder o;
    o.set("rows", static_cast<double>(rows));
    o.set("cols", static_cast<double>(cols));
    ev::Persistent d(makeFloat32Array(
        data, static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)));
    o.set("data", d.get());
    return o.build();
}

Value makeInt32Array(const std::vector<int>& v) {
    Value arr = ev::createTypedArray(ev::elements::Int32,
                                     static_cast<uint32_t>(v.size()));
    if (!v.empty()) {
        ev::fillTypedArray(
            arr, std::span<const uint8_t>(
                     reinterpret_cast<const uint8_t*>(v.data()),
                     v.size() * sizeof(int)));
    }
    return arr;
}

// "target" (default) / "prefix" / "both", or the equivalent 0 / 1 / 2.
bool readModTarget(Value v, brodiffusion::dit::QwenImage21ModTarget& out) {
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

// qwenImage21SetModDelta(delta: {rows,cols,data} | null, blockLo, blockHi,
//                        target?: 'target' | 'prefix' | 'both')
// delta is (1, 4*qwenImage21HiddenSize()); null/undefined clears. A
// prefix-side target resets the live prefix KV cache so the change lands.
Value qi21SetModDelta(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21SetModDelta");
    brotensor::Tensor delta;
    if (!args.empty() && ev::isObject(args[0])) {
        if (!tensorFromJs(args[0], delta)) {
            return ev::throwTypeError(
                "Pipeline.qwenImage21SetModDelta: delta must be {rows,cols,data}");
        }
    }
    if (args.size() < 3 || !ev::isNumber(args[1]) || !ev::isNumber(args[2])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21SetModDelta(delta, blockLo, blockHi, target?): "
            "integer range required");
    }
    brodiffusion::dit::QwenImage21ModTarget target;
    if (!readModTarget(argAt(args, 3), target)) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21SetModDelta: target must be "
            "'target' | 'prefix' | 'both'");
    }
    try {
        w->pipeline->qi21_set_mod_delta(delta, i32At(args, 1), i32At(args, 2),
                                        target);
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21SetModDelta failed: ") + e.what());
    }
}

// qwenImage21TimeMod(timestep) ->
//   { temb: (2, hidden), modTarget: (1, 4*hidden), modPrefix: (1, 4*hidden) }
// `timestep` is the 0..1000-scale value qwenImage21StepTimestep() returns.
// The mod rows are PRE-tanh — the space qwenImage21SetModDelta adds into.
Value qi21TimeMod(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21TimeMod");
    if (args.empty() || !ev::isNumber(args[0])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21TimeMod(timestep): numeric timestep required");
    }
    try {
        brotensor::Tensor temb, mod;
        w->pipeline->qi21_time_mod(static_cast<float>(ev::toDouble(args[0])),
                                   temb, mod);
        std::vector<float> mh = downloadTensorFloats(mod);
        const int cols = mod.cols;
        const bool two_rows = mod.rows >= 2;
        ObjectBuilder o;
        {
            ev::Persistent t(tensorToJs(temb));
            o.set("temb", t.get());
        }
        {
            ev::Persistent m(tensorFromHost(mh.data(), 1, cols));
            o.set("modTarget", m.get());
        }
        {
            // causal_condition off would collapse the two rows into one; hand
            // back the same row rather than an empty tensor so callers do not
            // have to branch.
            const float* row1 =
                mh.data() + (two_rows ? static_cast<std::size_t>(cols) : 0);
            ev::Persistent m(tensorFromHost(row1, 1, cols));
            o.set("modPrefix", m.get());
        }
        return o.build();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21TimeMod failed: ") + e.what());
    }
}

// qwenImage21SetGateScale(attnScale, mlpScale, txtScale, imgScale,
//                         blockLo, blockHi) — post-tanh multipliers.
Value qi21SetGateScale(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21SetGateScale");
    if (args.size() < 6 || !ev::isNumber(args[0]) || !ev::isNumber(args[1]) ||
        !ev::isNumber(args[2]) || !ev::isNumber(args[3]) ||
        !ev::isNumber(args[4]) || !ev::isNumber(args[5])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21SetGateScale(attnScale, mlpScale, txtScale, "
            "imgScale, blockLo, blockHi): numeric args required");
    }
    try {
        w->pipeline->qi21_set_gate_scale(
            static_cast<float>(ev::toDouble(args[0])),
            static_cast<float>(ev::toDouble(args[1])),
            static_cast<float>(ev::toDouble(args[2])),
            static_cast<float>(ev::toDouble(args[3])),
            i32At(args, 4), i32At(args, 5));
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21SetGateScale failed: ") + e.what());
    }
}

// qwenImage21SetGateMask(mask: {rows,cols,data} | null, blockLo, blockHi) —
// mask holds (textRows + imgLen) values in joint forward order.
Value qi21SetGateMask(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21SetGateMask");
    brotensor::Tensor mask;
    if (!args.empty() && ev::isObject(args[0])) {
        if (!tensorFromJs(args[0], mask)) {
            return ev::throwTypeError(
                "Pipeline.qwenImage21SetGateMask: mask must be {rows,cols,data}");
        }
    }
    if (args.size() < 3 || !ev::isNumber(args[1]) || !ev::isNumber(args[2])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21SetGateMask(mask, blockLo, blockHi): "
            "integer range required");
    }
    try {
        w->pipeline->qi21_set_gate_mask(mask, i32At(args, 1), i32At(args, 2));
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21SetGateMask failed: ") + e.what());
    }
}

// qwenImage21SetNormOutScaleDelta(delta: {rows,cols,data} | null) — (1, hidden)
// added to the final adaptive scale; null clears.
Value qi21SetNormOutScaleDelta(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21SetNormOutScaleDelta");
    brotensor::Tensor delta;
    if (!args.empty() && ev::isObject(args[0])) {
        if (!tensorFromJs(args[0], delta)) {
            return ev::throwTypeError(
                "Pipeline.qwenImage21SetNormOutScaleDelta: delta must be "
                "{rows,cols,data}");
        }
    }
    try {
        w->pipeline->qi21_set_norm_out_scale_delta(delta);
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21SetNormOutScaleDelta failed: ") +
            e.what());
    }
}

// qwenImage21CaptureGates(enable)
Value qi21CaptureGates(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21CaptureGates");
    const bool enable = !args.empty() && ev::toBool(args[0]);
    try {
        w->pipeline->qi21_capture_gates(enable);
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21CaptureGates failed: ") + e.what());
    }
}

// qwenImage21Gates() -> { rows, cols, data } — rows = qwenImage21NumLayers(),
// cols = textRows + imgLen (inferred from the flat buffer length).
Value qi21Gates(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21Gates");
    try {
        std::vector<float> flat = w->pipeline->qi21_gates();
        const int rows = w->pipeline->qi21_num_layers();
        const int cols =
            rows > 0 ? static_cast<int>(flat.size() / static_cast<size_t>(rows))
                     : 0;
        ObjectBuilder o;
        o.set("rows", static_cast<double>(rows));
        o.set("cols", static_cast<double>(cols));
        ev::Persistent d(makeFloat32Array(flat.data(), flat.size()));
        o.set("data", d.get());
        return o.build();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21Gates failed: ") + e.what());
    }
}

// qwenImage21HiddenSize() -> 4096
Value qi21HiddenSize(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21HiddenSize");
    try {
        return ev::fromDouble(w->pipeline->qi21_hidden_size());
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21HiddenSize failed: ") + e.what());
    }
}

// qwenImage21NumLayers() -> 32
Value qi21NumLayers(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21NumLayers");
    try {
        return ev::fromDouble(w->pipeline->qi21_num_layers());
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21NumLayers failed: ") + e.what());
    }
}

// qwenImage21TextHiddenDim() -> 4096 (the Qwen3-VL-8B width; the dimension a
// control dictionary or a setControlVector direction must have).
Value qi21TextHiddenDim(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21TextHiddenDim");
    try {
        return ev::fromDouble(w->pipeline->qi21_text_hidden_dim());
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21TextHiddenDim failed: ") + e.what());
    }
}

// qwenImage21EncodePrompt(prompt) -> { embeds, mask, ids } — the raw
// (n, 4096) Qwen3-VL-8B rows the DiT's txt_in consumes, an all-ones validity
// mask, and the FULL template token ids (system prefix included). Edit
// `embeds` and feed it back through qwenImage21PrimeFromText().
Value qi21EncodePrompt(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21EncodePrompt");
    if (!w->weights_loaded) {
        return ev::throwError(
            "Pipeline.qwenImage21EncodePrompt: call loadWeights() first");
    }
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21EncodePrompt(prompt): prompt string required");
    }
    std::string prompt = ev::toUtf8(args[0]);
    try {
        brodiffusion::qwenimage21::TextConditioning tc =
            w->pipeline->qi21_encode_prompt(prompt);
        ObjectBuilder o;
        {
            ev::Persistent e(tensorToJs(tc.embeds));
            o.set("embeds", e.get());
        }
        {
            ev::Persistent m(tensorToJs(tc.mask));
            o.set("mask", m.get());
        }
        {
            ev::Persistent ids(makeInt32Array(tc.token_ids));
            o.set("ids", ids.get());
        }
        o.set("dropIdx", static_cast<double>(tc.drop_idx));
        return o.build();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21EncodePrompt failed: ") + e.what());
    }
}

// qwenImage21PrimeFromText(embeds, mask, opts?, uncondEmbeds?, uncondMask?)
//   -> PipelineState. Prime from caller-supplied (n, 4096) rows instead of a
// prompt string. `mask` may be null (all rows valid). Omit the uncond pair to
// fall back to encoding opts.negativePrompt when guidanceScale > 1.
Value qi21PrimeFromText(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21PrimeFromText");
    if (!w->weights_loaded) {
        return ev::throwError(
            "Pipeline.qwenImage21PrimeFromText: call loadWeights() first");
    }

    ev::Persistent self(thisVal);
    brotensor::Tensor embeds, mask;
    if (args.empty() || !tensorFromJs(args[0], embeds)) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21PrimeFromText(embeds, mask, opts?, "
            "uncondEmbeds?, uncondMask?): embeds must be {rows,cols,data}");
    }
    if (args.size() >= 2 && ev::isObject(args[1]) &&
        !tensorFromJs(args[1], mask)) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21PrimeFromText: mask must be {rows,cols,data} "
            "or null");
    }
    auto opts = parseGenerateOptions(args.size() >= 3 ? args[2] : ev::undefined());

    brotensor::Tensor uembeds, umask;
    bool haveUncond = false;
    if (args.size() >= 4 && ev::isObject(args[3])) {
        if (!tensorFromJs(args[3], uembeds)) {
            return ev::throwTypeError(
                "Pipeline.qwenImage21PrimeFromText: uncondEmbeds must be "
                "{rows,cols,data}");
        }
        if (args.size() >= 5 && ev::isObject(args[4]) &&
            !tensorFromJs(args[4], umask)) {
            return ev::throwTypeError(
                "Pipeline.qwenImage21PrimeFromText: uncondMask must be "
                "{rows,cols,data} or null");
        }
        haveUncond = true;
    }

    try {
        auto sw = std::make_unique<PipelineStateWrapper>();
        sw->opts = opts;
        sw->state = w->pipeline->qi21_prime_from_text(
            embeds, mask, haveUncond ? &uembeds : nullptr,
            haveUncond ? &umask : nullptr, opts);
        Value st = g_pipelineStateClass.createInstance(std::move(sw));
        return attachPipelineToState(st, self.get());
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21PrimeFromText failed: ") + e.what());
    }
}

// qwenImage21TextRows(uncond?) -> { rows, cols, data } — the prepared
// (nValid, hidden) text rows of the most recent prime(), i.e. the joint
// sequence's text half after txt_in.
Value qi21TextRows(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21TextRows");
    const bool uncond = !args.empty() && ev::toBool(args[0]);
    try {
        return tensorToJs(w->pipeline->qi21_text_rows(uncond));
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21TextRows failed: ") + e.what());
    }
}

// qwenImage21SetTextRows(rows, uncond?) — replace them; the prefix KV cache
// is reset so the change lands on the next step.
Value qi21SetTextRows(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21SetTextRows");
    brotensor::Tensor rows;
    if (args.empty() || !tensorFromJs(args[0], rows)) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21SetTextRows(rows, uncond?): rows must be "
            "{rows,cols,data}");
    }
    const bool uncond = args.size() >= 2 && ev::toBool(args[1]);
    try {
        w->pipeline->qi21_set_text_rows(rows, uncond);
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21SetTextRows failed: ") + e.what());
    }
}

// qwenImage21ScalePrefixKv(layerLo, layerHi, kScale, vScale) — attenuate the
// live prefix KV cache for a layer range. Takes effect on the next step with
// no re-extraction; requires at least one step to have run.
Value qi21ScalePrefixKv(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ScalePrefixKv");
    if (args.size() < 4 || !ev::isNumber(args[0]) || !ev::isNumber(args[1]) ||
        !ev::isNumber(args[2]) || !ev::isNumber(args[3])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21ScalePrefixKv(layerLo, layerHi, kScale, "
            "vScale): numeric args required");
    }
    try {
        w->pipeline->qi21_scale_prefix_kv(
            i32At(args, 0), i32At(args, 1),
            static_cast<float>(ev::toDouble(args[2])),
            static_cast<float>(ev::toDouble(args[3])));
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21ScalePrefixKv failed: ") + e.what());
    }
}

// qwenImage21ResetCache() — drop the live prefix KV cache so the next step
// re-extracts. The hooks do this themselves; call it after editing
// conditioning out of band.
Value qi21ResetCache(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ResetCache");
    try {
        w->pipeline->qi21_reset_cache();
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21ResetCache failed: ") + e.what());
    }
}

// qwenImage21EncodeImage(pixels, H, W) -> { rows, cols, data, hLat, wLat } —
// `pixels` is a Float32Array of length 3*H*W, CHW in [0,1]; H and W must be
// multiples of 16. The result is a pipeline-scale latent, ready for
// PipelineState.setLatent() or qwenImage21Decode().
Value qi21EncodeImage(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21EncodeImage");
    if (!w->weights_loaded) {
        return ev::throwError(
            "Pipeline.qwenImage21EncodeImage: call loadWeights() first");
    }
    if (args.size() < 3 || !ev::isNumber(args[1]) || !ev::isNumber(args[2])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21EncodeImage(pixels, H, W): pixels must be a "
            "Float32Array of length 3*H*W");
    }
    const int H = i32At(args, 1);
    const int W = i32At(args, 2);
    const float* px = nullptr;
    size_t cnt = 0;
    if (!readFloat32Array(args[0], px, cnt) || H <= 0 || W <= 0 ||
        cnt != static_cast<size_t>(3) * static_cast<size_t>(H) *
                   static_cast<size_t>(W)) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21EncodeImage(pixels, H, W): pixels must be a "
            "Float32Array of length 3*H*W");
    }
    try {
        brotensor::Tensor latent = w->pipeline->qi21_encode_image(px, H, W);
        const int scale = w->pipeline->vae_scale_factor();
        ev::Persistent t(tensorToJs(latent));
        t.set(ev::setProperty(t.get(), "hLat",
                              ev::fromDouble(H / scale)));
        t.set(ev::setProperty(t.get(), "wLat",
                              ev::fromDouble(W / scale)));
        return t.get();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21EncodeImage failed: ") + e.what());
    }
}

// qwenImage21Decode(latent, hLat, wLat, opts?) -> { width, height, data,
// fp32? } — the same canvas-ready shape PipelineState.decode() returns, with
// alpha dropped. `latent` is {rows,cols,data} holding 64*hLat*wLat floats.
Value qi21Decode(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21Decode");
    brotensor::Tensor latent;
    if (args.size() < 3 || !tensorFromJs(args[0], latent) ||
        !ev::isNumber(args[1]) || !ev::isNumber(args[2])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21Decode(latent, hLat, wLat): latent must be "
            "{rows,cols,data} and the grid numeric");
    }
    const int hLat = i32At(args, 1);
    const int wLat = i32At(args, 2);
    const bool wantFp32 =
        args.size() >= 4 && ev::isObject(args[3]) && propBool(args[3], "fp32");
    try {
        std::vector<float> rgb = w->pipeline->qi21_decode(latent, hLat, wLat);
        const int scale = w->pipeline->vae_scale_factor();
        return makeImageResult(rgb, hLat * scale, wLat * scale, wantFp32);
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21Decode failed: ") + e.what());
    }
}

// qwenImage21ReleaseTextEncoder() — free the 8.5 GiB Qwen3-VL-8B backbone,
// which is idle from prime() onwards. A primed state keeps stepping;
// encoding a new prompt throws until it is reloaded.
Value qi21ReleaseTextEncoder(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ReleaseTextEncoder");
    try {
        w->pipeline->qi21_release_text_encoder();
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21ReleaseTextEncoder failed: ") +
            e.what());
    }
}

// qwenImage21TextEncoderResident() -> boolean
Value qi21TextEncoderResident(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21TextEncoderResident");
    try {
        return ev::fromBool(w->pipeline->qi21_text_encoder_resident());
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21TextEncoderResident failed: ") +
            e.what());
    }
}

// qwenImage21ReloadTextEncoder(modelDir, textEncoderPath?, opts?) — reload the
// backbone into a pipeline it was released from, or swap in a different one.
Value qi21ReloadTextEncoder(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ReloadTextEncoder");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21ReloadTextEncoder(modelDir, path?, opts?): "
            "modelDir string required");
    }
    const std::string dir = resolveDiffusionPath(ev::toUtf8(args[0]));
    std::string path;
    if (args.size() >= 2 && ev::isString(args[1])) {
        path = resolveDiffusionPath(ev::toUtf8(args[1]));
    }
    bool quantize = true;
    if (args.size() >= 3 && ev::isObject(args[2])) {
        quantize = propBool(args[2], "quantizeWeights", true);
    }
    try {
        w->pipeline->qi21_reload_text_encoder(dir, path, quantize);
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21ReloadTextEncoder failed: ") +
            e.what());
    }
}

} // namespace

void decoratePipelineQwenImage21Proto(ObjectBuilder& proto) {
    proto.def("qwenImage21SetModDelta", 4, qi21SetModDelta);
    proto.def("qwenImage21TimeMod", 1, qi21TimeMod);
    proto.def("qwenImage21SetGateScale", 6, qi21SetGateScale);
    proto.def("qwenImage21SetGateMask", 3, qi21SetGateMask);
    proto.def("qwenImage21SetNormOutScaleDelta", 1, qi21SetNormOutScaleDelta);
    proto.def("qwenImage21CaptureGates", 1, qi21CaptureGates);
    proto.def("qwenImage21Gates", 0, qi21Gates);
    proto.def("qwenImage21HiddenSize", 0, qi21HiddenSize);
    proto.def("qwenImage21NumLayers", 0, qi21NumLayers);
    proto.def("qwenImage21TextHiddenDim", 0, qi21TextHiddenDim);
    proto.def("qwenImage21EncodePrompt", 1, qi21EncodePrompt);
    proto.def("qwenImage21PrimeFromText", 5, qi21PrimeFromText);
    proto.def("qwenImage21TextRows", 1, qi21TextRows);
    proto.def("qwenImage21SetTextRows", 2, qi21SetTextRows);
    proto.def("qwenImage21ScalePrefixKv", 4, qi21ScalePrefixKv);
    proto.def("qwenImage21ResetCache", 0, qi21ResetCache);
    proto.def("qwenImage21EncodeImage", 3, qi21EncodeImage);
    proto.def("qwenImage21Decode", 3, qi21Decode);
    proto.def("qwenImage21ReleaseTextEncoder", 0, qi21ReleaseTextEncoder);
    proto.def("qwenImage21TextEncoderResident", 0, qi21TextEncoderResident);
    proto.def("qwenImage21ReloadTextEncoder", 3, qi21ReloadTextEncoder);
}

} // namespace brodiffusion::api
