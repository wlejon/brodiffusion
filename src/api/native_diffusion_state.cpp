// PipelineState — the step-wise inspection API: advance one denoising step
// (optionally capturing an attention trace or injecting per-layer logit bias),
// VAE-decode the working latent, read/overwrite that latent, and fork a state.
//
// Ported from the QuickJS-era src/js/diffusion_bindings.cpp
// registerPipelineStateClass block. The bronze rewrite had reduced
// stepOnce()/decode() to a counter bump and a black image; both now run the
// owning Pipeline, which the state retains through its `__pipeline` link.

#include "host_diffusion_internal.h"

#include <brotensor/ops/elementwise.h>

#include <stdexcept>

namespace brodiffusion::api {

namespace {

// stepOnce(ctrl?) -> { trace? } — advance one denoising step.
//   ctrl.trace    bool — capture per-layer head-averaged attention maps
//   ctrl.attnBias ({data:Float32Array, Lq, Lk} | null)[] — per-layer
//                 pre-softmax logit bias; length must equal numXAttnBlocks().
// Supplying attnBias forces trace mode internally (brodiffusion contract).
Value stateStepOnce(Value thisVal, std::span<const Value> args) {
    auto* sw = unwrapPipelineState(thisVal);
    if (!sw) return ev::throwTypeError("PipelineState.stepOnce: not a PipelineState");
    PipelineWrapper* pw = pipelineWrapperOfState(thisVal);  // thisVal is stale after this
    brodiffusion::pipeline::Pipeline* pipe = pw ? pw->pipeline.get() : nullptr;
    if (!pipe) return ev::throwError("PipelineState.stepOnce: pipeline handle lost");
    if (pw->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    if (pw->cancel_requested.load(std::memory_order_relaxed)) {
        ObjectBuilder b;
        b.set("cancelled", true);
        return b.build();
    }

    ev::Persistent ctrl(args.empty() ? ev::undefined() : args[0]);
    const bool wantTrace = propBool(ctrl.get(), "trace");

    // attn_logit_biases — owned tensors kept alive for the step_once call;
    // ptrs is the parallel null-preserving pointer vector brodiffusion takes.
    std::vector<brotensor::Tensor> owned;
    std::vector<const brotensor::Tensor*> ptrs;
    bool haveBias = false;
    if (ev::isObject(ctrl.get())) {
        Value bVal = ev::getProperty(ctrl.get(), "attnBias");
        if (ev::isUndefined(bVal) || ev::isNull(bVal)) {
            bVal = ev::getProperty(ctrl.get(), "logitBias");
        }
        ev::Persistent biasArr(bVal);
        if (ev::isObject(biasArr.get())) {
            haveBias = true;
            // Denoiser-generic block count: 16 for the SD1.5 UNet, 57 for the
            // Flux DiT. A denoiser with no trace support reports 0, so any
            // supplied attnBias array trips the length check with a clear
            // message instead of throwing past the binding.
            const int n = pipe->num_xattn_blocks();
            const uint32_t len = arrayLength(biasArr.get());
            if (static_cast<int>(len) != n) {
                return ev::throwRangeError(
                    "PipelineState.stepOnce: attnBias length " + std::to_string(len) +
                    " must equal numXAttnBlocks() " + std::to_string(n));
            }
            owned.reserve(static_cast<size_t>(n));
            ptrs.reserve(static_cast<size_t>(n));
            std::vector<bool> present(static_cast<size_t>(n), false);
            for (int i = 0; i < n; ++i) {
                ev::Persistent e(ev::getElement(biasArr.get(), static_cast<uint32_t>(i)));
                if (!ev::isObject(e.get())) continue;   // null / undefined: no bias
                present[static_cast<size_t>(i)] = true;
                int Lq = 0, Lk = 0;
                propInt(e.get(), "Lq", Lq);
                propInt(e.get(), "Lk", Lk);
                // Read the view LAST: the property reads above may allocate.
                Value dv = ev::getProperty(e.get(), "data");
                const float* fp = nullptr;
                size_t cnt = 0;
                const bool ok = readFloat32Array(dv, fp, cnt) && fp && Lq > 0 && Lk > 0 &&
                                cnt == static_cast<size_t>(Lq) * static_cast<size_t>(Lk);
                if (!ok) {
                    return ev::throwTypeError(
                        "PipelineState.stepOnce: attnBias[" + std::to_string(i) +
                        "] must be { data: Float32Array(Lq*Lk), Lq, Lk } or null");
                }
                owned.push_back(brotensor::Tensor::from_host(fp, Lq, Lk));
            }
            // Pointers are taken only once `owned` has stopped growing — a
            // push_back can reallocate the vector and move every element.
            size_t next = 0;
            for (int i = 0; i < n; ++i) {
                ptrs.push_back(present[static_cast<size_t>(i)] ? &owned[next++] : nullptr);
            }
        }
    }

    // Denoiser-generic trace buffer: a vector of (Lq, Lk) attention tensors,
    // one per cross-attention block, regardless of denoiser type.
    brodiffusion::AttentionTrace trace;
    brodiffusion::AttentionTrace* tracePtr = wantTrace ? &trace : nullptr;
    const std::vector<const brotensor::Tensor*>* biasPtr = haveBias ? &ptrs : nullptr;

    try {
        pipe->step_once(sw->state, sw->opts, tracePtr, biasPtr);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("PipelineState.stepOnce failed: ") + e.what());
    }

    ObjectBuilder res;
    if (wantTrace) {
        ev::Persistent arr(hostArrayOf(trace.size(), [&trace](size_t i) -> Value {
            const brotensor::Tensor& t = trace[i];
            std::vector<float> host = downloadTensorFloats(t);
            ObjectBuilder entry;
            entry.set("Lq", static_cast<double>(t.rows));
            entry.set("Lk", static_cast<double>(t.cols));
            ev::Persistent d(makeFloat32Array(host.data(), host.size()));
            entry.set("data", d.get());
            return entry.build();
        }));
        res.set("trace", arr.get());
    }
    return res.build();
}

// decode(opts?) -> { width, height, data } — VAE-decode the current latent.
// opts.includeFp32 attaches the raw NCHW FP32 buffer as `fp32`.
Value stateDecode(Value thisVal, std::span<const Value> args) {
    auto* sw = unwrapPipelineState(thisVal);
    if (!sw) return ev::throwTypeError("PipelineState.decode: not a PipelineState");
    PipelineWrapper* pw = pipelineWrapperOfState(thisVal);
    brodiffusion::pipeline::Pipeline* pipe = pw ? pw->pipeline.get() : nullptr;
    if (!pipe) return ev::throwError("PipelineState.decode: pipeline handle lost");
    if (pw->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);

    const bool includeFp32 = !args.empty() && propBool(args[0], "includeFp32");
    try {
        std::vector<float> img = pipe->decode(sw->state);
        // KL-VAE upsamples 8x (SD / Flux); Sana's DC-AE upsamples 32x.
        const int scale = pipe->vae_scale_factor();
        return makeImageResult(img, sw->state.H_lat * scale, sw->state.W_lat * scale, includeFp32);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("PipelineState.decode failed: ") + e.what());
    }
}

// latent() -> Float32Array — download the working latent (small; on demand).
Value stateLatent(Value thisVal, std::span<const Value>) {
    auto* sw = unwrapPipelineState(thisVal);
    if (!sw) return ev::throwTypeError("PipelineState.latent: not a PipelineState");
    try {
        std::vector<float> host = downloadTensorFloats(sw->state.latent);
        return makeFloat32Array(host.data(), host.size());
    } catch (const std::exception& e) {
        return ev::throwError(std::string("PipelineState.latent failed: ") + e.what());
    }
}

// setLatent(Float32Array) — overwrite the working latent in place (host FP32,
// length must equal the current latent's element count; cast to the state's
// working dtype if needed). For spatial paint compositing: blend two states'
// latents host-side each step, then push the blend back before the next step.
Value stateSetLatent(Value thisVal, std::span<const Value> args) {
    auto* sw = unwrapPipelineState(thisVal);
    if (!sw) return ev::throwTypeError("PipelineState.setLatent: not a PipelineState");
    const size_t expect = static_cast<size_t>(sw->state.latent.rows) *
                          static_cast<size_t>(sw->state.latent.cols);
    const float* fp = nullptr;
    size_t cnt = 0;
    if (args.empty() || !readFloat32Array(args[0], fp, cnt) || cnt != expect) {
        return ev::throwTypeError(
            "PipelineState.setLatent(data): Float32Array length must equal the current latent's "
            "element count (" + std::to_string(expect) + ")");
    }
    try {
        const brotensor::Dtype dt = sw->state.latent.dtype;
        brotensor::Tensor host = brotensor::Tensor::from_host(
            fp, sw->state.latent.rows, sw->state.latent.cols);
        if (dt == brotensor::Dtype::FP32) {
            sw->state.latent = host;
        } else {
            brotensor::Tensor casted;
            brotensor::cast(host, casted, dt);
            sw->state.latent = casted;
        }
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("PipelineState.setLatent failed: ") + e.what());
    }
}

// krea2StepTimestep() -> number — the active scheduler's timestep for this
// state's current step_index (0..1000 scale), the same value step_once() will
// feed the denoiser next. Krea 2 only.
Value stateKrea2StepTimestep(Value thisVal, std::span<const Value>) {
    auto* sw = unwrapPipelineState(thisVal);
    if (!sw) return ev::throwTypeError("PipelineState.krea2StepTimestep: not a PipelineState");
    PipelineWrapper* pw = pipelineWrapperOfState(thisVal);
    brodiffusion::pipeline::Pipeline* pipe = pw ? pw->pipeline.get() : nullptr;
    if (!pipe) return ev::throwError("PipelineState.krea2StepTimestep: pipeline handle lost");
    // The schedule it reads is rewritten by a background generate's prime().
    if (pw->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    try {
        return ev::fromDouble(static_cast<double>(pipe->krea_step_timestep(sw->state)));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("PipelineState.krea2StepTimestep failed: ") + e.what());
    }
}

// qwenImage21StepTimestep() -> number — the active scheduler's timestep for
// this state's current step_index (0..1000 scale), the same value step_once()
// will feed the denoiser next and the one qwenImage21TimeMod() takes.
// Qwen-Image 2.1 only.
Value stateQwenImage21StepTimestep(Value thisVal, std::span<const Value>) {
    auto* sw = unwrapPipelineState(thisVal);
    if (!sw) return ev::throwTypeError("PipelineState.qwenImage21StepTimestep: not a PipelineState");
    PipelineWrapper* pw = pipelineWrapperOfState(thisVal);
    brodiffusion::pipeline::Pipeline* pipe = pw ? pw->pipeline.get() : nullptr;
    if (!pipe) return ev::throwError("PipelineState.qwenImage21StepTimestep: pipeline handle lost");
    if (pw->busy.load(std::memory_order_acquire)) return ev::throwError(kPipelineBusy);
    try {
        return ev::fromDouble(static_cast<double>(pipe->qi21_step_timestep(sw->state)));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("PipelineState.qwenImage21StepTimestep failed: ") + e.what());
    }
}

// clone() -> PipelineState — deep-copy the state (one latent clone). The
// owning Pipeline handle is carried forward so the clone stays valid.
Value stateClone(Value thisVal, std::span<const Value>) {
    auto* sw = unwrapPipelineState(thisVal);
    if (!sw) return ev::throwTypeError("PipelineState.clone: not a PipelineState");
    ev::Persistent self(thisVal);
    try {
        auto nw = std::make_unique<PipelineStateWrapper>();
        nw->state = sw->state.clone();
        nw->opts = sw->opts;
        // The link is read first: a Value is stale after any allocation, and
        // the two would otherwise be evaluated in unspecified order.
        ev::Persistent pipe(ev::getProperty(self.get(), "__pipeline"));
        ev::Persistent st(g_pipelineStateClass.createInstance(std::move(nw)));
        return attachPipelineToState(st.get(), pipe.get());
    } catch (const std::exception& e) {
        return ev::throwError(std::string("PipelineState.clone failed: ") + e.what());
    }
}

} // namespace

void decoratePipelineStateProto(ObjectBuilder& proto) {
    proto.def("stepOnce", 1, stateStepOnce);
    proto.def("decode", 1, stateDecode);
    proto.def("latent", 0, stateLatent);
    proto.def("setLatent", 1, stateSetLatent);
    proto.def("krea2StepTimestep", 0, stateKrea2StepTimestep);
    proto.def("qwenImage21StepTimestep", 0, stateQwenImage21StepTimestep);
    proto.def("clone", 0, stateClone);

    proto.accessor("stepIndex", [](Value thisVal, std::span<const Value>) -> Value {
        auto* sw = unwrapPipelineState(thisVal);
        return ev::fromDouble(sw ? sw->state.step_index : 0);
    });
    proto.accessor("numSteps", [](Value thisVal, std::span<const Value>) -> Value {
        auto* sw = unwrapPipelineState(thisVal);
        return ev::fromDouble(sw ? sw->state.n_steps : 0);
    });
    // `totalSteps` is the bronze-era spelling of numSteps; kept so code written
    // against the ported surface keeps working.
    proto.accessor("totalSteps", [](Value thisVal, std::span<const Value>) -> Value {
        auto* sw = unwrapPipelineState(thisVal);
        return ev::fromDouble(sw ? sw->state.n_steps : 0);
    });
    proto.accessor("done", [](Value thisVal, std::span<const Value>) -> Value {
        auto* sw = unwrapPipelineState(thisVal);
        return ev::fromBool(sw && sw->state.step_index >= sw->state.n_steps);
    });
    proto.accessor("latentWidth", [](Value thisVal, std::span<const Value>) -> Value {
        auto* sw = unwrapPipelineState(thisVal);
        return ev::fromDouble(sw ? sw->state.W_lat : 0);
    });
    proto.accessor("latentHeight", [](Value thisVal, std::span<const Value>) -> Value {
        auto* sw = unwrapPipelineState(thisVal);
        return ev::fromDouble(sw ? sw->state.H_lat : 0);
    });
    proto.def("cancel", 0, [](Value thisVal, std::span<const Value>) -> Value {
        Value p = ev::getProperty(thisVal, "__pipeline");
        if (auto* pw = unwrapPipeline(p)) {
            pw->cancel_requested.store(true, std::memory_order_relaxed);
        }
        return ev::undefined();
    });
}

} // namespace brodiffusion::api
