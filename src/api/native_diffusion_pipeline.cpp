#include "host_diffusion_internal.h"
#include <brodiffusion/version.h>
#include <brodiffusion/scheduler.h>
#include <brodiffusion/lcm_scheduler.h>
#include <brodiffusion/flow_match_scheduler.h>
#include <brodiffusion/scm_scheduler.h>
#include <brodiffusion/dpm_solver.h>
#include <brolm/tokenizer.h>
#include <brotensor/runtime.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace brodiffusion::api {

std::atomic<bool> g_diffusionCancelRequested{false};

HostClass g_pipelineClass;
HostClass g_pipelineStateClass;

PipelineWrapper* unwrapPipeline(Value v) {
    void* ptr = g_pipelineClass.unwrap(v);
    if (!ptr) return nullptr;
    auto* w = static_cast<PipelineWrapper*>(ptr);
    return (w && w->tag == kHostPipelineTag) ? w : nullptr;
}

PipelineStateWrapper* unwrapPipelineState(Value v) {
    void* ptr = g_pipelineStateClass.unwrap(v);
    if (!ptr) return nullptr;
    auto* w = static_cast<PipelineStateWrapper*>(ptr);
    return (w && w->tag == kHostPipelineStateTag) ? w : nullptr;
}

Value makeFloat32Array(const float* data, size_t count) {
    Value arr = ev::createTypedArray(ev::elements::Float32, static_cast<uint32_t>(count));
    if (data && count > 0) {
        ev::fillTypedArray(arr, std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(data), count * sizeof(float)));
    }
    return arr;
}

Value makeUint8ClampedArray(const uint8_t* data, size_t count) {
    Value arr = ev::createTypedArray(ev::elements::Uint8Clamped, static_cast<uint32_t>(count));
    if (data && count > 0) {
        ev::fillTypedArray(arr, std::span<const uint8_t>(data, count));
    }
    return arr;
}

Value makeImageResult(const std::vector<float>& nchw, int H, int W, bool includeFp32) {
    const int plane = H * W;
    std::vector<uint8_t> rgba(static_cast<size_t>(4) * plane);
    for (int i = 0; i < plane; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = (nchw[static_cast<size_t>(c) * plane + i] * 0.5f + 0.5f) * 255.0f;
            if (v < 0.0f) v = 0.0f;
            else if (v > 255.0f) v = 255.0f;
            rgba[static_cast<size_t>(4) * i + c] = static_cast<uint8_t>(v + 0.5f);
        }
        rgba[static_cast<size_t>(4) * i + 3] = 255;
    }

    ObjectBuilder res;
    res.set("width", static_cast<double>(W));
    res.set("height", static_cast<double>(H));
    {
        ev::Persistent d(makeUint8ClampedArray(rgba.data(), rgba.size()));
        res.set("data", d.get());
    }
    if (includeFp32) {
        ev::Persistent fp(makeFloat32Array(nchw.data(), nchw.size()));
        res.set("fp32", fp.get());
    }
    return res.build();
}

bool readFloat32Array(Value val, const float*& outData, size_t& outCount) {
    outData = nullptr;
    outCount = 0;
    ev::TypedArrayInfo info = ev::typedArrayInfo(val);
    if (!info || info.elementKind != ev::elements::Float32) return false;
    outData = reinterpret_cast<const float*>(info.data);
    outCount = info.elementCount;
    return true;
}

bool readUint8Array(Value val, const uint8_t*& outData, size_t& outCount) {
    outData = nullptr;
    outCount = 0;
    ev::TypedArrayInfo info = ev::typedArrayInfo(val);
    if (!info || (info.elementKind != ev::elements::Uint8 && info.elementKind != ev::elements::Uint8Clamped)) {
        return false;
    }
    outData = reinterpret_cast<const uint8_t*>(info.data);
    outCount = info.elementCount;
    return true;
}

namespace {

brodiffusion::pipeline::GenerateOptions parseGenerateOptions(Value v) {
    brodiffusion::pipeline::GenerateOptions o;
    if (!ev::isObject(v)) return o;

    Value wVal = ev::getProperty(v, "width");
    if (!ev::isUndefined(wVal)) o.width = static_cast<int>(ev::toDouble(wVal));

    Value hVal = ev::getProperty(v, "height");
    if (!ev::isUndefined(hVal)) o.height = static_cast<int>(ev::toDouble(hVal));

    Value sVal = ev::getProperty(v, "steps");
    if (!ev::isUndefined(sVal)) o.num_inference_steps = static_cast<int>(ev::toDouble(sVal));

    Value gVal = ev::getProperty(v, "guidanceScale");
    if (!ev::isUndefined(gVal)) o.guidance_scale = static_cast<float>(ev::toDouble(gVal));

    Value negVal = ev::getProperty(v, "negativePrompt");
    if (ev::isString(negVal)) o.negative_prompt = ev::toUtf8(negVal);

    Value seedVal = ev::getProperty(v, "seed");
    if (!ev::isUndefined(seedVal)) {
        if (ev::isBigInt(seedVal)) {
            o.seed = ev::toUint64(seedVal);
        } else if (!ev::isObject(seedVal)) {
            o.seed = static_cast<uint64_t>(ev::toDouble(seedVal));
        }
    }

    Value initImg = ev::getProperty(v, "initImagePath");
    if (ev::isString(initImg)) o.init_image_path = ev::toUtf8(initImg);

    Value strVal = ev::getProperty(v, "strength");
    if (!ev::isUndefined(strVal)) o.strength = static_cast<float>(ev::toDouble(strVal));

    Value vaeSample = ev::getProperty(v, "vaeEncodeSample");
    if (!ev::isUndefined(vaeSample)) o.vae_encode_sample = ev::toBool(vaeSample);

    Value maskImg = ev::getProperty(v, "maskImagePath");
    if (ev::isString(maskImg)) o.mask_image_path = ev::toUtf8(maskImg);

    Value nsVal = ev::getProperty(v, "noiseSource");
    if (ev::isString(nsVal)) {
        std::string ns = ev::toUtf8(nsVal);
        if (ns == "torch") o.noise_source = brodiffusion::pipeline::NoiseSource::Torch;
        else if (ns == "internal") o.noise_source = brodiffusion::pipeline::NoiseSource::Internal;
    }

    Value initNoiseVal = ev::getProperty(v, "initNoise");
    const float* noiseData = nullptr;
    size_t noiseCount = 0;
    if (readFloat32Array(initNoiseVal, noiseData, noiseCount) && noiseCount > 0) {
        o.init_noise.assign(noiseData, noiseData + noiseCount);
    }

    Value controlsVal = ev::getProperty(v, "controls");
    if (ev::isObject(controlsVal)) {
        Value lenVal = ev::getProperty(controlsVal, "length");
        if (!ev::isUndefined(lenVal)) {
            uint32_t count = static_cast<uint32_t>(ev::toDouble(lenVal));
            o.controls.reserve(count);
            for (uint32_t i = 0; i < count; ++i) {
                Value cEntry = ev::getElement(controlsVal, i);
                brodiffusion::pipeline::ControlNetInput ci;
            if (ev::isObject(cEntry)) {
                Value ip = ev::getProperty(cEntry, "imagePath");
                if (ev::isString(ip)) ci.image_path = ev::toUtf8(ip);
                Value sc = ev::getProperty(cEntry, "scale");
                if (!ev::isUndefined(sc)) ci.scale = static_cast<float>(ev::toDouble(sc));
                Value ss = ev::getProperty(cEntry, "startStep");
                if (!ev::isUndefined(ss)) ci.start_step = static_cast<float>(ev::toDouble(ss));
                Value es = ev::getProperty(cEntry, "endStep");
                if (!ev::isUndefined(es)) ci.end_step = static_cast<float>(ev::toDouble(es));
            }
            o.controls.push_back(std::move(ci));
        }
        }
    }

    o.should_cancel = []() {
        return g_diffusionCancelRequested.load(std::memory_order_relaxed);
    };

    return o;
}

Value pipelineGenerate(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.generate: not a loaded Pipeline");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.generate(prompt, opts?): string prompt required");
    }

    std::string prompt = ev::toUtf8(args[0]);
    Value optVal = args.size() > 1 ? args[1] : ev::undefined();
    auto opts = parseGenerateOptions(optVal);

    try {
        g_diffusionCancelRequested.store(false, std::memory_order_relaxed);
        std::vector<float> nchw = w->pipeline->generate(prompt, opts);
        bool includeFp32 = ev::isObject(optVal) && ev::toBool(ev::getProperty(optVal, "includeFp32"));
        return makeImageResult(nchw, opts.height, opts.width, includeFp32);
    } catch (const brodiffusion::pipeline::GenerateCancelled&) {
        ObjectBuilder b;
        b.set("cancelled", true);
        return b.build();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.generate failed: ") + e.what());
    }
}

Value pipelineTextToImage(Value thisVal, std::span<const Value> args) {
    return pipelineGenerate(thisVal, args);
}

Value pipelineImageToImage(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.imageToImage: not a loaded Pipeline");
    if (args.size() < 2 || !ev::isString(args[0]) || !ev::isString(args[1])) {
        return ev::throwTypeError("Pipeline.imageToImage(imagePath, prompt, opts?): string imagePath and prompt required");
    }

    std::string imagePath = ev::toUtf8(args[0]);
    std::string prompt = ev::toUtf8(args[1]);
    Value optVal = args.size() > 2 ? args[2] : ev::createObject();
    auto opts = parseGenerateOptions(optVal);
    opts.init_image_path = imagePath;

    try {
        g_diffusionCancelRequested.store(false, std::memory_order_relaxed);
        std::vector<float> nchw = w->pipeline->generate(prompt, opts);
        return makeImageResult(nchw, opts.height, opts.width);
    } catch (const brodiffusion::pipeline::GenerateCancelled&) {
        ObjectBuilder b;
        b.set("cancelled", true);
        return b.build();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.imageToImage failed: ") + e.what());
    }
}

Value pipelineInpaint(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.inpaint: not a loaded Pipeline");
    if (args.size() < 3 || !ev::isString(args[0]) || !ev::isString(args[1]) || !ev::isString(args[2])) {
        return ev::throwTypeError("Pipeline.inpaint(imagePath, maskPath, prompt, opts?): string imagePath, maskPath, prompt required");
    }

    std::string imagePath = ev::toUtf8(args[0]);
    std::string maskPath = ev::toUtf8(args[1]);
    std::string prompt = ev::toUtf8(args[2]);
    Value optVal = args.size() > 3 ? args[3] : ev::createObject();
    auto opts = parseGenerateOptions(optVal);
    opts.init_image_path = imagePath;
    opts.mask_image_path = maskPath;

    try {
        g_diffusionCancelRequested.store(false, std::memory_order_relaxed);
        std::vector<float> nchw = w->pipeline->generate(prompt, opts);
        return makeImageResult(nchw, opts.height, opts.width);
    } catch (const brodiffusion::pipeline::GenerateCancelled&) {
        ObjectBuilder b;
        b.set("cancelled", true);
        return b.build();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.inpaint failed: ") + e.what());
    }
}

Value pipelineLoadWeights(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.loadWeights: not a loaded Pipeline");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.loadWeights(path, ...): path string required");
    }

    std::string p0 = ev::toUtf8(args[0]);
    try {
        if (args.size() >= 3 && ev::isString(args[1]) && ev::isString(args[2])) {
            auto tf = brotensor::safetensors::File::open(p0);
            auto uf = brotensor::safetensors::File::open(ev::toUtf8(args[1]));
            auto vf = brotensor::safetensors::File::open(ev::toUtf8(args[2]));
            w->pipeline->load_weights(tf, uf, vf);
        } else if (args.size() >= 2 && ev::isObject(args[1])) {
            std::string tp = strAt(args, 1);
            Value upVal = ev::getProperty(args[1], "unetPrefix");
            Value tpVal = ev::getProperty(args[1], "textPrefix");
            Value vpVal = ev::getProperty(args[1], "vaePrefix");
            std::string up = ev::isString(upVal) ? ev::toUtf8(upVal) : "model.diffusion_model.";
            std::string tpS = ev::isString(tpVal) ? ev::toUtf8(tpVal) : "cond_stage_model.transformer.text_model.";
            std::string vp = ev::isString(vpVal) ? ev::toUtf8(vpVal) : "first_stage_model.decoder.";
            auto f = brotensor::safetensors::File::open(p0);
            w->pipeline->load_weights(f, tpS, up, vp);
        } else {
            auto f = brotensor::safetensors::File::open(p0);
            w->pipeline->load_weights(f);
        }
        w->weights_loaded = true;
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.loadWeights failed: ") + e.what());
    }
}

Value pipelineApplyLora(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.applyLora: not a loaded Pipeline");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.applyLora(path, scale?): path string required");
    }

    std::string path = ev::toUtf8(args[0]);
    float scale = args.size() > 1 ? static_cast<float>(numAt(args, 1)) : 1.0f;
    try {
        auto f = brotensor::safetensors::File::open(path);
        int group = w->pipeline->apply_lora(f, scale);
        if (group >= 0) return ev::fromDouble(group);
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.applyLora failed: ") + e.what());
    }
}

Value pipelineSetLoraScale(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.setLoraScale: not a loaded Pipeline");
    if (args.size() < 2) return ev::throwTypeError("Pipeline.setLoraScale(index, scale): required");
    int index = i32At(args, 0);
    float scale = static_cast<float>(numAt(args, 1));
    try {
        w->pipeline->set_lora_scale(index, scale);
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.setLoraScale failed: ") + e.what());
    }
}

Value pipelineClearLoras(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.clearLoras: not a loaded Pipeline");
    try {
        w->pipeline->clear_loras();
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.clearLoras failed: ") + e.what());
    }
}

Value pipelineNumLoras(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.numLoras: not a loaded Pipeline");
    try {
        return ev::fromDouble(w->pipeline->num_loras());
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.numLoras failed: ") + e.what());
    }
}

Value pipelineAddControlNet(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.addControlNet: not a loaded Pipeline");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.addControlNet(path, cfg?): path string required");
    }

    std::string path = ev::toUtf8(args[0]);
    try {
        auto f = brotensor::safetensors::File::open(path);
        int idx = 0;
        if (args.size() > 1 && ev::isObject(args[1])) {
            brodiffusion::controlnet::ControlNetConfig cfg;
            Value inCh = ev::getProperty(args[1], "inChannels");
            if (!ev::isUndefined(inCh)) cfg.in_channels = static_cast<int>(ev::toDouble(inCh));
            idx = w->pipeline->add_controlnet(f, cfg);
        } else {
            idx = w->pipeline->add_controlnet(f);
        }
        return ev::fromDouble(idx);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.addControlNet failed: ") + e.what());
    }
}

Value pipelinePrime(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.prime: not a loaded Pipeline");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("Pipeline.prime(prompt, opts?): string prompt required");
    }

    std::string prompt = ev::toUtf8(args[0]);
    Value optVal = args.size() > 1 ? args[1] : ev::undefined();
    auto opts = parseGenerateOptions(optVal);

    try {
        auto stateWrapper = std::make_unique<PipelineStateWrapper>();
        stateWrapper->opts = opts;
        stateWrapper->state = w->pipeline->prime(prompt, opts);
        return g_pipelineStateClass.createInstance(std::move(stateWrapper));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.prime failed: ") + e.what());
    }
}

Value pipelineStepOnce(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.stepOnce: not a loaded Pipeline");
    if (args.empty()) return ev::throwTypeError("Pipeline.stepOnce(state): state required");

    auto* sw = unwrapPipelineState(args[0]);
    if (!sw) return ev::throwTypeError("Pipeline.stepOnce: expected PipelineState argument");

    try {
        w->pipeline->step_once(sw->state, sw->opts);
        bool hasMore = sw->state.step_index < sw->state.n_steps;
        return ev::fromBool(hasMore);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.stepOnce failed: ") + e.what());
    }
}

Value pipelineDecode(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapPipeline(thisVal);
    if (!w || !w->pipeline) return ev::throwTypeError("Pipeline.decode: not a loaded Pipeline");
    if (args.empty()) return ev::throwTypeError("Pipeline.decode(state): state required");

    auto* sw = unwrapPipelineState(args[0]);
    if (!sw) return ev::throwTypeError("Pipeline.decode: expected PipelineState argument");

    try {
        std::vector<float> nchw = w->pipeline->decode(sw->state);
        int scale = w->pipeline->vae_scale_factor();
        int H = sw->state.H_lat * scale;
        int W = sw->state.W_lat * scale;
        return makeImageResult(nchw, H, W);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.decode failed: ") + e.what());
    }
}

Value pipelineDispose(Value thisVal, std::span<const Value>) {
    auto* w = unwrapPipeline(thisVal);
    if (!w) return ev::throwTypeError("Pipeline.dispose: not a Pipeline");
    w->pipeline.reset();
    w->weights_loaded = false;
    return ev::undefined();
}

Value stateStepOnce(Value thisVal, std::span<const Value>) {
    auto* sw = unwrapPipelineState(thisVal);
    if (!sw) return ev::throwTypeError("PipelineState.stepOnce: not a PipelineState");
    sw->state.step_index++;
    return ev::fromBool(sw->state.step_index < sw->state.n_steps);
}

Value stateDecode(Value thisVal, std::span<const Value>) {
    auto* sw = unwrapPipelineState(thisVal);
    if (!sw) return ev::throwTypeError("PipelineState.decode: not a PipelineState");
    int H = sw->state.H_lat > 0 ? sw->state.H_lat * 8 : 512;
    int W = sw->state.W_lat > 0 ? sw->state.W_lat * 8 : 512;
    std::vector<float> dummy(static_cast<size_t>(3) * H * W, 0.0f);
    return makeImageResult(dummy, H, W);
}

void decoratePipelineProto(ObjectBuilder& proto) {
    proto.def("generate", 2, pipelineGenerate);
    proto.def("textToImage", 2, pipelineTextToImage);
    proto.def("imageToImage", 3, pipelineImageToImage);
    proto.def("inpaint", 4, pipelineInpaint);
    proto.def("loadWeights", 3, pipelineLoadWeights);
    proto.def("applyLora", 2, pipelineApplyLora);
    proto.def("setLoraScale", 2, pipelineSetLoraScale);
    proto.def("clearLoras", 0, pipelineClearLoras);
    proto.def("numLoras", 0, pipelineNumLoras);
    proto.def("addControlNet", 2, pipelineAddControlNet);
    proto.def("prime", 2, pipelinePrime);
    proto.def("stepOnce", 1, pipelineStepOnce);
    proto.def("decode", 1, pipelineDecode);
    proto.def("dispose", 0, pipelineDispose);
}

void decoratePipelineStateProto(ObjectBuilder& proto) {
    proto.def("stepOnce", 0, stateStepOnce);
    proto.def("decode", 0, stateDecode);
    proto.accessor("stepIndex", [](Value thisVal, std::span<const Value>) -> Value {
        auto* sw = unwrapPipelineState(thisVal);
        return sw ? ev::fromDouble(sw->state.step_index) : ev::fromDouble(0);
    });
    proto.accessor("totalSteps", [](Value thisVal, std::span<const Value>) -> Value {
        auto* sw = unwrapPipelineState(thisVal);
        return sw ? ev::fromDouble(sw->state.n_steps) : ev::fromDouble(0);
    });
}

} // namespace

void ensureDiffusionClassesInstalled() {
    static bool installed = false;
    if (installed) return;
    installed = true;

    g_pipelineClass.install("Pipeline", 0, nullptr, decoratePipelineProto);
    g_pipelineStateClass.install("PipelineState", 0, nullptr, decoratePipelineStateProto);
}

Value makeDiffusionNamespace() {
    ensureDiffusionClassesInstalled();
    ObjectBuilder diff;

    diff.accessor("version", [](Value, std::span<const Value>) -> Value {
        return ev::fromUtf8(brodiffusion::version_string());
    });

    diff.def("init", 0, [](Value, std::span<const Value>) -> Value {
        try {
            brotensor::init();
        } catch (const std::exception& e) {
            return ev::throwError(std::string("bro.diffusion.init failed: ") + e.what());
        }
        return ev::undefined();
    });

    diff.def("loadModel", 2, [](Value, std::span<const Value> args) -> Value {
        if (args.empty()) {
            return ev::throwTypeError("bro.diffusion.loadModel: path is required");
        }
        if (!ev::isString(args[0])) {
            return ev::throwTypeError("bro.diffusion.loadModel: path must be a string");
        }

        std::string dir = ev::toUtf8(args[0]);
        brodiffusion::pipeline::Pipeline::ModelDirOptions dirOpts;
        if (args.size() > 1 && ev::isObject(args[1])) {
            Value qv = ev::getProperty(args[1], "quantizeWeights");
            if (!ev::isUndefined(qv)) dirOpts.quantize = ev::toBool(qv);
            Value tev = ev::getProperty(args[1], "textEncoderPath");
            if (ev::isString(tev)) dirOpts.text_encoder_path = ev::toUtf8(tev);
        }
        dirOpts.should_cancel = []() {
            return g_diffusionCancelRequested.load(std::memory_order_relaxed);
        };

        if (!std::filesystem::exists(dir)) {
            return ev::throwError(std::string("loadModel failed: model dir not found: ") + dir);
        }

        try {
            brotensor::init();
            auto w = std::make_unique<PipelineWrapper>();
            w->pipeline = std::make_unique<brodiffusion::pipeline::Pipeline>(
                brodiffusion::pipeline::Pipeline::from_model_dir(dir, dirOpts));
            w->weights_loaded = true;
            return g_pipelineClass.createInstance(std::move(w));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("loadModel failed: ") + e.what());
        }
    });

    diff.def("createPipeline", 1, [](Value, std::span<const Value> args) -> Value {
        if (args.empty() || !ev::isObject(args[0])) {
            return ev::throwTypeError("bro.diffusion.createPipeline: config object is required");
        }

        Value vpVal = ev::getProperty(args[0], "vocabPath");
        if (!ev::isString(vpVal)) {
            return ev::throwTypeError("createPipeline: opts.vocabPath (string) required");
        }
        Value mpVal = ev::getProperty(args[0], "mergesPath");
        if (!ev::isString(mpVal)) {
            return ev::throwTypeError("createPipeline: opts.mergesPath (string) required");
        }

        std::string vocabPath = ev::toUtf8(vpVal);
        std::string mergesPath = ev::toUtf8(mpVal);
        std::string schedulerName = "ddim";
        Value sVal = ev::getProperty(args[0], "scheduler");
        if (ev::isString(sVal)) schedulerName = ev::toUtf8(sVal);

        bool lcm = (schedulerName == "lcm");
        bool flowmatch = (schedulerName == "flowmatch");
        bool scm = (schedulerName == "scm");
        bool dpm = (schedulerName == "dpm" || schedulerName == "dpmsolver");
        bool lcmDistilled = ev::toBool(ev::getProperty(args[0], "lcmDistilled"));
        bool quantize = ev::toBool(ev::getProperty(args[0], "quantizeWeights"));

        try {
            brotensor::init();
            auto tok = brolm::clip::Tokenizer::load(vocabPath, mergesPath);
            brodiffusion::pipeline::PipelineConfig cfg;
            if (lcm) {
                cfg.scheduler = brodiffusion::scheduler::LCMConfig{};
                if (lcmDistilled) cfg.unet.time_cond_proj_dim = 256;
            } else if (flowmatch) {
                cfg.scheduler = brodiffusion::scheduler::FlowMatchConfig{};
            } else if (scm) {
                cfg.scheduler = brodiffusion::scheduler::SCMConfig{};
            } else if (dpm) {
                cfg.scheduler = brodiffusion::scheduler::DPMSolverConfig{};
            }
            cfg.unet.quantize_weights = quantize;

            auto w = std::make_unique<PipelineWrapper>();
            w->pipeline = std::make_unique<brodiffusion::pipeline::Pipeline>(cfg, std::move(tok));
            return g_pipelineClass.createInstance(std::move(w));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("createPipeline failed: ") + e.what());
        }
    });

    diff.def("expandNoise", 2, [](Value, std::span<const Value> args) -> Value {
        if (args.empty()) {
            return ev::throwTypeError("bro.diffusion.expandNoise: src Float32Array is required");
        }

        const float* src = nullptr;
        size_t count = 0;
        if (!readFloat32Array(args[0], src, count) || count == 0) {
            return ev::throwTypeError("expandNoise: src must be a non-empty Float32Array");
        }

        int c = 4, h = 64, w = 64, k = 2;
        uint64_t seed = 0;
        if (args.size() > 1 && ev::isObject(args[1])) {
            Value cv = ev::getProperty(args[1], "channels");
            if (!ev::isUndefined(cv)) c = static_cast<int>(ev::toDouble(cv));
            Value hv = ev::getProperty(args[1], "height");
            if (!ev::isUndefined(hv)) h = static_cast<int>(ev::toDouble(hv));
            Value wv = ev::getProperty(args[1], "width");
            if (!ev::isUndefined(wv)) w = static_cast<int>(ev::toDouble(wv));
            Value fv = ev::getProperty(args[1], "factor");
            if (!ev::isUndefined(fv)) k = static_cast<int>(ev::toDouble(fv));
            Value sv = ev::getProperty(args[1], "seed");
            if (!ev::isUndefined(sv)) seed = static_cast<uint64_t>(ev::toDouble(sv));
        }

        if (static_cast<size_t>(c * h * w) > count) {
            return ev::throwTypeError("expandNoise: src buffer smaller than c*h*w");
        }

        try {
            std::vector<float> expanded = brodiffusion::pipeline::expand_init_noise(src, c, h, w, k, seed);
            return makeFloat32Array(expanded.data(), expanded.size());
        } catch (const std::exception& e) {
            return ev::throwError(std::string("expandNoise failed: ") + e.what());
        }
    });

    diff.def("cancel", 0, [](Value, std::span<const Value>) -> Value {
        g_diffusionCancelRequested.store(true, std::memory_order_relaxed);
        return ev::undefined();
    });

    diff.set("Pipeline", g_pipelineClass.constructor());
    diff.set("PipelineState", g_pipelineStateClass.constructor());
    ensureVaeClassesInstalled();
    diff.set("VAE", g_vaeClass.constructor());

    return diff.build();
}

} // namespace brodiffusion::api
