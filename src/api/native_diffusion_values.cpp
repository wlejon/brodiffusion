// Value translation shared by the diffusion binding's translation units:
// typed-array in/out, the canvas-ready image result, tensor <-> JS object,
// the GenerateOptions option bag, and the PipelineState -> Pipeline link.
//
// Ported from the QuickJS-era src/js/diffusion_bindings.cpp helpers so the
// shapes stay exactly what the old surface produced.

#include "host_diffusion_internal.h"
#include "api.h"

#include <brotensor/ops/elementwise.h>

namespace brodiffusion::api {

std::atomic<bool> g_diffusionCancelRequested{false};

namespace {

// Set once by the host (bro) at install time; empty in a standalone build, in
// which case every path is used exactly as the caller wrote it.
std::function<std::string(const std::string&)>& pathResolver() {
    static std::function<std::string(const std::string&)> r;
    return r;
}

} // namespace

void setPathResolver(std::function<std::string(const std::string&)> resolver) {
    pathResolver() = std::move(resolver);
}

std::string resolveDiffusionPath(const std::string& path) {
    auto& r = pathResolver();
    return r ? r(path) : path;
}

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

std::vector<float> downloadTensorFloats(const brotensor::Tensor& t) {
    if (t.dtype == brotensor::Dtype::FP16) {
        std::vector<uint16_t> bits = t.to_host_vector_fp16();
        std::vector<float> out(bits.size());
        for (size_t i = 0; i < bits.size(); ++i) out[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        return out;
    }
    if (t.dtype == brotensor::Dtype::BF16) {
        std::vector<uint16_t> bits = t.to_host_vector_bf16();
        std::vector<float> out(bits.size());
        for (size_t i = 0; i < bits.size(); ++i) out[i] = brotensor::bf16_bits_to_fp32(bits[i]);
        return out;
    }
    return t.to_host_vector();
}

bool tensorFromJs(Value v, brotensor::Tensor& out) {
    if (!ev::isObject(v)) return false;
    int rows = 0, cols = 0;
    propInt(v, "rows", rows);
    propInt(v, "cols", cols);
    Value dv = ev::getProperty(v, "data");
    const float* fp = nullptr;
    size_t cnt = 0;
    if (!readFloat32Array(dv, fp, cnt)) return false;
    if (!fp || rows <= 0 || cols <= 0 ||
        cnt != static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
        return false;
    }
    out = brotensor::Tensor::from_host(fp, rows, cols);
    return true;
}

Value tensorToJs(const brotensor::Tensor& t) {
    std::vector<float> host = downloadTensorFloats(t);
    ObjectBuilder o;
    o.set("rows", static_cast<double>(t.rows));
    o.set("cols", static_cast<double>(t.cols));
    ev::Persistent d(makeFloat32Array(host.data(), host.size()));
    o.set("data", d.get());
    return o.build();
}

Value textConditioningToJs(const brodiffusion::krea2::TextConditioning& tc) {
    ObjectBuilder o;
    {
        ev::Persistent e(tensorToJs(tc.prompt_embeds));
        o.set("embeds", e.get());
    }
    {
        ev::Persistent m(tensorToJs(tc.prompt_embeds_mask));
        o.set("mask", m.get());
    }
    return o.build();
}

brodiffusion::pipeline::GenerateOptions parseGenerateOptions(Value v) {
    brodiffusion::pipeline::GenerateOptions o;
    if (!ev::isObject(v)) {
        o.should_cancel = [] { return g_diffusionCancelRequested.load(std::memory_order_relaxed); };
        return o;
    }
    ev::Persistent root(v);

    propInt(root.get(), "width", o.width);
    propInt(root.get(), "height", o.height);
    propInt(root.get(), "steps", o.num_inference_steps);
    propNum(root.get(), "guidanceScale", o.guidance_scale);
    propStr(root.get(), "negativePrompt", o.negative_prompt);
    propSeed(root.get(), "seed", o.seed);

    // img2img / inpaint
    propStr(root.get(), "initImagePath", o.init_image_path);
    propNum(root.get(), "strength", o.strength);
    o.vae_encode_sample = propBool(root.get(), "vaeEncodeSample", o.vae_encode_sample);
    propStr(root.get(), "maskImagePath", o.mask_image_path);

    // noise source: 'internal' (default) or 'torch'
    {
        std::string ns;
        if (propStr(root.get(), "noiseSource", ns)) {
            if (ns == "torch") o.noise_source = brodiffusion::pipeline::NoiseSource::Torch;
            else if (ns == "internal") o.noise_source = brodiffusion::pipeline::NoiseSource::Internal;
        }
    }

    // initNoise: Float32Array of raw N(0,1) values, NCHW flat.
    {
        Value nv = ev::getProperty(root.get(), "initNoise");
        const float* noiseData = nullptr;
        size_t noiseCount = 0;
        if (readFloat32Array(nv, noiseData, noiseCount) && noiseCount > 0) {
            o.init_noise.assign(noiseData, noiseData + noiseCount);
        }
    }

    // controls: [{ imagePath, scale?, startStep?, endStep? }, ...], one entry
    // per registered ControlNet. brodiffusion validates the count at prime.
    {
        ev::Persistent controls(ev::getProperty(root.get(), "controls"));
        const uint32_t count = arrayLength(controls.get());
        o.controls.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            Value cEntry = ev::getElement(controls.get(), i);
            brodiffusion::pipeline::ControlNetInput ci;
            if (ev::isObject(cEntry)) {
                ev::Persistent entry(cEntry);
                propStr(entry.get(), "imagePath", ci.image_path);
                propNum(entry.get(), "scale", ci.scale);
                propNum(entry.get(), "startStep", ci.start_step);
                propNum(entry.get(), "endStep", ci.end_step);
            }
            o.controls.push_back(std::move(ci));
        }
    }

    o.should_cancel = [] { return g_diffusionCancelRequested.load(std::memory_order_relaxed); };
    return o;
}

Value attachPipelineToState(Value stateVal, Value pipelineVal) {
    ev::Persistent st(stateVal);
    ev::Persistent pipe(pipelineVal);
    st.set(ev::setProperty(st.get(), "__pipeline", pipe.get()));
    return st.get();
}

brodiffusion::pipeline::Pipeline* pipelineOfState(Value stateVal) {
    if (!ev::isObject(stateVal)) return nullptr;
    Value p = ev::getProperty(stateVal, "__pipeline");
    PipelineWrapper* pw = unwrapPipeline(p);
    return pw ? pw->pipeline.get() : nullptr;
}

} // namespace brodiffusion::api
