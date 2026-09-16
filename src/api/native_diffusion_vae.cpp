#include "host_diffusion_internal.h"
#include <broimage/decode.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace brodiffusion::api {

HostClass g_vaeClass;

VaeWrapper* unwrapVae(Value v) {
    void* ptr = g_vaeClass.unwrap(v);
    if (!ptr) return nullptr;
    auto* w = static_cast<VaeWrapper*>(ptr);
    return (w && w->tag == kHostVaeTag) ? w : nullptr;
}

namespace {

Value vaeLoadWeights(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapVae(thisVal);
    if (!w) return ev::throwTypeError("VAE.loadWeights: not a VAE instance");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("loadWeights(path, prefix?): path string required");
    }

    std::string path = ev::toUtf8(args[0]);
    std::string prefix = args.size() > 1 && ev::isString(args[1]) ? ev::toUtf8(args[1]) : "decoder.";

    if (!std::filesystem::exists(path)) {
        return ev::throwError("VAE.loadWeights failed: file not found: " + path);
    }

    try {
        brotensor::init();
        auto f = brotensor::safetensors::File::open(path);
        if (!w->decoder) {
            w->decoder = std::make_unique<brodiffusion::vae::Decoder>(brodiffusion::vae::DecoderConfig{});
        }
        w->decoder->load_weights(f, prefix);

        // Try loading encoder if encoder prefix exists
        std::string encPrefix = "encoder.";
        if (prefix == "first_stage_model.decoder.") encPrefix = "first_stage_model.encoder.";
        try {
            if (!w->encoder) {
                w->encoder = std::make_unique<brodiffusion::vae::Encoder>(brodiffusion::vae::EncoderConfig{});
            }
            w->encoder->load_weights(f, encPrefix);
        } catch (...) {
            // Optional if checkpoint only has decoder
        }

        w->weights_loaded = true;
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("VAE.loadWeights failed: ") + e.what());
    }
}

Value vaeDecode(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapVae(thisVal);
    if (!w || !w->decoder) return ev::throwTypeError("VAE.decode: decoder not initialized");
    if (args.empty()) return ev::throwTypeError("VAE.decode(latent, opts?): latent required");

    const float* latData = nullptr;
    size_t latCount = 0;
    int H_lat = 64, W_lat = 64;

    if (ev::isObject(args[0]) && !readFloat32Array(args[0], latData, latCount)) {
        Value dataVal = ev::getProperty(args[0], "data");
        readFloat32Array(dataVal, latData, latCount);
        Value wv = ev::getProperty(args[0], "width");
        Value hv = ev::getProperty(args[0], "height");
        if (!ev::isUndefined(wv)) W_lat = static_cast<int>(ev::toDouble(wv));
        if (!ev::isUndefined(hv)) H_lat = static_cast<int>(ev::toDouble(hv));
    }

    if (args.size() > 1 && ev::isObject(args[1])) {
        Value wv = ev::getProperty(args[1], "width");
        Value hv = ev::getProperty(args[1], "height");
        if (!ev::isUndefined(wv)) W_lat = static_cast<int>(ev::toDouble(wv));
        if (!ev::isUndefined(hv)) H_lat = static_cast<int>(ev::toDouble(hv));
    }

    if (!latData || latCount == 0) {
        return ev::throwTypeError("VAE.decode: latent must be a Float32Array or { data, width, height }");
    }

    const size_t inCh = static_cast<size_t>(w->decoder->config().in_channels);
    if (latCount < inCh * H_lat * W_lat) {
        return ev::throwTypeError("VAE.decode: latent buffer smaller than in_channels * H_lat * W_lat");
    }

    try {
        brotensor::init();
        brotensor::Tensor latent;
        if (brotensor::compute_dtype() == brotensor::Dtype::FP16) {
            std::vector<uint16_t> bits(latCount);
            for (size_t i = 0; i < latCount; ++i) bits[i] = brotensor::fp32_to_fp16_bits(latData[i]);
            latent = brotensor::Tensor::from_host_fp16(bits.data(), 1, static_cast<int>(inCh * H_lat * W_lat));
        } else {
            latent = brotensor::Tensor::from_host(latData, 1, static_cast<int>(inCh * H_lat * W_lat));
        }

        brotensor::Tensor out;
        w->decoder->decode(latent, H_lat, W_lat, out);
        brotensor::sync_all();

        std::vector<float> nchw;
        if (out.dtype == brotensor::Dtype::FP16) {
            std::vector<uint16_t> bits = out.to_host_vector_fp16();
            nchw.resize(bits.size());
            for (size_t i = 0; i < bits.size(); ++i) nchw[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        } else {
            nchw = out.to_host_vector();
        }

        int outH = H_lat * 8;
        int outW = W_lat * 8;
        return makeImageResult(nchw, outH, outW);
    } catch (const std::exception& e) {
        return ev::throwError(std::string("VAE.decode failed: ") + e.what());
    }
}

Value vaeEncode(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapVae(thisVal);
    if (!w || !w->encoder) return ev::throwTypeError("VAE.encode: encoder not initialized");
    if (args.empty()) return ev::throwTypeError("VAE.encode(image, opts?): image required");

    std::vector<uint8_t> rgba;
    int iw = 0, ih = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, iw, ih, err)) {
        return ev::throwTypeError(std::string("VAE.encode: ") + err);
    }

    if (iw % 8 != 0 || ih % 8 != 0) {
        return ev::throwTypeError("VAE.encode: image dimensions must be multiples of 8");
    }

    try {
        brotensor::init();
        const int plane = iw * ih;
        std::vector<float> nchw(static_cast<size_t>(3) * plane);
        for (int i = 0; i < plane; ++i) {
            for (int c = 0; c < 3; ++c) {
                float v = rgba[static_cast<size_t>(4) * i + c] / 255.0f * 2.0f - 1.0f;
                nchw[static_cast<size_t>(c) * plane + i] = v;
            }
        }

        brotensor::Tensor image;
        if (brotensor::compute_dtype() == brotensor::Dtype::FP16) {
            std::vector<uint16_t> bits(nchw.size());
            for (size_t i = 0; i < bits.size(); ++i) bits[i] = brotensor::fp32_to_fp16_bits(nchw[i]);
            image = brotensor::Tensor::from_host_fp16(bits.data(), 1, 3 * plane);
        } else {
            image = brotensor::Tensor::from_host(nchw.data(), 1, 3 * plane);
        }

        brotensor::Tensor outLat;
        w->encoder->encode(image, ih, iw, nullptr, outLat);
        brotensor::sync_all();

        std::vector<float> hostLat;
        if (outLat.dtype == brotensor::Dtype::FP16) {
            std::vector<uint16_t> bits = outLat.to_host_vector_fp16();
            hostLat.resize(bits.size());
            for (size_t i = 0; i < bits.size(); ++i) hostLat[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        } else {
            hostLat = outLat.to_host_vector();
        }

        ObjectBuilder res;
        res.set("width", static_cast<double>(iw / 8));
        res.set("height", static_cast<double>(ih / 8));
        res.set("channels", static_cast<double>(4));
        {
            ev::Persistent d(makeFloat32Array(hostLat.data(), hostLat.size()));
            res.set("data", d.get());
        }
        return res.build();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("VAE.encode failed: ") + e.what());
    }
}

void decorateVaeProto(ObjectBuilder& proto) {
    proto.def("loadWeights", 2, vaeLoadWeights);
    proto.def("decode", 2, vaeDecode);
    proto.def("encode", 2, vaeEncode);
}

} // namespace

void ensureVaeClassesInstalled() {
    static bool installed = false;
    if (installed) return;
    installed = true;

    g_vaeClass.install("VAE", 0, [](Value, std::span<const Value>) -> Value {
        auto w = std::make_unique<VaeWrapper>();
        w->decoder = std::make_unique<brodiffusion::vae::Decoder>(brodiffusion::vae::DecoderConfig{});
        w->encoder = std::make_unique<brodiffusion::vae::Encoder>(brodiffusion::vae::EncoderConfig{});
        return g_vaeClass.createInstance(std::move(w));
    }, decorateVaeProto);
}

} // namespace brodiffusion::api
