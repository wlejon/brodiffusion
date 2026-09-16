#include "host_diffusion_internal.h"
#include <broimage/decode.h>
#include <brotensor/runtime.h>
#include <brotensor/safetensors.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>

namespace brodiffusion::api {

std::atomic<bool> g_triposplatCancelRequested{false};
HostClass g_tripoSplatClass;

TripoSplatWrapper* unwrapTripoSplat(Value v) {
    void* ptr = g_tripoSplatClass.unwrap(v);
    if (!ptr) return nullptr;
    auto* w = static_cast<TripoSplatWrapper*>(ptr);
    return (w && w->tag == kHostTripoSplatTag) ? w : nullptr;
}

namespace {

constexpr int kCanvas = 1024;

inline float logitf(float p) {
    p = std::clamp(p, 1e-6f, 1.0f - 1e-6f);
    return std::log(p / (1.0f - p));
}

void preprocessImage(const std::vector<uint8_t>& rgba, int w, int h, std::vector<float>& rgb01) {
    rgb01.assign(static_cast<size_t>(kCanvas) * kCanvas * 3, 0.0f);

    const int er = std::max(1, static_cast<int>(std::lround(std::min(w, h) / 1024.0)));
    std::vector<uint8_t> alpha(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t m = 255;
            for (int dy = -er; dy <= er; ++dy) {
                const int yy = std::min(std::max(y + dy, 0), h - 1);
                for (int dx = -er; dx <= er; ++dx) {
                    const int xx = std::min(std::max(x + dx, 0), w - 1);
                    m = std::min(m, rgba[(static_cast<size_t>(yy) * w + xx) * 4 + 3]);
                }
            }
            alpha[static_cast<size_t>(y) * w + x] = m;
        }
    }

    int x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (alpha[static_cast<size_t>(y) * w + x] > 0) {
                x0 = std::min(x0, x); x1 = std::max(x1, x);
                y0 = std::min(y0, y); y1 = std::max(y1, y);
            }
        }
    }

    float cx = 0.0f, cy = 0.0f, half = 0.0f;
    if (x1 < 0) {
        cx = w * 0.5f; cy = h * 0.5f;
        half = std::min(w, h) * 0.5f;
        std::fill(alpha.begin(), alpha.end(), static_cast<uint8_t>(255));
    } else {
        cx = (x0 + x1) * 0.5f;
        cy = (y0 + y1) * 0.5f;
        half = std::max(x1 - x0, y1 - y0) * 0.5f * 1.2f;
        if (half < 1.0f) half = 1.0f;
    }

    auto px = [&](int x, int y, int c) -> float {
        if (x < 0 || y < 0 || x >= w || y >= h) return 0.0f;
        if (c == 3) return static_cast<float>(alpha[static_cast<size_t>(y) * w + x]);
        return static_cast<float>(rgba[(static_cast<size_t>(y) * w + x) * 4 + c]);
    };

    const float step = 2.0f * half / kCanvas;
    for (int ty = 0; ty < kCanvas; ++ty) {
        for (int tx = 0; tx < kCanvas; ++tx) {
            const float fx = cx - half + (tx + 0.5f) * step - 0.5f;
            const float fy = cy - half + (ty + 0.5f) * step - 0.5f;
            const int ix = static_cast<int>(std::floor(fx));
            const int iy = static_cast<int>(std::floor(fy));
            const float ax = fx - ix;
            const float ay = fy - iy;
            float v[4];
            for (int c = 0; c < 4; ++c) {
                const float c00 = px(ix, iy, c), c10 = px(ix + 1, iy, c);
                const float c01 = px(ix, iy + 1, c), c11 = px(ix + 1, iy + 1, c);
                v[c] = (c00 * (1 - ax) + c10 * ax) * (1 - ay) +
                       (c01 * (1 - ax) + c11 * ax) * ay;
            }
            const float a = v[3] / 255.0f;
            float* o = &rgb01[(static_cast<size_t>(ty) * kCanvas + tx) * 3];
            o[0] = v[0] / 255.0f * a;
            o[1] = v[1] / 255.0f * a;
            o[2] = v[2] / 255.0f * a;
        }
    }
}

brotensor::Tensor uploadCompute(const float* src, int rows, int cols) {
    if (brotensor::compute_dtype() == brotensor::Dtype::FP16) {
        std::vector<uint16_t> bits(static_cast<size_t>(rows) * cols);
        for (size_t i = 0; i < bits.size(); ++i) bits[i] = brotensor::fp32_to_fp16_bits(src[i]);
        return brotensor::Tensor::from_host_fp16(bits.data(), rows, cols);
    }
    return brotensor::Tensor::from_host(src, rows, cols);
}

std::vector<float> downloadF32(const brotensor::Tensor& t) {
    if (t.dtype == brotensor::Dtype::FP16) {
        std::vector<uint16_t> bits = t.to_host_vector_fp16();
        std::vector<float> out(bits.size());
        for (size_t i = 0; i < bits.size(); ++i) out[i] = brotensor::fp16_bits_to_fp32(bits[i]);
        return out;
    }
    return t.to_host_vector();
}

} // namespace

bool readImageInput(Value val, std::vector<uint8_t>& rgba, int& w, int& h, std::string& err) {
    if (ev::isString(val)) {
        std::string path = ev::toUtf8(val);
        broimage::Image img;
        if (!broimage::decode_file(path, img, &err)) {
            if (err.empty()) err = "Failed to decode image file: " + path;
            return false;
        }
        w = img.width;
        h = img.height;
        rgba = std::move(img.pixels);
        return true;
    }

    if (ev::isObject(val)) {
        Value wVal = ev::getProperty(val, "width");
        Value hVal = ev::getProperty(val, "height");
        if (!ev::isUndefined(wVal) && !ev::isUndefined(hVal)) {
            w = static_cast<int>(ev::toDouble(wVal));
            h = static_cast<int>(ev::toDouble(hVal));
        }
        Value dataVal = ev::getProperty(val, "data");
        const uint8_t* u8Data = nullptr;
        size_t u8Count = 0;
        if (readUint8Array(dataVal, u8Data, u8Count) && u8Data) {
            if (w <= 0 || h <= 0) {
                err = "Image { width, height } must be positive";
                return false;
            }
            const size_t need = static_cast<size_t>(w) * h * 4;
            if (u8Count < need) {
                err = "image.data too small for width*height*4 RGBA";
                return false;
            }
            rgba.assign(u8Data, u8Data + need);
            return true;
        }
    }

    err = "image must be a path string or { data, width, height }";
    return false;
}

bool saveSplatPLY(const brodiffusion::triposplat::GaussianSplats& splats, const std::string& path) {
    if (splats.empty()) return false;
    const size_t n = splats.count();
    const int degree = std::min(3, std::max(0, splats.shDegree));
    const int coeffs = (degree + 1) * (degree + 1);
    const int restPerChannel = coeffs - 1;
    const int restTotal = restPerChannel * 3;
    const int stride = splats.shStride();

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    std::string header = "ply\nformat binary_little_endian 1.0\n";
    header += "element vertex " + std::to_string(n) + "\n";
    header += "property float x\nproperty float y\nproperty float z\n";
    header += "property float nx\nproperty float ny\nproperty float nz\n";
    header += "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n";
    for (int i = 0; i < restTotal; ++i) {
        header += "property float f_rest_" + std::to_string(i) + "\n";
    }
    header += "property float opacity\n";
    header += "property float scale_0\nproperty float scale_1\nproperty float scale_2\n";
    header += "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n";
    header += "end_header\n";
    std::fwrite(header.c_str(), 1, header.size(), f);

    std::vector<float> rec;
    rec.reserve(static_cast<size_t>(6 + 3 + restTotal + 1 + 3 + 4));
    for (size_t v = 0; v < n; ++v) {
        rec.clear();
        rec.push_back(splats.positions[v * 3 + 0]);
        rec.push_back(splats.positions[v * 3 + 1]);
        rec.push_back(splats.positions[v * 3 + 2]);
        rec.push_back(0.0f); rec.push_back(0.0f); rec.push_back(0.0f);

        const float* sh = splats.sh.data() + v * static_cast<size_t>(stride);
        rec.push_back(sh[0]); rec.push_back(sh[1]); rec.push_back(sh[2]);
        for (int c = 0; c < 3; ++c) {
            for (int k = 0; k < restPerChannel; ++k) {
                rec.push_back(sh[(k + 1) * 3 + c]);
            }
        }

        rec.push_back(logitf(splats.opacities[v]));
        for (int a = 0; a < 3; ++a) {
            float s = splats.scales[v * 3 + a];
            rec.push_back(std::log(std::max(1e-12f, s)));
        }
        rec.push_back(splats.rotations[v * 4 + 3]); // w
        rec.push_back(splats.rotations[v * 4 + 0]); // x
        rec.push_back(splats.rotations[v * 4 + 1]); // y
        rec.push_back(splats.rotations[v * 4 + 2]); // z

        std::fwrite(rec.data(), sizeof(float), rec.size(), f);
    }

    std::fclose(f);
    return true;
}

bool saveSplatBinary(const brodiffusion::triposplat::GaussianSplats& splats, const std::string& path) {
    if (splats.empty()) return false;
    const size_t n = splats.count();
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    constexpr float C0 = 0.28209479177387814f;
    const int stride = splats.shStride();

    #pragma pack(push, 1)
    struct SplatRecord {
        float pos[3];
        float scale[3];
        uint8_t rgba[4];
        uint8_t rot[4];
    };
    #pragma pack(pop)
    static_assert(sizeof(SplatRecord) == 32, "SplatRecord size must be exactly 32 bytes");

    std::vector<SplatRecord> records(n);
    for (size_t i = 0; i < n; ++i) {
        auto& r = records[i];
        r.pos[0] = splats.positions[i * 3 + 0];
        r.pos[1] = splats.positions[i * 3 + 1];
        r.pos[2] = splats.positions[i * 3 + 2];

        r.scale[0] = splats.scales[i * 3 + 0];
        r.scale[1] = splats.scales[i * 3 + 1];
        r.scale[2] = splats.scales[i * 3 + 2];

        const float* sh = splats.sh.data() + i * static_cast<size_t>(stride);
        float cr = std::clamp(sh[0] * C0 + 0.5f, 0.0f, 1.0f);
        float cg = std::clamp(sh[1] * C0 + 0.5f, 0.0f, 1.0f);
        float cb = std::clamp(sh[2] * C0 + 0.5f, 0.0f, 1.0f);
        float ca = std::clamp(splats.opacities[i], 0.0f, 1.0f);
        r.rgba[0] = static_cast<uint8_t>(cr * 255.0f + 0.5f);
        r.rgba[1] = static_cast<uint8_t>(cg * 255.0f + 0.5f);
        r.rgba[2] = static_cast<uint8_t>(cb * 255.0f + 0.5f);
        r.rgba[3] = static_cast<uint8_t>(ca * 255.0f + 0.5f);

        for (int q = 0; q < 4; ++q) {
            float v = std::clamp(splats.rotations[i * 4 + q] * 128.0f + 128.0f, 0.0f, 255.0f);
            r.rot[q] = static_cast<uint8_t>(v);
        }
    }

    std::fwrite(records.data(), sizeof(SplatRecord), records.size(), f);
    std::fclose(f);
    return true;
}

namespace {

Value makeSplatsResult(const brodiffusion::triposplat::GaussianSplats& splats) {
    ObjectBuilder res;
    {
        ev::Persistent pos(makeFloat32Array(splats.positions.data(), splats.positions.size()));
        res.set("positions", pos.get());
    }
    {
        ev::Persistent sc(makeFloat32Array(splats.scales.data(), splats.scales.size()));
        res.set("scales", sc.get());
    }
    {
        ev::Persistent rot(makeFloat32Array(splats.rotations.data(), splats.rotations.size()));
        res.set("rotations", rot.get());
    }
    {
        ev::Persistent op(makeFloat32Array(splats.opacities.data(), splats.opacities.size()));
        res.set("opacities", op.get());
    }
    {
        ev::Persistent sh(makeFloat32Array(splats.sh.data(), splats.sh.size()));
        res.set("sh", sh.get());
    }
    res.set("shDegree", static_cast<double>(splats.shDegree));
    res.set("count", static_cast<double>(splats.count()));
    return res.build();
}

bool parseSplatsFromValue(Value val, brodiffusion::triposplat::GaussianSplats& out) {
    if (!ev::isObject(val)) return false;
    const float* pos = nullptr; size_t posCount = 0;
    const float* sc = nullptr;  size_t scCount = 0;
    const float* rot = nullptr; size_t rotCount = 0;
    const float* op = nullptr;  size_t opCount = 0;
    const float* sh = nullptr;  size_t shCount = 0;

    readFloat32Array(ev::getProperty(val, "positions"), pos, posCount);
    readFloat32Array(ev::getProperty(val, "scales"), sc, scCount);
    readFloat32Array(ev::getProperty(val, "rotations"), rot, rotCount);
    readFloat32Array(ev::getProperty(val, "opacities"), op, opCount);
    readFloat32Array(ev::getProperty(val, "sh"), sh, shCount);

    if (!pos || posCount == 0) return false;
    out.positions.assign(pos, pos + posCount);
    if (sc) out.scales.assign(sc, sc + scCount);
    if (rot) out.rotations.assign(rot, rot + rotCount);
    if (op) out.opacities.assign(op, op + opCount);
    if (sh) out.sh.assign(sh, sh + shCount);

    Value degVal = ev::getProperty(val, "shDegree");
    if (!ev::isUndefined(degVal)) out.shDegree = static_cast<int>(ev::toDouble(degVal));
    return true;
}

Value tripoGenerate(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapTripoSplat(thisVal);
    if (!w || !w->flow || !w->decoder || !w->vae) {
        return ev::throwTypeError("triposplat: pipeline not loaded");
    }
    if (args.empty()) {
        return ev::throwTypeError("generate(image, opts): image required");
    }

    g_triposplatCancelRequested.store(false, std::memory_order_relaxed);

    int seed = 42, steps = 20, numGaussians = 131072;
    float guidance = 3.0f, shift = 3.0f;
    if (args.size() > 1 && ev::isObject(args[1])) {
        Value sv = ev::getProperty(args[1], "seed");
        if (!ev::isUndefined(sv)) seed = static_cast<int>(ev::toDouble(sv));
        Value stv = ev::getProperty(args[1], "steps");
        if (!ev::isUndefined(stv)) steps = static_cast<int>(ev::toDouble(stv));
        Value ngv = ev::getProperty(args[1], "numGaussians");
        if (!ev::isUndefined(ngv)) numGaussians = static_cast<int>(ev::toDouble(ngv));
        Value gv = ev::getProperty(args[1], "guidanceScale");
        if (!ev::isUndefined(gv)) guidance = static_cast<float>(ev::toDouble(gv));
        Value shv = ev::getProperty(args[1], "shift");
        if (!ev::isUndefined(shv)) shift = static_cast<float>(ev::toDouble(shv));
    }

    std::vector<uint8_t> rgba;
    int iw = 0, ih = 0;
    std::string err;
    if (!readImageInput(args[0], rgba, iw, ih, err)) {
        return ev::throwTypeError(std::string("triposplat: ") + err);
    }

    std::vector<float> rgb01;
    preprocessImage(rgba, iw, ih, rgb01);

    try {
        const int HW = kCanvas * kCanvas;
        std::vector<float> vae_in(static_cast<size_t>(3) * HW);
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < HW; ++i) {
                vae_in[static_cast<size_t>(c) * HW + i] =
                    rgb01[static_cast<size_t>(i) * 3 + c] * 2.0f - 1.0f;
            }
        }
        brotensor::Tensor vae_px = uploadCompute(vae_in.data(), 1, 3 * HW);
        brotensor::Tensor vae_tok;
        w->vae->encode(vae_px, kCanvas, kCanvas, vae_tok);
        brotensor::sync_all();

        if (g_triposplatCancelRequested.load(std::memory_order_relaxed)) {
            ObjectBuilder b;
            b.set("cancelled", true);
            return b.build();
        }

        const int K = 4101; // 5 prefix + 4096 tokens
        const int D1 = 1280;
        const int D2 = 128;
        std::vector<float> f1(static_cast<size_t>(K) * D1, 0.0f);
        brotensor::Tensor feature1 = uploadCompute(f1.data(), K, D1);

        std::vector<float> vt = downloadF32(vae_tok);
        const int Tvae = vae_tok.rows;
        const int prefix = K - Tvae;
        std::vector<float> f2(static_cast<size_t>(K) * D2, 0.0f);
        if (prefix >= 0 && vt.size() <= f2.size() - static_cast<size_t>(prefix) * D2) {
            std::copy(vt.begin(), vt.end(), f2.begin() + static_cast<size_t>(prefix) * D2);
        }
        brotensor::Tensor feature2 = uploadCompute(f2.data(), K, D2);

        const auto& fc = w->flow->config();
        std::mt19937_64 rng(static_cast<uint64_t>(static_cast<uint32_t>(seed)));
        std::normal_distribution<float> norm(0.0f, 1.0f);
        std::vector<float> nlat(static_cast<size_t>(fc.q_token_length) * fc.in_channels);
        for (float& v : nlat) v = norm(rng);
        std::vector<float> ncam(static_cast<size_t>(fc.cam_channels));
        for (float& v : ncam) v = norm(rng);
        brotensor::Tensor noise_lat = uploadCompute(nlat.data(), fc.q_token_length, fc.in_channels);
        brotensor::Tensor noise_cam = uploadCompute(ncam.data(), 1, fc.cam_channels);

        brodiffusion::triposplat::FlowSampleOptions sopts;
        sopts.steps = steps;
        sopts.guidance_scale = guidance;
        sopts.shift = shift;
        sopts.should_cancel = []() {
            return g_triposplatCancelRequested.load(std::memory_order_relaxed);
        };

        brotensor::Tensor latent;
        brodiffusion::triposplat::sample_latent(*w->flow, feature1, feature2, noise_lat, noise_cam, sopts, latent);
        brotensor::sync_all();

        if (g_triposplatCancelRequested.load(std::memory_order_relaxed)) {
            ObjectBuilder b;
            b.set("cancelled", true);
            return b.build();
        }

        brodiffusion::triposplat::GaussianSplats splats =
            w->decoder->decode(latent, numGaussians, static_cast<uint64_t>(static_cast<uint32_t>(seed)));

        // model space (z-up) -> scene space (y-up): (x, y, z) -> (x, -z, y)
        const float s2 = 0.70710678118654752440f;
        float* P = splats.positions.data();
        float* R = splats.rotations.data();
        const size_t n = splats.count();
        for (size_t i = 0; i < n; ++i) {
            const float py = P[i * 3 + 1], pz = P[i * 3 + 2];
            P[i * 3 + 1] = -pz;
            P[i * 3 + 2] = py;
            const float qx = R[i * 4 + 0], qy = R[i * 4 + 1];
            const float qz = R[i * 4 + 2], qw = R[i * 4 + 3];
            R[i * 4 + 0] = s2 * (qx + qw);
            R[i * 4 + 1] = s2 * (qy - qz);
            R[i * 4 + 2] = s2 * (qz + qy);
            R[i * 4 + 3] = s2 * (qw - qx);
        }

        w->lastSplats = splats;
        return makeSplatsResult(splats);
    } catch (const brodiffusion::triposplat::SampleCancelled&) {
        ObjectBuilder b;
        b.set("cancelled", true);
        return b.build();
    } catch (const std::exception& e) {
        return ev::throwError(std::string("triposplat.generate failed: ") + e.what());
    }
}

Value tripoExportPLY(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapTripoSplat(thisVal);
    if (!w) return ev::throwTypeError("TripoSplatPipeline.exportPLY: not a TripoSplatPipeline");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("exportPLY(path): path string required");
    }
    std::string path = ev::toUtf8(args[0]);
    if (!saveSplatPLY(w->lastSplats, path)) {
        return ev::throwError("exportPLY: failed to write PLY to " + path);
    }
    return ev::fromBool(true);
}

Value tripoExportSplat(Value thisVal, std::span<const Value> args) {
    auto* w = unwrapTripoSplat(thisVal);
    if (!w) return ev::throwTypeError("TripoSplatPipeline.exportSplat: not a TripoSplatPipeline");
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("exportSplat(path): path string required");
    }
    std::string path = ev::toUtf8(args[0]);
    if (!saveSplatBinary(w->lastSplats, path)) {
        return ev::throwError("exportSplat: failed to write .splat to " + path);
    }
    return ev::fromBool(true);
}

void decorateTripoSplatProto(ObjectBuilder& proto) {
    proto.def("generate", 2, tripoGenerate);
    proto.def("imageTo3D", 2, tripoGenerate);
    proto.def("exportPLY", 1, tripoExportPLY);
    proto.def("savePly", 1, tripoExportPLY);
    proto.def("exportSplat", 1, tripoExportSplat);
    proto.def("saveSplat", 1, tripoExportSplat);
    proto.accessor("device", [](Value thisVal, std::span<const Value>) -> Value {
        auto* w = unwrapTripoSplat(thisVal);
        if (!w) return ev::fromUtf8("CPU");
        switch (w->device.type) {
            case brotensor::DeviceType::CUDA:  return ev::fromUtf8("CUDA");
            case brotensor::DeviceType::Metal: return ev::fromUtf8("Metal");
            default:                           return ev::fromUtf8("CPU");
        }
    });
}

} // namespace

void ensureTriposplatClassesInstalled() {
    static bool installed = false;
    if (installed) return;
    installed = true;

    g_tripoSplatClass.install("TripoSplatPipeline", 0, nullptr, decorateTripoSplatProto);
}

Value makeTriposplatNamespace() {
    ensureTriposplatClassesInstalled();
    ObjectBuilder tsp;

    tsp.def("init", 0, [](Value, std::span<const Value>) -> Value {
        try {
            brotensor::init();
        } catch (const std::exception& e) {
            return ev::throwError(std::string("triposplat.init failed: ") + e.what());
        }
        return ev::undefined();
    });

    tsp.def("load", 1, [](Value, std::span<const Value> args) -> Value {
        if (args.empty() || !ev::isObject(args[0])) {
            return ev::throwTypeError("load({ dinov3, vae, flow, decoder }) requires an options object");
        }

        Value dinoV = ev::getProperty(args[0], "dinov3");
        Value vaeV = ev::getProperty(args[0], "vae");
        Value flowV = ev::getProperty(args[0], "flow");
        Value decV = ev::getProperty(args[0], "decoder");

        if (!ev::isString(dinoV) || !ev::isString(vaeV) || !ev::isString(flowV) || !ev::isString(decV)) {
            return ev::throwTypeError("load: dinov3, vae, flow and decoder paths are all required");
        }

        std::string p_dino = ev::toUtf8(dinoV);
        std::string p_vae = ev::toUtf8(vaeV);
        std::string p_flow = ev::toUtf8(flowV);
        std::string p_dec = ev::toUtf8(decV);

        if (!std::filesystem::exists(p_dino)) {
            return ev::throwError("triposplat.load failed: cannot open dinov3 file " + p_dino);
        }
        if (!std::filesystem::exists(p_vae)) {
            return ev::throwError("triposplat.load failed: cannot open vae file " + p_vae);
        }
        if (!std::filesystem::exists(p_flow)) {
            return ev::throwError("triposplat.load failed: cannot open flow file " + p_flow);
        }
        if (!std::filesystem::exists(p_dec)) {
            return ev::throwError("triposplat.load failed: cannot open decoder file " + p_dec);
        }

        try {
            brotensor::init();
            brotensor::Device device = brotensor::Device::CPU;
            if (brotensor::is_available(brotensor::Device::CUDA)) device = brotensor::Device::CUDA;
            else if (brotensor::is_available(brotensor::Device::Metal)) device = brotensor::Device::Metal;

            Value devVal = ev::getProperty(args[0], "device");
            if (ev::isString(devVal)) {
                std::string dev = ev::toUtf8(devVal);
                if (dev == "cpu") device = brotensor::Device::CPU;
                else if (dev == "cuda") device = brotensor::Device::CUDA;
                else if (dev == "metal") device = brotensor::Device::Metal;
            }

            auto w = std::make_unique<TripoSplatWrapper>();
            w->device = device;

            w->vae = std::make_unique<brodiffusion::triposplat::Flux2VaeEncoder>();
            {
                auto f = brotensor::safetensors::File::open(p_vae);
                w->vae->load_weights(f);
            }

            w->flow = std::make_unique<brodiffusion::triposplat::FlowDiT>();
            {
                auto f = brotensor::safetensors::File::open(p_flow);
                w->flow->load_weights(f);
            }

            w->decoder = std::make_unique<brodiffusion::triposplat::OctreeGaussianDecoder>();
            {
                auto f = brotensor::safetensors::File::open(p_dec);
                w->decoder->load_weights(f);
            }

            return g_tripoSplatClass.createInstance(std::move(w));
        } catch (const std::exception& e) {
            return ev::throwError(std::string("triposplat.load failed: ") + e.what());
        }
    });

    tsp.def("cancel", 0, [](Value, std::span<const Value>) -> Value {
        g_triposplatCancelRequested.store(true, std::memory_order_relaxed);
        return ev::undefined();
    });

    tsp.def("exportPLY", 2, [](Value, std::span<const Value> args) -> Value {
        if (args.size() < 2 || !ev::isString(args[1])) {
            return ev::throwTypeError("exportPLY(splats, path): splats object and path string required");
        }
        brodiffusion::triposplat::GaussianSplats splats;
        if (!parseSplatsFromValue(args[0], splats)) {
            return ev::throwTypeError("exportPLY: invalid splats object");
        }
        std::string path = ev::toUtf8(args[1]);
        if (!saveSplatPLY(splats, path)) {
            return ev::throwError("exportPLY: failed to write PLY to " + path);
        }
        return ev::fromBool(true);
    });

    tsp.def("exportSplat", 2, [](Value, std::span<const Value> args) -> Value {
        if (args.size() < 2 || !ev::isString(args[1])) {
            return ev::throwTypeError("exportSplat(splats, path): splats object and path string required");
        }
        brodiffusion::triposplat::GaussianSplats splats;
        if (!parseSplatsFromValue(args[0], splats)) {
            return ev::throwTypeError("exportSplat: invalid splats object");
        }
        std::string path = ev::toUtf8(args[1]);
        if (!saveSplatBinary(splats, path)) {
            return ev::throwError("exportSplat: failed to write .splat to " + path);
        }
        return ev::fromBool(true);
    });

    tsp.set("TripoSplatPipeline", g_tripoSplatClass.constructor());
    return tsp.build();
}

} // namespace brodiffusion::api
