// Weights-gated half of brodiffusion_test_api: drives a real SD1.5 model dir
// through bro.diffusion so the paths that allocate between reads (option
// parsing, the async job, PipelineState stepping, the VAE latent reader) run,
// and under brodiffusion_test_api_gcstress run with a collection on every
// allocation.
//
// The model dir is BRODIFFUSION_SD15_DIR, else <repo>/weights/sd15; the test
// prints a skip line and passes when neither exists.
//
// Linked into brodiffusion_test_api; called from its main().

#define _CRT_SECURE_NO_WARNINGS  // std::getenv

#include "../src/api/api.h"
#include "embed/embed.h"
#include "eval/eval.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#ifndef BRODIFFUSION_WEIGHTS_DIR
#define BRODIFFUSION_WEIGHTS_DIR ""
#endif

namespace {

namespace ev = bronze::embed;
using bronze::Value;

[[noreturn]] void die(const std::string& message) {
    std::cerr << "  FAIL: " << message << std::endl;
    std::exit(1);
}

std::string errorMessage(Value thrown) {
    if (!ev::isObject(thrown)) return ev::toUtf8(thrown);
    return ev::toUtf8(ev::getProperty(thrown, "message"));
}

std::string sd15Dir() {
    if (const char* e = std::getenv("BRODIFFUSION_SD15_DIR"); e && *e) {
        return std::filesystem::exists(e) ? std::string(e) : std::string();
    }
    const std::filesystem::path p = std::filesystem::path(BRODIFFUSION_WEIGHTS_DIR) / "sd15";
    return std::filesystem::exists(p / "unet") ? p.generic_string() : std::string();
}

// Run `script` (an expression) and require the string "OK".
void expectOk(const std::string& what, const std::string& script) {
    ev::CallResult r = bronze::eval::evalScript(script);
    if (r.thrown) die(what + " threw: " + errorMessage(r.value));
    const std::string s = ev::toUtf8(r.value);
    if (s != "OK") die(what + " returned " + s);
    std::cout << "  " << what << ": OK" << std::endl;
}

// Pump the diffusion tick until `poll` (an expression) stops answering "WAIT".
std::string pollUntil(const char* poll, int timeoutS) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        brodiffusion::api::tickDiffusionAsync();
        ev::drainMicrotasks();
        ev::CallResult s = bronze::eval::evalScript(poll);
        std::string state = s.thrown ? "poll threw: " + errorMessage(s.value) : ev::toUtf8(s.value);
        if (state != "WAIT") return state;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(timeoutS)) return "TIMEOUT";
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

}  // namespace

// bro.triposplat without DINOv3 (optional): load, generate from an
// { data, width, height } image, and the two exporters — the namespace form
// reads a SplatCloud back out of JS.
void triposplatWithWeights() {
    const std::filesystem::path root = std::filesystem::path(BRODIFFUSION_WEIGHTS_DIR) / "triposplat";
    const std::string vae = (root / "vae/flux2-vae.safetensors").generic_string();
    const std::string flow = (root / "diffusion_models/triposplat_fp16.safetensors").generic_string();
    const std::string dec = (root / "vae/triposplat_vae_decoder_fp16.safetensors").generic_string();
    if (!std::filesystem::exists(vae) || !std::filesystem::exists(flow) || !std::filesystem::exists(dec)) {
        std::cout << "  triposplat skipped (no weights/triposplat)" << std::endl;
        return;
    }
    const std::string ply = (std::filesystem::temp_directory_path() / "brodiffusion_api_test.ply").generic_string();
    const std::string splat = (std::filesystem::temp_directory_path() / "brodiffusion_api_test.splat").generic_string();
    expectOk("triposplat load / generate / export", R"JS(
        (function() {
            const tp = bro.triposplat.load({ vae: ")JS" + vae + R"JS(", flow: ")JS" + flow +
                                           R"JS(", decoder: ")JS" + dec + R"JS(" });
            if (tp.backgroundRemoval) throw new Error("backgroundRemoval without birefnet");
            const W = 64, H = 64, px = new Uint8ClampedArray(W * H * 4);
            for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
                const i = (y * W + x) * 4, inside = x > 16 && x < 48 && y > 16 && y < 48;
                px[i] = 200; px[i + 1] = 60; px[i + 2] = 40; px[i + 3] = inside ? 255 : 0;
            }
            const s = tp.generate({ data: px, width: W, height: H }, { steps: 2, numGaussians: 4096, seed: 1 });
            if (s.cancelled) throw new Error("cancelled");
            if (!(s.count > 0) || s.positions.length !== 3 * s.count || s.rotations.length !== 4 * s.count)
                throw new Error("splats " + s.count);
            if (bro.triposplat.exportPLY(s, ")JS" + ply + R"JS(") !== true) throw new Error("exportPLY");
            if (tp.exportSplat(")JS" + splat + R"JS(") !== true) throw new Error("exportSplat");
            return "OK";
        })()
    )JS");
    std::error_code ec;
    if (std::filesystem::file_size(ply, ec) < 1000 || std::filesystem::file_size(splat, ec) < 1000) {
        die("triposplat export files are empty");
    }
    std::filesystem::remove(ply, ec);
    std::filesystem::remove(splat, ec);
}

void brodiffusionTestWithWeights() {
    std::cout << "[weights] bro.triposplat..." << std::endl;
    triposplatWithWeights();

    std::cout << "[weights] SD1.5 through bro.diffusion..." << std::endl;
    const std::string dir = sd15Dir();
    if (dir.empty()) {
        std::cout << "  skipped (no weights/sd15; set BRODIFFUSION_SD15_DIR)" << std::endl;
        return;
    }

    expectOk("loadModel + generate", R"JS(
        (function() {
            globalThis.__pipe = bro.diffusion.loadModel(")JS" + dir + R"JS(");
            const p = globalThis.__pipe;
            if (p.busy) throw new Error("busy before any generate");
            const img = p.generate("a red cube on a table", { width: 256, height: 256, steps: 2, seed: 7 });
            if (img.width !== 256 || img.height !== 256) throw new Error("size " + img.width + "x" + img.height);
            if (!(img.data instanceof Uint8ClampedArray) || img.data.length !== 256 * 256 * 4)
                throw new Error("data " + img.data.length);
            return "OK";
        })()
    )JS");

    expectOk("prime / stepOnce / clone / latent / setLatent / decode", R"JS(
        (function() {
            const p = globalThis.__pipe;
            const st = p.prime("a red cube on a table", { width: 256, height: 256, steps: 2, seed: 7 });
            if (st.numSteps !== 2 || st.stepIndex !== 0) throw new Error("state " + st.numSteps + "/" + st.stepIndex);
            st.stepOnce();
            const c = st.clone();
            if (c.stepIndex !== 1) throw new Error("clone stepIndex " + c.stepIndex);
            const lat = st.latent();
            if (lat.length !== 4 * 32 * 32) throw new Error("latent " + lat.length);
            c.setLatent(lat);
            c.stepOnce();
            if (!c.done) throw new Error("clone not done");
            const img = c.decode({ includeFp32: true });
            if (img.width !== 256 || !(img.fp32 instanceof Float32Array)) throw new Error("decode");
            return "OK";
        })()
    )JS");

    // A background generate: the pipeline reports busy, refuses a second
    // user, and hands the image to onDone on a later tick.
    expectOk("background generate launches and holds the pipeline", R"JS(
        (function() {
            const p = globalThis.__pipe;
            globalThis.__done = null;
            p.generate("a blue ball", { width: 256, height: 256, steps: 2, seed: 3,
                onDone: (img, info) => { globalThis.__done = { img, info }; } });
            if (!p.busy) throw new Error("not busy after launch");
            let refused = "";
            try { p.generate("x", { width: 64, height: 64, steps: 1 }); } catch (e) { refused = e.message; }
            if (!/in flight/.test(refused)) throw new Error("second generate: " + refused);
            try { p.prime("x", { width: 64, height: 64, steps: 1 }); refused = ""; } catch (e) { refused = e.message; }
            if (!/in flight/.test(refused)) throw new Error("prime while busy: " + refused);
            return "OK";
        })()
    )JS");
    {
        const std::string s = pollUntil(R"JS(
            (function() {
                const d = globalThis.__done;
                if (!d) return "WAIT";
                if (d.info.cancelled || d.info.error) return "info " + JSON.stringify(d.info);
                if (!d.img || d.img.width !== 256 || d.img.data.length !== 256 * 256 * 4) return "bad image";
                if (globalThis.__pipe.busy) return "still busy";
                return "DONE";
            })()
        )JS", 300);
        if (s != "DONE") die("background generate: " + s);
        std::cout << "  background generate delivered: OK" << std::endl;
    }

    // dispose() during a background generate stops and joins it before the
    // weights go; onDone still reports the cancellation.
    expectOk("dispose during a background generate", R"JS(
        (function() {
            const p = globalThis.__pipe;
            globalThis.__done = null;
            p.generate("a green hill", { width: 512, height: 512, steps: 30, seed: 5,
                onDone: (img, info) => { globalThis.__done = { img, info }; } });
            p.dispose();
            if (p.busy) throw new Error("busy after dispose");
            let msg = "";
            try { p.generate("x", { steps: 1 }); } catch (e) { msg = e.message; }
            if (!/not a loaded Pipeline/.test(msg)) throw new Error("generate after dispose: " + msg);
            return "OK";
        })()
    )JS");
    {
        const std::string s = pollUntil(R"JS(
            (function() {
                const d = globalThis.__done;
                if (!d) return "WAIT";
                return d.info.cancelled ? "DONE" : "not cancelled: " + JSON.stringify(d.info);
            })()
        )JS", 120);
        if (s != "DONE") die("dispose during generate: " + s);
        std::cout << "  cancelled job reported through onDone: OK" << std::endl;
    }

    // The standalone VAE: decode an { data, width, height } latent, then
    // encode the image back.
    const std::string vaeFile = dir + "/vae/diffusion_pytorch_model.fp16.safetensors";
    if (!std::filesystem::exists(vaeFile)) {
        std::cout << "  VAE skipped (no " << vaeFile << ")" << std::endl;
        return;
    }
    expectOk("VAE decode / encode", R"JS(
        (function() {
            const vae = new bro.diffusion.VAE();
            vae.loadWeights(")JS" + vaeFile + R"JS(", "decoder.");
            const lat = new Float32Array(4 * 16 * 24);
            for (let i = 0; i < lat.length; i++) lat[i] = Math.sin(i) * 0.5;
            const img = vae.decode({ data: lat, width: 24, height: 16 });
            if (img.width !== 192 || img.height !== 128) throw new Error("decode size " + img.width + "x" + img.height);
            const enc = vae.encode(img);
            if (enc.width !== 24 || enc.height !== 16 || enc.data.length < 4 * 16 * 24)
                throw new Error("encode " + enc.width + "x" + enc.height + " " + enc.data.length);
            return "OK";
        })()
    )JS");
}
