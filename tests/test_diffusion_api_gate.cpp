// bro.diffusion and bro.triposplat under the host's access gate
// (embed/embed_access.h; api.h, "Access"): each loader asks the gate about the
// paths it was given and each exporter about the file it writes, before
// anything is opened; a refusal is an Error carrying the gate's sentence, code
// EACCES and the path, a refused writer leaves no file, and with the gate
// allowing it the same call gets past the gate to its ordinary not-found /
// cannot-write error. No weights: every path under the refused directory is
// refused whether it exists or not. Run from test_diffusion_api.cpp's main.

#include "embed/embed.h"
#include "embed/embed_access.h"
#include "eval/eval.h"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ev = bronze::embed;

#define GCHECK(cond, msg)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "CHECK failed: " #cond " (" << __FILE__ << ":"        \
                      << __LINE__ << "): " << (msg) << std::endl;              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

std::string utf8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

// The path as the gate is handed it.
std::string gatePath(const fs::path& p) {
    return utf8(fs::absolute(p).lexically_normal());
}

std::string js(const fs::path& p) {
    std::string out = "'";
    for (char c : utf8(p)) {
        if (c == '\\' || c == '\'') out += '\\';
        out += c;
    }
    return out + "'";
}

std::string evalString(const std::string& src) {
    auto r = bronze::eval::evalScript(src);
    if (r.thrown) return "<thrown>";
    return ev::isString(r.value) ? ev::toUtf8(r.value) : "<not a string>";
}

// Runs `stmt`; "ok", the refusal as
// "denied|message|code|path", or "error|<code>|<message>" for any other throw.
std::string syncJs(const std::string& stmt) {
    return evalString(
        "(function() { try { " + stmt + "; return 'ok'; } catch (e) {\n"
        "  if (e && e.code === 'EACCES') return 'denied|' + e.message + '|' + e.code + '|' + e.path;\n"
        "  return 'error|' + (e && e.code) + '|' + (e && e.message); } })()");
}

std::string refusal(const fs::path& p) {
    const std::string path = gatePath(p);
    return "denied|no files access: " + path + "|EACCES|" + path;
}

bool isOrdinaryError(const std::string& r) {
    return r.rfind("error|", 0) == 0 && r.find("EACCES") == std::string::npos;
}

struct TestGate {
    std::mutex mu;
    std::vector<std::string> refused;
    std::atomic<int> calls{0};
    std::string ask(const std::string& path) {
        ++calls;
        std::lock_guard lock(mu);
        for (const auto& p : refused) {
            if (path.compare(0, p.size(), p) == 0) return "no files access: " + path;
        }
        return "";
    }
};

} // namespace

void brodiffusionTestAccessGate() {
    std::cout << "  access gate..." << std::endl;

    const fs::path root = fs::temp_directory_path() / "brodiffusion_api_gate";
    const fs::path ok = root / "ok";
    const fs::path hidden = root / "hidden";
    fs::create_directories(ok);
    fs::create_directories(hidden);
    std::ofstream(ok / "vocab.json", std::ios::binary | std::ios::trunc) << "{\"a\":1,\"a</w>\":2}";
    std::ofstream(ok / "merges.txt") << "#version: test\n";
    // A file that exists under the refused directory: refused all the same.
    std::ofstream(hidden / "w.safetensors") << "not safetensors";

    auto gate = std::make_shared<TestGate>();
    gate->refused.push_back(gatePath(hidden));
    ev::setAccessGate([gate](const ev::AccessRequest& r) -> std::string {
        if (r.kind != ev::AccessKind::Read && r.kind != ev::AccessKind::Write) return "";
        return gate->ask(std::string(r.subject));
    });

    // A Pipeline made from the tiny tokenizer above (the gate allows ok/).
    std::string r = syncJs("globalThis._gatePipe = bro.diffusion.createPipeline({ vocabPath: " +
                           js(ok / "vocab.json") + ", mergesPath: " + js(ok / "merges.txt") +
                           ", scheduler: 'euler' })");
    GCHECK(r == "ok", "createPipeline -> " + r);

    const std::string D = "bro.diffusion.", T = "bro.triposplat.", P = "globalThis._gatePipe.";
    const std::string splats =
        "{ positions: new Float32Array(3), scales: new Float32Array([1, 1, 1]), rotations: new Float32Array([0, 0, 0, 1]),"
        " opacities: new Float32Array([0.5]), sh: new Float32Array(3), shDegree: 0 }";
    struct Case { std::string call; fs::path refusedPath; bool writes; };
    const std::vector<Case> cases = {
        {D + "loadModel(" + js(hidden / "model") + ")", hidden / "model", false},
        {D + "loadModel(" + js(ok / "model") + ", { textEncoderPath: " + js(hidden / "te") + " })", hidden / "te", false},
        {D + "createPipeline({ vocabPath: " + js(ok / "vocab.json") + ", mergesPath: " + js(hidden / "merges.txt") + " })",
         hidden / "merges.txt", false},
        {D + "loadTerrain(" + js(hidden / "terrain") + ")", hidden / "terrain", false},
        {"new " + D + "VAE().loadWeights(" + js(hidden / "vae.safetensors") + ")", hidden / "vae.safetensors", false},
        {P + "loadWeights(" + js(hidden / "w.safetensors") + ")", hidden / "w.safetensors", false},
        {P + "loadWeights(" + js(ok / "t.safetensors") + ", " + js(ok / "u.safetensors") + ", " +
             js(hidden / "v.safetensors") + ")",
         hidden / "v.safetensors", false},
        {P + "loadControlDictionary(" + js(hidden / "dict.safetensors") + ")", hidden / "dict.safetensors", false},
        {P + "reloadTextEncoder(" + js(hidden / "model") + ", '')", hidden / "model", false},
        {T + "load({ vae: " + js(ok / "vae.st") + ", flow: " + js(ok / "flow.st") + ", decoder: " +
             js(hidden / "dec.st") + " })",
         hidden / "dec.st", false},
        {T + "load({ dinov3: " + js(ok / "d.st") + ", vae: " + js(ok / "vae.st") + ", flow: " + js(ok / "flow.st") +
             ", decoder: " + js(ok / "dec.st") + ", birefnet: " + js(hidden / "rmbg") + " })",
         hidden / "rmbg", false},
        {T + "exportPLY(" + splats + ", " + js(hidden / "out.ply") + ")", hidden / "out.ply", true},
        {T + "exportSplat(" + splats + ", " + js(hidden / "out.splat") + ")", hidden / "out.splat", true},
    };

    // Refused: the gate's sentence, EACCES and the absolute path, before the
    // native looks at the disk; a refused writer writes nothing.
    for (const auto& c : cases) {
        r = syncJs(c.call);
        GCHECK(r == refusal(c.refusedPath), c.call + " -> " + r);
        if (c.writes) GCHECK(!fs::exists(c.refusedPath), "a refused export wrote " + utf8(c.refusedPath));
    }
    GCHECK(fs::exists(hidden / "w.safetensors"), "the refused file went away");

    // Allowed: the same calls get past the gate, to their own errors. The
    // exports write into a directory that does not exist, so they fail too.
    {
        std::lock_guard lock(gate->mu);
        gate->refused.clear();
    }
    const int callsBefore = gate->calls.load();
    for (const auto& c : cases) {
        std::string call = c.call;
        if (c.writes) {
            const std::string from = js(c.refusedPath);
            call.replace(call.find(from), from.size(), js(hidden / "missing" / c.refusedPath.filename()));
        }
        r = syncJs(call);
        GCHECK(isOrdinaryError(r), "allowed " + call + " -> " + r);
    }
    GCHECK(gate->calls.load() > callsBefore, "the allowing gate was not asked");
    GCHECK(!fs::exists(hidden / "out.ply") && !fs::exists(hidden / "out.splat"), "an export wrote");

    // An allowed export into a directory that exists writes the file.
    r = syncJs(T + "exportPLY(" + splats + ", " + js(ok / "out.ply") + ")");
    GCHECK(r == "ok" && fs::exists(ok / "out.ply"), "allowed exportPLY -> " + r);

    ev::setAccessGate(nullptr);
    bronze::eval::evalScript("delete globalThis._gatePipe;");
    std::error_code ec;
    fs::remove_all(root, ec);
    std::cout << "  access gate [PASS]" << std::endl;
}
