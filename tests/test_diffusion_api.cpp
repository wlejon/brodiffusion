#include <brodiffusion/version.h>
#include "../src/api/api.h"
#include "../src/api/object_builder.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

// tests/test_diffusion_surface.cpp — the restored Pipeline / PipelineState
// members (docs/transition-drift.md row H7).
void brodiffusionTestRestoredSurface();
// tests/test_diffusion_api_weights.cpp — SD1.5 through bro.diffusion, skipped
// without weights.
void brodiffusionTestWithWeights();

// Failures exit rather than assert(): assert() is a no-op in the Release
// configuration this test runs in.
//
// Every Value held across an allocating embed call rides in a Persistent
// (embed.h GC contract); brodiffusion_test_api_gcstress runs this under
// BRONZE_GC_STRESS=1 BRONZE_GC_POISON=1, where a stale Value crashes.

namespace {

namespace ev = bronze::embed;
using bronze::Value;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond     \
                      << std::endl;                                              \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

// obj[name](...args) with obj as the receiver. `args` are Persistents so
// they survive the method lookup, which allocates.
ev::CallResult callMethod(const ev::Persistent& obj, const char* name,
                          const std::vector<const ev::Persistent*>& args = {}) {
    ev::Persistent fn(ev::getProperty(obj.get(), name));
    if (!ev::isFunction(fn.get())) {
        std::cerr << "FAIL: " << name << " is not a function" << std::endl;
        std::exit(1);
    }
    std::vector<Value> argv;
    argv.reserve(args.size());
    for (const ev::Persistent* a : args) argv.push_back(a->get());
    return ev::call(fn.get(), obj.get(), std::span<const Value>(argv));
}

std::string errorText(const ev::CallResult& r) { return ev::toUtf8(r.value); }

// A call that must throw, with a message containing one of `needles`.
std::string expectThrow(const ev::Persistent& obj, const char* name,
                        std::initializer_list<const char*> needles,
                        const std::vector<const ev::Persistent*>& args = {}) {
    ev::CallResult r = callMethod(obj, name, args);
    if (!r.thrown) {
        std::cerr << "FAIL: " << name << " did not throw" << std::endl;
        std::exit(1);
    }
    std::string msg = errorText(r);
    bool hit = needles.size() == 0;
    for (const char* n : needles) hit = hit || msg.find(n) != std::string::npos;
    if (!hit) {
        std::cerr << "FAIL: " << name << " threw an unexpected error: " << msg << std::endl;
        std::exit(1);
    }
    return msg;
}

void expectUndefinedResult(const ev::Persistent& obj, const char* name) {
    ev::CallResult r = callMethod(obj, name);
    CHECK(!r.thrown);
    CHECK(ev::isUndefined(r.value));
}

void expectObjectProp(const ev::Persistent& obj, const char* name) {
    Value v = ev::getProperty(obj.get(), name);
    if (!ev::isObject(v)) {
        std::cerr << "FAIL: missing " << name << std::endl;
        std::exit(1);
    }
}

}  // namespace

int main() {
    std::cout << "Installing diffusion API into Bronze realm..." << std::endl;
    brodiffusion::api::installDiffusion();

    ev::Persistent bro;
    {
        auto g = ev::globalValue("bro");
        CHECK(g.found);
        CHECK(ev::isObject(g.value));
        bro.set(g.value);
    }

    // Check bro.diffusion
    ev::Persistent diff(ev::getProperty(bro.get(), "diffusion"));
    CHECK(ev::isObject(diff.get()));

    {
        Value ver = ev::getProperty(diff.get(), "version");
        CHECK(ev::isString(ver));
        std::string v = ev::toUtf8(ver);
        CHECK(!v.empty());
        std::cout << "  bro.diffusion.version = " << v << std::endl;
    }

    expectUndefinedResult(diff, "init");
    expectObjectProp(diff, "Pipeline");
    expectObjectProp(diff, "PipelineState");
    expectObjectProp(diff, "VAE");

    std::cout << "  bro.diffusion.loadModel() threw expected error: "
              << expectThrow(diff, "loadModel", {"path"}) << std::endl;
    std::cout << "  bro.diffusion.createPipeline() threw expected error: "
              << expectThrow(diff, "createPipeline", {"config", "vocabPath"}) << std::endl;
    std::cout << "  bro.diffusion.expandNoise() threw expected error: "
              << expectThrow(diff, "expandNoise", {"Float32Array", "src"}) << std::endl;

    expectUndefinedResult(diff, "cancel");
    expectUndefinedResult(diff, "tick");

    // Terrain: the class is mounted and the loader validates before touching
    // any weights.
    expectObjectProp(diff, "TerrainWorld");
    std::cout << "  bro.diffusion.loadTerrain() threw expected error: "
              << expectThrow(diff, "loadTerrain", {"weightsDir"}) << std::endl;
    {
        ev::Persistent missing(ev::fromUtf8("/nonexistent/terrain-dir"));
        expectThrow(diff, "loadTerrain", {"config.json"}, {&missing});
    }

    // Check bro.triposplat
    ev::Persistent tsp(ev::getProperty(bro.get(), "triposplat"));
    CHECK(ev::isObject(tsp.get()));

    expectUndefinedResult(tsp, "init");
    expectObjectProp(tsp, "TripoSplatPipeline");

    std::cout << "  bro.triposplat.load() threw expected error: "
              << expectThrow(tsp, "load", {"requires an options object", "dinov3"}) << std::endl;
    expectUndefinedResult(tsp, "cancel");

    // exportPLY and exportSplat validation
    expectThrow(tsp, "exportPLY", {});
    expectThrow(tsp, "exportSplat", {});

    // triposplat.load dinov3 validation
    {
        brodiffusion::api::ObjectBuilder loadOpts;
        loadOpts.set("dinov3", "/tmp/nonexistent_dino_test.safetensors");
        loadOpts.set("vae", "/tmp/nonexistent_vae_test.safetensors");
        loadOpts.set("flow", "/tmp/nonexistent_flow_test.safetensors");
        loadOpts.set("decoder", "/tmp/nonexistent_dec_test.safetensors");
        ev::Persistent opts(loadOpts.build());
        std::cout << "  bro.triposplat.load() rejects non-existent dinov3: "
                  << expectThrow(tsp, "load", {"dinov3"}, {&opts}) << std::endl;
    }

    // createPipeline with the euler scheduler, then the Pipeline methods
    {
        auto tmp = std::filesystem::temp_directory_path();
        auto vp = tmp / "brodiffusion_api_test_vocab.json";
        auto mp = tmp / "brodiffusion_api_test_merges.txt";
        std::ofstream(vp, std::ios::binary | std::ios::trunc) << "{\"a\":1,\"a</w>\":2}";
        std::ofstream(mp) << "#version: test\n";

        brodiffusion::api::ObjectBuilder pipeOpts;
        pipeOpts.set("vocabPath", vp.string());
        pipeOpts.set("mergesPath", mp.string());
        pipeOpts.set("scheduler", "euler");
        ev::Persistent opts(pipeOpts.build());
        ev::CallResult pipeRes = callMethod(diff, "createPipeline", {&opts});
        if (pipeRes.thrown) {
            std::cerr << "FAIL: createPipeline threw: " << errorText(pipeRes) << std::endl;
            return 1;
        }
        CHECK(ev::isObject(pipeRes.value));
        ev::Persistent pipe(pipeRes.value);

        {
            ev::CallResult cfgRes = callMethod(pipe, "config");
            CHECK(!cfgRes.thrown);
            CHECK(ev::isObject(cfgRes.value));
            ev::Persistent cfg(cfgRes.value);
            std::string sched = ev::toUtf8(ev::getProperty(cfg.get(), "scheduler"));
            CHECK(sched == "euler");
            std::cout << "  bro.diffusion.createPipeline({ scheduler: 'euler' }) config().scheduler: "
                      << sched << std::endl;
        }

        std::cout << "  Pipeline.prototype.stepOnce validates arguments: "
                  << expectThrow(pipe, "stepOnce", {"state required"}) << std::endl;
        expectThrow(pipe, "generateAsync", {});
        expectUndefinedResult(pipe, "cancel");
        expectUndefinedResult(pipe, "tick");

        // A handle of another class is not a Pipeline: the brand check turns
        // it into a TypeError rather than a cast of the wrong payload.
        {
            ev::Persistent notPipe;
            {
                ev::Persistent tctor(ev::getProperty(tsp.get(), "TripoSplatPipeline"));
                ev::Persistent tproto(ev::getProperty(tctor.get(), "prototype"));
                notPipe.set(ev::getProperty(tproto.get(), "generate"));
            }
            if (ev::isFunction(notPipe.get())) {
                ev::CallResult r = ev::call(notPipe.get(), pipe.get(), {});
                if (!r.thrown) {
                    std::cerr << "FAIL: TripoSplatPipeline.generate accepted a Pipeline" << std::endl;
                    return 1;
                }
                std::cout << "  TripoSplatPipeline.prototype.generate rejects a Pipeline: "
                          << errorText(r) << std::endl;
            }
        }

        std::filesystem::remove(vp);
        std::filesystem::remove(mp);
    }

    // Methods restored after the QuickJS → bronze port dropped them.
    brodiffusionTestRestoredSurface();

    brodiffusionTestWithWeights();

    std::cout << "All brodiffusion_api standalone tests passed successfully!" << std::endl;
    return 0;
}
