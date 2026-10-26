#include <brodiffusion/version.h>
#include "../src/api/api.h"
#include "../src/api/object_builder.h"
#include "eval/eval.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

// tests/test_diffusion_surface.cpp — the restored Pipeline / PipelineState
// members.
void brodiffusionTestRestoredSurface();
// tests/test_diffusion_api_weights.cpp — SD1.5 through bro.diffusion, skipped
// without weights.
void brodiffusionTestWithWeights();
// test_diffusion_api_gate.cpp
void brodiffusionTestAccessGate();

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
    // A cloud whose arrays are shorter than its positions say is refused,
    // never read past.
    {
        auto r = bronze::eval::evalScript(
            "(function() { const s = { positions: new Float32Array(3) }; let n = 0;\n"
            "  for (const f of ['exportPLY', 'exportSplat']) {\n"
            "    try { bro.triposplat[f](s, 'never-written'); } catch (e) { if (e instanceof TypeError) ++n; }\n"
            "  } return n; })()");
        CHECK(!r.thrown && ev::isNumber(r.value) && ev::toDouble(r.value) == 2);
    }

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
        {
            ev::Persistent p1(ev::fromUtf8("/tmp/init.png"));
            ev::Persistent p2(ev::fromUtf8("/tmp/mask.png"));
            ev::Persistent pr(ev::fromUtf8("a test prompt"));
            expectThrow(pipe, "imageToImage", {"loadWeights"}, {&p1, &pr});
            expectThrow(pipe, "inpaint", {"loadWeights"}, {&p1, &p2, &pr});
        }
        {
            brodiffusion::api::ObjectBuilder cancellableBuilder;
            cancellableBuilder.set("cancelled", false);
            cancellableBuilder.def("cancel", 0, [](Value thisVal, std::span<const Value>) -> Value {
                ev::Persistent self(thisVal);
                ev::setProperty(self.get(), "cancelled", ev::fromBool(true));
                return ev::undefined();
            });
            ev::Persistent cancellable(cancellableBuilder.build());
            ev::CallResult cr = callMethod(diff, "cancel", {&cancellable});
            CHECK(!cr.thrown);
            CHECK(ev::toBool(ev::getProperty(cancellable.get(), "cancelled")));
        }
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

    brodiffusionTestAccessGate();

    brodiffusionTestWithWeights();

    // Re-installing into the realm refreshes all classes (Pipeline, VAE,
    // TerrainWorld, TripoSplatPipeline) without stale thread_local guards.
    brodiffusion::api::installDiffusion();
    {
        ev::Persistent bro2(ev::globalValue("bro").value);
        ev::Persistent diff2(ev::getProperty(bro2.get(), "diffusion"));
        ev::Persistent tsp2(ev::getProperty(bro2.get(), "triposplat"));
        expectObjectProp(diff2, "Pipeline");
        expectObjectProp(diff2, "PipelineState");
        expectObjectProp(diff2, "VAE");
        expectObjectProp(diff2, "TerrainWorld");
        expectObjectProp(tsp2, "TripoSplatPipeline");
    }

    std::cout << "All brodiffusion_api standalone tests passed successfully!" << std::endl;
    return 0;
}
