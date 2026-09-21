#include <brodiffusion/version.h>
#include "../src/api/api.h"
#include "../src/api/object_builder.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

// tests/test_diffusion_surface.cpp — the restored Pipeline / PipelineState
// members (docs/transition-drift.md row H7).
void brodiffusionTestRestoredSurface();

int main() {
    namespace ev = bronze::embed;

    std::cout << "Installing diffusion API into Bronze realm..." << std::endl;
    brodiffusion::api::installDiffusion();

    auto g = ev::globalValue("bro");
    assert(g.found);
    assert(ev::isObject(g.value));

    // Check bro.diffusion
    auto diff = ev::getProperty(g.value, "diffusion");
    assert(ev::isObject(diff));

    auto ver = ev::getProperty(diff, "version");
    assert(ev::isString(ver));
    assert(!ev::toUtf8(ver).empty());
    std::cout << "  bro.diffusion.version = " << ev::toUtf8(ver) << std::endl;

    auto diffInit = ev::getProperty(diff, "init");
    assert(ev::isObject(diffInit));
    auto initRes = ev::call(diffInit, diff, {});
    assert(!initRes.thrown);
    assert(ev::isUndefined(initRes.value));

    auto pipeCtor = ev::getProperty(diff, "Pipeline");
    assert(ev::isObject(pipeCtor));

    auto stateCtor = ev::getProperty(diff, "PipelineState");
    assert(ev::isObject(stateCtor));

    auto vaeCtor = ev::getProperty(diff, "VAE");
    assert(ev::isObject(vaeCtor));

    // Test argument validation on loadModel
    auto loadModelFn = ev::getProperty(diff, "loadModel");
    assert(ev::isObject(loadModelFn));
    auto badCall = ev::call(loadModelFn, diff, {});
    assert(badCall.thrown);
    std::string errMsg = ev::toUtf8(badCall.value);
    assert(errMsg.find("path") != std::string::npos);
    std::cout << "  bro.diffusion.loadModel() threw expected error: " << errMsg << std::endl;

    // Test argument validation on createPipeline
    auto createPipeFn = ev::getProperty(diff, "createPipeline");
    assert(ev::isObject(createPipeFn));
    auto badPipeCall = ev::call(createPipeFn, diff, {});
    assert(badPipeCall.thrown);
    std::string pipeErrMsg = ev::toUtf8(badPipeCall.value);
    assert(pipeErrMsg.find("config") != std::string::npos || pipeErrMsg.find("vocabPath") != std::string::npos);
    std::cout << "  bro.diffusion.createPipeline() threw expected error: " << pipeErrMsg << std::endl;

    // Test argument validation on expandNoise
    auto expandNoiseFn = ev::getProperty(diff, "expandNoise");
    assert(ev::isObject(expandNoiseFn));
    auto badNoiseCall = ev::call(expandNoiseFn, diff, {});
    assert(badNoiseCall.thrown);
    std::string noiseErrMsg = ev::toUtf8(badNoiseCall.value);
    assert(noiseErrMsg.find("Float32Array") != std::string::npos || noiseErrMsg.find("src") != std::string::npos);
    std::cout << "  bro.diffusion.expandNoise() threw expected error: " << noiseErrMsg << std::endl;

    // Test diffusion cancel
    auto diffCancel = ev::getProperty(diff, "cancel");
    assert(ev::isObject(diffCancel));
    auto cancelRes = ev::call(diffCancel, diff, {});
    assert(!cancelRes.thrown);
    assert(ev::isUndefined(cancelRes.value));

    // Test diffusion tick
    auto diffTick = ev::getProperty(diff, "tick");
    assert(ev::isObject(diffTick));
    auto tickRes = ev::call(diffTick, diff, {});
    assert(!tickRes.thrown);
    assert(ev::isUndefined(tickRes.value));

    // Check bro.triposplat
    auto tsp = ev::getProperty(g.value, "triposplat");
    assert(ev::isObject(tsp));

    auto tspInit = ev::getProperty(tsp, "init");
    assert(ev::isObject(tspInit));
    auto tspInitRes = ev::call(tspInit, tsp, {});
    assert(!tspInitRes.thrown);
    assert(ev::isUndefined(tspInitRes.value));

    auto tspPipeCtor = ev::getProperty(tsp, "TripoSplatPipeline");
    assert(ev::isObject(tspPipeCtor));

    auto tspLoadFn = ev::getProperty(tsp, "load");
    assert(ev::isObject(tspLoadFn));
    auto tspBadCall = ev::call(tspLoadFn, tsp, {});
    assert(tspBadCall.thrown);
    std::string tspErrMsg = ev::toUtf8(tspBadCall.value);
    assert(tspErrMsg.find("requires an options object") != std::string::npos ||
           tspErrMsg.find("dinov3") != std::string::npos);
    std::cout << "  bro.triposplat.load() threw expected error: " << tspErrMsg << std::endl;

    auto tspCancel = ev::getProperty(tsp, "cancel");
    assert(ev::isObject(tspCancel));
    auto tspCancelRes = ev::call(tspCancel, tsp, {});
    assert(!tspCancelRes.thrown);
    assert(ev::isUndefined(tspCancelRes.value));

    // Test exportPLY and exportSplat validation
    auto expPly = ev::getProperty(tsp, "exportPLY");
    assert(ev::isObject(expPly));
    auto badPlyCall = ev::call(expPly, tsp, {});
    assert(badPlyCall.thrown);

    auto expSplat = ev::getProperty(tsp, "exportSplat");
    assert(ev::isObject(expSplat));
    auto badSplatCall = ev::call(expSplat, tsp, {});
    assert(badSplatCall.thrown);

    // Test triposplat.load dinov3 validation
    {
        brodiffusion::api::ObjectBuilder loadOpts;
        loadOpts.set("dinov3", "/tmp/nonexistent_dino_test.safetensors");
        loadOpts.set("vae", "/tmp/nonexistent_vae_test.safetensors");
        loadOpts.set("flow", "/tmp/nonexistent_flow_test.safetensors");
        loadOpts.set("decoder", "/tmp/nonexistent_dec_test.safetensors");
        const ev::Value lArgs[1] = {loadOpts.build()};
        auto badLoadCall = ev::call(tspLoadFn, tsp, std::span<const ev::Value>(lArgs, 1));
        assert(badLoadCall.thrown);
        std::string err = ev::toUtf8(badLoadCall.value);
        assert(err.find("dinov3") != std::string::npos);
        std::cout << "  bro.triposplat.load() rejects non-existent dinov3: " << err << std::endl;
    }

    // Test createPipeline with euler scheduler and stepOnce
    {
        auto tmp = std::filesystem::temp_directory_path();
        auto vp = tmp / "brodiffusion_api_test_vocab.json";
        auto mp = tmp / "brodiffusion_api_test_merges.txt";
        std::ofstream(vp, std::ios::binary | std::ios::trunc) << "{\"a\":1,\"a</w>\":2}";
        std::ofstream(mp) << "#version: test\n";

        auto createPipeFn = ev::getProperty(diff, "createPipeline");
        assert(ev::isObject(createPipeFn));
        brodiffusion::api::ObjectBuilder pipeOpts;
        pipeOpts.set("vocabPath", vp.string());
        pipeOpts.set("mergesPath", mp.string());
        pipeOpts.set("scheduler", "euler");
        const ev::Value pArgs[1] = {pipeOpts.build()};
        auto pipeRes = ev::call(createPipeFn, diff, std::span<const ev::Value>(pArgs, 1));
        assert(!pipeRes.thrown);
        assert(ev::isObject(pipeRes.value));

        auto pipeCfgFn = ev::getProperty(pipeRes.value, "config");
        assert(ev::isObject(pipeCfgFn));
        auto cfgRes = ev::call(pipeCfgFn, pipeRes.value, {});
        assert(!cfgRes.thrown);
        assert(ev::isObject(cfgRes.value));
        std::string sched = ev::toUtf8(ev::getProperty(cfgRes.value, "scheduler"));
        assert(sched == "euler");
        std::cout << "  bro.diffusion.createPipeline({ scheduler: 'euler' }) config().scheduler: " << sched << std::endl;

        // Test Pipeline.stepOnce validation
        auto stepOnceFn = ev::getProperty(pipeRes.value, "stepOnce");
        assert(ev::isObject(stepOnceFn));
        auto badStep = ev::call(stepOnceFn, pipeRes.value, {});
        assert(badStep.thrown);
        std::string stepErr = ev::toUtf8(badStep.value);
        assert(stepErr.find("state required") != std::string::npos);
        std::cout << "  Pipeline.prototype.stepOnce validates arguments: " << stepErr << std::endl;

        // Test Pipeline.generateAsync validation
        auto genAsyncFn = ev::getProperty(pipeRes.value, "generateAsync");
        assert(ev::isObject(genAsyncFn));
        auto badGen = ev::call(genAsyncFn, pipeRes.value, {});
        assert(badGen.thrown);

        // Test Pipeline.cancel
        auto pipeCancelFn = ev::getProperty(pipeRes.value, "cancel");
        assert(ev::isObject(pipeCancelFn));
        auto pCancelRes = ev::call(pipeCancelFn, pipeRes.value, {});
        assert(!pCancelRes.thrown);
        assert(ev::isUndefined(pCancelRes.value));

        // Test Pipeline.tick
        auto pipeTickFn = ev::getProperty(pipeRes.value, "tick");
        assert(ev::isObject(pipeTickFn));
        auto pTickRes = ev::call(pipeTickFn, pipeRes.value, {});
        assert(!pTickRes.thrown);
        assert(ev::isUndefined(pTickRes.value));

        std::filesystem::remove(vp);
        std::filesystem::remove(mp);
    }

    // Methods restored after the QuickJS → bronze port dropped them.
    brodiffusionTestRestoredSurface();

    std::cout << "All brodiffusion_api standalone tests passed successfully!" << std::endl;
    return 0;
}
