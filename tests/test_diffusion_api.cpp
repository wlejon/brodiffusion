#include <brodiffusion/version.h>
#include "../src/api/api.h"
#include <cassert>
#include <iostream>
#include <string>

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

    std::cout << "All brodiffusion_api standalone tests passed successfully!" << std::endl;
    return 0;
}
