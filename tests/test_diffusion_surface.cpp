// Surface coverage for the Pipeline / PipelineState methods that the QuickJS →
// bronze port dropped (see bro's docs/transition-drift.md row H7). Each name
// gets two checks: it must be a callable own member of the class prototype,
// and calling it with a `this` that is not a live Pipeline / PipelineState must
// throw rather than return undefined — a missing handler does neither.
//
// Linked into brodiffusion_test_api; called from its main().
//
// Failures exit the process rather than assert(): assert() is a no-op in the
// Release configuration this test actually runs in, so a bare assert would
// let a regression report PASS.

#include "../src/api/api.h"
#include "embed/embed.h"

#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

namespace ev = bronze::embed;

[[noreturn]] void die(const std::string& message) {
    std::cerr << "  " << message << std::endl;
    std::exit(1);
}
using bronze::Value;

// Every method restored onto Pipeline.prototype.
const char* const kPipelineMethods[] = {
    // ControlNet registry
    "addControlNet", "removeControlNet", "clearControlNets",
    // model / schedule introspection
    "reloadTextEncoder", "numXAttnBlocks", "sigmas", "config",
    // conditioning-control dictionary + weights
    "loadControlDictionary", "setControl", "clearControl", "setControlBudget",
    "controlNorm", "controlAxes", "controlVector", "encodeConditioning",
    "setControlVector", "removeControl",
    // Sana identity anchor
    "setIdentityAnchor", "setIdentityWeight", "hasIdentityAnchor", "clearIdentityAnchor",
    // Krea 2 research hooks
    "krea2SetModDelta", "krea2TimeMod", "krea2SetGateScale", "krea2SetGateMask",
    "krea2CaptureGates", "krea2Gates", "krea2HiddenSize", "krea2NumLayers",
    "krea2EncodePromptTaps", "krea2EncodeText", "krea2EncodeImagePrompt",
    "krea2PrimeFromTaps",
};

// Every method restored onto PipelineState.prototype.
const char* const kStateMethods[] = {
    "stepOnce", "decode", "latent", "setLatent", "krea2StepTimestep", "clone",
};

// PipelineState accessors. They must answer on a non-state `this` without
// throwing (they report zeros), which is how the old getters behaved.
const char* const kStateAccessors[] = {
    "stepIndex", "numSteps", "totalSteps", "done", "latentWidth", "latentHeight",
};

Value prototypeOf(Value ns, const char* className) {
    ev::Persistent ctor(ev::getProperty(ns, className));
    if (!ev::isObject(ctor.get())) die(std::string("missing constructor: ") + className);
    Value proto = ev::getProperty(ctor.get(), "prototype");
    if (!ev::isObject(proto)) die(std::string(className) + " has no prototype");
    return proto;
}

// A method that exists and rejects a foreign receiver.
void checkMethod(Value proto, const char* name) {
    ev::Persistent fn(ev::getProperty(proto, name));
    if (!ev::isFunction(fn.get())) die(std::string("MISSING method: ") + name);
    ev::CallResult r = ev::call(fn.get(), ev::undefined(), {});
    if (!r.thrown) {
        die(std::string("method ") + name + " did not reject a foreign receiver");
    }
}

} // namespace

void brodiffusionTestRestoredSurface() {
    std::cout << "Checking the restored bro.diffusion surface..." << std::endl;

    ev::Persistent broNs(ev::globalValue("bro").value);
    if (!ev::isObject(broNs.get())) die("globalThis.bro is missing");
    ev::Persistent diff(ev::getProperty(broNs.get(), "diffusion"));
    if (!ev::isObject(diff.get())) die("bro.diffusion is missing");

    ev::Persistent pipeProto(prototypeOf(diff.get(), "Pipeline"));
    for (const char* name : kPipelineMethods) checkMethod(pipeProto.get(), name);
    std::cout << "  Pipeline.prototype: " << (sizeof(kPipelineMethods) / sizeof(char*))
              << " restored methods present and receiver-checked" << std::endl;

    ev::Persistent stateProto(prototypeOf(diff.get(), "PipelineState"));
    for (const char* name : kStateMethods) checkMethod(stateProto.get(), name);
    std::cout << "  PipelineState.prototype: " << (sizeof(kStateMethods) / sizeof(char*))
              << " restored methods present and receiver-checked" << std::endl;

    for (const char* name : kStateAccessors) {
        Value v = ev::getProperty(stateProto.get(), name);
        if (ev::isUndefined(v)) die(std::string("MISSING accessor: ") + name);
    }
    std::cout << "  PipelineState accessors: stepIndex/numSteps/totalSteps/done/"
                 "latentWidth/latentHeight present" << std::endl;

    // Arity: the dropped positional/option arguments are back, so the declared
    // lengths match what the old bindings registered.
    struct { const char* name; double len; } kArity[] = {
        {"loadWeights", 3}, {"applyLora", 2}, {"addControlNet", 2},
        {"generate", 2}, {"prime", 2}, {"setControlVector", 4},
        {"krea2PrimeFromTaps", 5}, {"krea2EncodeImagePrompt", 3},
    };
    for (const auto& a : kArity) {
        ev::Persistent fn(ev::getProperty(pipeProto.get(), a.name));
        if (!ev::isFunction(fn.get())) die(std::string("MISSING method: ") + a.name);
        double len = ev::toDouble(ev::getProperty(fn.get(), "length"));
        if (len != a.len) {
            die(std::string("arity ") + a.name + " = " + std::to_string(len) +
                ", expected " + std::to_string(a.len));
        }
    }
    // PipelineState.stepOnce/decode take their options argument again.
    for (const char* name : {"stepOnce", "decode", "setLatent"}) {
        ev::Persistent fn(ev::getProperty(stateProto.get(), name));
        double len = ev::toDouble(ev::getProperty(fn.get(), "length"));
        if (len != 1.0) {
            die(std::string("arity PipelineState.") + name + " = " +
                std::to_string(len) + ", expected 1");
        }
    }
    std::cout << "  declared arities match the pre-transition surface" << std::endl;

    // setControl's object-map form is reached through Object.keys, so it must
    // reject a foreign receiver rather than silently enumerate nothing.
    {
        ev::Persistent fn(ev::getProperty(pipeProto.get(), "setControl"));
        ev::Persistent map(ev::parseJson("{\"warmth\":0.5}").value);
        const Value args[1] = {map.get()};
        ev::CallResult r = ev::call(fn.get(), ev::undefined(), std::span<const Value>(args, 1));
        if (!r.thrown) die("setControl({...}) accepted a foreign receiver");
    }

    std::cout << "Restored bro.diffusion surface OK." << std::endl;
}
