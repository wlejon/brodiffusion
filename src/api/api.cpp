#include "api.h"
#include "host_diffusion_internal.h"

namespace brodiffusion::api {

void installDiffusion() {
    ensureDiffusionClassesInstalled();
    ensureTriposplatClassesInstalled();
    ensureVaeClassesInstalled();

    // Every Value below that outlives an allocating call rides in a
    // Persistent (embed.h GC contract): getProperty, createObject and
    // setProperty may each move everything.
    ev::Persistent globalThisP;
    {
        auto gt = ev::globalValue("globalThis");
        if (gt.found && ev::isObject(gt.value)) globalThisP.set(gt.value);
    }

    ev::Persistent broP;
    {
        auto bg = ev::globalValue("bro");
        if (bg.found && ev::isObject(bg.value)) broP.set(bg.value);
    }
    if (!ev::isObject(broP.get()) && ev::isObject(globalThisP.get())) {
        Value candidate = ev::getProperty(globalThisP.get(), "bro");
        if (ev::isObject(candidate)) broP.set(candidate);
    }
    if (!ev::isObject(broP.get())) {
        broP.set(ev::createObject());
        ev::registerGlobal("bro", broP.get());
        if (ev::isObject(globalThisP.get())) {
            globalThisP.set(ev::setProperty(globalThisP.get(), "bro", broP.get()));
        }
    }

    // Mount bro.diffusion
    Value diffVal = makeDiffusionNamespace();
    broP.set(ev::setProperty(broP.get(), "diffusion", diffVal));

    // Mount bro.triposplat
    Value tripoVal = makeTriposplatNamespace();
    broP.set(ev::setProperty(broP.get(), "triposplat", tripoVal));
}

} // namespace brodiffusion::api
