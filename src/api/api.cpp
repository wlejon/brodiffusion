#include "api.h"
#include "host_diffusion_internal.h"

namespace brodiffusion::api {

void installDiffusion() {
    ensureDiffusionClassesInstalled();
    ensureTriposplatClassesInstalled();
    ensureVaeClassesInstalled();

    Value globalThisVal = ev::undefined();
    auto gt = ev::globalValue("globalThis");
    if (gt.found && ev::isObject(gt.value)) {
        globalThisVal = gt.value;
    }

    Value broVal = ev::globalValue("bro").found ? ev::globalValue("bro").value : ev::undefined();
    if (!ev::isObject(broVal)) {
        if (!ev::isUndefined(globalThisVal)) {
            Value candidate = ev::getProperty(globalThisVal, "bro");
            if (ev::isObject(candidate)) {
                broVal = candidate;
            }
        }
    }
    if (!ev::isObject(broVal)) {
        broVal = ev::createObject();
        ev::registerGlobal("bro", broVal);
        if (!ev::isUndefined(globalThisVal)) {
            ev::setProperty(globalThisVal, "bro", broVal);
        }
    }

    ev::Persistent broP(broVal);

    // Mount bro.diffusion
    Value diffVal = makeDiffusionNamespace();
    broP.set(ev::setProperty(broP.get(), "diffusion", diffVal));

    // Mount bro.triposplat
    Value tripoVal = makeTriposplatNamespace();
    broP.set(ev::setProperty(broP.get(), "triposplat", tripoVal));
}

} // namespace brodiffusion::api
