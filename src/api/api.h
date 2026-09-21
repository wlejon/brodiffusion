#pragma once

#include "embed/embed.h"

#include <functional>
#include <string>

namespace brodiffusion::api {

/// Mounts `bro.diffusion` and `bro.triposplat` onto `bro` in the current Bronze realm.
void installDiffusion();

/// Pump asynchronous diffusion jobs on the current realm/thread.
void tickDiffusionAsync();

/// Cancel and drain all active async diffusion jobs on this thread.
void shutdownDiffusionAsync();

/// Install the host's asset-path resolver. Model dirs, weight files, LoRAs,
/// ControlNets and control dictionaries are then reachable by app-relative and
/// mounted (`/app/...`) paths, as they were before the bronze port — the old
/// binding's setDiffusionAppContext(). Without a resolver every path is used
/// verbatim, which is what a standalone build wants.
void setPathResolver(std::function<std::string(const std::string&)> resolver);

} // namespace brodiffusion::api

using brodiffusion::api::installDiffusion;
