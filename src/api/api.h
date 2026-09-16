#pragma once

#include "embed/embed.h"

namespace brodiffusion::api {

/// Mounts `bro.diffusion` and `bro.triposplat` onto `bro` in the current Bronze realm.
void installDiffusion();

} // namespace brodiffusion::api

using brodiffusion::api::installDiffusion;
