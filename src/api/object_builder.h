#pragma once

// The builder is bronze's own (embed/embed_object_builder.h): one definition,
// so a decorate callback written against ObjectBuilder here is the type
// bronze::embed::HostClass::install takes (host_class.h).

#include "embed/embed.h"
#include "embed/embed_object_builder.h"

#include <functional>
#include <string>
#include <string_view>
#include <utility>

namespace brodiffusion::api {

namespace ev = bronze::embed;
using Value = bronze::Value;

using ObjectBuilder = bronze::embed::ObjectBuilder;

} // namespace brodiffusion::api
