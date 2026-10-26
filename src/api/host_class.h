#pragma once

// bronze's HostClass (embed/embed_host_class.h): a constructor and prototype
// per realm, branded handles, and unwrap() checked across inherit() lineages.
// It used to be a copy kept per library, one constructor and prototype per
// thread; the realms of one thread then shared a class's mutable prototype,
// so one realm could change what another's instances do. One definition
// keeps every library's classes per realm and every fix made once.

#include "embed/embed_host_class.h"
#include "object_builder.h"

#include <functional>
#include <memory>
#include <span>
#include <string>

namespace brodiffusion::api {

using HostClass = bronze::embed::HostClass;

} // namespace brodiffusion::api
