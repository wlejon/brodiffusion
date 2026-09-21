#include "brodiffusion/detail/jit_fusion.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace brodiffusion::detail {

namespace {

bool env_default_on() {
    const char* v = std::getenv("BRODIFFUSION_JIT");
    if (!v || !*v) return true;
    return !(std::strcmp(v, "0") == 0 || std::strcmp(v, "off") == 0 ||
             std::strcmp(v, "false") == 0);
}

// Read the environment once, then let set_jit_enabled() override it. Relaxed is
// enough: the value changes at most once, from the CLI, before any worker runs.
std::atomic<bool>& jit_flag() {
    static std::atomic<bool> on{env_default_on()};
    return on;
}

}  // namespace

bool jit_enabled() {
    return jit_flag().load(std::memory_order_relaxed);
}

void set_jit_enabled(bool on) {
    jit_flag().store(on, std::memory_order_relaxed);
}

bool JitSite::bound_to(std::initializer_list<const void*> ptrs) const {
    if (!handle_) return false;
    if (ptrs_.size() != ptrs.size()) return false;
    size_t i = 0;
    for (const void* p : ptrs) {
        if (ptrs_[i++] != p) return false;
    }
    return true;
}

void JitSite::bind(brotensor::TraceHandle h, std::initializer_list<const void*> ptrs) {
    handle_ = std::move(h);
    ptrs_.assign(ptrs.begin(), ptrs.end());
}

void JitSite::disable(const char* why) {
    disabled_ = true;
    handle_ = brotensor::TraceHandle();
    ptrs_.clear();
    std::fprintf(stderr, "[brodiffusion] jit fusion unavailable at '%s' (%s); using eager path\n",
                 name_, why ? why : "unknown");
    std::fflush(stderr);
}

}  // namespace brodiffusion::detail
