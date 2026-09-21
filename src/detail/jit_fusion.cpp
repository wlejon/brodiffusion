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

namespace {

bool same(const std::vector<const void*>& a,
          std::initializer_list<const void*> b) {
    if (a.size() != b.size()) return false;
    std::size_t i = 0;
    for (const void* p : b) {
        if (a[i++] != p) return false;
    }
    return true;
}

}  // namespace

int JitSite::find(std::initializer_list<const void*> tokens) const {
    // The caller loops over one binding at a time, so the previous hit is
    // almost always the answer.
    if (recent_ >= 0 && static_cast<std::size_t>(recent_) < bindings_.size() &&
        same(bindings_[static_cast<std::size_t>(recent_)].tokens, tokens)) {
        return recent_;
    }
    for (std::size_t i = 0; i < bindings_.size(); ++i) {
        if (same(bindings_[i].tokens, tokens)) {
            recent_ = static_cast<int>(i);
            return recent_;
        }
    }
    return -1;
}

void JitSite::bind(brotensor::TraceHandle h,
                   std::initializer_list<const void*> tokens) {
    Binding entry;
    entry.tokens.assign(tokens.begin(), tokens.end());
    entry.handle = std::move(h);
    if (bindings_.size() < kMaxBindings) {
        bindings_.push_back(std::move(entry));
        recent_ = static_cast<int>(bindings_.size()) - 1;
        return;
    }
    // Full: round-robin, so a site that genuinely cycles through more than the
    // cap degrades to re-tracing rather than growing without bound.
    bindings_[next_evict_] = std::move(entry);
    recent_ = static_cast<int>(next_evict_);
    next_evict_ = (next_evict_ + 1) % kMaxBindings;
}

void JitSite::disable(const char* why) {
    disabled_ = true;
    bindings_.clear();
    recent_ = -1;
    // One report per process. A backend the compiler has no fused form for —
    // the CPU trace compiler has no broadcast operands at all — would
    // otherwise print the same sentence once for every site in the model.
    static std::atomic<bool> reported{false};
    bool expected = false;
    if (!reported.compare_exchange_strong(expected, true)) return;
    std::fprintf(stderr,
                 "[brodiffusion] jit fusion unavailable at '%s' (%s); using the eager "
                 "path here and at any other site that hits the same wall\n",
                 name_, why ? why : "unknown");
    std::fflush(stderr);
}

}  // namespace brodiffusion::detail
