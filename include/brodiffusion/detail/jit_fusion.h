#pragma once

// ─── Trace-JIT fusion sites ─────────────────────────────────────────────────
//
// brotensor's trace JIT collapses an expression written in ordinary tensor ops
// into one kernel. The model code here uses it at the seams where the eager
// form writes a whole activation to HBM only to read it straight back: the
// DiT's AdaLN and residual gates, the SwiGLU tail and the scheduler's Euler
// update.
//
// Two things make that practical in a denoising loop.
//
// First, a trace has to be built once and replayed. Re-tracing is not free
// even on a cache hit — it re-runs the user code, and every traced operator
// allocates a full-size output tensor that the fused kernel then never uses.
// At DiT width that is tens of megabytes per re-trace. So a JitSite keeps a
// compiled trace for each distinct set of buffers it has seen and replays the
// matching one, which after the first block of the first step means always.
//
// Second, a site that the compiler cannot fuse must not take the model down.
// try_fused() catches, disables that one site for the rest of the process,
// warns once, and returns false so the caller runs its eager path. Correct
// output is never contingent on the JIT.
//
// BRODIFFUSION_JIT=0 (or set_jit_enabled(false), which is the CLI's --no-jit)
// turns every site off, so before/after is one flag.

#include "brotensor/jit/trace.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

namespace brodiffusion::detail {

// Process-wide switch. Defaults to on, or to off when BRODIFFUSION_JIT is set
// to "0". Read once.
bool jit_enabled();
void set_jit_enabled(bool on);

class JitSite {
public:
    // How many distinct bindings one site remembers. A DiT site sees one per
    // row range per resolution; a VAE norm sees one per (gain, feature map)
    // pair, which is a couple of dozen. Past the cap the oldest is dropped and
    // that binding re-traces the next time it comes round.
    static constexpr std::size_t kMaxBindings = 64;

    explicit JitSite(const char* name) : name_(name) {}

    bool disabled() const noexcept { return disabled_; }

    // Index of the binding compiled for exactly these identity tokens, or -1.
    int find(std::initializer_list<const void*> tokens) const;

    void bind(brotensor::TraceHandle h, std::initializer_list<const void*> tokens);
    void replay(int index) { bindings_[static_cast<std::size_t>(index)].handle.execute(); }

    // Turns this site off for the rest of the process and reports why, once.
    void disable(const char* why);

    const char* name() const noexcept { return name_; }
    std::size_t binding_count() const noexcept { return bindings_.size(); }

private:
    struct Binding {
        std::vector<const void*> tokens;
        brotensor::TraceHandle handle;
    };

    const char* name_;
    std::vector<Binding> bindings_;
    // The binding the last call used. Every site in the model is called in a
    // loop that repeats one binding, so checking it first turns the lookup
    // into a single compare.
    mutable int recent_ = -1;
    std::size_t next_evict_ = 0;
    bool disabled_ = false;
};

// `tokens` identifies the binding: the compiled kernel is replayed only when
// every token matches. List every buffer the expression touches — a stale
// pointer would otherwise replay the kernel against freed memory — and also
// any literal the tracer bakes into the code as an immediate, since replay
// would otherwise silently reuse the old constant. `token_of(float)` turns
// one of those into a token.
inline const void* token_of(float f) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    // Shifted up and tagged odd: every buffer address here is at least
    // 4-byte aligned, so a constant can never collide with one.
    return reinterpret_cast<const void*>(
        (static_cast<std::uintptr_t>(bits) << 1) | 1u);
}

// Runs `expr` — which must be the traced expression and nothing else — as one
// fused kernel. Returns false when the caller should fall back to its eager
// path: the JIT is off, or this site has been disabled by an earlier failure,
// or this attempt failed.
template <class F>
bool try_fused(JitSite& site, std::initializer_list<const void*> tokens, F&& expr) {
    if (site.disabled() || !jit_enabled()) return false;
    const int hit = site.find(tokens);
    if (hit >= 0) {
        site.replay(hit);
        return true;
    }
    try {
        brotensor::begin_trace();
        expr();
        site.bind(brotensor::end_trace(), tokens);
        return true;
    } catch (const std::exception& e) {
        if (brotensor::is_tracing()) brotensor::abort_trace();
        site.disable(e.what());
        return false;
    }
}

}  // namespace brodiffusion::detail
