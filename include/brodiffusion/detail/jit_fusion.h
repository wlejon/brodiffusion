#pragma once

// ─── Trace-JIT fusion sites ─────────────────────────────────────────────────
//
// brotensor's trace JIT collapses an expression written in ordinary tensor ops
// into one kernel. The model code here uses it at the seams where the eager
// form writes a whole activation to HBM only to read it straight back: the
// DiT's AdaLN and residual gates, the SwiGLU tail, the VAE resnet's norm and
// activation, and the scheduler's Euler update.
//
// Two things make that practical in a denoising loop.
//
// First, a trace has to be built once and replayed, not rebuilt per block. A
// cache-hit rebuild is ~30 us — trivial against a step, but 128 of them per
// step is not, and most of that cost is allocating the tracer's output
// tensors. A JitSite remembers the buffers its compiled trace was bound to
// and replays the kernel whenever they are unchanged, which in this model
// means "always, after the first block of the first step".
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
    explicit JitSite(const char* name) : name_(name) {}

    bool disabled() const noexcept { return disabled_; }

    // True when a compiled trace is held and it was bound to exactly these
    // buffers, in this order.
    bool bound_to(std::initializer_list<const void*> ptrs) const;

    void bind(brotensor::TraceHandle h, std::initializer_list<const void*> ptrs);
    void replay() { handle_.execute(); }

    // Turns this site off for the rest of the process and reports why, once.
    void disable(const char* why);

    const char* name() const noexcept { return name_; }
    const brotensor::TraceHandle& handle() const noexcept { return handle_; }

private:
    const char* name_;
    brotensor::TraceHandle handle_;
    std::vector<const void*> ptrs_;
    bool disabled_ = false;
};

// Runs `expr` — which must be the traced expression and nothing else — as one
// fused kernel. Returns false when the caller should fall back to its eager
// path: the JIT is off, or this site has been disabled by an earlier failure,
// or this attempt failed.
//
// `ptrs` identifies the binding. List every buffer the expression touches; a
// stale pointer would otherwise replay the kernel against freed memory.
template <class F>
bool try_fused(JitSite& site, std::initializer_list<const void*> ptrs, F&& expr) {
    if (site.disabled() || !jit_enabled()) return false;
    if (site.bound_to(ptrs)) {
        site.replay();
        return true;
    }
    try {
        brotensor::begin_trace();
        expr();
        site.bind(brotensor::end_trace(), ptrs);
        return true;
    } catch (const std::exception& e) {
        if (brotensor::is_tracing()) brotensor::abort_trace();
        site.disable(e.what());
        return false;
    }
}

}  // namespace brodiffusion::detail
