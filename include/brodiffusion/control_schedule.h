#pragma once

// Between-step control schedules — a control-dictionary direction re-applied
// with a per-step alpha DURING denoising.
//
// CondControl (cond_control.h) is a prime-time seam: the weighted sum of the
// active axes is added to the conditioning once, before the denoiser has seen
// anything, and there is no way to move it afterwards. That is the right shape
// for a fader and the wrong shape for a schedule, and qwen-image-research's
// round 3 ran into exactly that wall: all of the model's real steering
// authority lives in these axes, every in-network knob that IS schedulable is
// weak, and the desk fitted at 8 steps loses a third of its travel at 40
// because the one strong surface cannot be re-aimed as the picture settles.
//
// A ControlSchedule is the missing half. It holds an ordered list of slots,
// each a direction (a named dictionary axis, or an explicit vector the caller
// minted) with a per-step alpha curve and a step window, and at every step it
// hands back the ONE CondControl that realises that step's stack. The owner
// then rebuilds the conditioning rows as
//
//     rows(s) = txt_in( base + Σ_k alpha_k[s] * scale_k * dir_k )
//
// where `base` is the conditioning the generation was primed with. Because
// the base is the primed embedding — which already carries whatever prime-time
// setControl() asked for — a schedule COMPOSES with the static desk and its
// own contribution is independent of it.
//
// Rebuilding the rows invalidates the prefix KV cache, so a step where the
// alpha stack moves costs one extra prefill. advance() therefore reports
// whether anything actually changed: a flat schedule pays that cost once, and
// a window pays it twice (entering and leaving).
//
// Ranges follow the rest of the Qwen-Image 2.1 surface: [lo_step, hi_step) is
// HALF-OPEN, and the alpha array is indexed by the ABSOLUTE step index, not by
// the offset from lo_step.

#include "brodiffusion/cond_control.h"

#include <string>
#include <vector>

namespace brodiffusion {

// One armed schedule slot.
struct ControlScheduleSlot {
    // The dictionary axis this slot came from, purely for introspection and
    // error messages. Empty when the caller supplied an explicit direction.
    std::string name;
    // The resolved direction, dim wide. Taken as-is (the same contract
    // CondControl::set_vector has): the caller normalises, and the injected
    // vector is alpha * scale * dir.
    std::vector<float> dir;
    float scale = 1.0f;
    // Per ABSOLUTE step index. A step past the end contributes nothing, so a
    // short array is a schedule that stops rather than an error.
    std::vector<float> alpha;
    int lo_step = 0;
    int hi_step = -1;   // half-open; < 0 means "no upper bound"

    // This slot's contribution at `step`: 0 outside [lo_step, hi_step) or past
    // the end of `alpha`.
    float at(int step) const;
};

class ControlSchedule {
public:
    // Append a slot; returns its index. Throws when `dir` is empty, when it
    // disagrees in width with the slots already armed, or when `alpha` is
    // empty.
    int add(ControlScheduleSlot slot);

    // Replace the whole list with this one slot; returns 0.
    int set(ControlScheduleSlot slot);

    // Empty the list. Does NOT reset the applied mark — the owner still has
    // rows carrying the last applied stack and has to put them back, which it
    // discovers through advance() returning true for an all-zero stack.
    void clear() { slots_.clear(); }

    int  count() const { return static_cast<int>(slots_.size()); }
    bool empty() const { return slots_.empty(); }
    int  dim() const { return dim_; }
    const std::vector<ControlScheduleSlot>& slots() const { return slots_; }

    // Forget what was last applied — the rows are back at the base. Called
    // when a generation is primed, since prime() rebuilds them from the
    // conditioning.
    void reset_applied() {
        applied_ = false;
        last_.clear();
    }

    // True when nothing has been applied since the last reset_applied(), i.e.
    // the rows are still the primed ones.
    bool at_base() const { return !applied_; }

    // The step-`step` stack, as a CondControl ready to apply() to a copy of
    // the base conditioning, held to `budget` exactly as the prime-time seam
    // holds its own stack.
    //
    // Returns FALSE when the rows already carry this stack — the alpha vector
    // is identical to the one last applied, or nothing has ever been applied
    // and this step's stack is all zero. That is the whole point of the call:
    // a rebuild resets the prefix KV cache, and a schedule that is flat over a
    // run must not pay for a re-extract on every step.
    //
    // Returns TRUE having filled `out` and recorded the stack as applied. An
    // all-zero `out` (inactive()) is a legitimate TRUE: it means the schedule
    // just left its window and the owner must put the base rows back.
    bool advance(int step, float budget, CondControl& out);

private:
    std::vector<ControlScheduleSlot> slots_;
    std::vector<float> last_;   // the alpha vector last applied, per slot
    bool applied_ = false;
    int  dim_ = 0;
};

}  // namespace brodiffusion
