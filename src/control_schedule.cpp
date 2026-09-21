// Between-step control schedules. See control_schedule.h.

#include "brodiffusion/control_schedule.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brodiffusion {

float ControlScheduleSlot::at(int step) const {
    if (step < lo_step) return 0.0f;
    if (hi_step >= 0 && step >= hi_step) return 0.0f;
    if (step < 0 || static_cast<std::size_t>(step) >= alpha.size()) return 0.0f;
    return alpha[static_cast<std::size_t>(step)];
}

int ControlSchedule::add(ControlScheduleSlot slot) {
    if (slot.dir.empty()) {
        throw std::runtime_error(
            "ControlSchedule::add: the direction is empty — a schedule needs a "
            "dictionary axis or an explicit (dim,) vector");
    }
    if (slot.alpha.empty()) {
        throw std::runtime_error(
            "ControlSchedule::add: the alpha curve is empty — pass one value "
            "per step");
    }
    const int d = static_cast<int>(slot.dir.size());
    if (dim_ > 0 && d != dim_) {
        throw std::runtime_error(
            "ControlSchedule::add: direction width " + std::to_string(d) +
            " != the armed schedules' width " + std::to_string(dim_));
    }
    dim_ = d;
    slots_.push_back(std::move(slot));
    // A new slot changes the stack whatever its alpha says, so the next
    // advance() must rebuild even if every other slot sat still.
    last_.clear();
    return static_cast<int>(slots_.size()) - 1;
}

int ControlSchedule::set(ControlScheduleSlot slot) {
    slots_.clear();
    dim_ = 0;
    last_.clear();
    return add(std::move(slot));
}

bool ControlSchedule::advance(int step, float budget, CondControl& out) {
    const std::size_t n = slots_.size();
    std::vector<float> a(n, 0.0f);
    bool any = false;
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = slots_[i].at(step);
        if (a[i] != 0.0f) any = true;
    }
    // Nothing armed yet and nothing to arm: the rows are already the base.
    if (!applied_ && !any) return false;
    // The same stack as last time: the rows already carry it, and rebuilding
    // would cost a prefix re-extract for a bit-identical result.
    if (applied_ && a == last_) return false;

    out = CondControl{};
    out.set_budget(budget);
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] == 0.0f) continue;
        // One runtime axis per slot rather than one pre-summed vector: the
        // budget is a statement about a STACK of axes, and folding the stack
        // first would change what it measures.
        out.set_vector("sched:" + std::to_string(i), a[i], slots_[i].dir,
                       slots_[i].scale);
    }
    last_ = std::move(a);
    applied_ = true;
    return true;
}

}  // namespace brodiffusion
