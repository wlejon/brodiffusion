// Qwen-Image 2.1 — the between-step control schedule.
//
// The rest of the qwenImage21* surface is in-network: modulation deltas, gate
// scales and deltas, gate masks, the prefix-KV dial. qwen-image-research spent
// a round measuring every one of them against the conditioning axes and came
// back with the same answer each time — the best minted text axis moves its
// readout 25σ where the best in-network dial manages 4.4σ, and every
// modulation-side dial has spent 97% of its authority by step 2.
//
// So the one surface worth scheduling was the one that could not be: an axis
// applied through setControl() lands once, before step 0, and the desk fitted
// at 8 steps loses 30-37% of its travel at 40 with nothing in-network able to
// recover it. These bindings are that axis, re-aimed per step.
//
// Shape follows the multi-slot hooks next door exactly — Set replaces the
// list, Add appends and returns an index, Clear empties it, Count reports it —
// with two entry points per form, one naming a bank axis and one taking the
// direction outright, because the research mints diff-of-means axes that are
// in no bank.

#include "native_diffusion_qwenimage21_detail.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::api {

namespace {

// The per-step alpha curve: a Float32Array, or any array-like of numbers (a
// plain JS array of step coefficients is what a fitted schedule looks like in
// the research, and forcing a Float32Array around it buys nothing).
bool readAlphas(Value v, std::vector<float>& out) {
    out.clear();
    const float* f = nullptr;
    std::size_t n = 0;
    if (readFloat32Array(v, f, n)) {
        out.assign(f, f + n);
        return !out.empty();
    }
    ev::Persistent arr(v);  // arrayLength and getElement allocate
    const uint32_t len = arrayLength(arr.get());
    if (len == 0) return false;
    out.reserve(len);
    for (uint32_t i = 0; i < len; ++i) {
        Value e = ev::getElement(arr.get(), i);
        if (!ev::isNumber(e)) return false;
        out.push_back(static_cast<float>(ev::toDouble(e)));
    }
    return true;
}

// [loStep, hiStep) with the same defaults everywhere: omitted loStep is 0 and
// omitted hiStep is "to the end of the run" (-1), so the two-argument form is
// "this curve, over the whole generation".
void readWindow(std::span<const Value> args, std::size_t base, int& lo,
                int& hi) {
    lo = 0;
    hi = -1;
    if (args.size() > base && ev::isNumber(args[base])) {
        lo = static_cast<int>(ev::toDouble(args[base]));
    }
    if (args.size() > base + 1 && ev::isNumber(args[base + 1])) {
        hi = static_cast<int>(ev::toDouble(args[base + 1]));
    }
}

// The name/direction split is decided by argument 0's type, so one native
// function backs both documented signatures: a bank axis by name, or an
// explicit Float32Array direction with an optional scale in its place.
//
//   (name, alphaPerStep, loStep?, hiStep?)
//   (dir,  alphaPerStep, loStep?, hiStep?, scale?)
//
// `replace` picks Set (replace the list) over Add (append).
Value scheduleImpl(Value thisVal, std::span<const Value> args, bool replace,
                   const char* who) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21(who);

    std::vector<float> alpha;
    if (args.size() < 2 || !readAlphas(args[1], alpha)) {
        return ev::throwTypeError(
            std::string("Pipeline.") + who +
            "(nameOrDir, alphaPerStep, loStep?, hiStep?): alphaPerStep must "
            "be a non-empty Float32Array or number array, one value per step");
    }
    int lo = 0, hi = -1;
    readWindow(args, 2, lo, hi);

    try {
        if (!args.empty() && ev::isString(args[0])) {
            const std::string name = ev::toUtf8(args[0]);
            const int slot =
                replace ? w->pipeline->qi21_set_control_schedule(name, alpha,
                                                                 lo, hi)
                        : w->pipeline->qi21_add_control_schedule(name, alpha,
                                                                 lo, hi);
            return ev::fromDouble(static_cast<double>(slot));
        }
        const float* dir = nullptr;
        std::size_t n = 0;
        if (args.empty() || !readFloat32Array(args[0], dir, n) || n == 0) {
            return ev::throwTypeError(
                std::string("Pipeline.") + who +
                ": the first argument must be a control-dictionary axis name "
                "or a non-empty Float32Array direction of "
                "qwenImage21TextHiddenDim() floats");
        }
        float scale = 1.0f;
        if (args.size() > 4 && ev::isNumber(args[4])) {
            scale = static_cast<float>(ev::toDouble(args[4]));
        }
        const std::vector<float> v(dir, dir + n);
        const int slot =
            replace ? w->pipeline->qi21_set_control_schedule_dir(v, scale,
                                                                 alpha, lo, hi)
                    : w->pipeline->qi21_add_control_schedule_dir(v, scale,
                                                                 alpha, lo, hi);
        return ev::fromDouble(static_cast<double>(slot));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.") + who + " failed: " +
                              e.what());
    }
}

// qwenImage21SetControlSchedule(nameOrDir, alphaPerStep, loStep?, hiStep?,
//                               scale?) -> 0
// Replace every armed schedule with this one.
Value qi21SetControlSchedule(Value thisVal, std::span<const Value> args) {
    return scheduleImpl(thisVal, args, /*replace=*/true,
                        "qwenImage21SetControlSchedule");
}

// qwenImage21AddControlSchedule(...) -> slot index. Schedules compose: every
// armed slot contributes alpha_k[step] * scale_k * dir_k to the same step.
Value qi21AddControlSchedule(Value thisVal, std::span<const Value> args) {
    return scheduleImpl(thisVal, args, /*replace=*/false,
                        "qwenImage21AddControlSchedule");
}

// qwenImage21ClearControlSchedules() — and, when a schedule had already moved
// a live generation's rows, put the primed rows back.
Value qi21ClearControlSchedules(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ClearControlSchedules");
    try {
        w->pipeline->qi21_clear_control_schedules();
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21ClearControlSchedules failed: ") +
            e.what());
    }
}

Value qi21ControlScheduleCount(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ControlScheduleCount");
    return ev::fromDouble(
        static_cast<double>(w->pipeline->qi21_control_schedule_count()));
}

// qwenImage21ApplyControlStep(state, step) -> boolean.
// step_once() does this itself; the explicit form is for a caller driving the
// denoiser out of band (or measuring what a re-extract costs). True means the
// rows were rebuilt and the prefix cache dropped.
Value qi21ApplyControlStep(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ApplyControlStep");
    auto* st = unwrapPipelineState(args.empty() ? ev::undefined() : args[0]);
    if (!st) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21ApplyControlStep(state, step?): a "
            "PipelineState is required");
    }
    int step = st->state.step_index;
    if (args.size() >= 2 && ev::isNumber(args[1])) {
        step = static_cast<int>(ev::toDouble(args[1]));
    }
    try {
        return ev::fromBool(
            w->pipeline->qi21_apply_control_step(st->state, step));
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21ApplyControlStep failed: ") +
            e.what());
    }
}

}  // namespace

void decoratePipelineQwenImage21SchedProto(ObjectBuilder& proto) {
    proto.def("qwenImage21SetControlSchedule", 4, qi21SetControlSchedule);
    proto.def("qwenImage21AddControlSchedule", 4, qi21AddControlSchedule);
    proto.def("qwenImage21ClearControlSchedules", 0, qi21ClearControlSchedules);
    proto.def("qwenImage21ControlScheduleCount", 0, qi21ControlScheduleCount);
    proto.def("qwenImage21ApplyControlStep", 2, qi21ApplyControlStep);
}

}  // namespace brodiffusion::api
