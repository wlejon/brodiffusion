// Qwen-Image 2.1 — the multi-slot half of the qwenImage21* surface.
//
// Three groups, all of which exist because the research effort ran into the
// single-binding version of them.
//
//   1. Add/Clear/Count for every ranged hook. Each hook holds an ORDERED LIST
//      of bindings now, and all of them apply to the same generation. The
//      old surface held exactly one: a second qwenImage21SetGateScale
//      REPLACED the first, so a dial vector could not put a scale on the
//      shallow blocks and a delta on the deep ones at the same time, and the
//      whole shape of a control panel had to be squeezed through one
//      (lo, hi, target). Set* is still there, unchanged, as "replace the list
//      with this one entry".
//
//   2. qwenImage21ScalePrefixKv as an idempotent DIAL. It used to multiply
//      the live cache in place, which compounds: firing it once per step ran
//      away, so it could not be used through generate() at all and had to be
//      driven from a hand-rolled stepwise loop that fired it exactly once.
//      The factor is now applied where the DiT reads the cache, so repeated
//      calls mean what they say and the cache stays pristine.
//
//   3. The prefix cache SLOTS. The C API had qi_save_prefix / qi_blend_prefix
//      and JS had nothing, so blending two prompts had to happen a level up,
//      on the conditioning rows. Here it happens on the post-RoPE K/V — below
//      the text rows, below txt_in, and below the text encoder, which may
//      have been released by then.
//
// Plus the prepared text rows and the prompt memo, which are the other two
// things that outlive the encoder.

#include "native_diffusion_qwenimage21_detail.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::api {

namespace {

using MT = brodiffusion::dit::QwenImage21ModTarget;

// The four shapes below repeat once per hook. Written out rather than
// macro'd: each one carries its own argument names in its error message, and
// a JS caller reading "blockLo, blockHi" in a throw is the point.

// ── modulation delta ───────────────────────────────────────────────────────

// qwenImage21AddModDelta(delta, blockLo, blockHi, target?) -> slot index.
// Appends to the list instead of replacing it. Every binding covering a block
// is SUMMED into that block's modulation, before the gates' tanh.
Value qi21AddModDelta(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21AddModDelta");
    brotensor::Tensor delta;
    if (args.empty() || !ev::isObject(args[0]) ||
        !tensorFromJs(args[0], delta)) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21AddModDelta(delta, blockLo, blockHi, "
            "target?): delta must be {rows,cols,data} — use "
            "qwenImage21ClearModDeltas() to clear");
    }
    if (args.size() < 3 || !ev::isNumber(args[1]) || !ev::isNumber(args[2])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21AddModDelta(delta, blockLo, blockHi, "
            "target?): integer range required");
    }
    MT target;
    if (!readModTarget(argAt(args, 3), target)) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21AddModDelta: target must be "
            "'target' | 'prefix' | 'both'");
    }
    try {
        return ev::fromDouble(static_cast<double>(
            w->pipeline->qi21_add_mod_delta(delta, i32At(args, 1),
                                            i32At(args, 2), target)));
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21AddModDelta failed: ") + e.what());
    }
}

Value qi21ClearModDeltas(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ClearModDeltas");
    try {
        w->pipeline->qi21_clear_mod_deltas();
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21ClearModDeltas failed: ") +
            e.what());
    }
}

Value qi21ModDeltaCount(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ModDeltaCount");
    try {
        return ev::fromDouble(
            static_cast<double>(w->pipeline->qi21_mod_delta_count()));
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21ModDeltaCount failed: ") +
            e.what());
    }
}

// ── gate scale ─────────────────────────────────────────────────────────────

// qwenImage21AddGateScale(attnScale, mlpScale, txtScale, imgScale, blockLo,
//                         blockHi) -> slot index.
// Scales covering the same block MULTIPLY.
Value qi21AddGateScale(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21AddGateScale");
    if (args.size() < 6) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21AddGateScale(attnScale, mlpScale, txtScale, "
            "imgScale, blockLo, blockHi): six arguments required");
    }
    try {
        return ev::fromDouble(static_cast<double>(
            w->pipeline->qi21_add_gate_scale(
                static_cast<float>(ev::toDouble(args[0])),
                static_cast<float>(ev::toDouble(args[1])),
                static_cast<float>(ev::toDouble(args[2])),
                static_cast<float>(ev::toDouble(args[3])), i32At(args, 4),
                i32At(args, 5))));
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21AddGateScale failed: ") + e.what());
    }
}

// qwenImage21SetGateScaleRows(attnTxt, attnImg, mlpTxt, mlpImg, blockLo,
//                             blockHi)
// qwenImage21AddGateScaleRows(...) -> slot index.
//
// The four multipliers set INDEPENDENTLY instead of as the rank-1
// attn/mlp x txt/img product. The product cannot express "the attention gate
// on the image rows only": raising attnScale raises the prefix's product
// too, which re-extracts the cache (+13%/step). attnImg alone leaves the
// prefix at 1, so the cache — and every armed qwenImage21ScalePrefixKv
// edit — stays live.
Value qi21GateScaleRows(Value thisVal, std::span<const Value> args,
                        bool add, const char* who) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21(who);
    if (args.size() < 6) {
        return ev::throwTypeError(
            std::string("Pipeline.") + who +
            "(attnTxt, attnImg, mlpTxt, mlpImg, blockLo, blockHi): six "
            "arguments required");
    }
    const float at = static_cast<float>(ev::toDouble(args[0]));
    const float ai = static_cast<float>(ev::toDouble(args[1]));
    const float mt = static_cast<float>(ev::toDouble(args[2]));
    const float mi = static_cast<float>(ev::toDouble(args[3]));
    const int lo = i32At(args, 4), hi = i32At(args, 5);
    try {
        if (!add) {
            w->pipeline->qi21_set_gate_scale_rows(at, ai, mt, mi, lo, hi);
            return ev::undefined();
        }
        return ev::fromDouble(static_cast<double>(
            w->pipeline->qi21_add_gate_scale_rows(at, ai, mt, mi, lo, hi)));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.") + who + " failed: " +
                              e.what());
    }
}

Value qi21SetGateScaleRows(Value thisVal, std::span<const Value> args) {
    return qi21GateScaleRows(thisVal, args, /*add=*/false,
                             "qwenImage21SetGateScaleRows");
}

Value qi21AddGateScaleRows(Value thisVal, std::span<const Value> args) {
    return qi21GateScaleRows(thisVal, args, /*add=*/true,
                             "qwenImage21AddGateScaleRows");
}

Value qi21ClearGateScales(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ClearGateScales");
    try {
        w->pipeline->qi21_clear_gate_scales();
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21ClearGateScales failed: ") +
            e.what());
    }
}

Value qi21GateScaleCount(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21GateScaleCount");
    try {
        return ev::fromDouble(
            static_cast<double>(w->pipeline->qi21_gate_scale_count()));
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21GateScaleCount failed: ") +
            e.what());
    }
}

// ── gate delta ─────────────────────────────────────────────────────────────

// qwenImage21AddGateDelta(delta, blockLo, blockHi, target?) -> slot index.
// Deltas covering the same block ADD, after every covering scale has
// multiplied the tanh.
Value qi21AddGateDelta(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21AddGateDelta");
    brotensor::Tensor delta;
    if (args.empty() || !ev::isObject(args[0]) ||
        !tensorFromJs(args[0], delta)) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21AddGateDelta(delta, blockLo, blockHi, "
            "target?): delta must be {rows,cols,data} — use "
            "qwenImage21ClearGateDeltas() to clear");
    }
    if (args.size() < 3 || !ev::isNumber(args[1]) || !ev::isNumber(args[2])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21AddGateDelta(delta, blockLo, blockHi, "
            "target?): integer range required");
    }
    MT target;
    if (!readModTarget(argAt(args, 3), target)) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21AddGateDelta: target must be "
            "'target' | 'prefix' | 'both'");
    }
    try {
        return ev::fromDouble(static_cast<double>(
            w->pipeline->qi21_add_gate_delta(delta, i32At(args, 1),
                                             i32At(args, 2), target)));
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21AddGateDelta failed: ") + e.what());
    }
}

Value qi21ClearGateDeltas(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ClearGateDeltas");
    try {
        w->pipeline->qi21_clear_gate_deltas();
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21ClearGateDeltas failed: ") +
            e.what());
    }
}

Value qi21GateDeltaCount(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21GateDeltaCount");
    try {
        return ev::fromDouble(
            static_cast<double>(w->pipeline->qi21_gate_delta_count()));
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21GateDeltaCount failed: ") +
            e.what());
    }
}

// ── gate mask ──────────────────────────────────────────────────────────────

// qwenImage21AddGateMask(mask, blockLo, blockHi,
//                        which?: 'both' | 'attn' | 'mlp') -> slot index.
// Masks covering the same block AND the same sublayer MULTIPLY elementwise,
// so two masks are an intersection of what they keep. An 'attn' mask and an
// 'mlp' mask over the same blocks are independent: that is how you keep a
// region's attention wide open while damping its MLP.
Value qi21AddGateMask(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21AddGateMask");
    brotensor::Tensor mask;
    if (args.empty() || !ev::isObject(args[0]) || !tensorFromJs(args[0], mask)) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21AddGateMask(mask, blockLo, blockHi, which?): "
            "mask must be {rows,cols,data} — use qwenImage21ClearGateMasks() "
            "to clear");
    }
    if (args.size() < 3 || !ev::isNumber(args[1]) || !ev::isNumber(args[2])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21AddGateMask(mask, blockLo, blockHi, which?): "
            "integer range required");
    }
    brodiffusion::dit::QwenImage21GateSublayer which{};
    if (!readGateSublayer(args.size() > 3 ? args[3] : ev::undefined(), which)) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21AddGateMask: which must be 'both', 'attn' "
            "or 'mlp'");
    }
    try {
        return ev::fromDouble(static_cast<double>(
            w->pipeline->qi21_add_gate_mask(mask, i32At(args, 1),
                                            i32At(args, 2), which)));
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21AddGateMask failed: ") + e.what());
    }
}

Value qi21ClearGateMasks(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ClearGateMasks");
    try {
        w->pipeline->qi21_clear_gate_masks();
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21ClearGateMasks failed: ") +
            e.what());
    }
}

Value qi21GateMaskCount(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21GateMaskCount");
    try {
        return ev::fromDouble(
            static_cast<double>(w->pipeline->qi21_gate_mask_count()));
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21GateMaskCount failed: ") +
            e.what());
    }
}

// ── the prefix KV dial ─────────────────────────────────────────────────────

// qwenImage21ScalePrefixKv(layerLo, layerHi, kScale, vScale,
//                          rowMask?: {rows,cols,data} | null) — replace the
// scale list with this one binding. Idempotent set-semantics: the factor is
// applied where the cached prefix is READ, so calling this twice with the
// same value is the same as calling it once, it works through generate(),
// and it survives a cache reset. 1/1 over the full range clears.
//
// rowMask, when given, holds ONE WEIGHT per prefix row saying how much of
// kScale/vScale that row gets — k_row[r] = 1 + rowMask[r] * (kScale - 1) —
// so all ones is the broadcast and all zeros the identity. That is per-token
// prompt weighting over the cache, with no re-encode: 1 on one phrase's rows
// and 0 on the rest attenuates only that phrase. Its length must equal
// qwenImage21TextRows()'s row count plus any condition-image rows, or the
// next step throws. null/omitted = every row.
Value qi21PrefixKvScale(Value thisVal, std::span<const Value> args, bool add,
                        const char* who) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21(who);
    if (args.size() < 4 || !ev::isNumber(args[0]) || !ev::isNumber(args[1]) ||
        !ev::isNumber(args[2]) || !ev::isNumber(args[3])) {
        return ev::throwTypeError(
            std::string("Pipeline.") + who +
            "(layerLo, layerHi, kScale, vScale, rowMask?): numeric args "
            "required");
    }
    brotensor::Tensor rows;
    if (args.size() > 4 && ev::isObject(args[4])) {
        if (!tensorFromJs(args[4], rows)) {
            return ev::throwTypeError(
                std::string("Pipeline.") + who +
                ": rowMask must be {rows,cols,data} or null");
        }
    }
    const int lo = i32At(args, 0), hi = i32At(args, 1);
    const float ks = static_cast<float>(ev::toDouble(args[2]));
    const float vs = static_cast<float>(ev::toDouble(args[3]));
    try {
        if (!add) {
            w->pipeline->qi21_scale_prefix_kv(lo, hi, ks, vs, rows);
            return ev::undefined();
        }
        return ev::fromDouble(static_cast<double>(
            w->pipeline->qi21_add_prefix_kv_scale(lo, hi, ks, vs, rows)));
    } catch (const std::exception& e) {
        return ev::throwError(std::string("Pipeline.") + who + " failed: " +
                              e.what());
    }
}

Value qi21ScalePrefixKv(Value thisVal, std::span<const Value> args) {
    return qi21PrefixKvScale(thisVal, args, /*add=*/false,
                             "qwenImage21ScalePrefixKv");
}

// qwenImage21AddPrefixKvScale(layerLo, layerHi, kScale, vScale, rowMask?)
// -> slot index. Bindings covering the same layer MULTIPLY.
Value qi21AddPrefixKvScale(Value thisVal, std::span<const Value> args) {
    return qi21PrefixKvScale(thisVal, args, /*add=*/true,
                             "qwenImage21AddPrefixKvScale");
}

Value qi21ClearPrefixKvScales(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ClearPrefixKvScales");
    try {
        w->pipeline->qi21_clear_prefix_kv_scales();
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21ClearPrefixKvScales failed: ") +
            e.what());
    }
}

Value qi21PrefixKvScaleCount(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21PrefixKvScaleCount");
    try {
        return ev::fromDouble(
            static_cast<double>(w->pipeline->qi21_prefix_kv_scale_count()));
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21PrefixKvScaleCount failed: ") +
            e.what());
    }
}

// ── the prefix cache slots ─────────────────────────────────────────────────

// qwenImage21SavePrefixCache(slot) — deep-copy the live extracted prefix K/V
// (both CFG branches) into `slot`. Needs one step to have run.
Value qi21SavePrefixCache(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21SavePrefixCache");
    if (args.empty() || !ev::isNumber(args[0])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21SavePrefixCache(slot): slot index required");
    }
    try {
        w->pipeline->qi21_save_prefix_cache(i32At(args, 0));
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21SavePrefixCache failed: ") +
            e.what());
    }
}

// qwenImage21BlendPrefixCache(slot, alpha) — blend the live prefix towards a
// saved one: k = (1-alpha)*k + alpha*saved.k, likewise v. The two must
// describe the same layout (same prefix length, same target grid), which in
// practice means two prompts of equal token length; a mismatch throws.
Value qi21BlendPrefixCache(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21BlendPrefixCache");
    if (args.size() < 2 || !ev::isNumber(args[0]) || !ev::isNumber(args[1])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21BlendPrefixCache(slot, alpha): numeric args "
            "required");
    }
    try {
        w->pipeline->qi21_blend_prefix_cache(
            i32At(args, 0), static_cast<float>(ev::toDouble(args[1])));
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21BlendPrefixCache failed: ") +
            e.what());
    }
}

Value qi21ClearPrefixSlots(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ClearPrefixSlots");
    try {
        w->pipeline->qi21_clear_prefix_slots();
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21ClearPrefixSlots failed: ") +
            e.what());
    }
}

// qwenImage21PrefixSlots() -> the slot capacity (4).
Value qi21PrefixSlots(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21PrefixSlots");
    return ev::fromDouble(static_cast<double>(
        brodiffusion::pipeline::Pipeline::kQi21PrefixSlots));
}

// qwenImage21PrefixSlotValid(slot) -> boolean.
Value qi21PrefixSlotValid(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21PrefixSlotValid");
    if (args.empty() || !ev::isNumber(args[0])) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21PrefixSlotValid(slot): slot index required");
    }
    return ev::fromBool(w->pipeline->qi21_prefix_slot_valid(i32At(args, 0)));
}

// ── the prepared text rows ─────────────────────────────────────────────────

// qwenImage21TextRows(uncond?) -> { rows, cols, data } — the prepared
// (nValid, hidden) text rows of the most recent prime(), i.e. the joint
// sequence's text half after txt_in.
Value qi21TextRows(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21TextRows");
    const bool uncond = !args.empty() && ev::toBool(args[0]);
    try {
        return tensorToJs(w->pipeline->qi21_text_rows(uncond));
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21TextRows failed: ") + e.what());
    }
}

// qwenImage21SetTextRows(rows, uncond?) — replace them; the prefix KV cache
// is reset so the change lands on the next step.
Value qi21SetTextRows(Value thisVal, std::span<const Value> args) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21SetTextRows");
    brotensor::Tensor rows;
    if (args.empty() || !tensorFromJs(args[0], rows)) {
        return ev::throwTypeError(
            "Pipeline.qwenImage21SetTextRows(rows, uncond?): rows must be "
            "{rows,cols,data}");
    }
    const bool uncond = args.size() >= 2 && ev::toBool(args[1]);
    try {
        w->pipeline->qi21_set_text_rows(rows, uncond);
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21SetTextRows failed: ") + e.what());
    }
}

// qwenImage21ResetCache() — drop the live prefix KV cache so the next step
// re-extracts. The hooks do this themselves; call it after editing
// conditioning out of band.
Value qi21ResetCache(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ResetCache");
    try {
        w->pipeline->qi21_reset_cache();
        return ev::undefined();
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21ResetCache failed: ") + e.what());
    }
}

// ── the prompt memo ────────────────────────────────────────────────────────

// qwenImage21MemoizedPrompts() -> string[], most recently encoded first.
// These are exactly the prompts prime()/generate() can still serve once
// qwenImage21ReleaseTextEncoder() has freed the backbone.
Value qi21MemoizedPrompts(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21MemoizedPrompts");
    try {
        const std::vector<std::string> prompts =
            w->pipeline->qi21_memoized_prompts();
        return hostArrayOf(prompts.size(), [&prompts](std::size_t i) {
            return ev::fromUtf8(prompts[i]);
        });
    } catch (const std::exception& e) {
        return ev::throwError(
            std::string("Pipeline.qwenImage21MemoizedPrompts failed: ") +
            e.what());
    }
}

Value qi21ClearPromptMemo(Value thisVal, std::span<const Value>) {
    auto* w = qi21Pipeline(thisVal);
    if (!w) return notQi21("qwenImage21ClearPromptMemo");
    w->pipeline->qi21_clear_prompt_memo();
    return ev::undefined();
}

}  // namespace

void decoratePipelineQwenImage21SlotsProto(ObjectBuilder& proto) {
    proto.def("qwenImage21AddModDelta", 4, qi21AddModDelta);
    proto.def("qwenImage21ClearModDeltas", 0, qi21ClearModDeltas);
    proto.def("qwenImage21ModDeltaCount", 0, qi21ModDeltaCount);

    proto.def("qwenImage21AddGateScale", 6, qi21AddGateScale);
    proto.def("qwenImage21SetGateScaleRows", 6, qi21SetGateScaleRows);
    proto.def("qwenImage21AddGateScaleRows", 6, qi21AddGateScaleRows);
    proto.def("qwenImage21ClearGateScales", 0, qi21ClearGateScales);
    proto.def("qwenImage21GateScaleCount", 0, qi21GateScaleCount);

    proto.def("qwenImage21AddGateDelta", 4, qi21AddGateDelta);
    proto.def("qwenImage21ClearGateDeltas", 0, qi21ClearGateDeltas);
    proto.def("qwenImage21GateDeltaCount", 0, qi21GateDeltaCount);

    proto.def("qwenImage21AddGateMask", 4, qi21AddGateMask);
    proto.def("qwenImage21ClearGateMasks", 0, qi21ClearGateMasks);
    proto.def("qwenImage21GateMaskCount", 0, qi21GateMaskCount);

    proto.def("qwenImage21ScalePrefixKv", 5, qi21ScalePrefixKv);
    proto.def("qwenImage21AddPrefixKvScale", 5, qi21AddPrefixKvScale);
    proto.def("qwenImage21ClearPrefixKvScales", 0, qi21ClearPrefixKvScales);
    proto.def("qwenImage21PrefixKvScaleCount", 0, qi21PrefixKvScaleCount);

    proto.def("qwenImage21SavePrefixCache", 1, qi21SavePrefixCache);
    proto.def("qwenImage21BlendPrefixCache", 2, qi21BlendPrefixCache);
    proto.def("qwenImage21ClearPrefixSlots", 0, qi21ClearPrefixSlots);
    proto.def("qwenImage21PrefixSlots", 0, qi21PrefixSlots);
    proto.def("qwenImage21PrefixSlotValid", 1, qi21PrefixSlotValid);

    proto.def("qwenImage21TextRows", 1, qi21TextRows);
    proto.def("qwenImage21SetTextRows", 2, qi21SetTextRows);
    proto.def("qwenImage21ResetCache", 0, qi21ResetCache);

    proto.def("qwenImage21MemoizedPrompts", 0, qi21MemoizedPrompts);
    proto.def("qwenImage21ClearPromptMemo", 0, qi21ClearPromptMemo);
}

}  // namespace brodiffusion::api
