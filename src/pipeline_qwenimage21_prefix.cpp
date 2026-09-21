// Qwen-Image 2.1 on Pipeline — the prefix KV cache surface, the prepared text
// rows, and the prompt memo.
//
// Everything here is about the two things that sit BELOW the text encoder and
// outlive it.
//
//   The prefix KV cache. Under causal_condition the text and condition-image
//   rows are modulated from t = 0, so their post-RoPE K/V are step
//   independent: the first forward extracts them and every later one decodes
//   from the copy. After that extract step the cache IS the conditioning as
//   far as the image is concerned — which makes it a control surface, not
//   just an optimisation. Two operations on it:
//
//     * attenuation (qi21_scale_prefix_kv). A per-layer factor on the cached
//       K and V. It is applied by the DiT where it READS the cache, not by
//       writing into the cache, which is what makes it an idempotent dial
//       rather than a multiply that compounds every time it is called.
//     * blending (qi21_save_prefix_cache / qi21_blend_prefix_cache). N deep
//       copies of an extracted prefix, so prompt A's conditioning can be
//       mixed into prompt B's live conditioning with neither re-encoded.
//       This is the JS half of the C API's qi_save_prefix / qi_blend_prefix,
//       which had no Pipeline counterpart at all.
//
//   The prompt memo. qi21_release_text_encoder() frees 8.5 GiB, and used to
//   make prime(prompt) throw for EVERY prompt — including one whose rows the
//   pipeline had just produced. The memo keeps the last K encodings, so the
//   recipe is: encode (or generate) every prompt you need, release, then
//   prime freely. A prompt that was never encoded still throws, naming it.
//
// The research hooks themselves are in pipeline_qwenimage21_hooks.cpp.

#include "brodiffusion/pipeline.h"

#include "pipeline_detail.h"

#include "brodiffusion/denoiser.h"
#include "brodiffusion/dit/qwenimage21.h"
#include "brodiffusion/qwenimage21_text.h"

#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/tensor.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace brodiffusion::pipeline {

namespace bt = ::brotensor;

using detail_pipe::fail;
using detail_pipe::qi21_denoiser;
using detail_pipe::qi21_model;

// ── prefix cache: reset and attenuate ──────────────────────────────────────

void Pipeline::qi21_reset_cache() {
    if (model_class_ != ModelClass::QwenImage21) return;
    auto prepared = last_prepared_.lock();
    if (!prepared) return;   // nothing primed, or the state was dropped
    auto& den = qi21_denoiser(model_class_, denoiser_, "qi21_reset_cache");
    den.reset_cache(*prepared);
}

void Pipeline::qi21_scale_prefix_kv(int layer_lo, int layer_hi, float k_scale,
                                    float v_scale,
                                    const bt::Tensor& row_scale) {
    // No "has anything been primed" check any more: the scale is a dial on
    // the model, so arming it before the first step is legitimate and it
    // survives the re-extract that follows a prefix-side hook.
    qi21_model(model_class_, denoiser_, "qi21_scale_prefix_kv")
        .set_prefix_kv_scale(layer_lo, layer_hi, k_scale, v_scale, row_scale);
}

int Pipeline::qi21_add_prefix_kv_scale(int layer_lo, int layer_hi,
                                       float k_scale, float v_scale,
                                       const bt::Tensor& row_scale) {
    return qi21_model(model_class_, denoiser_, "qi21_add_prefix_kv_scale")
        .add_prefix_kv_scale(layer_lo, layer_hi, k_scale, v_scale, row_scale);
}

void Pipeline::qi21_clear_prefix_kv_scales() {
    qi21_model(model_class_, denoiser_, "qi21_clear_prefix_kv_scales")
        .clear_prefix_kv_scales();
}

int Pipeline::qi21_prefix_kv_scale_count() const {
    return static_cast<int>(
        qi21_model(model_class_, denoiser_, "qi21_prefix_kv_scale_count")
            .prefix_kv_scales().size());
}

// ── prefix cache: the slots ────────────────────────────────────────────────

namespace {

void check_slot(int slot, const char* who) {
    if (slot < 0 || slot >= Pipeline::kQi21PrefixSlots) {
        fail(std::string(who) + ": slot must be in [0, " +
             std::to_string(Pipeline::kQi21PrefixSlots) + ")");
    }
}

}  // namespace

void Pipeline::qi21_save_prefix_cache(int slot) {
    check_slot(slot, "qi21_save_prefix_cache");
    auto& den = qi21_denoiser(model_class_, denoiser_,
                              "qi21_save_prefix_cache");
    auto prepared = last_prepared_.lock();
    if (!prepared) {
        fail("qi21_save_prefix_cache: nothing primed — call prime() (and run "
             "at least one step, so the prefix has been extracted) first");
    }
    auto& cache = den.prefix_cache(*prepared, /*uncond=*/false);
    if (!cache.valid()) {
        fail("qi21_save_prefix_cache: the prefix has not been extracted yet "
             "— run one step first");
    }
    Qi21PrefixSlot& dst = qi21_prefix_slots_[static_cast<std::size_t>(slot)];
    // Tensor's copy assignment is a device-aware DEEP copy, so this is a real
    // snapshot: the live cache is free to be reset, re-extracted or blended
    // afterwards without touching what was saved.
    dst.cond = cache;
    dst.uncond_valid = false;
    if (den.has_uncond(*prepared)) {
        auto& ucache = den.prefix_cache(*prepared, /*uncond=*/true);
        if (ucache.valid()) {
            dst.uncond = ucache;
            dst.uncond_valid = true;
        }
    }
}

void Pipeline::qi21_blend_prefix_cache(int slot, float alpha) {
    check_slot(slot, "qi21_blend_prefix_cache");
    auto& den = qi21_denoiser(model_class_, denoiser_,
                              "qi21_blend_prefix_cache");
    auto prepared = last_prepared_.lock();
    if (!prepared) {
        fail("qi21_blend_prefix_cache: nothing primed — call prime() (and run "
             "at least one step, so the prefix has been extracted) first");
    }
    Qi21PrefixSlot& src = qi21_prefix_slots_[static_cast<std::size_t>(slot)];
    if (!src.cond.valid()) {
        fail("qi21_blend_prefix_cache: slot " + std::to_string(slot) +
             " is empty — qi21_save_prefix_cache() into it first");
    }
    auto& cache = den.prefix_cache(*prepared, /*uncond=*/false);
    if (!cache.valid()) {
        fail("qi21_blend_prefix_cache: the live prefix has not been extracted "
             "yet — run one step first");
    }
    // blend_from() throws on a layout mismatch, which is the common mistake
    // here (two prompts of different token length). Let it.
    cache.blend_from(src.cond, alpha);
    if (src.uncond_valid && den.has_uncond(*prepared)) {
        auto& ucache = den.prefix_cache(*prepared, /*uncond=*/true);
        if (ucache.valid()) ucache.blend_from(src.uncond, alpha);
    }
}

void Pipeline::qi21_clear_prefix_slots() {
    for (auto& s : qi21_prefix_slots_) {
        s.cond.reset();
        s.uncond.reset();
        s.uncond_valid = false;
    }
    // The slots held one (prefix_len, hidden) K and V per layer per branch;
    // hand the blocks back rather than leaving them cached against a DiT that
    // wants the room.
    bt::device_mem_trim(bt::default_device());
}

bool Pipeline::qi21_prefix_slot_valid(int slot) const {
    if (slot < 0 || slot >= kQi21PrefixSlots) return false;
    return qi21_prefix_slots_[static_cast<std::size_t>(slot)].cond.valid();
}

// ── the prepared text rows ─────────────────────────────────────────────────

bt::Tensor Pipeline::qi21_text_rows(bool uncond) const {
    // text_rows() is a mutable accessor on the denoiser; this const overload
    // hands back a copy, which is what a reader wants anyway.
    auto& den = qi21_denoiser(model_class_, denoiser_, "qi21_text_rows");
    auto prepared = last_prepared_.lock();
    if (!prepared) fail("qi21_text_rows: nothing primed — call prime() first");
    return den.text_rows(*prepared, uncond);
}

void Pipeline::qi21_set_text_rows(const bt::Tensor& rows, bool uncond) {
    auto& den = qi21_denoiser(model_class_, denoiser_, "qi21_set_text_rows");
    auto prepared = last_prepared_.lock();
    if (!prepared) {
        fail("qi21_set_text_rows: nothing primed — call prime() first");
    }
    bt::Tensor& dst = den.text_rows(*prepared, uncond);
    const int H = den.config().hidden_size();
    if (rows.cols != H || rows.rows <= 0) {
        fail("qi21_set_text_rows: rows must be (n, qi21_hidden_size())");
    }
    bt::Tensor src = rows.to(bt::default_device());
    if (src.dtype != den.compute_dtype()) {
        bt::Tensor t;
        bt::cast(src, t, den.compute_dtype());
        src = std::move(t);
    }
    dst = std::move(src);
    // The joint sequence's text half just changed; the cached prefix K/V
    // describe the old one.
    den.reset_cache(*prepared);
}

// ── the between-step control schedule ──────────────────────────────────────
//
// See control_schedule.h for the why. The mechanics here are three lines of
// real work: build a one-shot CondControl for this step's stack, add it to the
// conditioning the generation was primed with, and push the result back
// through txt_in. What makes it worth a surface rather than a recipe is the
// bookkeeping around those lines — resolving a bank name to a direction, only
// rebuilding when the alpha actually moved (a rebuild costs a prefix
// re-extract), and putting the primed rows back when the schedule is cleared
// or leaves its window.

namespace {

brodiffusion::ControlScheduleSlot make_slot(std::string name,
                                            std::vector<float> dir,
                                            float scale,
                                            const std::vector<float>& alpha,
                                            int lo_step, int hi_step) {
    brodiffusion::ControlScheduleSlot s;
    s.name    = std::move(name);
    s.dir     = std::move(dir);
    s.scale   = scale;
    s.alpha   = alpha;
    s.lo_step = lo_step < 0 ? 0 : lo_step;
    s.hi_step = hi_step;
    return s;
}

}  // namespace

void Pipeline::qi21_rebuild_text_rows_(PreparedConditioning& prepared,
                                       const CondControl& delta) {
    auto& den = qi21_denoiser(model_class_, denoiser_,
                              "qi21_apply_control_step");
    if (conditioning_.text_embeddings.size() == 0) {
        fail("qi21_apply_control_step: nothing primed — prime() first");
    }
    // A deep copy (brotensor's copy assignment clones): the primed embedding
    // is the BASE every step rebuilds from, so it must survive the injection.
    bt::Tensor emb = conditioning_.text_embeddings;
    // The same call, on the same rows, with the same row policy prime() used
    // for this model class — which is what makes a flat alpha=1 schedule
    // reproduce the prime-time setControl render to the pixel.
    delta.apply(emb, /*row_end=*/-1, /*row_start=*/0);
    den.set_text_rows_from_embeds(prepared, emb,
                                  conditioning_.text_embeddings_mask,
                                  /*uncond=*/false);
}

int Pipeline::qi21_add_control_schedule(const std::string& name,
                                        const std::vector<float>& alpha,
                                        int lo_step, int hi_step) {
    if (model_class_ != ModelClass::QwenImage21) {
        fail("qi21_add_control_schedule: Qwen-Image 2.1 only");
    }
    if (!cond_control_.loaded()) {
        fail("qi21_add_control_schedule: no control dictionary loaded — "
             "load_control_dictionary() (or a setControlVector axis) first, "
             "or schedule an explicit direction instead");
    }
    // Throws naming the axis when it is not in the bank. Runtime axes
    // registered through set_vector() resolve here too, so a minted
    // diff-of-means direction can be scheduled by name.
    std::vector<float> dir = cond_control_.direction(name);
    const float scale = cond_control_.axis_scale(name);
    return qi21_ctl_sched_.add(
        make_slot(name, std::move(dir), scale, alpha, lo_step, hi_step));
}

int Pipeline::qi21_add_control_schedule_dir(const std::vector<float>& dir,
                                            float scale,
                                            const std::vector<float>& alpha,
                                            int lo_step, int hi_step) {
    if (model_class_ != ModelClass::QwenImage21) {
        fail("qi21_add_control_schedule_dir: Qwen-Image 2.1 only");
    }
    return qi21_ctl_sched_.add(
        make_slot(std::string(), dir, scale, alpha, lo_step, hi_step));
}

int Pipeline::qi21_set_control_schedule(const std::string& name,
                                        const std::vector<float>& alpha,
                                        int lo_step, int hi_step) {
    qi21_ctl_sched_.clear();
    return qi21_add_control_schedule(name, alpha, lo_step, hi_step);
}

int Pipeline::qi21_set_control_schedule_dir(const std::vector<float>& dir,
                                            float scale,
                                            const std::vector<float>& alpha,
                                            int lo_step, int hi_step) {
    qi21_ctl_sched_.clear();
    return qi21_add_control_schedule_dir(dir, scale, alpha, lo_step, hi_step);
}

void Pipeline::qi21_clear_control_schedules() {
    const bool rows_moved = !qi21_ctl_sched_.at_base();
    qi21_ctl_sched_.clear();
    qi21_ctl_sched_.reset_applied();
    if (!rows_moved || model_class_ != ModelClass::QwenImage21) return;
    // A schedule had already edited a live generation's rows. Clearing the
    // list must mean the schedule stops, not that its last alpha sticks for
    // the rest of the denoise, so put the primed rows back.
    auto prepared = last_prepared_.lock();
    if (!prepared) return;
    if (conditioning_.text_embeddings.size() == 0) return;
    qi21_rebuild_text_rows_(*prepared, CondControl{});
}

int Pipeline::qi21_control_schedule_count() const {
    return qi21_ctl_sched_.count();
}

bool Pipeline::qi21_apply_control_step(PipelineState& state, int step) {
    if (model_class_ != ModelClass::QwenImage21) return false;
    // Nothing armed and nothing left over from a schedule that was: the rows
    // are the primed ones and there is nothing to do. This is the hot path —
    // every step of every generation runs it.
    if (qi21_ctl_sched_.empty() && qi21_ctl_sched_.at_base()) return false;
    if (!state.prepared) return false;
    CondControl delta;
    // The prime-time stack budget governs a scheduled stack too: it is a
    // statement about how far the conditioning may be pushed off its manifold,
    // and the denoiser cannot tell which seam pushed it.
    if (!qi21_ctl_sched_.advance(step, cond_control_.budget(), delta)) {
        return false;
    }
    qi21_rebuild_text_rows_(*state.prepared, delta);
    return true;
}

// ── the prompt memo ────────────────────────────────────────────────────────

void Pipeline::qi21_memo_put_(const std::string& prompt,
                              const qwenimage21::TextConditioning& tc) {
    for (std::size_t i = 0; i < qi21_prompt_memo_.size(); ++i) {
        if (qi21_prompt_memo_[i].prompt == prompt) {
            qi21_prompt_memo_.erase(qi21_prompt_memo_.begin() +
                                    static_cast<std::ptrdiff_t>(i));
            break;
        }
    }
    Qi21PromptMemo e;
    e.prompt = prompt;
    e.cond = tc;   // deep copy: the caller is free to edit its own rows
    qi21_prompt_memo_.insert(qi21_prompt_memo_.begin(), std::move(e));
    while (qi21_prompt_memo_.size() > kQi21PromptMemo) {
        qi21_prompt_memo_.pop_back();
    }
}

const qwenimage21::TextConditioning* Pipeline::qi21_memo_get_(
    const std::string& prompt) {
    for (std::size_t i = 0; i < qi21_prompt_memo_.size(); ++i) {
        if (qi21_prompt_memo_[i].prompt != prompt) continue;
        if (i != 0) {   // most recently used to the front
            Qi21PromptMemo hit = std::move(qi21_prompt_memo_[i]);
            qi21_prompt_memo_.erase(qi21_prompt_memo_.begin() +
                                    static_cast<std::ptrdiff_t>(i));
            qi21_prompt_memo_.insert(qi21_prompt_memo_.begin(),
                                     std::move(hit));
        }
        return &qi21_prompt_memo_.front().cond;
    }
    return nullptr;
}

qwenimage21::TextConditioning Pipeline::qi21_encode_or_memo_(
    const std::string& prompt) {
    if (model_class_ != ModelClass::QwenImage21) {
        fail("qi21_encode_or_memo: Qwen-Image 2.1 only");
    }
    if (!qwen3vl_model_ || !qwen3vl_tokenizer_) {
        // The encoder is gone. Serve the memo, or say exactly which prompt is
        // missing from it — a bare "no text encoder" sends the caller looking
        // in the wrong place when nine of ten prompts still work.
        const qwenimage21::TextConditioning* hit = qi21_memo_get_(prompt);
        if (hit != nullptr) return *hit;
        fail("prime: the Qwen3-VL text encoder has been released and the "
             "prompt \"" + prompt + "\" is not in the prompt memo (the last " +
             std::to_string(kQi21PromptMemo) +
             " encoded prompts). Encode it before releasing the encoder, "
             "reload the encoder with qi21_reload_text_encoder(), or prime "
             "from caller-supplied rows with qi21_prime_from_text().");
    }
    qwenimage21::TextConditioning tc = qwenimage21::encode_prompt(
        *qwen3vl_tokenizer_, *qwen3vl_model_, prompt);
    qi21_memo_put_(prompt, tc);
    return tc;
}

std::vector<std::string> Pipeline::qi21_memoized_prompts() const {
    std::vector<std::string> out;
    out.reserve(qi21_prompt_memo_.size());
    for (const auto& e : qi21_prompt_memo_) out.push_back(e.prompt);
    return out;
}

void Pipeline::qi21_clear_prompt_memo() { qi21_prompt_memo_.clear(); }

}  // namespace brodiffusion::pipeline
