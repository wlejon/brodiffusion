#pragma once

// Conditioning-space control axes — the runtime form of the sana-research seam.
//
// A loadable DICTIONARY of named control directions in the text encoder's
// embedding space (unit vectors of width = the encoder's hidden dim), each with
// a natural-unit `scale`. The Pipeline applies the weighted sum of the active
// axes additively to the POSITIVE conditioning, on every token row except BOS
// (row 0) for models with one, just before the denoiser projects it — i.e.
//
//     text_embeddings[r] += Σ_k  weight_k * scale_k * dir_k   for r in [row_start, L)
//
// This is encoder-agnostic: the directions are facts about the encoder's
// geometry (Gemma-2 for Sana, T5 / CLIP for Flux / SD), discovered offline; the
// same apply() drives any model whose Conditioning carries a text-embedding
// sequence. Weights are zero until set, so a fresh Pipeline is unaffected.
//
// Dictionary file format (little-endian binary):
//   char[4]   magic  = "BCD1"
//   int32     n_axes
//   int32     dim                       (encoder hidden width)
//   repeat n_axes:
//     int32   name_len
//     char[name_len] name               (utf-8, no terminator)
//     float32 scale                     (natural-unit injection norm)
//     float32[dim] dir                  (unit direction)

#include "brotensor/tensor.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace brodiffusion {

class CondControl {
public:
    // Load a dictionary file, replacing any currently-loaded axes (and resetting
    // all weights to zero). Throws std::runtime_error on a malformed file.
    void load(const std::string& path);

    bool loaded() const { return dim_ > 0; }
    int  dim() const { return dim_; }
    const std::vector<std::string>& names() const { return names_; }

    // Set axis `name`'s weight (alpha, in natural units). Throws if unknown.
    void set(const std::string& name, float alpha);

    // The stored direction of axis `name` (a dim()-length copy) and its baked
    // scale. Introspection of the axes themselves — e.g. decomposing a
    // freshly minted runtime axis against the dictionary's named directions
    // to explain WHAT it moves. Throws if unknown.
    std::vector<float> direction(const std::string& name) const;
    float axis_scale(const std::string& name) const;

    // Register (or replace) a runtime axis with an explicit direction + scale,
    // and set its weight. Unlike load()'s dictionary axes these are built live —
    // e.g. a diff-of-means search from user phrases. `dir` is taken as-is (the
    // caller normalizes / MASSIVE-zeros as the recipe requires); its length sets
    // dim() on the first call and must match it afterward. Coexists with loaded
    // dictionary axes; apply()/clear()/names()/set() all see it uniformly.
    void set_vector(const std::string& name, float alpha,
                    const std::vector<float>& dir, float scale = 1.0f);

    // Remove a single axis (runtime or dictionary) by name. No-op if unknown.
    void remove(const std::string& name);

    // Reset every axis weight to zero (no injection); keeps the loaded dictionary.
    void clear();
    // True iff any axis has a nonzero weight (apply() is a no-op otherwise).
    bool active() const;

    // ── stack budget ────────────────────────────────────────────────────────
    // Each axis weight is bounded by whatever range a UI gives its slider, but
    // the SUM of a stack is not — and what reaches the denoiser is the sum. Once
    // that combined vector grows comparable to the conditioning's own token norm,
    // the prompt stops being the thing that gets rendered: the model draws the
    // injection's semantics instead (on Krea 2 a ten-axis stack past ~12 alpha
    // reliably turns any scene into dramatic crowds, whatever the prompt said).
    //
    // set_budget() caps the injection's LENGTH, measured in the same alpha units
    // the weights use (a length of `budget * mean scale of the active axes`).
    // When a stack exceeds it, apply() scales every active axis by ONE common
    // factor — the dialled-in mix is preserved exactly, only the overdrive is
    // shed. 0 (the default) leaves the stack uncapped.
    void  set_budget(float alpha) { budget_ = alpha > 0.0f ? alpha : 0.0f; }
    float budget() const { return budget_; }

    // The current stack's length in those same alpha units — i.e. a lone axis at
    // this weight would inject the same magnitude. What a UI puts on a stack
    // meter, and what apply() compares against the budget. 0 when inactive.
    float active_norm() const;

    // Add the active weighted control vector to rows [row_start, end) of `emb`
    // in place, where end = row_end if 0 < row_end <= emb.rows, else emb.rows.
    // No-op when inactive or fewer than 2 steerable rows. `emb` is (L, dim) on
    // any device/dtype (FP32 / FP16 / BF16). Throws if emb.cols != dim(). The
    // injection is built in FP32 and cast to emb.dtype.
    //
    // row_end exists for the CLIP fixed-77 path: pass the EOS index so only the
    // CONTENT rows [row_start, eos) are steered, not the live EOS/padding tail
    // (clip-research found injecting the padding rows over-drives generation).
    // Sana's conditioning carries no padding, so it leaves row_end at the
    // default.
    //
    // row_start defaults to 1 (BOS row 0 untouched — SD/Flux/Sana/CLIP all
    // prepend a BOS-equivalent row). Krea 2's fused conditioning has no BOS
    // row, so its Pipeline passes row_start=0 to steer every valid token.
    void apply(brotensor::Tensor& emb, int row_end = -1,
              int row_start = 1) const;

private:
    // The weighted sum Σ weight_k * scale_k * dir_k, plus its length in alpha
    // units (see active_norm()). Empty when no axis is active.
    std::vector<float> combined(float& alpha_norm) const;

    int dim_ = 0;
    float budget_ = 0.0f;                          // 0 = uncapped
    std::vector<std::string>             names_;
    std::vector<float>                   scale_;   // [n_axes]
    std::vector<float>                   dirs_;    // [n_axes * dim] row-major
    std::vector<float>                   weight_;  // [n_axes], default 0
    std::unordered_map<std::string, int> index_;
};

}  // namespace brodiffusion
