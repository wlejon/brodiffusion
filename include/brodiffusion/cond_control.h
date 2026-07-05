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
    int dim_ = 0;
    std::vector<std::string>             names_;
    std::vector<float>                   scale_;   // [n_axes]
    std::vector<float>                   dirs_;    // [n_axes * dim] row-major
    std::vector<float>                   weight_;  // [n_axes], default 0
    std::unordered_map<std::string, int> index_;
};

}  // namespace brodiffusion
