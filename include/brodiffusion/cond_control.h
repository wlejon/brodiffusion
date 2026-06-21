#pragma once

// Conditioning-space control axes — the runtime form of the sana-research seam.
//
// A loadable DICTIONARY of named control directions in the text encoder's
// embedding space (unit vectors of width = the encoder's hidden dim), each with
// a natural-unit `scale`. The Pipeline applies the weighted sum of the active
// axes additively to the POSITIVE conditioning, on every token row except BOS
// (row 0), just before the denoiser projects it — i.e.
//
//     text_embeddings[r] += Σ_k  weight_k * scale_k * dir_k     for r in [1, L)
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
    // Reset every axis weight to zero (no injection); keeps the loaded dictionary.
    void clear();
    // True iff any axis has a nonzero weight (apply() is a no-op otherwise).
    bool active() const;

    // Add the active weighted control vector to rows [1, emb.rows) of `emb`
    // in place (BOS row 0 untouched). No-op when inactive or emb has < 2 rows.
    // `emb` is (L, dim) on any device/dtype (FP32 / FP16 / BF16). Throws if
    // emb.cols != dim(). The injection is built in FP32 and cast to emb.dtype.
    void apply(brotensor::Tensor& emb) const;

private:
    int dim_ = 0;
    std::vector<std::string>             names_;
    std::vector<float>                   scale_;   // [n_axes]
    std::vector<float>                   dirs_;    // [n_axes * dim] row-major
    std::vector<float>                   weight_;  // [n_axes], default 0
    std::unordered_map<std::string, int> index_;
};

}  // namespace brodiffusion
