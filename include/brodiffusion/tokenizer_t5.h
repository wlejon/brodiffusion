#pragma once

// SentencePiece Unigram tokenizer for T5's text branch.
//
// Parsed from a HuggingFace `tokenizer.json` (the Unigram model). Produces a
// fixed-length sequence of int32 token IDs: the Unigram pieces, then the eos
// token </s>, then padded with <pad> to max_length, matching what Flux's T5
// encoder expects.
//
// Pre-tokenization is metaspace-based: every ASCII space becomes the
// metaspace character U+2581 ("▁"), and one metaspace is prepended to the
// whole string (add_prefix_space). Heavy Unicode normalization is skipped —
// for English prompts metaspace is the part that matters. Segmentation is a
// Unigram Viterbi over the transformed UTF-8 string.

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brodiffusion::t5 {

class Tokenizer {
public:
    // Parse a HF tokenizer.json. Throws std::runtime_error on I/O / parse
    // error.
    static Tokenizer load(const std::string& tokenizer_json_path);

    // Encode `text` to exactly `max_length` int32 ids: the Unigram pieces,
    // then the eos token </s>, then padded with <pad> to max_length. If the
    // content + eos exceeds max_length it is truncated with eos in the last
    // slot.
    std::vector<int32_t> encode(std::string_view text, int max_length) const;

    // Unigram pieces only — no eos, no padding.
    std::vector<int32_t> tokenize(std::string_view text) const;

    std::size_t vocab_count() const { return vocab_.size(); }
    int pad_id() const { return pad_id_; }
    int eos_id() const { return eos_id_; }
    int unk_id() const { return unk_id_; }

private:
    Tokenizer() = default;

    // piece-string -> {id, log-prob score}.
    struct Entry { int32_t id; double score; };
    std::unordered_map<std::string, Entry> vocab_;

    int pad_id_ = 0;
    int eos_id_ = 1;
    int unk_id_ = 2;

    // Longest vocab piece byte-length — caps the Viterbi scan window.
    std::size_t max_piece_bytes_ = 1;
    // Smallest vocab score — the unk-fallback penalty is derived from it.
    double min_vocab_score_ = 0.0;
};

}  // namespace brodiffusion::t5
