#pragma once

// CLIP BPE tokenizer for SD1.5's text branch.
//
// Loads Hugging Face-format vocab.json + merges.txt. Produces a fixed-length
// (77) sequence of int32 token IDs framed with <|startoftext|> (49406) and
// padded/terminated with <|endoftext|> (49407), matching what SD1.5's CLIP
// ViT-L/14 expects.
//
// Pre-tokenization is ASCII-focused: lowercase, contraction split ('s, 't,
// 're, 've, 'm, 'll, 'd), runs of letters, single digits, and runs of other
// non-whitespace punctuation. Non-ASCII bytes still flow through (the
// byte-level BPE encoding handles them) but they're treated as part of a
// "punctuation run" rather than the Unicode letter/digit categories HF uses.
// In practice this matches HF behavior on English prompts, which covers the
// SD1.5 use case.

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brodiffusion::clip {

constexpr int max_length = 77;
constexpr int bos_id     = 49406;
constexpr int eos_id     = 49407;
constexpr int pad_id     = 49407;  // CLIP pads with EOS
constexpr int vocab_size = 49408;

class Tokenizer {
public:
    // Load HF-format vocab.json + merges.txt. Throws std::runtime_error on
    // I/O or parse failure.
    static Tokenizer load(const std::string& vocab_json_path,
                          const std::string& merges_txt_path);

    // Encode text into exactly `max_length` token IDs. Always starts with
    // bos_id, ends with eos_id (or is EOS-padded if the content is shorter).
    // If the content would exceed max_length-2 tokens, it's truncated and
    // eos_id replaces the last slot.
    std::vector<int32_t> encode(std::string_view text) const;

    // Lower-level: pre-tokenize + BPE, return token IDs without BOS/EOS or
    // padding. Useful for tests and for batching multiple prompts.
    std::vector<int32_t> tokenize(std::string_view text) const;

    std::size_t vocab_count() const { return vocab_.size(); }
    std::size_t merge_count() const { return merge_ranks_.size(); }

private:
    Tokenizer() = default;

    // BPE-merge one pre-token (already byte-encoded into the GPT-2/CLIP
    // unicode space) into a list of vocab strings.
    std::vector<std::string> bpe_(const std::string& token) const;

    // BPE-encode a UTF-8 pre-token and look up the merged pieces in vocab_,
    // appending token IDs to `out`. Unknown pieces are dropped (matches HF
    // behavior; with a complete vocab this shouldn't happen).
    void encode_piece_(std::string_view piece, std::vector<int32_t>& out) const;

    std::unordered_map<std::string, int32_t> vocab_;
    // Key: "first\x01second" (\x01 cannot appear in a byte-encoded token).
    std::unordered_map<std::string, int32_t> merge_ranks_;
    // Byte (0..255) -> UTF-8 representation of the GPT-2/CLIP unicode mapping.
    std::string byte_to_unicode_[256];
};

}
