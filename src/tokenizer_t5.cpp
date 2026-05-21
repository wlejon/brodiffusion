#include "brodiffusion/tokenizer_t5.h"

#include "brodiffusion/detail/json.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brodiffusion::t5 {

namespace json = ::brodiffusion::detail::json;

namespace {

// The metaspace character U+2581 ("▁"), UTF-8 0xE2 0x96 0x81.
const char kMetaspace[] = "\xE2\x96\x81";

// Byte-length of the UTF-8 character starting at s[i]. Assumes valid UTF-8.
std::size_t utf8_char_len(std::string_view s, std::size_t i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if      ((c & 0x80) == 0x00) return 1;
    else if ((c & 0xE0) == 0xC0) return 2;
    else if ((c & 0xF0) == 0xE0) return 3;
    else if ((c & 0xF8) == 0xF0) return 4;
    return 1;  // invalid lead byte — treat as 1 to make progress
}

// Metaspace pre-tokenization: replace every ASCII space with U+2581 and
// prepend one U+2581 (add_prefix_space).
std::string metaspace(std::string_view text) {
    std::string out;
    out.reserve(text.size() + text.size() / 4 + 3);
    out += kMetaspace;
    for (char c : text) {
        if (c == ' ') out += kMetaspace;
        else          out += c;
    }
    return out;
}

}  // namespace

// ─── load ──────────────────────────────────────────────────────────────────

Tokenizer Tokenizer::load(const std::string& tokenizer_json_path) {
    std::ifstream f(tokenizer_json_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("t5::Tokenizer: cannot open '" +
                                 tokenizer_json_path + "'");
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();

    json::Value root = json::parse(text);  // throws on malformed input
    const json::Value* model = root.find("model");
    if (!model || !model->is_object()) {
        throw std::runtime_error("t5::Tokenizer: tokenizer.json has no "
                                 "'model' object");
    }
    const json::Value* vocab = model->find("vocab");
    if (!vocab || !vocab->is_array()) {
        throw std::runtime_error("t5::Tokenizer: model has no 'vocab' array");
    }

    Tokenizer t;
    t.unk_id_ = model->get_int("unk_id", 2);

    const auto& entries = vocab->as_array();
    int eos_id = -1;
    int pad_id = -1;
    double min_score = std::numeric_limits<double>::infinity();
    std::size_t max_bytes = 1;

    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& pair = entries[i];
        if (!pair.is_array() || pair.as_array().size() < 2) {
            throw std::runtime_error("t5::Tokenizer: malformed vocab entry");
        }
        const std::string& piece = pair.as_array()[0].as_string();
        const double score = pair.as_array()[1].as_number();
        const int32_t id = static_cast<int32_t>(i);

        t.vocab_.emplace(piece, Tokenizer::Entry{id, score});
        if (!piece.empty() && piece.size() > max_bytes) {
            max_bytes = piece.size();
        }
        if (score < min_score) min_score = score;
        if (piece == "</s>") eos_id = static_cast<int>(i);
        if (piece == "<pad>") pad_id = static_cast<int>(i);
    }

    if (entries.empty()) {
        throw std::runtime_error("t5::Tokenizer: empty vocab");
    }

    t.eos_id_ = (eos_id >= 0) ? eos_id : 1;
    t.pad_id_ = (pad_id >= 0) ? pad_id : 0;
    t.max_piece_bytes_ = max_bytes;
    t.min_vocab_score_ = min_score;
    return t;
}

// ─── tokenize (Unigram Viterbi) ────────────────────────────────────────────

std::vector<int32_t> Tokenizer::tokenize(std::string_view text) const {
    const std::string s = metaspace(text);
    const std::size_t n = s.size();
    if (n == 0) return {};

    const double NEG_INF = -std::numeric_limits<double>::infinity();
    const double unk_penalty = min_vocab_score_ - 10.0;

    // best_score[i] = best log-prob to reach byte position i.
    // back[i] = {start_index, token_id} of the piece ending at i.
    std::vector<double>  best_score(n + 1, NEG_INF);
    std::vector<std::size_t> back_start(n + 1, 0);
    std::vector<int32_t> back_id(n + 1, unk_id_);
    best_score[0] = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        if (best_score[i] == NEG_INF) continue;

        // Scan piece candidates s[i:j] at UTF-8 character boundaries, up to
        // the longest vocab piece byte-length.
        const std::size_t scan_end =
            std::min(n, i + max_piece_bytes_);
        std::size_t j = i;
        bool first_char = true;
        std::size_t first_char_end = i;
        while (j < scan_end) {
            j += utf8_char_len(s, j);
            if (j > scan_end) break;  // would overrun the scan window
            const std::string piece = s.substr(i, j - i);
            auto it = vocab_.find(piece);
            if (it != vocab_.end()) {
                const double cand = best_score[i] + it->second.score;
                if (cand > best_score[j]) {
                    best_score[j] = cand;
                    back_start[j] = i;
                    back_id[j]    = it->second.id;
                }
            }
            if (first_char) {
                first_char_end = j;
                first_char = false;
            }
        }

        // Unknown-char fallback: the single character s[i:i+charlen] always
        // gets a transition via the unk token, guaranteeing a complete path.
        const std::size_t one_char_end = i + utf8_char_len(s, i);
        if (one_char_end <= n) {
            const std::string one = s.substr(i, one_char_end - i);
            const bool known = vocab_.find(one) != vocab_.end();
            if (!known) {
                const double cand = best_score[i] + unk_penalty;
                if (cand > best_score[one_char_end]) {
                    best_score[one_char_end] = cand;
                    back_start[one_char_end] = i;
                    back_id[one_char_end]    = unk_id_;
                }
            }
        }
        (void)first_char_end;
    }

    // Backtrack from the end to recover the token-id sequence.
    std::vector<int32_t> ids;
    std::size_t pos = n;
    while (pos > 0) {
        if (best_score[pos] == NEG_INF) {
            // No path reached `pos` — shouldn't happen with the unk
            // fallback, but guard anyway by stepping back one character.
            break;
        }
        ids.push_back(back_id[pos]);
        pos = back_start[pos];
    }
    std::reverse(ids.begin(), ids.end());
    return ids;
}

// ─── encode ────────────────────────────────────────────────────────────────

std::vector<int32_t> Tokenizer::encode(std::string_view text,
                                       int max_length) const {
    std::vector<int32_t> ids = tokenize(text);

    std::vector<int32_t> out;
    if (max_length <= 0) return out;
    out.reserve(static_cast<std::size_t>(max_length));

    // Content fills up to max_length-1 slots; eos always takes the last used
    // slot. If content + eos overflows, truncate the content.
    const std::size_t content_cap =
        static_cast<std::size_t>(max_length) - 1;
    const std::size_t n =
        (ids.size() > content_cap) ? content_cap : ids.size();
    for (std::size_t i = 0; i < n; ++i) out.push_back(ids[i]);

    out.push_back(eos_id_);
    while (out.size() < static_cast<std::size_t>(max_length)) {
        out.push_back(pad_id_);
    }
    return out;
}

}  // namespace brodiffusion::t5
