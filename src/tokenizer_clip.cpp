#include "brodiffusion/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace brodiffusion::clip {

// ─── GPT-2 / CLIP byte→unicode mapping ─────────────────────────────────────
//
// CLIP's BPE works in a "byte-level unicode" space: every byte 0..255 maps to
// a unicode codepoint, and the resulting UTF-8 string is what BPE merges
// operate on. Visible printable bytes map to themselves; the rest map to
// codepoints starting at U+0100 in the order they appear.

static void encode_codepoint_utf8(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

static void build_byte_to_unicode(std::string out[256]) {
    // Bytes that map to themselves (codepoint == byte value).
    bool self_map[256] = {false};
    auto mark_range = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b) self_map[b] = true;
    };
    mark_range(33, 126);
    mark_range(161, 172);
    mark_range(174, 255);

    int next_cp = 256;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = self_map[b] ? static_cast<uint32_t>(b)
                                  : static_cast<uint32_t>(next_cp++);
        out[b].clear();
        encode_codepoint_utf8(cp, out[b]);
    }
}

// ─── Tiny JSON loader for vocab.json ───────────────────────────────────────
//
// vocab.json is a flat object: `{"token_string": id, ...}`. Strings may
// contain \", \\, \/, \n, \t, \r, \b, \f, \uXXXX. Surrogate pairs (\uD8xx
// \uDCxx) are not expected in CLIP vocab (all tokens fit in the BMP); we
// pass them through as-is if encountered.

namespace {

struct JsonReader {
    const char* p;
    const char* end;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("vocab.json: " + msg);
    }

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }
    bool eat(char c) {
        skip_ws();
        if (p < end && *p == c) { ++p; return true; }
        return false;
    }
    void expect(char c) {
        if (!eat(c)) fail(std::string("expected '") + c + "'");
    }

    std::string parse_string() {
        skip_ws();
        if (p >= end || *p != '"') fail("expected string");
        ++p;
        std::string out;
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                char e = p[1];
                p += 2;
                switch (e) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'n':  out += '\n'; break;
                    case 't':  out += '\t'; break;
                    case 'r':  out += '\r'; break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'u': {
                        if (p + 4 > end) fail("truncated \\u escape");
                        uint32_t cp = 0;
                        for (int i = 0; i < 4; ++i) {
                            char c = p[i];
                            int v = (c >= '0' && c <= '9') ? c - '0'
                                  : (c >= 'a' && c <= 'f') ? c - 'a' + 10
                                  : (c >= 'A' && c <= 'F') ? c - 'A' + 10
                                  : -1;
                            if (v < 0) fail("bad hex in \\u escape");
                            cp = (cp << 4) | static_cast<uint32_t>(v);
                        }
                        p += 4;
                        encode_codepoint_utf8(cp, out);
                        break;
                    }
                    default: fail("unsupported escape");
                }
            } else {
                out += *p++;
            }
        }
        if (p >= end) fail("unterminated string");
        ++p;
        return out;
    }

    int32_t parse_int() {
        skip_ws();
        bool neg = false;
        if (p < end && *p == '-') { neg = true; ++p; }
        if (p >= end || *p < '0' || *p > '9') fail("expected integer");
        int64_t v = 0;
        while (p < end && *p >= '0' && *p <= '9') {
            v = v * 10 + (*p - '0');
            ++p;
        }
        return static_cast<int32_t>(neg ? -v : v);
    }
};

}  // namespace

static std::unordered_map<std::string, int32_t>
load_vocab_json(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("vocab.json: cannot open '" + path + "'");
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();

    JsonReader r{text.data(), text.data() + text.size()};
    r.expect('{');
    std::unordered_map<std::string, int32_t> vocab;
    bool first = true;
    while (true) {
        r.skip_ws();
        if (r.eat('}')) break;
        if (!first) r.expect(',');
        first = false;
        std::string key = r.parse_string();
        r.expect(':');
        int32_t id = r.parse_int();
        vocab.emplace(std::move(key), id);
    }
    return vocab;
}

// ─── merges.txt loader ─────────────────────────────────────────────────────

static std::unordered_map<std::string, int32_t>
load_merges_txt(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("merges.txt: cannot open '" + path + "'");

    std::unordered_map<std::string, int32_t> ranks;
    std::string line;
    int32_t rank = 0;
    bool first = true;
    while (std::getline(f, line)) {
        // Strip trailing \r (Windows line endings).
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (first && line.size() >= 1 && line[0] == '#') {
            first = false;
            continue;
        }
        first = false;
        auto sp = line.find(' ');
        if (sp == std::string::npos) {
            throw std::runtime_error("merges.txt: malformed line: '" + line + "'");
        }
        std::string a = line.substr(0, sp);
        std::string b = line.substr(sp + 1);
        std::string key = a + '\x01' + b;
        ranks.emplace(std::move(key), rank++);
    }
    return ranks;
}

// ─── BPE ───────────────────────────────────────────────────────────────────

namespace {

// Split a UTF-8 string into a list of unicode codepoints (each as its own
// UTF-8-encoded substring). The byte-encoded input is always valid UTF-8 by
// construction.
std::vector<std::string> split_codepoints(const std::string& s) {
    std::vector<std::string> out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t n = 1;
        if      ((c & 0x80) == 0x00) n = 1;
        else if ((c & 0xE0) == 0xC0) n = 2;
        else if ((c & 0xF0) == 0xE0) n = 3;
        else if ((c & 0xF8) == 0xF0) n = 4;
        out.emplace_back(s.substr(i, n));
        i += n;
    }
    return out;
}

}  // namespace

std::vector<std::string> Tokenizer::bpe_(const std::string& token) const {
    if (token.empty()) return {};

    // word = codepoints, with </w> glued onto the last one.
    std::vector<std::string> word = split_codepoints(token);
    if (word.empty()) return {};
    word.back() += "</w>";
    if (word.size() == 1) return word;

    while (true) {
        // Find the lowest-rank adjacent pair.
        int best_rank = -1;
        std::size_t best_i = 0;
        for (std::size_t i = 0; i + 1 < word.size(); ++i) {
            std::string key = word[i];
            key += '\x01';
            key += word[i + 1];
            auto it = merge_ranks_.find(key);
            if (it != merge_ranks_.end()) {
                if (best_rank < 0 || it->second < best_rank) {
                    best_rank = it->second;
                    best_i = i;
                }
            }
        }
        if (best_rank < 0) break;

        // Merge every non-overlapping occurrence of (word[best_i],
        // word[best_i+1]) — matches the canonical BPE pass which scans the
        // whole word once after picking the best pair.
        const std::string a = word[best_i];
        const std::string b = word[best_i + 1];
        std::vector<std::string> next;
        next.reserve(word.size());
        std::size_t i = 0;
        while (i < word.size()) {
            if (i + 1 < word.size() && word[i] == a && word[i + 1] == b) {
                next.emplace_back(a + b);
                i += 2;
            } else {
                next.emplace_back(word[i]);
                i += 1;
            }
        }
        word = std::move(next);
        if (word.size() == 1) break;
    }
    return word;
}

// ─── Pre-tokenization (ASCII-focused) ──────────────────────────────────────
//
// HF CLIP regex (simplified):
//   <|startoftext|> | <|endoftext|>
//   | 's | 't | 're | 've | 'm | 'll | 'd
//   | letters+ | digit | non-whitespace-non-letter-non-digit+
//
// We approximate this for ASCII; non-ASCII bytes are lumped into the
// "punctuation run" category (still flow through byte-encoded BPE correctly,
// just don't get the same chunking as HF's Unicode-aware split).

namespace {

bool is_ascii_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
bool is_ascii_digit(unsigned char c) { return c >= '0' && c <= '9'; }
bool is_ascii_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

bool starts_with(std::string_view s, std::size_t i, const char* lit) {
    std::size_t n = std::strlen(lit);
    if (i + n > s.size()) return false;
    return std::memcmp(s.data() + i, lit, n) == 0;
}

// Lowercase ASCII in place; non-ASCII bytes passed through.
std::string ascii_lower(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        unsigned char u = static_cast<unsigned char>(c);
        if (u >= 'A' && u <= 'Z') u = u - 'A' + 'a';
        out += static_cast<char>(u);
    }
    return out;
}

// Collapse runs of whitespace to a single space and strip leading/trailing
// whitespace. HF's `_clean_text` does roughly this.
std::string collapse_ws(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    bool prev_space = true;  // suppresses leading whitespace
    for (char c : s) {
        if (is_ascii_space(static_cast<unsigned char>(c))) {
            if (!prev_space) out += ' ';
            prev_space = true;
        } else {
            out += c;
            prev_space = false;
        }
    }
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::vector<std::string> pre_tokenize(std::string_view text) {
    std::vector<std::string> pieces;
    std::size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (is_ascii_space(c)) { ++i; continue; }

        // Contractions: ' followed by s|t|re|ve|m|ll|d (lowercased input).
        if (c == '\'') {
            // Try multi-char ones first.
            if (starts_with(text, i, "'re")) { pieces.emplace_back("'re"); i += 3; continue; }
            if (starts_with(text, i, "'ve")) { pieces.emplace_back("'ve"); i += 3; continue; }
            if (starts_with(text, i, "'ll")) { pieces.emplace_back("'ll"); i += 3; continue; }
            if (starts_with(text, i, "'s"))  { pieces.emplace_back("'s");  i += 2; continue; }
            if (starts_with(text, i, "'t"))  { pieces.emplace_back("'t");  i += 2; continue; }
            if (starts_with(text, i, "'m"))  { pieces.emplace_back("'m");  i += 2; continue; }
            if (starts_with(text, i, "'d"))  { pieces.emplace_back("'d");  i += 2; continue; }
            // Bare apostrophe falls into the punct branch below.
        }

        if (is_ascii_letter(c)) {
            std::size_t j = i;
            while (j < text.size() &&
                   is_ascii_letter(static_cast<unsigned char>(text[j]))) ++j;
            pieces.emplace_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        if (is_ascii_digit(c)) {
            // HF emits one digit at a time.
            pieces.emplace_back(text.substr(i, 1));
            i += 1;
            continue;
        }

        // Everything else: run of non-whitespace, non-letter, non-digit bytes.
        std::size_t j = i;
        while (j < text.size()) {
            unsigned char u = static_cast<unsigned char>(text[j]);
            if (is_ascii_space(u) || is_ascii_letter(u) || is_ascii_digit(u)) break;
            // Also break on apostrophe-contraction lookahead so "don't" splits
            // cleanly between "don" / "'t".
            if (u == '\'' && j != i) {
                std::string_view rest = text.substr(j);
                if (rest.substr(0, 3) == "'re" || rest.substr(0, 3) == "'ve" ||
                    rest.substr(0, 3) == "'ll" ||
                    rest.substr(0, 2) == "'s"  || rest.substr(0, 2) == "'t"  ||
                    rest.substr(0, 2) == "'m"  || rest.substr(0, 2) == "'d") {
                    break;
                }
            }
            ++j;
        }
        pieces.emplace_back(text.substr(i, j - i));
        i = j;
    }
    return pieces;
}

}  // namespace

// ─── Tokenizer driver ──────────────────────────────────────────────────────

Tokenizer Tokenizer::load(const std::string& vocab_json_path,
                          const std::string& merges_txt_path) {
    Tokenizer t;
    t.vocab_ = load_vocab_json(vocab_json_path);
    t.merge_ranks_ = load_merges_txt(merges_txt_path);
    build_byte_to_unicode(t.byte_to_unicode_);
    return t;
}

void Tokenizer::encode_piece_(std::string_view piece,
                              std::vector<int32_t>& out) const {
    if (piece.empty()) return;

    // Byte-encode the piece into the GPT-2/CLIP unicode space.
    std::string encoded;
    encoded.reserve(piece.size() * 2);
    for (char c : piece) {
        encoded += byte_to_unicode_[static_cast<unsigned char>(c)];
    }

    auto units = bpe_(encoded);
    for (const auto& u : units) {
        auto it = vocab_.find(u);
        if (it != vocab_.end()) out.push_back(it->second);
        // Unknown unit: drop. With a complete CLIP vocab this can't happen
        // because every single byte-encoded character is itself in the vocab.
    }
}

std::vector<int32_t> Tokenizer::tokenize(std::string_view text) const {
    std::string cleaned = collapse_ws(ascii_lower(text));
    auto pieces = pre_tokenize(cleaned);
    std::vector<int32_t> ids;
    ids.reserve(pieces.size() * 2);
    for (const auto& p : pieces) {
        encode_piece_(p, ids);
    }
    return ids;
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
    auto ids = tokenize(text);

    std::vector<int32_t> out;
    out.reserve(max_length);
    out.push_back(bos_id);

    const std::size_t content_cap = static_cast<std::size_t>(max_length) - 2;
    bool truncated = ids.size() > content_cap;
    std::size_t n = truncated ? content_cap : ids.size();
    for (std::size_t i = 0; i < n; ++i) out.push_back(ids[i]);

    out.push_back(eos_id);
    while (out.size() < static_cast<std::size_t>(max_length)) out.push_back(pad_id);
    return out;
}

}  // namespace brodiffusion::clip
