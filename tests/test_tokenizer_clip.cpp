#include "brodiffusion/tokenizer.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace clip = brodiffusion::clip;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// Build a tiny vocab.json + merges.txt that exercises:
//   - single-character byte tokens (lowercase a-z + space marker Ġ)
//   - </w> end-of-word suffix
//   - multi-step BPE merge cascade ("hi" -> "h" "i</w>" -> "hi</w>")
//
// IDs are arbitrary; we don't need them to match real CLIP.
static void write_fixture(const std::filesystem::path& vocab_path,
                          const std::filesystem::path& merges_path) {
    {
        std::ofstream f(vocab_path, std::ios::binary | std::ios::trunc);
        f << "{";
        // Per-character entries we need. The byte-encoded space is Ġ
        // (U+0120, UTF-8 C4 A0). The </w> suffix is ASCII.
        bool first = true;
        auto add = [&](const std::string& tok, int id) {
            if (!first) f << ",";
            first = false;
            f << "\"";
            for (char c : tok) {
                if (c == '"' || c == '\\') f << '\\' << c;
                else f << c;
            }
            f << "\":" << id;
        };
        add("h",       1);
        add("i",       2);
        add("c",       3);
        add("a",       4);
        add("t",       5);
        add("h</w>",   6);
        add("i</w>",   7);
        add("c</w>",   8);
        add("a</w>",   9);
        add("t</w>",  10);
        add("hi</w>", 11);
        add("cat</w>",12);
        add("ca",     13);
        add("\xc4\xa0",  14);    // U+0120 'Ġ' — byte-encoded space (no </w>)
        add("!</w>",  15);
        add("!",      16);
        f << "}";
    }
    {
        std::ofstream f(merges_path);
        f << "#version: test\n";
        // Rank 0: merge 'h' + 'i</w>' -> 'hi</w>'
        f << "h i</w>\n";
        // Rank 1: merge 'c' + 'a' -> 'ca'
        f << "c a\n";
        // Rank 2: merge 'ca' + 't</w>' -> 'cat</w>'
        f << "ca t</w>\n";
    }
}

int main() {
    auto tmp = std::filesystem::temp_directory_path();
    auto vp = tmp / "brodiffusion_clip_vocab.json";
    auto mp = tmp / "brodiffusion_clip_merges.txt";
    write_fixture(vp, mp);

    auto tok = clip::Tokenizer::load(vp.string(), mp.string());

    // Vocab count includes both forms: bare and </w>-suffixed entries.
    CHECK(tok.vocab_count() > 0);
    CHECK(tok.merge_count() == 3);

    // "hi" → ['h', 'i</w>'] then merge rank 0 → ['hi</w>'] → [11]
    {
        auto ids = tok.tokenize("hi");
        CHECK(ids.size() == 1);
        CHECK(ids[0] == 11);
    }

    // "cat" → ['c','a','t</w>'] → rank 1 merges (c,a) → ['ca','t</w>']
    //       → rank 2 merges (ca,t</w>) → ['cat</w>'] → [12]
    {
        auto ids = tok.tokenize("cat");
        CHECK(ids.size() == 1);
        CHECK(ids[0] == 12);
    }

    // "hi cat" → "hi" + "cat" pieces (space is dropped by pre-tokenizer).
    // Per HF behavior, multiple words just produce their tokens concatenated;
    // the space is encoded into the leading byte of the next pre-token via
    // the byte-encoder only when the pre-tokenizer keeps it. In our ASCII
    // pre-tokenizer we drop whitespace, matching the *common* SD prompt path
    // where consecutive content words each tokenize independently.
    {
        auto ids = tok.tokenize("hi cat");
        CHECK(ids.size() == 2);
        CHECK(ids[0] == 11);
        CHECK(ids[1] == 12);
    }

    // Case-insensitivity: "HI" → same as "hi".
    {
        auto ids = tok.tokenize("HI");
        CHECK(ids.size() == 1);
        CHECK(ids[0] == 11);
    }

    // Whitespace collapse: leading/trailing/repeated whitespace is dropped.
    {
        auto ids = tok.tokenize("   hi   cat   ");
        CHECK(ids.size() == 2);
        CHECK(ids[0] == 11);
        CHECK(ids[1] == 12);
    }

    // Punctuation: "!" is a non-letter/non-digit byte; pre-tokenizer keeps
    // it. After BPE on byte-encoded "!", with no merges, we look up "!</w>"
    // (15). Punctuation-as-its-own-run still gets </w>.
    {
        auto ids = tok.tokenize("!");
        CHECK(ids.size() == 1);
        CHECK(ids[0] == 15);
    }

    // encode() framing: starts with BOS, ends with EOS, padded to 77 with PAD.
    {
        auto ids = tok.encode("hi cat");
        CHECK(ids.size() == static_cast<std::size_t>(clip::max_length));
        CHECK(ids[0] == clip::bos_id);
        CHECK(ids[1] == 11);
        CHECK(ids[2] == 12);
        CHECK(ids[3] == clip::eos_id);
        // Padding is EOS, all the way to the end.
        for (std::size_t i = 4; i < ids.size(); ++i) {
            CHECK(ids[i] == clip::pad_id);
        }
    }

    // Empty input: BOS, immediate EOS, then padding.
    {
        auto ids = tok.encode("");
        CHECK(ids.size() == static_cast<std::size_t>(clip::max_length));
        CHECK(ids[0] == clip::bos_id);
        CHECK(ids[1] == clip::eos_id);
        for (std::size_t i = 2; i < ids.size(); ++i) {
            CHECK(ids[i] == clip::pad_id);
        }
    }

    // Truncation: produce more than 75 content tokens by repeating "hi".
    // Result length should still be max_length, last slot should be EOS.
    {
        std::string long_text;
        for (int i = 0; i < 100; ++i) long_text += "hi ";
        auto ids = tok.encode(long_text);
        CHECK(ids.size() == static_cast<std::size_t>(clip::max_length));
        CHECK(ids[0] == clip::bos_id);
        CHECK(ids[clip::max_length - 1] == clip::eos_id);
    }

    std::error_code ec;
    std::filesystem::remove(vp, ec);
    std::filesystem::remove(mp, ec);

    if (g_failures == 0) std::printf("tokenizer_clip: OK\n");
    else std::fprintf(stderr, "tokenizer_clip: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
