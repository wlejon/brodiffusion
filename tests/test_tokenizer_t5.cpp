// T5 SentencePiece Unigram tokenizer smoke test.
//
// Writes a synthetic HF tokenizer.json with a small Unigram vocab where the
// optimal segmentation is known, loads it, and verifies that the Viterbi
// segmentation, unknown-char fallback, and encode() framing behave as
// specified.

#include "brodiffusion/tokenizer_t5.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace t5 = brodiffusion::t5;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// Build a tiny Unigram tokenizer.json. The metaspace character U+2581 is
// written as its UTF-8 bytes (E2 96 81). The vocab list index IS the token id.
//   0:<pad>:0  1:</s>:0  2:<unk>:0  3:▁:-2.0  4:▁a:-3.0  5:a:-6.0
//   6:cat:-3.5  7:▁cat:-3.0  8:c:-7  9:at:-7
static void write_fixture(const std::filesystem::path& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    const std::string M = "\xE2\x96\x81";  // U+2581 metaspace
    f << "{";
    f << "\"model\":{";
    f << "\"type\":\"Unigram\",";
    f << "\"unk_id\":2,";
    f << "\"vocab\":[";
    f << "[\"<pad>\",0.0],";
    f << "[\"</s>\",0.0],";
    f << "[\"<unk>\",0.0],";
    f << "[\"" << M << "\",-2.0],";
    f << "[\"" << M << "a\",-3.0],";
    f << "[\"a\",-6.0],";
    f << "[\"cat\",-3.5],";
    f << "[\"" << M << "cat\",-3.0],";
    f << "[\"c\",-7.0],";
    f << "[\"at\",-7.0]";
    f << "]";
    f << "}";
    f << "}";
}

int main() {
    auto tmp = std::filesystem::temp_directory_path();
    auto path = tmp / "brodiffusion_t5_tokenizer.json";
    write_fixture(path);

    auto tok = t5::Tokenizer::load(path.string());

    CHECK(tok.vocab_count() == 10);
    CHECK(tok.pad_id() == 0);
    CHECK(tok.eos_id() == 1);
    CHECK(tok.unk_id() == 2);

    // "a cat" -> metaspace "▁a▁cat". Candidate segmentations:
    //   [▁a][▁cat]      = -3.0 + -3.0  = -6.0   (best)
    //   [▁a][▁][cat]    = -3.0 -2.0 -3.5 = -8.5
    //   [▁][a][▁][cat]  = -2 -6 -2 -3.5 = -13.5
    // Expect ids {4, 7}.
    {
        auto ids = tok.tokenize("a cat");
        CHECK(ids.size() == 2);
        if (ids.size() == 2) {
            CHECK(ids[0] == 4);
            CHECK(ids[1] == 7);
        }
    }

    // An unknown character must route through unk_id. "z" is not in the
    // vocab; "▁z" -> [▁][<unk>], so the result contains unk_id (2).
    {
        auto ids = tok.tokenize("z");
        bool saw_unk = false;
        for (int32_t id : ids) if (id == tok.unk_id()) saw_unk = true;
        CHECK(saw_unk);
    }

    // encode("a cat", 8) -> {4, 7, 1, 0, 0, 0, 0, 0}: pieces, eos, pad-filled.
    {
        auto ids = tok.encode("a cat", 8);
        CHECK(ids.size() == 8);
        const int32_t expected[8] = {4, 7, 1, 0, 0, 0, 0, 0};
        for (std::size_t i = 0; i < ids.size() && i < 8; ++i) {
            CHECK(ids[i] == expected[i]);
        }
    }

    // encode() with content + eos overflowing max_length truncates with eos
    // kept in the last slot.
    {
        auto ids = tok.encode("a cat", 2);
        CHECK(ids.size() == 2);
        if (ids.size() == 2) {
            CHECK(ids[0] == 4);
            CHECK(ids[1] == tok.eos_id());
        }
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    if (g_failures == 0) std::printf("tokenizer_t5: OK\n");
    else std::fprintf(stderr, "tokenizer_t5: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
