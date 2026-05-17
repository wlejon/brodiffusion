// Pipeline construction smoke test.
//
// Verifies that Pipeline constructs from sub-module configs and a tokenizer,
// and that generate() rejects malformed image dimensions before touching any
// weights. End-to-end generate() requires combined CLIP+UNet+VAE weights —
// that lives in a separate real-weights integration test.

#include "brodiffusion/pipeline.h"
#include "brodiffusion/tokenizer.h"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>

namespace pl   = brodiffusion::pipeline;
namespace clip = brodiffusion::clip;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

static void write_tiny_vocab(const std::filesystem::path& vp,
                              const std::filesystem::path& mp) {
    std::ofstream(vp, std::ios::binary | std::ios::trunc) << "{\"a\":1,\"a</w>\":2}";
    std::ofstream(mp) << "#version: test\n";
}

int main() {
    auto tmp = std::filesystem::temp_directory_path();
    auto vp = tmp / "brodiffusion_pipeline_vocab.json";
    auto mp = tmp / "brodiffusion_pipeline_merges.txt";
    write_tiny_vocab(vp, mp);
    auto tok = clip::Tokenizer::load(vp.string(), mp.string());

    pl::PipelineConfig cfg;
    pl::Pipeline pipeline(cfg, std::move(tok));
    CHECK(pipeline.config().unet.in_channels == 4);

    // generate() with bad dimensions should reject before touching weights.
    pl::GenerateOptions bad;
    bad.height = 7;   // not a multiple of 8
    bad.width  = 8;
    bool threw = false;
    try {
        pipeline.generate("hi", bad);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    return g_failures == 0 ? 0 : 1;
}
