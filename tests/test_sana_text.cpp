// Sana Gemma-2 text encoder smoke test (prompt -> last_hidden_state).
//
// Two parts:
//   (a) JSON -Infinity round-trip — always runs. Parses a tiny document with a
//       Python-json `-Infinity` literal through brodiffusion's bundled parser
//       and asserts it maps to -inf (the fix that lets real Sana scheduler
//       configs, which carry "lambda_min_clipped": -Infinity, load).
//   (b) Real checkpoint — gated on weights/sana-600m. When the text encoder or
//       tokenizer is absent it prints "skipped (no weights)" and exits 0. When
//       present it: loads the model config via load_model_config (which also
//       parses the scheduler's -Infinity), builds the Gemma tokenizer, the
//       Gemma2Model, loads the 2 fp16 shards, and encodes a real prompt plus an
//       empty negative prompt. Asserts (L, 2304) with L>10 (the CHI makes it
//       long), all-finite, and determinism across two calls.

#define _CRT_SECURE_NO_WARNINGS   // std::getenv for the gated checkpoint path

#include "brodiffusion/sana_text.h"
#include "brodiffusion/model_config.h"
#include "brodiffusion/detail/json.h"

#include "brolm/gemma2.h"
#include "brolm/gemma_tokenizer.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace bt   = brotensor;
namespace st   = brotensor::safetensors;
namespace gm   = brolm::gemma;
namespace sana = brodiffusion::sana;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

#ifndef BRODIFFUSION_WEIGHTS_DIR
#define BRODIFFUSION_WEIGHTS_DIR ""
#endif

static std::string weights_dir() {
    if (const char* e = std::getenv("BRODIFFUSION_WEIGHTS_DIR")) {
        if (e[0]) return e;
    }
    return BRODIFFUSION_WEIGHTS_DIR;
}

static void run_json_infinity() {
    namespace json = brodiffusion::detail::json;
    auto doc = json::parse(R"({"a": -Infinity, "b": Infinity, "c": NaN})");
    const double a = doc.at("a").as_number();
    const double b = doc.at("b").as_number();
    const double c = doc.at("c").as_number();
    CHECK(std::isinf(a) && a < 0.0);
    CHECK(std::isinf(b) && b > 0.0);
    CHECK(std::isnan(c));
    std::printf("sana_text: json -Infinity round-trip OK\n");
}

static int finite_count_check(const std::vector<float>& h) {
    int nonfinite = 0;
    for (float v : h) if (!bdtest::bd_finite(v)) ++nonfinite;
    return nonfinite;
}

static void run_real_checkpoint() {
    const std::string root   = weights_dir() + "/sana-600m";
    const std::string te_dir = root + "/text_encoder";
    const std::string tk_json = root + "/tokenizer/tokenizer.json";
    if (!std::filesystem::exists(te_dir) ||
        !std::filesystem::exists(tk_json)) {
        std::printf("sana_text: skipped (no weights)\n");
        return;
    }

    try {
        // Exercises the JSON -Infinity fix via the scheduler config.
        brodiffusion::ModelConfig mc = brodiffusion::load_model_config(root);
        CHECK(mc.model_class == brodiffusion::ModelClass::Sana);
        CHECK(mc.gemma.hidden_size == 2304);
        std::printf("sana_text: load_model_config OK (gemma hidden=%d layers=%d "
                    "vocab=%d)\n", mc.gemma.hidden_size,
                    mc.gemma.num_hidden_layers, mc.gemma.vocab_size);

        auto tok = gm::Tokenizer::load(tk_json);

        // Collect the *.safetensors shards in text_encoder/.
        std::vector<std::string> shard_paths;
        for (const auto& entry : std::filesystem::directory_iterator(te_dir)) {
            if (entry.path().extension() == ".safetensors")
                shard_paths.push_back(entry.path().string());
        }
        CHECK(!shard_paths.empty());
        std::vector<st::File> files;
        files.reserve(shard_paths.size());
        for (const auto& sp : shard_paths) files.push_back(st::File::open(sp));
        std::vector<const st::File*> shard_ptrs;
        for (const auto& f : files) shard_ptrs.push_back(&f);

        gm::Gemma2Model gemma(mc.gemma);
        gemma.load_weights(shard_ptrs);   // prefix "" — keys are unprefixed
        std::printf("sana_text: loaded Gemma-2 text encoder (%zu shard(s))\n",
                    shard_paths.size());

        // Positive prompt.
        bt::Tensor pos = sana::encode_prompt(gemma, tok, "a photo of a cat");
        bt::sync_all();
        CHECK(pos.cols == 2304);
        CHECK(pos.rows >= 1);
        std::vector<float> ph = bdtest::bd_download(pos);
        CHECK(finite_count_check(ph) == 0);
        std::printf("sana_text: positive prompt -> (%d, %d)\n",
                    pos.rows, pos.cols);

        // Determinism: a second identical call must reproduce the hidden states.
        bt::Tensor pos2 = sana::encode_prompt(gemma, tok, "a photo of a cat");
        bt::sync_all();
        CHECK(pos2.rows == pos.rows && pos2.cols == pos.cols);
        std::vector<float> ph2 = bdtest::bd_download(pos2);
        CHECK(ph == ph2);

        // Empty negative prompt — encoded the same way (CHI only).
        bt::Tensor neg = sana::encode_prompt(gemma, tok, "");
        bt::sync_all();
        CHECK(neg.cols == 2304);
        CHECK(neg.rows >= 1);
        std::vector<float> nh = bdtest::bd_download(neg);
        CHECK(finite_count_check(nh) == 0);
        std::printf("sana_text: negative (empty) prompt -> (%d, %d)\n",
                    neg.rows, neg.cols);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sana_text: exception: %s\n", e.what());
        ++g_failures;
    }
}

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    run_json_infinity();
    run_real_checkpoint();

    if (g_failures == 0) std::printf("sana_text: OK\n");
    else std::fprintf(stderr, "sana_text: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
