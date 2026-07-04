// Krea 2 text-conditioning smoke + real-weights test (prompt -> tapped
// Qwen3-VL hidden states + validity mask).
//
// Two parts:
//   (a) Template invariants — always runs. Asserts the fixed layer taps and
//       shape constants the DiT's text-fusion stage depends on (12 taps
//       {2,5,...,35}, 2560-wide, 512-token block). No model needed.
//   (b) Real checkpoint — gated on weights/krea-2-raw (text_encoder +
//       tokenizer). Loads the Qwen3-VL-4B backbone and tokenizer, encodes a
//       real prompt, and asserts the (512*12, 2560) layout, dtype, mask
//       structure (content rows then a 5-row suffix at 507..511), all-finite
//       valid rows, zero filler rows, and determinism across two calls.
//       Numerical parity vs diffusers is checked separately by
//       scripts/krea2_text_parity.sh (not part of ctest).

#define _CRT_SECURE_NO_WARNINGS   // std::getenv for the gated checkpoint path

#include "brodiffusion/krea2_text.h"
#include "brodiffusion/detail/compute.h"

#include "brolm/qwen3vl_config.h"
#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"

#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include "test_compute.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace bt = brotensor;
namespace st = brotensor::safetensors;
namespace k2 = brodiffusion::krea2;

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

// ─── Part (a): template invariants ─────────────────────────────────────────

static void test_invariants() {
    const std::vector<int> expect = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35};
    CHECK(k2::text_encoder_select_layers() == expect);
    CHECK(k2::kNumTextLayers == 12);
    CHECK(k2::kTextHiddenDim == 2560);
    CHECK(k2::kMaxSequenceLength == 512);
    CHECK(static_cast<int>(k2::text_encoder_select_layers().size()) ==
          k2::kNumTextLayers);
}

// ─── Part (b): real-weights end-to-end ─────────────────────────────────────

static void test_real_weights() {
    const std::string root = weights_dir() + "/krea-2-raw";
    const std::string te = root + "/text_encoder/model.safetensors";
    const std::string cfg_json = root + "/text_encoder/config.json";
    const std::string vocab = root + "/tokenizer/vocab.json";
    const std::string merges = root + "/tokenizer/merges.txt";
    if (!std::filesystem::exists(te) || !std::filesystem::exists(vocab) ||
        !std::filesystem::exists(merges)) {
        std::printf("krea2_text: skipped (no weights)\n");
        return;
    }

    auto tok = brolm::qwen3vl::Tokenizer::load(vocab, merges);
    auto cfg = brolm::qwen3vl::Qwen3VLConfig::load(cfg_json);
    brolm::qwen3vl::TextModel model(cfg.text);
    auto f = st::File::open(te);
    model.load_weights(f, "language_model.");

    const std::string prompt =
        "a photorealistic red fox sitting in freshly fallen snow";
    auto cond = k2::encode_prompt(tok, model, prompt);
    bt::sync_all();

    const int rows = k2::kMaxSequenceLength * k2::kNumTextLayers;
    CHECK(cond.prompt_embeds.rows == rows);
    CHECK(cond.prompt_embeds.cols == k2::kTextHiddenDim);
    CHECK(cond.prompt_embeds.dtype == brodiffusion::compute_dtype());
    CHECK(cond.prompt_embeds_mask.rows == k2::kMaxSequenceLength);
    CHECK(cond.prompt_embeds_mask.cols == 1);

    std::vector<float> mask = cond.prompt_embeds_mask.to_host_vector();
    int valid = 0;
    for (float m : mask) valid += (m > 0.5f) ? 1 : 0;
    // Content tokens (>0) plus the 5-token suffix at rows 507..511.
    CHECK(valid > 5 && valid < 512);
    for (int s = 0; s < 5; ++s) CHECK(mask[static_cast<std::size_t>(507 + s)] > 0.5f);
    // The gap between content and suffix must be filler (mask 0).
    CHECK(mask[506] < 0.5f);
    std::printf("krea2_text: real-weights valid tokens %d\n", valid);

    // Valid rows finite; filler rows exactly zero.
    std::vector<float> emb;
    {
        bt::Tensor f32;
        if (cond.prompt_embeds.dtype != bt::Dtype::FP32) {
            bt::cast(cond.prompt_embeds, f32, bt::Dtype::FP32);
            bt::sync_all();
            emb = f32.to_host_vector();
        } else {
            emb = cond.prompt_embeds.to_host_vector();
        }
    }
    const int D = k2::kTextHiddenDim;
    int nonfinite = 0, filler_nonzero = 0;
    for (int t = 0; t < k2::kMaxSequenceLength; ++t) {
        const bool valid_tok = mask[static_cast<std::size_t>(t)] > 0.5f;
        for (int l = 0; l < k2::kNumTextLayers; ++l) {
            const std::size_t base =
                (static_cast<std::size_t>(t) * k2::kNumTextLayers + l) * D;
            for (int c = 0; c < D; ++c) {
                const float v = emb[base + static_cast<std::size_t>(c)];
                if (valid_tok) { if (!std::isfinite(v)) ++nonfinite; }
                else if (v != 0.0f) ++filler_nonzero;
            }
        }
    }
    CHECK(nonfinite == 0);
    CHECK(filler_nonzero == 0);

    // Determinism.
    auto cond2 = k2::encode_prompt(tok, model, prompt);
    bt::sync_all();
    std::vector<float> emb2;
    {
        bt::Tensor f32;
        if (cond2.prompt_embeds.dtype != bt::Dtype::FP32) {
            bt::cast(cond2.prompt_embeds, f32, bt::Dtype::FP32);
            bt::sync_all();
            emb2 = f32.to_host_vector();
        } else {
            emb2 = cond2.prompt_embeds.to_host_vector();
        }
    }
    CHECK(emb == emb2);
}

int main() {
    try {
        bt::init();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "init failed: %s\n", e.what());
        return 1;
    }

    test_invariants();
    try {
        test_real_weights();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "krea2_text: real-weights test exception: %s\n", e.what());
        return 1;
    }

    if (g_failures == 0) std::printf("krea2_text: OK\n");
    else std::fprintf(stderr, "krea2_text: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
