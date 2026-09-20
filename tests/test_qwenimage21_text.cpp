// Qwen-Image 2.1 text-conditioning smoke + real-weights test (prompt -> the
// Qwen3-VL-8B last-hidden-state rows the DiT's txt_in consumes).
//
// Three parts:
//   (a) Template invariants — always runs. The fixed span ids and the shape
//       constants the DiT depends on. No tokenizer, no model.
//   (b) Tokenizer-only — gated on weights/qwen-image-2.1/processor. Builds the
//       T2I token sequence for several prompts and asserts the layout
//       (system prefix, user header, content, assistant suffix), the
//       empty-prompt substitution and content truncation. Cheap: no model.
//   (c) Real checkpoint — gated additionally on text_encoder/. Loads the 8B
//       backbone INT8 (BF16 would not leave room next to anything else),
//       encodes a real prompt, and asserts the (n_valid, 4096) layout, the
//       all-ones mask, drop_idx bookkeeping, finiteness and determinism.
//       Numerical parity vs diffusers is scripts/qwenimage21_text_parity.sh
//       (not part of ctest).

#define _CRT_SECURE_NO_WARNINGS   // std::getenv for the gated checkpoint path

#include "brodiffusion/qwenimage21_text.h"
#include "brodiffusion/detail/safetensors_dir.h"

#include "brolm/qwen3vl_config.h"
#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"

#include "brotensor/ops.h"
#include "brotensor/runtime.h"
#include "brotensor/safetensors.h"
#include "brotensor/tensor.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace bt = brotensor;
namespace st = brotensor::safetensors;
namespace q21 = brodiffusion::qwenimage21;

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

static std::string model_root() { return weights_dir() + "/qwen-image-2.1"; }

// ─── Part (a): template invariants ─────────────────────────────────────────

static void test_invariants() {
    CHECK(q21::kTextHiddenDim == 4096);
    CHECK(q21::kDropIdx == 14);

    // Stock Qwen special ids (processor/added_tokens.json): <|im_start|>
    // 151644, <|im_end|> 151645. The whole drop/keep split hangs off these.
    const std::vector<int> prefix = {151644, 8948, 198, 1092, 30782, 408, 323,
                                     23643, 279, 3897, 9934, 13, 151645, 198};
    const std::vector<int> header = {151644, 872, 198};
    const std::vector<int> suffix = {151645, 198, 151644, 77091, 198};
    CHECK(q21::system_prefix_ids() == prefix);
    CHECK(q21::user_header_ids() == header);
    CHECK(q21::assistant_suffix_ids() == suffix);
    CHECK(static_cast<int>(q21::system_prefix_ids().size()) == q21::kDropIdx);
}

// ─── Part (b): tokenizer-only template assembly ────────────────────────────

static void test_template_tokens() {
    const std::string proc = model_root() + "/processor";
    const std::string vocab = proc + "/vocab.json";
    const std::string merges = proc + "/merges.txt";
    if (!std::filesystem::exists(vocab) || !std::filesystem::exists(merges)) {
        std::printf("qwenimage21_text: tokenizer part skipped (no processor)\n");
        return;
    }
    auto tok = brolm::qwen3vl::Tokenizer::load(vocab, merges);

    const std::vector<int>& prefix = q21::system_prefix_ids();
    const std::vector<int>& header = q21::user_header_ids();
    const std::vector<int>& suffix = q21::assistant_suffix_ids();
    const std::size_t fixed = prefix.size() + header.size() + suffix.size();

    auto layout_ok = [&](const std::vector<int>& ids) {
        if (ids.size() < fixed) return false;
        for (std::size_t i = 0; i < prefix.size(); ++i) {
            if (ids[i] != prefix[i]) return false;
        }
        for (std::size_t i = 0; i < header.size(); ++i) {
            if (ids[prefix.size() + i] != header[i]) return false;
        }
        const std::size_t tail = ids.size() - suffix.size();
        for (std::size_t i = 0; i < suffix.size(); ++i) {
            if (ids[tail + i] != suffix[i]) return false;
        }
        return true;
    };

    for (const char* p : {"a red fox",
                          "a sign that reads \"OPEN\"",
                          "multi\nline\tprompt 123"}) {
        auto ids = q21::build_t2i_tokens(tok, p);
        CHECK(layout_ok(ids));
        // The content must be exactly what the prompt alone BPE-encodes to.
        auto alone = tok.encode(p, /*add_special=*/false);
        CHECK(ids.size() == fixed + alone.size());
        bool content_ok = true;
        for (std::size_t i = 0; i < alone.size(); ++i) {
            if (ids[prefix.size() + header.size() + i] !=
                static_cast<int>(alone[i])) content_ok = false;
        }
        CHECK(content_ok);
    }

    // An empty prompt becomes " " — never zero content tokens, since Qwen has
    // no BOS row for the encoder to read.
    auto empty_ids = q21::build_t2i_tokens(tok, "");
    CHECK(layout_ok(empty_ids));
    CHECK(empty_ids.size() > fixed);
    auto space_ids = q21::build_t2i_tokens(tok, " ");
    CHECK(empty_ids == space_ids);

    // Truncation caps the CONTENT only; the fixed spans always survive.
    auto trunc = q21::build_t2i_tokens(
        tok, "one two three four five six seven eight nine ten", /*max=*/3);
    CHECK(trunc.size() == fixed + 3);
    CHECK(layout_ok(trunc));
}

// ─── Part (c): real-weights end-to-end ─────────────────────────────────────

static void test_real_weights() {
    const std::string root = model_root();
    const std::string cfg_json = root + "/text_encoder/config.json";
    const std::string vocab = root + "/processor/vocab.json";
    const std::string merges = root + "/processor/merges.txt";
    if (!std::filesystem::exists(cfg_json) ||
        !std::filesystem::exists(vocab) ||
        !std::filesystem::exists(merges)) {
        std::printf("qwenimage21_text: real-weights skipped (no weights)\n");
        return;
    }

    auto tok = brolm::qwen3vl::Tokenizer::load(vocab, merges);
    auto cfg = brolm::qwen3vl::Qwen3VLConfig::load(cfg_json);
    CHECK(cfg.text.hidden_size == q21::kTextHiddenDim);
    CHECK(cfg.text.num_hidden_layers == 36);
    // INT8: the BF16 8B backbone is ~17 GiB, which a test has no business
    // pinning. Numbers are checked by the parity script, not here.
    cfg.text.quantize_weights = (bt::default_device() != bt::Device::CPU);
    // Never any logits from this backbone, so skip the 1.2 GiB untied head.
    cfg.text.tie_word_embeddings = true;
    brolm::qwen3vl::TextModel model(cfg.text);

    auto files = brodiffusion::detail::open_component_files(
        root + "/text_encoder");
    std::vector<const st::File*> ptrs;
    for (const auto& f : files) ptrs.push_back(&f);
    model.load_weights(ptrs, "model.language_model.");
    bt::sync_all();

    const std::string prompt =
        "a photorealistic red fox sitting in freshly fallen snow";
    auto cond = q21::encode_prompt(tok, model, prompt);
    bt::sync_all();

    const int n_valid = cond.embeds.rows;
    CHECK(cond.drop_idx == q21::kDropIdx);
    CHECK(static_cast<int>(cond.token_ids.size()) == n_valid + q21::kDropIdx);
    CHECK(cond.embeds.cols == q21::kTextHiddenDim);
    CHECK(cond.mask.rows == n_valid);
    CHECK(cond.mask.cols == 1);
    // 3 header + N content + 5 suffix, nothing padded.
    CHECK(n_valid == 3 + 5 + static_cast<int>(
              tok.encode(prompt, /*add_special=*/false).size()));
    std::printf("qwenimage21_text: real-weights n_valid %d\n", n_valid);

    // The mask carries no information at batch 1 — it is all ones.
    std::vector<float> mask = cond.mask.to(bt::Device::CPU).to_host_vector();
    bool all_ones = true;
    for (float m : mask) if (m != 1.0f) all_ones = false;
    CHECK(all_ones);

    // The last five surviving ids are the assistant suffix, in place.
    const std::vector<int>& suffix = q21::assistant_suffix_ids();
    bool suffix_ok = true;
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        if (cond.token_ids[cond.token_ids.size() - suffix.size() + i] !=
            suffix[i]) suffix_ok = false;
    }
    CHECK(suffix_ok);

    auto to_host = [](const bt::Tensor& t) {
        if (t.dtype == bt::Dtype::FP32) return t.to(bt::Device::CPU).to_host_vector();
        bt::Tensor f32;
        bt::cast(t, f32, bt::Dtype::FP32);
        bt::sync_all();
        return f32.to(bt::Device::CPU).to_host_vector();
    };

    std::vector<float> emb = to_host(cond.embeds);
    int nonfinite = 0;
    double sumsq = 0.0;
    for (float v : emb) {
        if (!std::isfinite(v)) ++nonfinite;
        else sumsq += static_cast<double>(v) * v;
    }
    CHECK(nonfinite == 0);
    CHECK(sumsq > 0.0);   // not an all-zero block

    // Determinism across two calls with the same prompt.
    auto cond2 = q21::encode_prompt(tok, model, prompt);
    bt::sync_all();
    CHECK(to_host(cond2.embeds) == emb);

    // A different prompt must produce different conditioning (guards against
    // the prompt tokens being dropped along with the system prefix).
    auto cond3 = q21::encode_prompt(tok, model, "a blue whale underwater");
    bt::sync_all();
    CHECK(cond3.token_ids != cond.token_ids);
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
        test_template_tokens();
        test_real_weights();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "qwenimage21_text: exception: %s\n", e.what());
        return 1;
    }

    if (g_failures == 0) std::printf("qwenimage21_text: OK\n");
    else std::fprintf(stderr, "qwenimage21_text: %d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
