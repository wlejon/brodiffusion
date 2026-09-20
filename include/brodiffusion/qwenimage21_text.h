#pragma once

// Qwen-Image 2.1 prompt → Qwen3-VL-8B last-hidden-state text conditioning.
//
// Reproduces diffusers' `QwenImage21Pipeline._get_qwen_prompt_embeds` for a
// single prompt. Unlike Krea 2 (which taps 12 intermediate decoder layers and
// fuses them), 2.1 conditions on ONE tensor: the output of the LAST decoder
// layer, taken BEFORE the text model's final RMSNorm, with the leading
// system-role rows dropped.
//
// ── The template, and what survives it ──────────────────────────────────────
//
//   "<|im_start|>system\n"
//   "Comprehend and analyze the provided prompt.<|im_end|>\n"     <- dropped
//   "<|im_start|>user\n{prompt}<|im_end|>\n"                      <- kept
//   "<|im_start|>assistant\n"                                     <- kept
//
// The reference computes the drop count by tokenizing the system message on
// its own through the processor's chat template; for this checkpoint that is
// 14 tokens, which is exactly the "<|im_start|>system\n…<|im_end|>\n" prefix
// above. Everything after it is kept — including the trailing
// "<|im_end|>\n<|im_start|>assistant\n", which the reference does NOT strip
// (Krea 2's encoder, by contrast, relocates its suffix). So the valid
// conditioning for a prompt of N content tokens is
//
//     3 (<|im_start|>user\n) + N + 5 (<|im_end|>\n<|im_start|>assistant\n)
//
// rows, in that order, with no padding anywhere. The whole template — prefix
// included — is run through the encoder first: attention is causal, so the
// kept rows sit at absolute positions 14…L-1 and their RoPE phases and keys
// depend on the prefix being there. Dropping happens only on the output.
//
// ── Special-token ids ───────────────────────────────────────────────────────
//
// This checkpoint uses the STOCK Qwen ids (verified against
// processor/added_tokens.json): <|im_start|> 151644, <|im_end|> 151645,
// <|image_pad|> 151655, <|vision_start|> 151652, <|vision_end|> 151653.
// Since the template's fixed spans are compile-time-constant strings, their
// ids are constant too — they are hard-coded here and only the user prompt is
// BPE-encoded, which is byte-identical between brolm and HF (checked across
// prompts containing punctuation, digits, quotes, newlines and tabs: the
// template's ids always equal prefix ++ tokenize(prompt) ++ suffix).
//
// ── The edit path ───────────────────────────────────────────────────────────
//
// Image-conditioned generation uses the SAME encoder with a different template
// ("<image1><|vision_start|><|image_pad|><|vision_end|>{prompt}" in the user
// turn) and the `<|image_pad|>` rows' input embeddings replaced by vision-tower
// output. encode_tokens() is the seam for that: hand it the token sequence and
// an `EncodeOverrides` carrying the substituted input embeddings, the 3-axis
// M-RoPE positions and the DeepStack splices, and it returns the same
// TextConditioning shape. The vision tower itself is not wired up here.

#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"
#include "brotensor/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brodiffusion::qwenimage21 {

// Qwen3-VL-8B hidden width == the DiT's context_in_dim.
constexpr int kTextHiddenDim = 4096;

// Number of leading (system-role) rows dropped from the encoder output.
// diffusers derives this from the tokenized system message; for this
// checkpoint's processor it is 14. Asserted at encode time.
constexpr int kDropIdx = 14;

// Upper bound on the PROMPT's own token count. The reference applies no
// truncation at all, so for any prompt shorter than this the two agree
// exactly; the cap only exists so a pathological prompt cannot blow up the
// DiT's joint sequence (and with it the prefix KV cache) without warning.
constexpr int kMaxContentTokens = 1024;

struct TextConditioning {
    // (n_valid, kTextHiddenDim) at the text model's compute dtype — the last
    // decoder layer's residual stream, pre-final-norm, rows [drop_idx, L).
    brotensor::Tensor embeds;
    // (n_valid, 1) FP32, all ones. Nothing is padded at batch 1, so the mask
    // carries no information; it exists because Conditioning /
    // QwenImage21Denoiser::prepare() take one, and because a batched or
    // truncated caller would need it.
    brotensor::Tensor mask;

    // The FULL template token sequence that was encoded (prefix included), and
    // how many of its leading rows were dropped. `token_ids.size() - drop_idx
    // == embeds.rows`. Kept for debugging, for the parity harness, and so the
    // edit path can recover which surviving rows are `<|image_pad|>` slots.
    std::vector<int> token_ids;
    int drop_idx = kDropIdx;

    int n_valid() const { return embeds.rows; }
};

// The fixed template spans, as this checkpoint's tokenizer emits them.
//   system_prefix_ids()      14 ids — "<|im_start|>system\n…<|im_end|>\n"
//   user_header_ids()         3 ids — "<|im_start|>user\n"
//   assistant_suffix_ids()    5 ids — "<|im_end|>\n<|im_start|>assistant\n"
const std::vector<int>& system_prefix_ids();
const std::vector<int>& user_header_ids();
const std::vector<int>& assistant_suffix_ids();

// Build the full text-to-image template token sequence for `prompt`:
//   system_prefix ++ user_header ++ BPE(prompt) ++ assistant_suffix
// An empty prompt becomes " " (Qwen has no BOS token, so an empty string would
// leave the encoder with nothing to read — diffusers substitutes a space too).
// The prompt's own tokens are truncated to `max_content_tokens`.
std::vector<int> build_t2i_tokens(const brolm::qwen3vl::Tokenizer& tokenizer,
                                  const std::string& prompt,
                                  int max_content_tokens = kMaxContentTokens);

// Optional per-call overrides for encode_tokens(). All null / empty in the
// text-only case, which is what encode_prompt() uses.
struct EncodeOverrides {
    // (L, hidden_size) input embeddings replacing embed_tokens(ids) — the seam
    // for splicing vision-tower output over `<|image_pad|>` rows. Must have
    // exactly ids.size() rows.
    const brotensor::Tensor* inputs_embeds = nullptr;
    // 3-axis M-RoPE positions, length L each. When null, plain sequential
    // positions 0…L-1 are used on all three axes (correct for pure text).
    // Supply all three or none.
    const std::vector<std::int64_t>* mrope_t = nullptr;
    const std::vector<std::int64_t>* mrope_h = nullptr;
    const std::vector<std::int64_t>* mrope_w = nullptr;
    // DeepStack feature injections for the image rows (see brolm's
    // qwen3vl::DeepstackSplice). Empty for text-only.
    std::vector<brolm::qwen3vl::DeepstackSplice> deepstack;
};

// Run `ids` through `model` as one causal prefill and return rows
// [drop_idx, ids.size()) of the LAST decoder layer's output, before the final
// RMSNorm. `model` must be the Qwen3-VL-8B text backbone from the checkpoint's
// text_encoder/ (hidden 4096, 36 layers).
//
// This is the lower-level entry the edit path calls with an
// `inputs_embeds` override; encode_prompt() is the text-only wrapper.
TextConditioning encode_tokens(brolm::qwen3vl::TextModel& model,
                               const std::vector<int>& ids,
                               int drop_idx = kDropIdx,
                               const EncodeOverrides& ov = {});

// Encode `prompt` into Qwen-Image 2.1's text conditioning: build the T2I
// template, run one causal prefill, drop the system rows. Throws
// std::runtime_error if `model`'s config does not match the 8B backbone the
// checkpoint ships (hidden_size != kTextHiddenDim).
TextConditioning encode_prompt(const brolm::qwen3vl::Tokenizer& tokenizer,
                               brolm::qwen3vl::TextModel& model,
                               const std::string& prompt,
                               int max_content_tokens = kMaxContentTokens);

}  // namespace brodiffusion::qwenimage21
