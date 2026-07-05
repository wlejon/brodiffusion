#pragma once

// Krea 2 prompt → Qwen3-VL tapped-hidden-states text conditioning.
//
// Krea 2's image DiT (dit/krea2.h) does not condition on a single text-encoder
// output. It taps 12 intermediate decoder layers of a Qwen3-VL-4B text backbone
// and fuses that layer stack internally (Krea2TextFusion). This module produces
// exactly the two tensors that fusion stage consumes, reproducing diffusers'
// `Krea2Pipeline.get_text_hidden_states` for a single prompt.
//
// Prompt template (fixed, from the pipeline):
//   prefix = "<|im_start|>system\nDescribe the image by detailing the color,
//             shape, size, texture, quantity, text, spatial relationships of the
//             objects and background:<|im_end|>\n<|im_start|>user\n"   (34 tok)
//   suffix = "<|im_end|>\n<|im_start|>assistant\n"                      ( 5 tok)
// The encoder sees `[prefix | prompt | suffix]`; the 34 prefix rows are dropped
// from the output. The result is a fixed 512-token block: the real prompt
// content, a pad/filler region, then the 5 suffix tokens sitting at the END
// (rows 507..511), with a validity mask marking the content and suffix rows.
//
// ── The mid-sequence padding, and why we encode a COMPACTED sequence ─────────
//
// diffusers pads in the MIDDLE: `[prefix | prompt | PAD... | suffix]` tokenized
// to a fixed 546-token block, then relies on the Qwen3-VL encoder combining
// causal masking with a key-padding mask so the suffix (physically AFTER the
// pad block) never attends to the pad tokens, and on cumulative-valid-token
// positions so the suffix's mRoPE phase counts only real tokens. brolm's
// qwen3vl::TextModel has no key-padding-within-causal capability — only a
// contiguous causal cutoff. Rather than add that, we run the encoder over the
// COMPACTED sequence with NO pad tokens: `[prefix(34) | content(actual_len) |
// suffix(5)]` at plain sequential positions 0..(34+actual_len+5-1). This is
// numerically IDENTICAL to the reference for every row that survives to the
// output:
//   - prefix/content rows precede the pad block in the reference, so causal
//     masking alone already excludes the pad block — compacting is a no-op for
//     them.
//   - suffix rows attend, in both schemes, exactly `[prefix, content,
//     suffix-so-far]` at the same cumulative positions (the reference's
//     cumsum position trick == compacted sequential positions here), and there
//     are no pad rows in the compacted run to attend to. Same RoPE angles,
//     same keys, same result.
// We then place the surviving rows into the fixed 512-row output ourselves
// (pure host-side placement, no attention): content at rows [0, actual_len),
// zero filler at [actual_len, 507), suffix at [507, 512). Filler content is
// irrelevant — the mask marks it invalid and every downstream consumer drops
// invalid keys. `actual_len = min(prompt_token_count, 507)` (truncation).
//
// ── Special-token ids ────────────────────────────────────────────────────────
//
// The Krea 2 checkpoint's tokenizer PERMUTES the chat-control special-token ids
// relative to stock Qwen (here <|im_start|>=151645, <|im_end|>=151643,
// <|endoftext|>=151644), and brolm's qwen3vl::Tokenizer injects the stock ids.
// Since prefix and suffix are compile-time-constant strings, their token ids
// are constant too: we hard-code the exact 34 + 5 ids the checkpoint's HF
// tokenizer emits and BPE-encode ONLY the user prompt (plain text, no specials
// — identical between brolm and HF). Encoding the prompt alone is byte-identical
// to the reference's `tokenize(prefix+prompt)[34:]` (verified across prompts).

#include "brolm/qwen3vl_prompt.h"
#include "brolm/qwen3vl_text.h"
#include "brolm/qwen3vl_tokenizer.h"
#include "brolm/qwen3vl_vision.h"
#include "brolm/qwen3vl_vl.h"  // ImageInput
#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brodiffusion::krea2 {

// The 12 Qwen3-VL decoder-layer taps (1-based, HF `output_hidden_states`
// indices) Krea 2 conditions on.
const std::vector<int>& text_encoder_select_layers();

// The fixed Krea 2 text conditioning: number of tapped layers and the token
// hidden width.
constexpr int kNumTextLayers = 12;
constexpr int kTextHiddenDim = 2560;
constexpr int kMaxSequenceLength = 512;

struct TextConditioning {
    // (max_sequence_length * num_text_layers, text_hidden_dim) — token-major,
    // layer-minor: row t*num_text_layers + l is token t's tap of
    // text_encoder_select_layers()[l]. At the model compute dtype. This is the
    // exact flattening Krea2TextFusion.forward reshapes on.
    brotensor::Tensor prompt_embeds;
    // (max_sequence_length, 1) FP32 — 1.0 for a valid (content or suffix) token,
    // 0.0 for a filler/pad row.
    brotensor::Tensor prompt_embeds_mask;
};

// Encode `prompt` into Krea 2's tapped-hidden-states conditioning.
//
// `model` is an already-loaded Qwen3-VL-4B text backbone (36 dense decoder
// layers); `tokenizer` its paired BPE tokenizer. Runs a single compacted causal
// prefill (see header comment) and assembles the fixed 512-row output. Throws
// std::runtime_error if the fixed prefix/suffix do not tokenize to their
// expected lengths under `tokenizer` (a checkpoint-mismatch guard).
TextConditioning encode_prompt(
    const brolm::qwen3vl::Tokenizer& tokenizer,
    brolm::qwen3vl::TextModel& model,
    const std::string& prompt,
    int max_sequence_length = kMaxSequenceLength);

// Encode an image into the SAME tapped-hidden-states conditioning shape
// encode_prompt() produces for text — interchangeable with it (both flow
// into Krea2TextFusion / Pipeline::krea_prime_from_taps() unchanged). Mirrors
// krea-research's s7_extract.py: the checkpoint's Qwen3-VL-4B text backbone
// ships its full vision tower too (unused by plain text prompting), so an
// image can be run through the exact same fixed system-prompt template as
// text — substituting a `<|vision_start|><|image_pad|>...<|image_pad|>
// <|vision_end|>` run (expanded to the image's post-merger token count) for
// the prompt's content tokens — and tapped at the identical 12 decoder
// layers. No separate training or fine-tuning: a VLM's hidden space is
// modality-shared, so the text-trained fusion stack consumes image taps
// exactly as it does text taps.
//
// `vision` is the Qwen3-VL-4B vision tower paired with `model` (same
// checkpoint); `pp` its preprocessor config. Throws std::runtime_error if
// the image's post-merger token count doesn't fit the content budget
// (max_sequence_length - 5, i.e. 507 by default) — unlike text, image
// tokens can't be truncated mid-splice.
TextConditioning encode_image_prompt(
    const brolm::qwen3vl::Tokenizer& tokenizer,
    brolm::qwen3vl::TextModel& model,
    brolm::qwen3vl::VisionTower& vision,
    const brolm::qwen3vl::PreprocessConfig& pp,
    const brolm::qwen3vl::ImageInput& image,
    int max_sequence_length = kMaxSequenceLength);

}  // namespace brodiffusion::krea2
