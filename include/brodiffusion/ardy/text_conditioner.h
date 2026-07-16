#pragma once
//
// ARDY text conditioning — turn a prompt string into the single 4096-dim
// LLM2Vec feature the motion denoiser cross-attends to.
//
// ARDY conditions the diffusion denoiser on ONE pooled text token (llm_shape
// [1, 4096]). That token is LLM2Vec's masked-mean sentence embedding of the
// prompt, and reproducing it bit-for-bit is what keeps the generated motion on
// the text. This module is the thin, ARDY-specific glue on top of the
// (separately golden-tested) brolm pieces — the Llama-3 byte-level-BPE
// tokenizer and the bidirectional LLM2Vec encoder:
//
//   1. Wrap the prompt in the Llama-3-Instruct user turn and append the ARDY
//      instruction/text split exactly as ardy's LLM2Vec.prepare_for_tokenization
//      + tokenize do:
//        <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
//        {prompt}<|eot_id|>
//      (ardy inserts a `!@#$%^&*()` instruction/text delimiter with an EMPTY
//      instruction, which the tokenizer strips back out — so the delimiter never
//      reaches the model and the effective sequence is the turn above.)
//   2. Build the LLM2Vec "embed_mask": 1 over the trailing {prompt tokens +
//      <|eot_id|>} span, 0 over the leading BOS + role/header tokens
//      (skip_instruction=True → the pool excludes the instruction/header).
//   3. Run the bidirectional encoder and mean-pool the per-token hidden states
//      over the mask → the (4096) conditioning feature.
//
// The ardy port must apply the header wrap UNCONDITIONALLY: ardy gates it on the
// base checkpoint's `_name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct"`,
// which is always true for the shipped preset, and a locally-merged checkpoint
// that drops the field must not silently skip the wrap.
//
// Reference: ardy/model/llm2vec/{llm2vec.py (prepare_for_tokenization, tokenize,
// get_pooling), llm2vec_wrapper.py (LLM2VecEncoder.__call__)}.

#include <cstdint>
#include <string>
#include <vector>

namespace brolm::llama3 { class Tokenizer; }
namespace brolm::llm2vec { class Encoder; }

namespace brodiffusion::ardy {

// Build ARDY's LLM2Vec token ids + pooling mask for a prompt (pure host work;
// no model forward). `ids` is the full input sequence (leading BOS, the
// Llama-3 user-turn header, the prompt tokens, trailing <|eot_id|>). `pool_mask`
// is parallel to `ids`: 1.0 over the trailing {prompt + <|eot_id|>} tokens that
// LLM2Vec mean-pools, 0.0 over the BOS + role/header prefix it skips.
//
// Trailing whitespace in `prompt` is stripped (ardy applies `.strip()` inside
// prepare_for_tokenization); leading whitespace is preserved.
void build_ardy_text_tokens(const brolm::llama3::Tokenizer& tok,
                            const std::string& prompt,
                            std::vector<std::int32_t>& ids,
                            std::vector<float>& pool_mask);

// Produce ARDY's (hidden_size == 4096) text conditioning feature for a prompt:
// build tokens, run the bidirectional LLM2Vec encoder, mean-pool over the mask.
// Writes hidden_size host FP32 values to `out` (downcast from the encoder's
// compute dtype). brotensor::init() must have run once before calling.
void ardy_text_feat(const brolm::llama3::Tokenizer& tok,
                    brolm::llm2vec::Encoder& enc,
                    const std::string& prompt,
                    std::vector<float>& out);

}  // namespace brodiffusion::ardy
