#pragma once

// Sana prompt → Gemma-2 hidden-states text encoder.
//
// NVIDIA Sana conditions its Linear DiT (see dit/sana.h) on the *last hidden
// state* of a Gemma-2 2B decoder, run as a causal encoder over a constructed
// prompt. This module reproduces diffusers' `SanaPipeline._get_gemma_prompt_
// embeds` faithfully and returns the Gemma `last_hidden_state` as a
// brotensor::Tensor of shape (L, 2304) at the model's compute dtype — exactly
// what feeds `Conditioning::text_embeddings` (positive prompt) and
// `Conditioning::uncond_embeddings` (negative prompt) for the SanaDenoiser.
//
// Prompt construction (matches diffusers, clean_caption=False default):
//   1. The user prompt is lower-cased and stripped (`text.lower().strip()`).
//   2. The default "complex human instruction" (CHI) — a fixed list of
//      instruction lines — is joined with '\n' and *prepended* to the
//      (preprocessed) user prompt: full = "\n".join(CHI) + user.
//   3. `full` is tokenized by the Gemma tokenizer with a leading <bos> and no
//      <eos> (HF add_special_tokens=True), then truncated to
//      `len(encode(CHI)) + max_seq_len - 2` tokens (diffusers `max_length_all`;
//      with no CHI the cap is simply `max_seq_len`).
//   4. Gemma runs causally over those ids; the post-final-norm hidden states
//      are returned.
//
//   5. diffusers' token selection (SanaPipeline "Section 3.1"):
//      select_index = [0] + range(-max_seq_len+1, 0). With the CHI present this
//      keeps the <bos> plus the trailing window starting at the last CHI token,
//      dropping the CHI body (so the DiT never cross-attends over the
//      instruction's example prompts) and spanning the user prompt. With no CHI
//      the window covers the whole sequence (all rows kept).
//
// No padding and no attention mask: the returned tensor holds exactly the
// selected valid (non-pad) rows, and the Sana DiT cross-attends over precisely
// those rows. The negative / unconditional prompt is encoded the SAME way — an
// empty negative prompt still gets the CHI, so after selection its sequence is
// just the <bos> + final CHI token(s).
//
// Construct + load the model and tokenizer once, then call encode_prompt() for
// both the positive and the negative prompt (each call (re)allocates the model
// KV cache to fit and runs a fresh forward):
//
//   auto mc  = brodiffusion::load_model_config("weights/sana-600m");
//   auto tok = brolm::gemma::Tokenizer::load(
//                  "weights/sana-600m/tokenizer/tokenizer.json");
//   brolm::gemma::Gemma2Model gemma(mc.gemma);
//   // shards: text_encoder/model.fp16-0000{1,2}-of-00002.safetensors
//   gemma.load_weights(shard_ptrs);          // prefix "" — keys are unprefixed
//   bt::Tensor pos = brodiffusion::sana::encode_prompt(gemma, tok, prompt);
//   bt::Tensor neg = brodiffusion::sana::encode_prompt(gemma, tok, "");

#include "brolm/gemma2.h"
#include "brolm/gemma_tokenizer.h"
#include "brotensor/tensor.h"

#include <string>
#include <vector>

namespace brodiffusion::sana {

// Sana's default complex_human_instruction (verbatim from diffusers
// SanaPipeline). Returns a reference to a process-wide constant list.
const std::vector<std::string>& default_complex_human_instruction();

// Encode `prompt` into the Gemma-2 last_hidden_state for the Sana DiT.
//
// Returns a (L, 2304) Tensor at brolm::compute_dtype() (FP16 on a GPU backend,
// FP32 on CPU): the post-final-norm hidden states for the constructed,
// CHI-prefixed, BOS-framed, truncated token sequence — trimmed to exactly the
// L valid tokens (no padding). Feed directly into Conditioning::text_embeddings
// (positive) or Conditioning::uncond_embeddings (negative).
//
//   max_seq_len  — Sana's `max_sequence_length` (the user-prompt token budget;
//                  the total cap is len(encode(CHI)) + max_seq_len - 2).
//   chi          — the complex human instruction lines (default = Sana's).
//                  Pass an empty vector to disable the CHI (cap == max_seq_len).
//
// Throws std::runtime_error on an empty token sequence or a non-positive
// max_seq_len. (Re)allocates the model's KV cache to fit L and runs forward.
brotensor::Tensor encode_prompt(
    brolm::gemma::Gemma2Model& model,
    const brolm::gemma::Tokenizer& tokenizer,
    const std::string& prompt,
    int max_seq_len = 300,
    const std::vector<std::string>& chi = default_complex_human_instruction());

}  // namespace brodiffusion::sana
