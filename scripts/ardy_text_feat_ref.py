#!/usr/bin/env python3
"""ARDY text-conditioning parity reference.

Reproduces exactly how ardy turns a prompt into the single (4096) LLM2Vec
feature the motion denoiser cross-attends to, to diff against the C++
brodiffusion::ardy::TextConditioner (`brodiffusion ardy-text-feat`).

Two gates:

  TOKENS (default) — reproduce ardy LLM2Vec.prepare_for_tokenization + tokenize
    with the AUTHORITATIVE Hugging Face tokenizer (weights/llm2vec-llama3-8b),
    and dump the golden input-id sequence + the skip-instruction embed_mask.
    This is the real risk in the port: brolm's byte-level-BPE Llama-3 tokenizer
    (a GPT-2 approximation) reproducing HF's ids, and the mask landing on the
    trailing {prompt + <|eot_id|>} span. No model forward — fast.

  FULL (ARDY_TEXT_FULL=1) — additionally run the BIDIRECTIONAL LLM2Vec forward
    on the real merged checkpoint (fp32 CPU, reusing brolm/scripts/llm2vec_ref.py
    llama_forward with causal=False) and mean-pool over the embed_mask, dumping
    the golden (4096) text_feat. Verifies the C++ encode_pooled masked-mean end
    to end. Slow (8B on CPU).

The ardy path (llm2vec.py): encode() injects an empty instruction so the string
becomes "!@#$%^&*(){prompt}"; prepare_for_tokenization wraps it in the Llama-3
user turn "<|start_header_id|>user<|end_header_id|>\\n\\n" + it.strip() +
"<|eot_id|>"; tokenize() splits the "!@#$%^&*()" delimiter back out, tokenizes
the rejoined string WITH special tokens (BOS prepended) as `original`, tokenizes
the post-delimiter "{prompt}<|eot_id|>" WITHOUT special tokens as the embed_mask
span (last len positions = 1); skip_instruction pools the mean over that span.

Dumps (raw little-endian) into <outdir> (default .parity):
  ardy_text_ids_ref.i32    int32 golden input ids            (L,)
  ardy_text_mask_ref.f32   float32 golden pool mask (0/1)    (L,)
  ardy_text_dims.txt       "L pooled"
  ardy_text_feat_ref.f32   float32 golden pooled feature     (4096,)  [FULL only]

Env: ARDY_TEXT_WEIGHTS (default D:/projects/brolm/weights/llm2vec-llama3-8b),
     ARDY_TEXT_PROMPT   (default "a person walks forward and waves"),
     ARDY_TEXT_FULL     (set to 1 to also dump the pooled embedding),
     BROLM_SCRIPTS      (default ../brolm/scripts — for llm2vec_ref import).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / ".parity"
OUT.mkdir(exist_ok=True)

WEIGHTS = Path(os.environ.get("ARDY_TEXT_WEIGHTS",
                              "D:/projects/brolm/weights/llm2vec-llama3-8b"))
PROMPT = os.environ.get("ARDY_TEXT_PROMPT", "a person walks forward and waves")
FULL = os.environ.get("ARDY_TEXT_FULL", "") not in ("", "0")

HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
EOT = "<|eot_id|>"


BOS_ID = 128000  # <|begin_of_text|> — Llama-3's only auto special (no auto-EOS)


def build_tokens(prompt: str):
    """ardy prepare_for_tokenization + tokenize, via the HF fast tokenizer.

    Loads the EXACT tokenizer.json the C++ side uses (PreTrainedTokenizerFast,
    sidestepping the merged dir's non-standard tokenizer_config class). Encodes
    with add_special_tokens=False (added-token strings like <|start_header_id|>
    are still matched atomically) and prepends BOS by hand, which is exactly what
    ardy's add_special_tokens=True `original` sequence is for Llama-3."""
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=str(WEIGHTS / "tokenizer.json"))
    content = ("!@#$%^&*()" + prompt).strip()          # ardy .strip()
    parts = content.split("!@#$%^&*()")                # delimiter split back out
    text_only = parts[1] if len(parts) > 1 else ""     # "{prompt}"
    original_texts = HEADER + text_only + EOT          # tokenize() "".join(split)

    core = tok(original_texts, add_special_tokens=False)["input_ids"]
    ids = [BOS_ID] + core                              # == add_special_tokens=True
    tail = tok(text_only + EOT, add_special_tokens=False)["input_ids"]
    mask = np.zeros(len(ids), dtype="<f4")
    n = min(len(tail), len(ids))
    if n:
        mask[len(ids) - n:] = 1.0
    assert ids[0] == BOS_ID and ids[-1] == 128009, (ids[0], ids[-1])
    return np.asarray(ids, dtype="<i4"), mask


def pooled_feature(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Bidirectional LLM2Vec forward (real weights, fp32 CPU) + masked mean."""
    brolm_scripts = Path(os.environ.get("BROLM_SCRIPTS", REPO.parent / "brolm" / "scripts"))
    sys.path.insert(0, str(brolm_scripts))
    import llm2vec_ref  # noqa: E402  (brolm reference forward)
    import torch

    cfg, w = llm2vec_ref.load_real(str(WEIGHTS))
    with torch.no_grad():
        hidden = llm2vec_ref.llama_forward(
            w, torch.from_numpy(ids.astype("int64")), cfg, causal=False)  # (L, H)
    h = hidden.float().cpu().numpy()
    m = mask.astype(np.float64)
    feat = (h * m[:, None]).sum(0) / max(m.sum(), 1.0)
    return feat.astype("<f4")


def main() -> int:
    ids, mask = build_tokens(PROMPT)
    ids.tofile(OUT / "ardy_text_ids_ref.i32")
    mask.tofile(OUT / "ardy_text_mask_ref.f32")
    pooled = int(mask.sum())
    (OUT / "ardy_text_dims.txt").write_text(f"{len(ids)} {pooled}\n")
    print(f'prompt: "{PROMPT}"')
    print(f"L={len(ids)} pooled={pooled}")
    print(f"ids   = {ids.tolist()}")
    print(f"mask  = {mask.astype(int).tolist()}")

    if FULL:
        feat = pooled_feature(ids, mask)
        feat.tofile(OUT / "ardy_text_feat_ref.f32")
        print(f"text_feat({feat.shape[0]}) mean {feat.mean():.6f} std {feat.std():.6f}")
    else:
        print("(tokens-only; set ARDY_TEXT_FULL=1 for the pooled-embedding gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
