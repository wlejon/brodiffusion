#!/usr/bin/env python
# Reference for the Qwen-Image 2.1 text-conditioning pathway (prompt -> the
# Qwen3-VL-8B last-hidden-state rows the DiT's txt_in consumes), to diff
# against brodiffusion's qwenimage21::encode_prompt (CLI `qi21-text-fwd`).
#
# Runs the REAL diffusers `QwenImage21Pipeline._get_qwen_prompt_embeds` bound
# to a lightweight shim carrying just the processor / text_encoder, so the
# template, the `_drop_idx` derivation and the final-norm-neutralizing forward
# hook are exactly the shipped implementation. Writes:
#   .parity/qi21_text_ref.f32       — prompt_embeds (n_valid, 4096) flat f32
#   .parity/qi21_text_ref_mask.f32  — encoder_attention_mask (n_valid,) f32
#   .parity/qi21_text_ref_ids.i32   — the post-drop token ids (n_valid,) int32
#   .parity/qi21_text_prompt.txt    — the prompt string (the C++ side reads it)
#
# Usage:
#   python scripts/qwenimage21_text_ref.py [weights/qwen-image-2.1] [outdir] [prompt]
#
# Env:
#   QI21_TEXT_REF_DTYPE   bfloat16 (default) | float32
#   QI21_TEXT_REF_DEVICE  cuda (default) | cpu
#
# float32/cpu is the gold reference. It matters here: this encoder's
# `<|im_start|>` rows carry massive activations that survive as a small
# difference of large numbers, and in bfloat16 (8 mantissa bits) that row's
# value is dominated by rounding — the reference's own bf16 result sits at
# cosine ~0.90 against its fp32 result on that row alone. Judging an
# implementation against the bf16 run therefore understates it; brodiffusion
# (FP16 compute) matches fp32 to cosine 0.99996 while the bf16 reference
# matches it to 0.9961.
import os
import sys

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline

root = sys.argv[1] if len(sys.argv) > 1 else "weights/qwen-image-2.1"
out = sys.argv[2] if len(sys.argv) > 2 else ".parity"
prompt = sys.argv[3] if len(sys.argv) > 3 else \
    "a photorealistic red fox sitting in freshly fallen snow at golden hour"
os.makedirs(out, exist_ok=True)

dtype = getattr(torch, os.environ.get("QI21_TEXT_REF_DTYPE", "bfloat16"))
device = os.environ.get("QI21_TEXT_REF_DEVICE", "cuda")
print("reference dtype", dtype, "device", device)
processor = AutoProcessor.from_pretrained(os.path.join(root, "processor"))
text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    os.path.join(root, "text_encoder"), dtype=dtype
).to(device).eval()


class Shim:
    # `_get_qwen_prompt_embeds` reads `self.processor`, `self.text_encoder`,
    # `self.prompt_template_t2i`, `self._drop_idx`, `self._img_token_id`,
    # `self._execution_device` and the pipeline's own
    # `_extract_masked_hidden` — nothing else.
    _execution_device = torch.device(device)
    _extract_masked_hidden = QwenImage21Pipeline._extract_masked_hidden


shim = Shim()
shim.processor = processor
shim.text_encoder = text_encoder
sys_prompt = "Comprehend and analyze the provided prompt."
shim.prompt_template_t2i = (
    f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
    f"<|im_start|>user\n{{}}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
sys_message = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}]
sys_tokens = processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
shim._drop_idx = len(sys_tokens[0])
shim._img_token_id = processor.tokenizer.encode("<|image_pad|>")[0]
print("drop_idx", shim._drop_idx, "img_token_id", shim._img_token_id)

with torch.no_grad():
    embeds, mask, image_pad_mask = QwenImage21Pipeline._get_qwen_prompt_embeds(
        shim, prompt, image=None, device=device
    )

# embeds: (1, n_valid, 4096); mask: (1, n_valid)
e = embeds[0].float().cpu().numpy().astype("<f4")
m = mask[0].float().cpu().numpy().astype("<f4")
print("prompt_embeds", e.shape, "valid tokens", int(m.sum()))
assert int(m.sum()) == e.shape[0], "batch 1 should not pad"
assert int(image_pad_mask.sum()) == 0, "text-only prompt has no image slots"

# The surviving token ids, for the tokenization half of the gate.
full_ids = processor(
    text=[shim.prompt_template_t2i.format(prompt)],
    padding=True,
    padding_side="left",
    return_tensors="pt",
).input_ids[0].tolist()
kept = np.asarray(full_ids[shim._drop_idx:], dtype="<i4")
assert kept.shape[0] == e.shape[0], (kept.shape, e.shape)

e.tofile(os.path.join(out, "qi21_text_ref.f32"))
m.tofile(os.path.join(out, "qi21_text_ref_mask.f32"))
kept.tofile(os.path.join(out, "qi21_text_ref_ids.i32"))
with open(os.path.join(out, "qi21_text_prompt.txt"), "w", encoding="utf-8") as f:
    f.write(prompt)
print("wrote reference to", out)
