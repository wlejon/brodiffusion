#!/usr/bin/env python
# Reference for the Krea 2 text-conditioning pathway (prompt → the tapped
# Qwen3-VL hidden states + validity mask the DiT's text-fusion stage consumes),
# to diff against brodiffusion's krea2::encode_prompt (CLI `krea2-text-fwd`).
#
# Runs the REAL diffusers `Krea2Pipeline.get_text_hidden_states` (bound to a
# lightweight shim carrying just the tokenizer / text_encoder / template
# attributes) so the mid-sequence-padding + cumsum-position machinery is exactly
# the shipped implementation. Writes:
#   .parity/krea2_text_ref_embeds.f32  — hidden_states (512, 12, 2560) flat f32
#   .parity/krea2_text_ref_mask.f32    — attention_mask (512,) f32 (1/0)
#   .parity/krea2_text_prompt.txt      — the prompt string (C++ side reads it)
#
# The C++ path assembles its filler rows as zeros while the reference computes
# real (garbage) content there; the mask marks those rows invalid. Compare only
# mask==1 rows (the parity script does this).
#
# Usage: python scripts/krea2_text_ref.py [weights/krea-2-raw] [outdir] [prompt]
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3VLModel

from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline

root = sys.argv[1] if len(sys.argv) > 1 else "weights/krea-2-raw"
out = sys.argv[2] if len(sys.argv) > 2 else ".parity"
prompt = sys.argv[3] if len(sys.argv) > 3 else \
    "a photorealistic red fox sitting in freshly fallen snow at golden hour"
os.makedirs(out, exist_ok=True)

device = "cuda"
tok = AutoTokenizer.from_pretrained(os.path.join(root, "tokenizer"))
te = Qwen3VLModel.from_pretrained(
    os.path.join(root, "text_encoder"), torch_dtype=torch.bfloat16
).to(device).eval()


class Shim:
    pass


shim = Shim()
shim.tokenizer = tok
shim.text_encoder = te
shim.text_encoder_select_layers = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
shim.prompt_template_encode_prefix = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
shim.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
shim.prompt_template_encode_start_idx = 34
shim.prompt_template_encode_num_suffix_tokens = 5

with torch.no_grad():
    hs, mask = Krea2Pipeline.get_text_hidden_states(
        shim, prompt, max_sequence_length=512, device=device
    )

# hs: (1, 512, 12, 2560); mask: (1, 512) bool
hs = hs[0].float().cpu().numpy().astype("<f4")
mask = mask[0].float().cpu().numpy().astype("<f4")
print("hidden_states", hs.shape, "valid tokens", int(mask.sum()))

hs.tofile(os.path.join(out, "krea2_text_ref_embeds.f32"))
mask.tofile(os.path.join(out, "krea2_text_ref_mask.f32"))
with open(os.path.join(out, "krea2_text_prompt.txt"), "w", encoding="utf-8") as f:
    f.write(prompt)
print("wrote reference to", out)
