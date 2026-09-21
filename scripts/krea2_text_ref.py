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
# With KREA2_TEXT_REF_IMAGE set to an image path, the IMAGE-prompt pathway is
# referenced instead: the content tokens become a
# "<|vision_start|><|image_pad|>...<|vision_end|>" run, the vision tower runs,
# and the same 12 decoder layers are tapped — brodiffusion's
# krea2::encode_image_prompt (CLI `krea2-text-fwd --image`). diffusers has no
# such path (it is a property of the checkpoint, not of the pipeline), so this
# mode drives Qwen3VLModel directly with the template ids brodiffusion uses.
# One more file appears:
#   .parity/krea2_text_cond.png      — the RESIZED image the reference encoded,
#                                      which the C++ side must read VERBATIM so
#                                      no resampler sits between the two.
#
# Usage: python scripts/krea2_text_ref.py [weights/krea-2-raw] [outdir] [prompt]
# Env:   KREA2_TEXT_REF_IMAGE        image path (default: none, i.e. text mode)
#        KREA2_TEXT_REF_DTYPE        bfloat16 (default, what the pipeline ships
#                                    with) or float32. BF16 is the accuracy
#                                    FLOOR of this gate, not its ceiling: on an
#                                    image prompt the reference's own bf16
#                                    rounding costs ~0.9994 flat cosine and
#                                    drives a handful of massive-activation
#                                    tokens to ~0.83 per-token, which is larger
#                                    than the C++ path's total error. Set this
#                                    to float32 to measure the implementation
#                                    instead of the reference's noise.
#        KREA2_TEXT_REF_VISION_DUMP  directory; when set, the vision tower's
#                                    stage boundaries (patchified pixels, patch
#                                    embedding, interpolated pos embed, every
#                                    block output, the DeepStack merger outputs
#                                    and the main merger output) are written as
#                                    raw FP32 <dir>/vis_ref_<name>.f32. The C++
#                                    side writes the same names with the
#                                    vis_mine_ prefix under BROLM_VISION_DUMP,
#                                    so the two dumps diff stage by stage.
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
ref_dtype = getattr(torch, os.environ.get("KREA2_TEXT_REF_DTYPE", "bfloat16"))
tok = AutoTokenizer.from_pretrained(os.path.join(root, "tokenizer"))
te = Qwen3VLModel.from_pretrained(
    os.path.join(root, "text_encoder"), torch_dtype=ref_dtype
).to(device).eval()
print("reference dtype", ref_dtype)


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

image_path = os.environ.get("KREA2_TEXT_REF_IMAGE", "")

if not image_path:
    with torch.no_grad():
        hs, mask = Krea2Pipeline.get_text_hidden_states(
            shim, prompt, max_sequence_length=512, device=device
        )
    # hs: (1, 512, 12, 2560); mask: (1, 512) bool
    hs = hs[0].float().cpu().numpy().astype("<f4")
    mask = mask[0].float().cpu().numpy().astype("<f4")
else:
    # ── image prompt ──────────────────────────────────────────────────────
    #
    # The same fixed template, with the prompt's content tokens replaced by a
    # vision run. The template ids are spelled out rather than tokenized so
    # this reference cannot drift from the C++ side through a tokenizer
    # difference: they are exactly the ids src/krea2_text.cpp emits.
    from PIL import Image
    from transformers import AutoImageProcessor

    PREFIX = [151645, 8948, 198, 74785, 279, 2168, 553, 44193, 279, 1894, 11,
              6083, 11, 1379, 11, 10434, 11, 12194, 11, 1467, 11, 27979,
              11871, 315, 279, 6171, 323, 4004, 25, 151643, 198, 151645, 872,
              198]
    SUFFIX = [151643, 198, 151645, 77091, 198]
    VISION_START, VISION_END, IMAGE_PAD = 151652, 151653, 151655
    MAX_SEQ = 512

    # 512x512 is 32x32 patches of 16 at merge 2 -> 256 merged tokens, well
    # inside the 507-token content budget. Resized here and written out, so
    # the C++ side encodes the identical pixels.
    proc = AutoImageProcessor.from_pretrained(os.path.join(root, "processor")
                                              if os.path.isdir(
                                                  os.path.join(root, "processor"))
                                              else "Qwen/Qwen3-VL-4B-Instruct")
    img = Image.open(image_path).convert("RGB").resize((512, 512),
                                                       Image.LANCZOS)
    cond_png = os.path.join(out, "krea2_text_cond.png")
    img.save(cond_png)
    print("condition image", image_path, "-> 512x512 ->", cond_png)

    pv = proc(images=img, return_tensors="pt")
    grid = pv["image_grid_thw"]
    n_img = int(grid.prod(-1).item()) // 4          # 2x2 spatial merge
    content = [VISION_START] + [IMAGE_PAD] * n_img + [VISION_END]
    ids = PREFIX + content + SUFFIX
    print("image -> %d merged tokens, %d content rows" % (n_img, len(content)))

    # Stage-boundary dump of the vision tower, for bisecting an image-prompt
    # divergence down to the first stage that disagrees. Hooks sit on the
    # modules themselves so nothing about the real forward changes.
    vis_dump = os.environ.get("KREA2_TEXT_REF_VISION_DUMP", "")
    if vis_dump:
        os.makedirs(vis_dump, exist_ok=True)

        def write(name, t):
            arr = t.detach().float().cpu().numpy().astype("<f4")
            arr.tofile(os.path.join(vis_dump, "vis_ref_%s.f32" % name))

        vis = te.visual
        handles = []
        handles.append(vis.patch_embed.register_forward_hook(
            lambda m, i, o: write("patch_embed", o)))
        for bi, blk in enumerate(vis.blocks):
            handles.append(blk.register_forward_hook(
                (lambda idx: lambda m, i, o: write("block%d" % idx, o))(bi)))
        for di, mrg in enumerate(vis.deepstack_merger_list):
            handles.append(mrg.register_forward_hook(
                (lambda idx: lambda m, i, o: write("ds%d" % idx, o))(di)))
        handles.append(vis.merger.register_forward_hook(
            lambda m, i, o: write("merged", o)))
        write("patches", pv["pixel_values"])
        with torch.no_grad():
            # pos_embed is not a module boundary; recompute it the way the
            # tower does, so the dump has the same stage granularity.
            from transformers.vision_utils import (
                get_vision_bilinear_indices_and_weights,
            )
            bidx, bw = get_vision_bilinear_indices_and_weights(
                grid.to(device),
                num_grid_per_side=vis.num_grid_per_side,
                spatial_merge_size=vis.config.spatial_merge_size,
            )
            write("pos_embed", (vis.pos_embed(bidx) * bw[:, :, None]).sum(0))
            pe = vis.patch_embed(pv["pixel_values"].to(device, ref_dtype))
            write("block_in", pe + (vis.pos_embed(bidx) * bw[:, :, None]).sum(0)
                  .to(pe.dtype))
    else:
        handles = []

    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        outs = te(
            input_ids=input_ids,
            output_hidden_states=True,
            pixel_values=pv["pixel_values"].to(device, ref_dtype),
            image_grid_thw=grid.to(device),
            # M-RoPE needs the image-token positions marked.
            mm_token_type_ids=(input_ids == IMAGE_PAD).long(),
        )
    for h in handles:
        h.remove()
    taps = torch.stack(
        [outs.hidden_states[k][0] for k in shim.text_encoder_select_layers], 1)
    taps = taps.float().cpu().numpy()                # (len(ids), 12, 2560)

    # The same mid-sequence padding get_text_hidden_states applies: content at
    # the head, suffix pinned to the tail, filler rows in between masked out.
    n = len(content)
    hs = np.zeros((MAX_SEQ, len(shim.text_encoder_select_layers), 2560),
                  dtype="<f4")
    mask = np.zeros(MAX_SEQ, dtype="<f4")
    hs[:n] = taps[len(PREFIX):len(PREFIX) + n]
    mask[:n] = 1.0
    hs[MAX_SEQ - len(SUFFIX):] = taps[len(PREFIX) + n:]
    mask[MAX_SEQ - len(SUFFIX):] = 1.0

print("hidden_states", hs.shape, "valid tokens", int(mask.sum()))

hs.tofile(os.path.join(out, "krea2_text_ref_embeds.f32"))
mask.tofile(os.path.join(out, "krea2_text_ref_mask.f32"))
with open(os.path.join(out, "krea2_text_prompt.txt"), "w", encoding="utf-8") as f:
    f.write(prompt)
print("wrote reference to", out)
