#!/usr/bin/env python
# End-to-end reference for Qwen-Image 2.1's IMAGE-CONDITIONED generation, to
# diff against brodiffusion's `txt2img --model weights/qwen-image-2.1 --image`.
#
# Same shape as qwenimage21_pipeline_ref.py (the text-to-image reference), with
# the two deviations that make a parity number mean something — the initial
# latent is drawn here and handed over, and the final latent is captured
# through callback_on_step_end — plus one more that only matters here:
#
#   * the RESIZED condition images are written out. The pipeline resizes with
#     PIL's resampler; brodiffusion resizes with broimage's. Both are Lanczos-3
#     and they agree closely, but "closely" is not a useful baseline for a
#     component this far upstream — a resize difference would arrive at the
#     parity number indistinguishable from a DiT difference. Handing the C++
#     side the reference's own resized pixels removes the resampler from the
#     comparison entirely.
#
# Writes (into `outdir`):
#   qi21_edit_init.f32     — the initial noise, (64, h, w) NCHW flat f32
#   qi21_edit_ref.f32      — the final denoised latent, same layout
#   qi21_edit_ref.png      — the decoded image
#   qi21_edit_cond_<i>.png — the resized condition images the run actually used
#   qi21_edit_meta.txt     — width / height / steps / cfg / seed / n_images /
#                            prompt, read by the parity script so the two sides
#                            cannot drift apart
#
# Usage:
#   python scripts/qwenimage21_edit_ref.py <image.png>[,<image2.png>] \
#          [weights/qwen-image-2.1] [outdir]
#
# Env:
#   QI21_EDIT_PROMPT   the edit instruction (default: the sunset sky below)
#   QI21_EDIT_RES      output_resolution (default 512)
#   QI21_EDIT_STEPS    inference steps (default 8)
#   QI21_EDIT_CFG      true_cfg_scale (default 1.0 — no uncond branch)
#   QI21_EDIT_SEED     torch.Generator seed for the initial latent (default 0)
#   QI21_EDIT_DTYPE    bfloat16 (default) | float32
#   QI21_EDIT_OFFLOAD  1 (default) model-cpu-offload | 0 keep it on the GPU
import os
import sys

import torch
from PIL import Image

from diffusers import QwenImage21Pipeline
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

if len(sys.argv) < 2:
    raise SystemExit(
        "usage: qwenimage21_edit_ref.py <image.png>[,<image2.png>] "
        "[weights/qwen-image-2.1] [outdir]")
image_paths = [p for p in sys.argv[1].split(",") if p]
root = sys.argv[2] if len(sys.argv) > 2 else "weights/qwen-image-2.1"
out = sys.argv[3] if len(sys.argv) > 3 else ".parity"
os.makedirs(out, exist_ok=True)

prompt = os.environ.get("QI21_EDIT_PROMPT", "make the sky a deep sunset orange")
resolution = int(os.environ.get("QI21_EDIT_RES", "512"))
steps = int(os.environ.get("QI21_EDIT_STEPS", "8"))
cfg = float(os.environ.get("QI21_EDIT_CFG", "1.0"))
seed = int(os.environ.get("QI21_EDIT_SEED", "0"))
dtype = getattr(torch, os.environ.get("QI21_EDIT_DTYPE", "bfloat16"))
offload = os.environ.get("QI21_EDIT_OFFLOAD", "1") != "0"

print("prompt   :", prompt)
print("images   :", image_paths)
print("res %d  steps %d  true_cfg %.2f  seed %d  dtype %s"
      % (resolution, steps, cfg, seed, dtype))

pipe = QwenImage21Pipeline.from_pretrained(root, torch_dtype=dtype)
if offload:
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to("cuda")

# Resize every condition image exactly as __call__ does, and hand the pipeline
# the resized copies. __call__ will run calculate_dimensions on them again;
# the rule is idempotent (both sides already a multiple of 32 at the target
# area), so the pipeline sees the same pixels the C++ side will.
images = []
for i, path in enumerate(image_paths):
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    iw, ih = img.size
    cw, ch, _ = calculate_dimensions(resolution * resolution, iw / ih)
    resized = pipe.image_processor.resize(img, width=cw, height=ch)
    resized.save(os.path.join(out, "qi21_edit_cond_%d.png" % i))
    print("condition image %d: %dx%d -> %dx%d" % (i, iw, ih, cw, ch))
    images.append(resized)

# The canvas the pipeline derives from the LAST image's aspect — computed here
# so the initial latent can be drawn at the right shape.
width, height, _ = calculate_dimensions(
    resolution * resolution, images[-1].size[0] / images[-1].size[1])
vsf = pipe.vae_scale_factor                # 16
z = pipe.transformer.config.in_channels    # 64
h = 2 * (height // (vsf * 2))
w = 2 * (width // (vsf * 2))
print("canvas   : %dx%d  latent (%d, %d, %d)" % (width, height, z, h, w))

gen = torch.Generator(device="cpu").manual_seed(seed)
noise = torch.randn((1, 1, z, h, w), generator=gen, dtype=torch.float32)
noise.reshape(-1).numpy().astype("<f4").tofile(
    os.path.join(out, "qi21_edit_init.f32"))
packed = QwenImage21Pipeline._pack_latents(noise.to(dtype), 1, z, h, w)

captured = {}


def grab(pipeline, i, t, kwargs):
    captured["latents"] = kwargs["latents"].detach().clone()
    return {}


image = pipe(
    prompt=prompt,
    image=images,
    negative_prompt=" ",
    width=width,
    height=height,
    num_inference_steps=steps,
    true_cfg_scale=cfg,
    output_resolution=resolution,
    latents=packed,
    callback_on_step_end=grab,
    callback_on_step_end_tensor_inputs=["latents"],
).images[0]
image.save(os.path.join(out, "qi21_edit_ref.png"))

final = QwenImage21Pipeline._unpack_latents(captured["latents"], height, width, vsf)
final.float().cpu().numpy().reshape(-1).astype("<f4").tofile(
    os.path.join(out, "qi21_edit_ref.f32"))
print("final latent", tuple(final.shape))

with open(os.path.join(out, "qi21_edit_meta.txt"), "w", encoding="utf-8") as f:
    f.write("%d\n%d\n%d\n%.6f\n%d\n%d\n%s\n"
            % (width, height, steps, cfg, seed, len(images), prompt))
print("wrote reference to", out)
