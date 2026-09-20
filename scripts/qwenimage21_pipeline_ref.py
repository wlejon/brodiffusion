#!/usr/bin/env python
# End-to-end reference for the Qwen-Image 2.1 text-to-image pipeline, to diff
# against brodiffusion's `txt2img --model weights/qwen-image-2.1`.
#
# Runs the real diffusers QwenImage21Pipeline. Two deliberate deviations from a
# plain `pipe(prompt)` call, both so the comparison measures the thing we care
# about:
#
#   * The initial latent is drawn HERE rather than inside prepare_latents, so
#     the exact same N(0,1) field can be handed to brodiffusion via
#     --latent-in. A parity number that folds in an RNG disagreement measures
#     nothing. The pipeline takes a pre-packed `latents` argument and uses it
#     verbatim, so nothing else about the run changes.
#   * The final latent is captured through callback_on_step_end, which fires
#     after the last step, so we get both the latent AND the pipeline's own
#     decoded image out of one run.
#
# The whole model (14 GB DiT + 17 GB text encoder at BF16) does not fit a 24 GB
# card, so this runs under diffusers' sequential module offload: each component
# is moved to the GPU for its own turn. It is slow; it is a reference.
#
# Writes (into `outdir`):
#   qi21_pipe_init.f32     — the initial noise, (64, h, w) NCHW flat f32.
#                            Feed to brodiffusion as --latent-in.
#   qi21_pipe_ref.f32      — the final denoised latent, unpacked to the same
#                            (64, h, w) NCHW layout --latent-out writes.
#   qi21_pipe_ref.png      — the decoded image.
#   qi21_pipe_meta.txt     — size / steps / cfg / seed / prompt, read by the
#                            parity script so both sides cannot drift apart.
#
# Usage:
#   python scripts/qwenimage21_pipeline_ref.py [weights/qwen-image-2.1] [outdir]
#
# Env:
#   QI21_PIPE_PROMPT   prompt (default: the fox below)
#   QI21_PIPE_SIZE     square edge in pixels (default 512)
#   QI21_PIPE_STEPS    inference steps (default 8)
#   QI21_PIPE_CFG      true_cfg_scale (default 1.0 — the reference default,
#                      which runs no uncond branch at all)
#   QI21_PIPE_SEED     torch.Generator seed for the initial latent (default 0)
#   QI21_PIPE_DTYPE    bfloat16 (default) | float32
#   QI21_PIPE_OFFLOAD  1 (default) model-cpu-offload | 0 keep it all on the GPU
import os
import sys

import torch

from diffusers import QwenImage21Pipeline

root = sys.argv[1] if len(sys.argv) > 1 else "weights/qwen-image-2.1"
out = sys.argv[2] if len(sys.argv) > 2 else ".parity"
os.makedirs(out, exist_ok=True)

prompt = os.environ.get(
    "QI21_PIPE_PROMPT",
    "a photorealistic red fox sitting in freshly fallen snow at golden hour")
size = int(os.environ.get("QI21_PIPE_SIZE", "512"))
steps = int(os.environ.get("QI21_PIPE_STEPS", "8"))
cfg = float(os.environ.get("QI21_PIPE_CFG", "1.0"))
seed = int(os.environ.get("QI21_PIPE_SEED", "0"))
dtype = getattr(torch, os.environ.get("QI21_PIPE_DTYPE", "bfloat16"))
offload = os.environ.get("QI21_PIPE_OFFLOAD", "1") != "0"

print("prompt   :", prompt)
print("size     : %dx%d  steps %d  true_cfg %.2f  seed %d  dtype %s"
      % (size, size, steps, cfg, seed, dtype))

pipe = QwenImage21Pipeline.from_pretrained(root, torch_dtype=dtype)
if offload:
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to("cuda")

vsf = pipe.vae_scale_factor               # 16
z = pipe.transformer.config.in_channels   # 64
h = 2 * (size // (vsf * 2))
w = 2 * (size // (vsf * 2))
print("latent   : (%d, %d, %d)  vae_scale_factor %d" % (z, h, w, vsf))

# The initial latent, drawn on the CPU so the seed means the same thing on any
# device, then packed exactly as prepare_latents would have.
gen = torch.Generator(device="cpu").manual_seed(seed)
noise = torch.randn((1, 1, z, h, w), generator=gen, dtype=torch.float32)
noise.reshape(-1).numpy().astype("<f4").tofile(
    os.path.join(out, "qi21_pipe_init.f32"))
packed = QwenImage21Pipeline._pack_latents(noise.to(dtype), 1, z, h, w)

captured = {}


def grab(pipeline, i, t, kwargs):
    # Fires after every step, so what survives is the final latent.
    captured["latents"] = kwargs["latents"].detach().clone()
    return {}


image = pipe(
    prompt=prompt,
    negative_prompt=" ",
    width=size,
    height=size,
    num_inference_steps=steps,
    true_cfg_scale=cfg,
    latents=packed,
    callback_on_step_end=grab,
    callback_on_step_end_tensor_inputs=["latents"],
).images[0]
image.save(os.path.join(out, "qi21_pipe_ref.png"))

# Back to the (64, h, w) NCHW layout brodiffusion's --latent-out writes.
final = QwenImage21Pipeline._unpack_latents(captured["latents"], size, size, vsf)
final.float().cpu().numpy().reshape(-1).astype("<f4").tofile(
    os.path.join(out, "qi21_pipe_ref.f32"))
print("final latent", tuple(final.shape))

with open(os.path.join(out, "qi21_pipe_meta.txt"), "w", encoding="utf-8") as f:
    f.write("%d\n%d\n%.6f\n%d\n%s\n" % (size, steps, cfg, seed, prompt))
print("wrote reference to", out)
