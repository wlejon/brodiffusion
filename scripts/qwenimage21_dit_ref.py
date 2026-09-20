#!/usr/bin/env python
# Reference for the Qwen-Image 2.1 image DiT (QwenImage21Transformer2DModel),
# to diff against brodiffusion's dit::QwenImage21Transformer2DModel
# (CLI `qi21-fwd`).
#
# Two modes:
#   synth  (default) — build a SMALL random-weight QwenImage21Transformer2DModel
#       in float32, save it (config.json + safetensors) to a synthetic dir, run
#       TWO forwards — an "extract" prefill and a "cached" decode from the
#       prefix KV cache — and dump both velocities plus all inputs. Both sides
#       run FP32 so this is an EXACT architecture parity.
#   real   — load the real transformer (weights/qwen-image-2.1/transformer) in
#       bfloat16 on CUDA (or float32 on CPU when QI21_REF_DEVICE=cpu) and run
#       the same two forwards on a small grid.
#
# Dumps (raw little-endian float32) into <outdir>:
#   qi21_dit_{latent,embeds,velocity,velocity2}.f32 and the config/weights dir.
# The C++ harness reads latent/embeds and compares its velocities.
#
# Usage: python scripts/qwenimage21_dit_ref.py [synth|real] [outdir]
import json
import math
import os
import sys

import numpy as np
import torch

from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21KVCache,
    QwenImage21Transformer2DModel,
)

mode = sys.argv[1] if len(sys.argv) > 1 else "synth"
out = sys.argv[2] if len(sys.argv) > 2 else ".parity"
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)

T1 = 0.7
T2 = 0.4


def dump(name, t):
    t.detach().float().cpu().numpy().astype("<f4").tofile(os.path.join(out, name))


def run(model, latent, ehs, hp, wp, device, dtype, tag):
    """Text-to-image: the joint sequence is [text ; target image].

    `img_mask` spans the joint sequence — False over the prompt, True over the
    one-slot-per-2x2-latents tail the pipeline appends for the target image.
    """
    text_seq = ehs.shape[1]
    img_len = hp * wp
    img_mask = torch.zeros(1, text_seq + img_len // 4, dtype=torch.bool, device=device)
    img_mask[0, text_seq:] = True
    img_shapes = [[(1, hp, wp)]]

    cache = QwenImage21KVCache(len(model.transformer_blocks))
    outs = []
    for i, t in enumerate((T1, T2)):
        timestep = torch.tensor([t], dtype=dtype, device=device)
        with torch.no_grad():
            vel = model(
                hidden_states=latent,
                encoder_hidden_states=ehs,
                timestep=timestep,
                img_shapes=img_shapes,
                img_mask=img_mask,
                kv_cache=cache,
                kv_cache_mode="extract" if i == 0 else "cached",
                return_dict=False,
            )[0]
        vel = vel[:, -img_len:]
        print(tag, "step%d velocity" % i, tuple(vel.shape),
              "mean %.5f std %.5f" % (float(vel.mean()), float(vel.std())))
        outs.append(vel)

    dump("qi21_dit_latent.f32", latent[0])
    dump("qi21_dit_embeds.f32", ehs[0])
    dump("qi21_dit_velocity.f32", outs[0][0])
    dump("qi21_dit_velocity2.f32", outs[1][0])
    with open(os.path.join(out, "qi21_dit_dims.txt"), "w") as f:
        f.write("%d %d %d %.4f %.4f\n" % (hp, wp, text_seq, T1, T2))


if mode == "synth":
    cfg = dict(
        patch_size=1, in_channels=64, out_channels=64, num_layers=2,
        attention_head_dim=64, num_attention_heads=2, context_in_dim=128,
        mlp_ratio=3, axes_dims_rope=(8, 28, 28), eps=1e-6, causal_condition=True,
    )
    model = QwenImage21Transformer2DModel(**cfg).to(torch.float32).eval()
    # Randomize every parameter (incl. the zero-init ZeroCenterRMSNorm gain and
    # the ones-init q/k RMSNorm gains) so no path degenerates to identity.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.1)

    sdir = os.path.join(out, "qi21_synth_transformer")
    os.makedirs(sdir, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              os.path.join(sdir, "diffusion_pytorch_model.safetensors"))
    json.dump({k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
              open(os.path.join(sdir, "config.json"), "w"))

    # hp*wp must be a multiple of 4: the pipeline appends one img_mask slot per
    # 2x2 group of target latents.
    hp, wp, text_seq = 6, 4, 5
    latent = torch.randn(1, hp * wp, cfg["in_channels"], dtype=torch.float32)
    ehs = torch.randn(1, text_seq, cfg["context_in_dim"], dtype=torch.float32)
    run(model, latent, ehs, hp, wp, torch.device("cpu"), torch.float32, "synth")

else:
    tdir = os.environ.get("QI21_DIR", "weights/qwen-image-2.1") + "/transformer"
    dev = os.environ.get("QI21_REF_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    model = QwenImage21Transformer2DModel.from_pretrained(
        tdir, torch_dtype=dtype).to(dev).eval()
    cfg = model.config
    hp = int(os.environ.get("QI21_HP", 8))
    wp = int(os.environ.get("QI21_WP", 8))
    text_seq = int(os.environ.get("QI21_SEQ", 24))
    g = torch.Generator().manual_seed(1234)
    latent = torch.randn(1, hp * wp, cfg.in_channels, generator=g,
                         dtype=torch.float32).to(dev, dtype)
    ehs = torch.randn(1, text_seq, cfg.context_in_dim, generator=g,
                      dtype=torch.float32).to(dev, dtype)
    run(model, latent, ehs, hp, wp, torch.device(dev), dtype, "real")

print("wrote reference to", out)
