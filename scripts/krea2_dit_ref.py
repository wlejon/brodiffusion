#!/usr/bin/env python
# Reference for the Krea 2 image DiT (Krea2Transformer2DModel), to diff against
# brodiffusion's dit::Krea2Transformer2DModel (CLI `krea2-fwd`).
#
# Two modes:
#   synth  (default) — build a SMALL random-weight Krea2Transformer2DModel in
#       float32, save it (config.json + safetensors) to a synthetic dir, run one
#       forward, and dump the velocity + all inputs. Both sides run FP32 so this
#       is an EXACT architecture parity (no dtype drift, always fits in memory).
#   real   — load the real weights/krea-2-raw/transformer and run one forward on
#       CPU float32 (the BF16 model is ~24 GB, over a single card) using the
#       stage-1 reference conditioning; dumps velocity + inputs. Slow.
#
# Dumps (raw little-endian float32) into <outdir>:
#   krea2_dit_{latent,embeds,mask,velocity}.f32   and  the config/weights dir.
# The C++ harness reads latent/embeds/mask and compares its velocity.
#
# Usage: python scripts/krea2_dit_ref.py [synth|real] [outdir]
import os
import sys

import numpy as np
import torch

from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel
from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline

mode = sys.argv[1] if len(sys.argv) > 1 else "synth"
out = sys.argv[2] if len(sys.argv) > 2 else ".parity"
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)


def dump(name, t):
    t.detach().float().cpu().numpy().astype("<f4").tofile(os.path.join(out, name))


def run(model, hidden, ehs, timestep, mask, hp, wp, tag):
    text_seq = ehs.shape[1]
    pos = Krea2Pipeline.prepare_position_ids(text_seq, hp, wp, device=hidden.device)
    with torch.no_grad():
        vel = model(
            hidden_states=hidden,
            encoder_hidden_states=ehs,
            timestep=timestep,
            position_ids=pos,
            encoder_attention_mask=mask,
            return_dict=True,
        ).sample
    print(tag, "velocity", tuple(vel.shape),
          "mean %.5f std %.5f" % (float(vel.mean()), float(vel.std())))
    dump("krea2_dit_latent.f32", hidden[0])                      # (img_len, in_ch)
    dump("krea2_dit_embeds.f32", ehs[0].reshape(-1, ehs.shape[-1]))  # (seq*NL, TH)
    dump("krea2_dit_mask.f32", mask[0].float())                 # (seq,)
    dump("krea2_dit_velocity.f32", vel[0])                      # (img_len, in_ch)


if mode == "synth":
    cfg = dict(
        in_channels=64, num_layers=2, attention_head_dim=8, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, timestep_embed_dim=16,
        text_hidden_dim=24, num_text_layers=3, text_num_attention_heads=2,
        text_num_key_value_heads=2, text_intermediate_size=48,
        num_layerwise_text_blocks=2, num_refiner_text_blocks=2,
        axes_dims_rope=(4, 2, 2), rope_theta=1000.0, norm_eps=1e-5,
    )
    model = Krea2Transformer2DModel(**cfg).to(torch.float32).eval()
    # Randomize every parameter (incl. the zero-init RMSNorm gains and tables) so
    # no code path degenerates to identity.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.1)

    sdir = os.path.join(out, "krea2_synth_transformer")
    os.makedirs(sdir, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              os.path.join(sdir, "diffusion_pytorch_model.safetensors"))
    import json
    json.dump(cfg, open(os.path.join(sdir, "config.json"), "w"))

    hp, wp, text_seq = 4, 4, 5
    img_len = hp * wp
    hidden = torch.randn(1, img_len, cfg["in_channels"], dtype=torch.float32)
    ehs = torch.randn(1, text_seq, cfg["num_text_layers"], cfg["text_hidden_dim"],
                      dtype=torch.float32)
    mask = torch.tensor([[True, True, True, False, True]])   # a mid-sequence pad
    timestep = torch.tensor([0.7], dtype=torch.float32)
    run(model, hidden, ehs, timestep, mask, hp, wp, "synth")
    with open(os.path.join(out, "krea2_dit_dims.txt"), "w") as f:
        f.write("%d %d %d\n" % (hp, wp, text_seq))

else:
    tdir = "weights/krea-2-raw/transformer"
    model = Krea2Transformer2DModel.from_pretrained(
        tdir, torch_dtype=torch.float32).to("cpu").eval()
    cfg = model.config
    # Small hp/wp/text_seq: REAL full 12.9B weights, but a short sequence so the
    # CPU FP32 forward (dominated by weight loading, not L) finishes quickly.
    hp, wp, text_seq = int(os.environ.get("KREA2_DIT_HP", 2)), \
        int(os.environ.get("KREA2_DIT_WP", 2)), \
        int(os.environ.get("KREA2_DIT_SEQ", 12))
    img_len = hp * wp
    g = torch.Generator().manual_seed(1234)
    hidden = torch.randn(1, img_len, cfg.in_channels, generator=g, dtype=torch.float32)
    ehs = torch.randn(1, text_seq, cfg.num_text_layers, cfg.text_hidden_dim,
                      generator=g, dtype=torch.float32)
    mask = torch.zeros(1, text_seq, dtype=torch.bool)
    n_valid = max(1, text_seq // 2)
    mask[0, :n_valid] = True
    timestep = torch.tensor([0.7], dtype=torch.float32)
    run(model, hidden, ehs, timestep, mask, hp, wp, "real")
    with open(os.path.join(out, "krea2_dit_dims.txt"), "w") as f:
        f.write("%d %d %d\n" % (hp, wp, text_seq))

print("wrote reference to", out)
