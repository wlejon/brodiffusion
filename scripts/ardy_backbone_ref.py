#!/usr/bin/env python3
"""ARDY denoiser backbone (TransformerEncoderBlock) parity reference.

Instantiates the real ardy `TransformerEncoderBlock` (backbone.py) with the g152
denoiser config, loads one stage's weights from `denoiser.safetensors`
(`denoiser.backbone.{root_model,body_model}.*`), and runs a single forward on a
deterministic pre-projected motion input + text/timestep/heading conditioning —
the first-generation-window case (token_index = arange(T), all-attendable).
Dumps inputs and the block output to .parity/ for the C++ `ardy-backbone-fwd`
subcommand to match.

Real trained weights (nvidia/ARDY-G1-RP-25FPS-Horizon52); only the module + its
own weights are used (no omegaconf / model tower).

Env: ARDY_REPO overrides the ardy checkout (default ../ardy);
     ARDY_CKPT overrides the checkpoint dir (default weights/ardy-g152);
     ARDY_STAGE selects root|body (default root).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]
ARDY = Path(os.environ.get("ARDY_REPO", REPO.parent / "ardy"))
# backbone.py does `from ardy.tools import validate` — make the package importable.
sys.path.insert(0, str(ARDY))
CKPT = Path(os.environ.get("ARDY_CKPT", REPO / "weights" / "ardy-g152"))
STAGE = os.environ.get("ARDY_STAGE", "root")

PARITY = REPO / ".parity"
PARITY.mkdir(exist_ok=True)

# Load backbone.py directly (needs torch + einops + pydantic + omegaconf.ListConfig).
_spec = importlib.util.spec_from_file_location(
    "ardy_backbone", ARDY / "ardy" / "model" / "backbone.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TransformerEncoderBlock = _mod.TransformerEncoderBlock


# ── minimal skeleton stub (backbone only reads skeleton.nbjoints) ──
class _Skel:
    nbjoints = 34


# g152 denoiser config (config.yaml)
LATENT = 1024
HEADS = 8
FF = 2048
LAYERS = 8
LLM_SHAPE = [1, 4096]
OUTPUT_DIM = 128 if STAGE == "body" else 20

T = 13  # gen_horizon 52 / fpt 4
torch.manual_seed(20260715 + (1 if STAGE == "body" else 0))

block = TransformerEncoderBlock(
    input_dim=-1,
    output_dim=OUTPUT_DIM,
    skeleton=_Skel(),
    llm_shape=LLM_SHAPE,
    use_text_mask=False,
    latent_dim=LATENT,
    ff_size=FF,
    num_layers=LAYERS,
    num_heads=HEADS,
    activation="gelu",
    dropout=0.0,
    pe_dropout=0.0,
    norm_first=False,
    input_first_heading_angle=True,
    add_input_proj=False,
    positional_encoding_mode="learned_prefix_zero_at_first_generation",
)

# Load stage weights (strip 'denoiser.backbone.<stage>_model.').
full = load_file(str(CKPT / "denoiser.safetensors"))
prefix = f"denoiser.backbone.{STAGE}_model."
sd = {k[len(prefix):]: v for k, v in full.items() if k.startswith(prefix)}
missing, unexpected = block.load_state_dict(sd, strict=False)
assert not missing, f"missing backbone params: {missing}"
block.eval()

# Deterministic inputs.
x = torch.randn(1, T, LATENT)                 # pre-projected motion tokens
text_feat = torch.randn(1, 1, 4096)           # one 4096-dim text token
timesteps = torch.tensor([3], dtype=torch.long)
first_heading = torch.tensor([0.6])           # radians
token_index = torch.arange(T)[None, :]        # (1, T), first window origin 0
x_pad_mask = torch.ones(1, T, dtype=torch.bool)
text_pad_mask = torch.ones(1, 1, dtype=torch.bool)

with torch.no_grad():
    out = block(
        x, x_pad_mask, text_feat, text_pad_mask,
        timesteps, first_heading, token_index,
    )                                          # (1, T, OUTPUT_DIM)

out = out.squeeze(0)


def dump(name, t):
    a = t.detach().cpu().numpy().astype("<f4").ravel()
    a.tofile(PARITY / name)
    return a


dump(f"ardy_bb_{STAGE}_x.f32", x.squeeze(0))
dump(f"ardy_bb_{STAGE}_text.f32", text_feat.squeeze(0))
ref_out = dump(f"ardy_bb_{STAGE}_ref_out.f32", out)

print(f"stage={STAGE} T={T} out_dim={OUTPUT_DIM} timestep=3 heading=0.6")
print(f"out mean {ref_out.mean():.6f} std {ref_out.std():.6f}")
print("dumped x/text + ref out to .parity/")
