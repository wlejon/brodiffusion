#!/usr/bin/env python3
"""ARDY spaced-DDIM text-to-motion window parity reference.

Drives the REAL ardy `Diffusion` + `DDIMSampler` through the full generation
loop (ardy_model.py denoising_step / _generate_window) for a text-only,
no-history window: a fresh 52-frame / 13-token block denoised from a fixed noise
seed. Uses the real `AutoLatentTwostageDenoiser` (g152) with separated CFG done
explicitly (text / uncond passes; the constraint pass is bit-identical to uncond
for a constraint-free window, so it collapses to uncond + w*(text-uncond) — the
same collapse the C++ `ardy-sample` relies on).

Dumps the noise seed, text embedding, and the final denoised hybrid to .parity/
for the C++ `ardy-sample` subcommand to match.

Real trained weights (nvidia/ARDY-G1-RP-25FPS-Horizon52).

Env: ARDY_REPO (default ../ardy); ARDY_CKPT (default weights/ardy-g152).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]
ARDY = Path(os.environ.get("ARDY_REPO", REPO.parent / "ardy"))
CKPT = Path(os.environ.get("ARDY_CKPT", REPO / "weights" / "ardy-g152"))
PARITY = REPO / ".parity"
PARITY.mkdir(exist_ok=True)

sys.path.insert(0, str(ARDY))
import ardy  # noqa: E402  (empty __init__)

# Lightweight ardy.model package (skip the heavy __init__ = hydra/transformers).
_model_pkg = types.ModuleType("ardy.model")
_model_pkg.__path__ = [str(ARDY / "ardy" / "model")]
sys.modules["ardy.model"] = _model_pkg
_loading = types.ModuleType("ardy.model.loading")
_loading.load_checkpoint_state_dict = lambda *a, **k: {}
sys.modules["ardy.model.loading"] = _loading

from ardy.model.auto_latent_twostage_denoiser import AutoLatentTwostageDenoiser  # noqa: E402
from ardy.model.diffusion import DDIMSampler, Diffusion  # noqa: E402
from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep  # noqa: E402
from ardy.skeleton import G1Skeleton34  # noqa: E402

# ── build motion rep + denoiser (g152 config) ──
motion_rep = ArdyMotionRep(
    skeleton=G1Skeleton34(),
    fps=25,
    stats_path=str(CKPT / "stats" / "motion"),
)
kwargs = dict(
    latent_dim=1024,
    positional_encoding_mode="learned_prefix_zero_at_first_generation",
    llm_shape=[1, 4096],
    use_text_mask=False,
    ff_size=2048,
    num_layers=8,
    num_heads=8,
    activation="gelu",
    dropout=0.0,
    pe_dropout=0.0,
    norm_first=False,
    input_first_heading_angle=True,
)
dec = AutoLatentTwostageDenoiser(
    motion_rep=motion_rep,
    motion_mask_mode="concat",
    num_frames_per_token=4,
    nframe_root_dim=20,
    latent_embedding_dim=128,
    **kwargs,
)
full = load_file(str(CKPT / "denoiser.safetensors"))
sd = {k.replace("denoiser.", "", 1): v for k, v in full.items() if k.startswith("denoiser")}
sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
dec.load_state_dict(sd)
dec.eval()

# ── generation-only window ──
T_TOK = 13
FPT = 4
NUM_FRAMES = T_TOK * FPT  # 52
HYB = 148
NUM_BASE_STEPS = 10
NUM_DENOISING_STEPS = 10
CFG_WEIGHT = 2.5
FIRST_HEADING = 0.6

torch.manual_seed(240240)
x_init = torch.randn(1, T_TOK, HYB)
text_feat = torch.randn(1, 1, 4096)
first_heading = torch.tensor([FIRST_HEADING])
text_pad_mask = torch.ones(1, 1, dtype=torch.bool)

# masks: history_len 0, all-generation, no future.
history_len = torch.tensor([0])
generation_len = torch.tensor([NUM_FRAMES])
future_len = torch.tensor([0])
history_mask = torch.zeros(1, NUM_FRAMES, dtype=torch.bool)
generation_mask = torch.ones(1, NUM_FRAMES, dtype=torch.bool)
future_mask = torch.zeros(1, NUM_FRAMES, dtype=torch.bool)
history_token_mask = torch.zeros(1, T_TOK, dtype=torch.bool)
generation_token_mask = torch.ones(1, T_TOK, dtype=torch.bool)
future_token_mask = torch.zeros(1, T_TOK, dtype=torch.bool)

diffusion = Diffusion(num_base_steps=NUM_BASE_STEPS)
sampler = DDIMSampler(diffusion)
use_timesteps, map_tensor = diffusion.space_timesteps(NUM_DENOISING_STEPS)
diffusion.calc_diffusion_vars(use_timesteps)


def denoise_pass(x, t_map, tf):
    """One x0-prediction denoiser forward at base timestep t_map."""
    with torch.no_grad():
        return dec(
            x, history_len, generation_len, future_len,
            history_mask, generation_mask, future_mask,
            history_token_mask, generation_token_mask, future_token_mask,
            tf, text_pad_mask, t_map, first_heading,
            motion_mask=None, observed_motion=None,
        )


x = x_init.clone()
indices = list(range(NUM_DENOISING_STEPS))[::-1]
for i in indices:
    t = torch.tensor([i])
    t_map = map_tensor[t]
    out_text = denoise_pass(x, t_map, text_feat)
    out_uncond = denoise_pass(x, t_map, 0 * text_feat)
    x0 = out_uncond + CFG_WEIGHT * (out_text - out_uncond)  # separated-CFG collapse
    x = sampler(x, x0, t)

out = x.squeeze(0)


def dump(name, t):
    a = t.detach().cpu().numpy().astype("<f4").ravel()
    a.tofile(PARITY / name)
    return a


dump("ardy_smp_xinit.f32", x_init.squeeze(0))
dump("ardy_smp_text.f32", text_feat.squeeze(0))
ref_out = dump("ardy_smp_ref_out.f32", out)

print(f"T_tok={T_TOK} steps={NUM_DENOISING_STEPS} cfg={CFG_WEIGHT} heading={FIRST_HEADING}")
print(f"use_timesteps={use_timesteps.tolist()}")
print(f"final hybrid mean {ref_out.mean():.6f} std {ref_out.std():.6f}")
print("dumped x_init/text + ref out to .parity/")
