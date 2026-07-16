#!/usr/bin/env python3
"""ARDY two-stage denoiser parity reference — history-conditioned window.

Same real g152 `AutoLatentTwostageDenoiser` as ardy_denoiser_ref.py, but exercises
the AUTOREGRESSIVE history path: a window of NUM_HISTORY clean history tokens
followed by 13 generation tokens (history_len = NUM_HISTORY*fpt, no future / no
constraints). This drives the history-token projections (global/local_root_
hybrid_proj), the negative token-index origin (history_len // fpt), and the
history/generation fusing that the C++ `ardy-denoiser-fwd --history-tok N` path
must match. Dumps the full (NUM_HISTORY+13, 148) hybrid input, text embedding, and
the predicted clean hybrid to .parity/.

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

# ── history + generation window inputs ──
NUM_HISTORY = int(os.environ.get("ARDY_HIST_TOK", "3"))
GEN_TOK = 13
T_TOK = NUM_HISTORY + GEN_TOK
FPT = 4
NUM_FRAMES = T_TOK * FPT
HYB = 148

torch.manual_seed(880088)
x = torch.randn(1, T_TOK, HYB)              # [history tokens | noisy generation tokens]
text_feat = torch.randn(1, 1, 4096)
timesteps = torch.tensor([3], dtype=torch.long)
first_heading = torch.tensor([0.6])

hist_frames = NUM_HISTORY * FPT
history_len = torch.tensor([hist_frames])
generation_len = torch.tensor([GEN_TOK * FPT])
future_len = torch.tensor([0])

idx = torch.arange(NUM_FRAMES)[None, :]
history_mask = idx < hist_frames
generation_mask = (idx >= hist_frames) & (idx < NUM_FRAMES)
future_mask = torch.zeros(1, NUM_FRAMES, dtype=torch.bool)

tidx = torch.arange(T_TOK)[None, :]
history_token_mask = tidx < NUM_HISTORY
generation_token_mask = (tidx >= NUM_HISTORY) & (tidx < T_TOK)
future_token_mask = torch.zeros(1, T_TOK, dtype=torch.bool)
text_pad_mask = torch.ones(1, 1, dtype=torch.bool)

with torch.no_grad():
    out = dec(
        x, history_len, generation_len, future_len,
        history_mask, generation_mask, future_mask,
        history_token_mask, generation_token_mask, future_token_mask,
        text_feat, text_pad_mask, timesteps, first_heading,
        motion_mask=None, observed_motion=None,
    )

out = out.squeeze(0)


def dump(name, t):
    a = t.detach().cpu().numpy().astype("<f4").ravel()
    a.tofile(PARITY / name)
    return a


dump("ardy_dnh_hybrid.f32", x.squeeze(0))
dump("ardy_dnh_text.f32", text_feat.squeeze(0))
ref_out = dump("ardy_dnh_ref_out.f32", out)

print(f"history={NUM_HISTORY} gen={GEN_TOK} T_tok={T_TOK} num_frames={NUM_FRAMES} timestep=3 heading=0.6")
print(f"pred hybrid mean {ref_out.mean():.6f} std {ref_out.std():.6f}")
print("dumped hybrid/text + ref out to .parity/")
