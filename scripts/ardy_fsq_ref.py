#!/usr/bin/env python3
"""ARDY FSQ motion-decoder (detokenize) parity reference.

Instantiates the real ardy DoubleCondDecoderTransformer, loads the g152
`tokenizer.safetensors` decoder weights + post-quantization stats, and runs the
`FSQVAETransformer.detokenize` computation (unnormalize -> re-round to the FSQ
grid -> conditioned decode) on a deterministic set of token embeddings + local
root, dumping inputs and the decoded pose to .parity/ for the C++
`brodiffusion ardy-fsq-detok` subcommand to match.

Real trained weights (nvidia/ARDY-G1-RP-25FPS-Horizon52). Only the module + its
own weights are used — no vector_quantize_pytorch dependency (the FSQ re-round
is a scalar op we replicate directly, exactly as fsq.py does at inference).

Env: ARDY_REPO overrides the ardy checkout (default ../ardy);
     ARDY_CKPT overrides the checkpoint dir (default weights/ardy-g152).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]
ARDY = Path(os.environ.get("ARDY_REPO", REPO.parent / "ardy"))
CKPT = Path(os.environ.get("ARDY_CKPT", REPO / "weights" / "ardy-g152"))

# Load transformer.py directly (it only needs torch + einops); importing via the
# ardy.model package would drag in omegaconf and the whole model tower.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ardy_dec_transformer",
    ARDY / "ardy" / "model" / "autoencoder" / "transformer.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DoubleCondDecoderTransformer = _mod.DoubleCondDecoderTransformer

PARITY = REPO / ".parity"
PARITY.mkdir(exist_ok=True)

# ── g152 decoder config (config.yaml + derived motion-rep dims) ──
TOKEN_DIM = 128           # num_fsq_levels / latent_embedding_dim
OUTPUT_DIM = 413          # pose = local_root(4) + body(409)
FPT = 4                   # num_frames_per_token
LATENT = 512
HEADS = 4
FF = 1024
LAYERS = 8
EXT_DIM = 4               # local root
FSQ_LEVEL = 64            # half_width = 32
EPS = 1e-5

T_TOK = 13                # 13 tokens -> 52 frames (gen_horizon)
NUM_FRAMES = T_TOK * FPT

torch.manual_seed(4321)

dec = DoubleCondDecoderTransformer(
    input_dim=TOKEN_DIM,
    output_dim=OUTPUT_DIM,
    num_frames_per_token=FPT,
    latent_dim=LATENT,
    num_heads=HEADS,
    ff_size=FF,
    dropout=0.0,
    activation="gelu",
    norm_first=False,
    num_layers=LAYERS,
    pe_dropout=0.0,
    is_causal=True,
    target_cond_dim=409,   # body (present in weights, unused without target cond)
    external_cond_dim=EXT_DIM,
)

# Load decoder.* weights from tokenizer.safetensors (strip 'pose_net.decoder.').
full = load_file(str(CKPT / "tokenizer.safetensors"))
prefix = "pose_net.decoder."
sd = {k[len(prefix):]: v for k, v in full.items() if k.startswith(prefix)}
missing, unexpected = dec.load_state_dict(sd, strict=False)
# target_cond_blocks are in the module; everything the module needs must load.
assert not missing, f"missing decoder params: {missing}"
dec.eval()

# Post-quantization stats.
pq_mean = torch.from_numpy(np.load(CKPT / "stats/post_quantization/mean.npy")).float()
pq_std = torch.from_numpy(np.load(CKPT / "stats/post_quantization/std.npy")).float()
std_eps = torch.sqrt(pq_std**2 + EPS)

# Deterministic inputs: normalized token embeddings + local-root condition.
tokens_norm = torch.randn(1, T_TOK, TOKEN_DIM)            # (1, T_tok, 128)
local_root = torch.randn(1, NUM_FRAMES, EXT_DIM) * 0.3    # (1, num_frames, 4)

# detokenize front half: unnormalize -> re-round to FSQ grid.
half = FSQ_LEVEL // 2  # 32
tq = tokens_norm * std_eps + pq_mean
tq = torch.round(tq.clamp(-1, 1) * half) / half

with torch.no_grad():
    out = dec(tq, external_cond=local_root)               # (1, num_frames, 413)

out = out.squeeze(0)                                       # (num_frames, 413)


def dump(name, t):
    a = t.detach().cpu().numpy().astype("<f4").ravel()
    a.tofile(PARITY / name)
    return a


dump("ardy_fsq_tokens.f32", tokens_norm.squeeze(0))        # (T_tok, 128)
dump("ardy_fsq_localroot.f32", local_root.squeeze(0))      # (num_frames, 4)
ref_out = dump("ardy_fsq_ref_out.f32", out)                # (num_frames, 413)

print(f"T_tok={T_TOK} num_frames={NUM_FRAMES} out_dim={OUTPUT_DIM}")
print(f"decoded pose mean {ref_out.mean():.6f} std {ref_out.std():.6f}")
print("dumped tokens/localroot + ref out to .parity/")
