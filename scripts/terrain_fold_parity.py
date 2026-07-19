#!/usr/bin/env python3
"""Parity gate for the terrain-diffusion weight fold.

Proves that pre-folding MPConv's inference-time weight transform into the stored
weights is *exact* — i.e. that the converted checkpoint, run through a network
whose MPConv does a bare conv2d with no normalize and no gain, reproduces the
upstream network bit-for-bit (to fp32 rounding).

This is the load-bearing claim of convert-terrain-diffusion.py. If it holds, the
C++ port needs no weight normalization at runtime, no bias tensors, and no
learned-gain plumbing — every MPConv is a plain conv2d.

Both sides run the REAL upstream EDMUnet2D from ../terrain-diffusion with the
REAL downloaded checkpoint. The only difference is the weights loaded and a
one-line monkey-patch of MPConv.forward.

    reference: original weights + MPConv.forward  (normalize -> *gain/sqrt(fan_in) -> conv2d)
    mine:      folded weights   + bare conv2d

Usage:
  scripts/terrain_fold_parity.py [--src DIR] [--dst DIR] [--td-root DIR] [--size N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

STAGES = (("coarse", "coarse_model"), ("base", "base_model"), ("decoder", "decoder_model"))


def bare_mpconv_forward(self, x, gain=1):
    """MPConv.forward with the weight transform removed — the folded weights
    already carry normalize() * gain/sqrt(fan_in), so `gain` is ignored."""
    w = self.weight
    if w.ndim == 2:
        return F.linear(x, w)
    assert w.ndim == 4
    return F.conv2d(x, w, padding=(0 if self.no_padding else w.shape[-1] // 2,), groups=self.groups)


def make_inputs(stage: str, cfg: dict, size: int, gen: torch.Generator):
    """Deterministic inputs matching each stage's conditioning signature."""
    b = 1
    ch_in = cfg["in_channels"]
    x = torch.randn(b, ch_in, size, size, generator=gen)
    # Noise labels live on the TrigFlow arc t = atan(sigma/sigma_data) in (0, pi/2).
    noise = torch.full((b,), 1.2)
    conds = []
    for kind, dim, _w in (cfg.get("conditional_inputs") or []):
        if kind == "float":
            conds.append(torch.randn(b, generator=gen))
        elif kind == "tensor":
            conds.append(torch.randn(b, dim, generator=gen))
        else:
            raise SystemExit(f"{stage}: unsupported conditional input kind {kind!r}")
    return x, noise, conds


def compare(ref: torch.Tensor, mine: torch.Tensor) -> tuple[float, float, float]:
    r = ref.detach().reshape(-1).double()
    m = mine.detach().reshape(-1).double()
    cos = float(torch.dot(r, m) / (r.norm() * m.norm()))
    rel = float((r - m).norm() / (r.norm() + 1e-30))
    mad = float((r - m).abs().max())
    return cos, rel, mad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="weights/terrain-diffusion-30m", type=Path)
    ap.add_argument("--dst", default=None, type=Path)
    ap.add_argument("--td-root", default=Path("../terrain-diffusion"), type=Path)
    ap.add_argument("--size", default=64, type=int,
                    help="spatial size for the test forward (nets are fully convolutional)")
    args = ap.parse_args()

    src: Path = args.src
    dst: Path = args.dst or src.parent / f"{src.name}-bro"
    td_root: Path = args.td_root.resolve()

    if not (td_root / "terrain_diffusion").is_dir():
        raise SystemExit(f"error: --td-root {td_root} has no terrain_diffusion/ package")
    if not dst.is_dir():
        raise SystemExit(f"error: {dst} missing — run convert-terrain-diffusion.py first")

    sys.path.insert(0, str(td_root))
    from terrain_diffusion.models.edm_unet import EDMUnet2D  # noqa: E402
    from terrain_diffusion.models.mp_layers import MPConv     # noqa: E402

    torch.manual_seed(0)
    status = 0

    for stage, sub in STAGES:
        cfg = json.loads((src / sub / "config.json").read_text())
        cfg_clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
        orig_sd = load_file(str(src / sub / "diffusion_pytorch_model.safetensors"))
        fold_sd = load_file(str(dst / f"{stage}.safetensors"))

        size = args.size
        # The coarse model is a single-level 16x16 net; keep it at its native size.
        if len(cfg_clean.get("model_channel_mults") or [1, 2, 3, 4]) == 1:
            size = min(size, 16)

        gen = torch.Generator().manual_seed(1234)
        x, noise, conds = make_inputs(stage, cfg_clean, size, gen)

        # ---- reference: original weights, original MPConv.forward
        ref_model = EDMUnet2D.from_config(cfg_clean)
        missing, unexpected = ref_model.load_state_dict(orig_sd, strict=False)
        if missing or unexpected:
            raise SystemExit(f"{stage}: reference load mismatch {missing=} {unexpected=}")
        ref_model.eval()  # critical: train mode mutates self.weight in MPConv.forward
        with torch.no_grad():
            ref = ref_model(x, noise_labels=noise, conditional_inputs=conds)

        # ---- mine: folded weights, bare conv2d
        mine_model = EDMUnet2D.from_config(cfg_clean)
        m2, u2 = mine_model.load_state_dict(fold_sd, strict=False)
        # Folded checkpoints deliberately omit the gain scalars (folded in) and
        # the training-only logvar head. Anything else missing is a real bug.
        unexpected_missing = [k for k in m2
                              if not (k.endswith("emb_gain") or k == "out_gain"
                                      or k.startswith("logvar_"))]
        if unexpected_missing or u2:
            raise SystemExit(f"{stage}: folded load mismatch {unexpected_missing=} {u2=}")
        mine_model.eval()
        orig_forward = MPConv.forward
        try:
            MPConv.forward = bare_mpconv_forward
            with torch.no_grad():
                mine = mine_model(x, noise_labels=noise, conditional_inputs=conds)
        finally:
            MPConv.forward = orig_forward

        cos, rel, mad = compare(ref, mine)
        ok = cos > 0.99999999 and rel < 1e-5
        status |= 0 if ok else 1
        print(f"== {stage:8s} {size}x{size} out {tuple(ref.shape)}")
        print(f"   cosine {cos:.10f}  relL2 {rel:.3e}  maxabsdiff {mad:.3e}")
        print(f"   ref std {ref.std():.6f}  mine std {mine.std():.6f}")
        print(f"   {'PARITY OK' if ok else 'PARITY FAILED'}")

    print("\nALL PARITY OK" if status == 0 else "\nPARITY FAILED")
    return status


if __name__ == "__main__":
    sys.exit(main())
