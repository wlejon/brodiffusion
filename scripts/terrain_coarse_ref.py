#!/usr/bin/env python3
"""Reference coarse world map, straight out of the real PyTorch WorldPipeline.

Drives upstream's own `WorldPipeline.coarse` InfiniteTensor rather than
reimplementing the stage, so the reference is upstream by construction: if the
port and this disagree, the port is wrong.

Note this reads the tensor's raw 7-channel weighted form (`value*w` in 0..5,
`w` in 6) and, unless --raw, divides to recover the weighted mean — the same
thing WorldPipeline.get does downstream. Comparing the raw form as well matters:
a bug that scaled both the value and the weight identically would cancel out of
the normalized view and stay invisible until the next stage consumed it.

Usage: scripts/terrain_coarse_ref.py --weights DIR --seed N --out F [--raw]
                                     [--i1 A --j1 B --i2 C --j2 D]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SIBLINGS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_SIBLINGS / "terrain-diffusion"))
sys.path.insert(0, str(_SIBLINGS / "infinite-tensor"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="UPSTREAM checkpoint dir (not the -bro one)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", action="store_true", help="emit the weighted form")
    ap.add_argument("--latent", action="store_true", help="read the latent stage instead")
    # The latent stage is two TrigFlow steps across two tensors. Reading the
    # first alone halves the composition depth, which is what says whether a
    # discrepancy accumulates per step or arrives from the coarse input.
    ap.add_argument("--latent-init", action="store_true")
    ap.add_argument("--residual", action="store_true")
    # CUDA by default, because that is the path that ships. The reference's own
    # CPU-vs-CUDA spread on this stage is 1.5e-4 relative, well under the gate's
    # bar, so running on GPU costs no discrimination — and CPU costs about three
    # orders of magnitude in wall clock here.
    ap.add_argument("--device", default="cuda")
    # The port computes in FP16 on CUDA (brotensor::compute_dtype), so
    # --dtype fp16 is what makes the reference comparable rather than
    # merely close. FP32 is kept available to separate "the port is wrong"
    # from "the port is running in half precision".
    ap.add_argument("--dtype", default=None, choices=[None, "fp16", "bf16"])
    ap.add_argument("--i1", type=int, default=0)
    ap.add_argument("--j1", type=int, default=0)
    ap.add_argument("--i2", type=int, default=64)
    ap.add_argument("--j2", type=int, default=64)
    args = ap.parse_args()

    # Resolve before the chdir below, since both are given relative to brodiffusion.
    weights = str(Path(args.weights).resolve())
    out_path = str(Path(args.out).resolve())

    # Upstream resolves its synthetic-map stats cache off the CWD, and will
    # otherwise try to recompute it — which prompts for a WorldClim download and
    # dies on EOF. Run from the checkout root so it finds the cache built by
    # scripts/build-terrain-synthetic-stats.py.
    import os
    os.chdir(_SIBLINGS / "terrain-diffusion")

    import torch
    from terrain_diffusion.inference.world_pipeline import WorldPipeline

    torch.set_grad_enabled(False)

    # seed and caching_strategy are constructor kwargs (both are in
    # ignore_for_config, so the shipped config carries neither). 'direct' keeps
    # the store in memory, so nothing persists between runs and each invocation
    # regenerates from scratch — which is the point: a cached tile would hide a
    # nondeterminism bug rather than expose it.
    pipe = WorldPipeline.from_pretrained(
        weights, seed=int(args.seed), caching_strategy="direct", dtype=args.dtype)
    pipe.to(args.device)
    pipe.bind()

    ch = 2 if args.residual else (6 if (args.latent or args.latent_init) else 7)
    tensor = pipe.residual if args.residual else (pipe.latents if args.latent else pipe.coarse)
    if args.latent_init:
        tensor, ch = pipe.latents.args[0], 6
    r = tensor[0:ch, args.i1:args.i2, args.j1:args.j2]
    arr = r.cpu().numpy()
    if not args.raw:
        arr = arr[:ch - 1] / arr[ch - 1:ch]

    np.asarray(arr, dtype="<f4").tofile(out_path)
    print(f"{'latent' if args.latent else 'coarse'}{' raw' if args.raw else ''} {arr.shape} "
          f"sum {float(arr.sum()):.4f} "
          f"elev [{float(arr[0].min()):.3f}, {float(arr[0].max()):.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
