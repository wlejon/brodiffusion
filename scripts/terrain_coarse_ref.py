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
    ap.add_argument("--raw", action="store_true", help="emit the 7-channel weighted form")
    # CUDA by default, because that is the path that ships. The reference's own
    # CPU-vs-CUDA spread on this stage is 1.5e-4 relative, well under the gate's
    # bar, so running on GPU costs no discrimination — and CPU costs about three
    # orders of magnitude in wall clock here.
    ap.add_argument("--device", default="cuda")
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
        weights, seed=int(args.seed), caching_strategy="direct")
    pipe.to(args.device)
    pipe.bind()

    r = pipe.coarse[0:7, args.i1:args.i2, args.j1:args.j2]
    arr = r.cpu().numpy()
    if not args.raw:
        arr = arr[:6] / arr[6:7]

    np.asarray(arr, dtype="<f4").tofile(out_path)
    print(f"coarse{' raw' if args.raw else ''} {arr.shape} "
          f"sum {float(arr.sum()):.4f} "
          f"elev [{float(arr[0].min()):.3f}, {float(arr[0].max()):.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
