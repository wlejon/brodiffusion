#!/usr/bin/env python3
"""Reference scenario for the infinite-tensor evaluator port.

Runs a small two-node DAG through the real Python `infinite-tensor` and dumps the
results, so the C++ port can be compared against it bit-for-bit.

The compute functions here deliberately return only SMALL INTEGERS. Every value
that flows through the graph — including the sums produced when overlapping
windows accumulate — stays exactly representable in float32, so a mismatch can
only mean the C++ visited a different set of windows, applied a different offset,
or accumulated a different number of times. That isolates the graph structure,
which is the whole point: float tolerances would let a genuine off-by-one in the
window lattice hide inside rounding noise.

The scenario is shaped to hit the parts of the arithmetic that actually bite:

  * A's output windows OVERLAP (size 4, stride 3), so reading A exercises
    read-time summation of several windows per pixel.
  * B's output windows overlap too (size 4, stride 2), one level up.
  * B reads A through an arg window with a NEGATIVE OFFSET and unit stride
    (size 3, stride 1, offset -1) — the same shape as the real pipeline's
    coarse->latent edge, and the case where floor-vs-truncate division goes
    wrong.
  * Requested ranges include negative world coordinates.
  * The same range is read twice, which must be idempotent (dedup via
    is_window_processed) rather than double-counted.
  * The whole thing runs at several batch sizes, which must not change a single
    bit — `f` is pure, so batching is purely a scheduling concern.

Usage: scripts/terrain_itensor_ref.py --out <file.f32> [--batch N] [--case K]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "infinite-tensor"))

from infinite_tensor import InfiniteTensor, MemoryTileStore, TensorWindow

# (c0, c1) channel extent, then the y/x pixel ranges to read from B.
CASES = [
    ((0, 2), (0, 10), (0, 10)),      # origin, spans several windows
    ((0, 2), (-7, 5), (-13, -1)),    # straddles the origin on both axes
    ((0, 2), (5, 9), (5, 9)),        # small interior window
    ((0, 2), (-40, -32), (24, 32)),  # far from the origin, mixed signs
]


def f_a(ctx):
    """Leaf tile: a value determined entirely by the window index."""
    v = (ctx[1] * 7 + ctx[2] * 13) % 32
    out = torch.empty((2, 4, 4), dtype=torch.float32)
    for c in range(2):
        for y in range(4):
            for x in range(4):
                out[c, y, x] = float(v + c * 64 + y * 2 + x)
    return out


def f_b(ctxs, a_tiles):
    """Batched: one output per window index, each folding in its whole arg slice.

    Summing the entire arg slice is what makes this a real test — if the C++
    hands `f` a slice taken at the wrong offset, or one window too few, the sum
    changes and the mismatch is unmissable.
    """
    outs = []
    for ctx, a in zip(ctxs, a_tiles):
        s = float(a.sum())
        out = torch.empty((2, 4, 4), dtype=torch.float32)
        for c in range(2):
            for y in range(4):
                for x in range(4):
                    out[c, y, x] = s + c * 8 + y + x * 3
        outs.append(out)
    return outs


def build(batch_size: int):
    store = MemoryTileStore()
    a = InfiniteTensor(
        shape=(2, None, None),
        f=f_a,
        output_window=TensorWindow(size=(2, 4, 4), stride=(2, 3, 3)),
        tile_store=store,
        tensor_id="A",
    )
    b = InfiniteTensor(
        shape=(2, None, None),
        f=f_b,
        output_window=TensorWindow(size=(2, 4, 4), stride=(2, 2, 2)),
        args=(a,),
        args_windows=(TensorWindow(size=(2, 3, 3), stride=(2, 1, 1), offset=(0, -1, -1)),),
        batch_size=batch_size,
        tile_store=store,
        tensor_id="B",
    )
    return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--case", type=int, default=0)
    args = ap.parse_args()

    b = build(args.batch)
    (c0, c1), (y0, y1), (x0, x1) = CASES[args.case]

    r = b[c0:c1, y0:y1, x0:x1]
    # Read the identical range a second time. The store has already processed
    # every window involved, so this must return exactly the same values — if
    # accumulation happened at write time instead of read time, the second read
    # would come back doubled.
    r2 = b[c0:c1, y0:y1, x0:x1]
    if not torch.equal(r, r2):
        print("REFERENCE SELF-CHECK FAILED: repeated read is not idempotent", file=sys.stderr)
        return 1

    arr = r.numpy().astype("<f4")
    arr.tofile(args.out)
    print(f"case {args.case} batch {args.batch} shape {tuple(r.shape)} "
          f"sum {float(r.sum()):.1f} min {float(r.min()):.1f} max {float(r.max()):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
