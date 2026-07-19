#!/usr/bin/env bash
# Parity gate for the infinite-tensor evaluator (window arithmetic, tile store,
# lazy recursive materialisation).
#
# Demands BIT-EXACT equality with the Python infinite-tensor. That is achievable
# here because the scenario's compute functions return only small integers, so
# every value — including the sums produced when overlapping windows accumulate —
# is exactly representable in float32. A mismatch therefore cannot be rounding:
# it means the C++ visited a different set of windows, applied a different
# offset, or accumulated a different number of times. Float tolerances would let
# a genuine off-by-one in the window lattice hide inside rounding noise.
#
# The scenario (mirrored in `brodiffusion terrain-itensor` and
# scripts/terrain_itensor_ref.py) is the real coarse->latent edge in miniature:
#   * A's output windows overlap (size 4, stride 3) -> read-time summation
#   * B's output windows overlap (size 4, stride 2) -> summation one level up
#   * B reads A through size 3, stride 1, offset -1 -> the negative-offset
#     unit-stride arg edge the real pipeline uses
#   * requested ranges include negative world coordinates
#   * each range is read twice (the CLI self-checks idempotence), which catches
#     accumulation having been moved to write time
#   * every case runs at batch 1/4/16 and must agree bit-for-bit: `f` is pure, so
#     batching is purely a scheduling concern and must not reach the output
#
# Note on what this canNOT catch: substituting truncating division for the
# floor/ceil division in intersecting_windows provably cannot change the output.
# Truncation errs only toward a SUPERSET of windows, and a window that does not
# actually overlap the request contributes nothing to read_pixels. That makes the
# floor/ceil handling a cost guard (a spurious window is a full UNet evaluation
# in the real pipeline), not an output-correctness one. Verified by experiment,
# not assumed.
#
# Usage: scripts/terrain_itensor_parity.sh [path/to/brodiffusion.exe]
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

BIN="${1:-build_cpu/Release/brodiffusion.exe}"
[ -x "$BIN" ] || BIN="build_cuda/Release/brodiffusion.exe"
[ -x "$BIN" ] || BIN="build/Release/brodiffusion.exe"
[ -x "$BIN" ] || { echo "error: no brodiffusion.exe found" >&2; exit 2; }
BIN="$REPO/$BIN"
mkdir -p "$REPO/.parity"

CASES="0 1 2 3"
BATCHES="1 4 16"
status=0

for c in $CASES; do
  for b in $BATCHES; do
    REF="$REPO/.parity/it_ref_c${c}_b${b}.f32"
    MINE="$REPO/.parity/it_mine_c${c}_b${b}.f32"
    rm -f "$REF" "$MINE"

    python "$REPO/scripts/terrain_itensor_ref.py" --out "$REF" --case "$c" --batch "$b" >/dev/null \
      || { echo "case $c batch $b: REFERENCE FAILED"; status=1; continue; }

    # The CLI also self-checks that reading the same range twice is idempotent,
    # and exits nonzero if not — so a nonzero exit here is meaningful, not just
    # a missing subcommand.
    "$BIN" terrain-itensor --case "$c" --batch "$b" --out "$MINE" >/dev/null \
      || { echo "case $c batch $b: CLI FAILED"; status=1; continue; }
    [ -s "$MINE" ] || { echo "case $c batch $b: CLI WROTE NOTHING (stale binary?)"; status=1; continue; }

    REF="$REF" MINE="$MINE" C="$c" B="$b" python - <<'PY' || status=1
import os, sys
import numpy as np
ref  = np.fromfile(os.environ["REF"],  dtype="<f4")
mine = np.fromfile(os.environ["MINE"], dtype="<f4")
tag = f"case {os.environ['C']} batch {os.environ['B']:>2}"
if ref.shape != mine.shape:
    print(f"{tag}: SHAPE MISMATCH {ref.shape} vs {mine.shape}"); sys.exit(1)
nbad = int((ref.view(np.uint32) != mine.view(np.uint32)).sum())
frac = float(np.abs(ref - np.round(ref)).max())
# Guard the guard: if the scenario ever stops producing exact integers, the
# bit-exact bar silently becomes a much harder (and wrong) demand.
if frac != 0.0:
    print(f"{tag}: scenario no longer integer-valued (max frac {frac}) — bit-exact bar invalid")
    sys.exit(1)
print(f"{tag}: n={ref.size:4d}  differing bit patterns {nbad}  "
      f"{'OK' if nbad == 0 else 'FAILED'}")
sys.exit(0 if nbad == 0 else 1)
PY
  done
done

[ "$status" -eq 0 ] && echo "ALL INFINITE-TENSOR PARITY OK" || echo "INFINITE-TENSOR PARITY FAILED"
exit "$status"
