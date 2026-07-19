#!/usr/bin/env bash
# Parity gate for the tile-seeded Gaussian noise field.
#
# Unlike the network gates, this one demands BIT-EXACT equality with upstream.
# The field is integer-derived (PCG64 + Marsaglia polar), so matching it exactly
# costs nothing and buys seed compatibility with upstream-generated worlds.
# Anything short of exact means the tile lattice or the RNG stream has drifted.
#
# Covers the two portability traps specifically:
#   * negative world coordinates  — Python's // floors, C++ / truncates. Getting
#     this wrong shifts the tile lattice across the origin.
#   * windows straddling tile boundaries — the whole point of the design.
#
# Plus the seam invariant (overlapping windows agree exactly), which is what
# InfiniteDiffusion's blending depends on and which needs no Python at all.
#
# Usage: scripts/terrain_rng_parity.sh [path/to/brodiffusion.exe]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-build_cpu/Release/brodiffusion.exe}"
[ -x "$BIN" ] || BIN="build_cuda/Release/brodiffusion.exe"
[ -x "$BIN" ] || BIN="build/Release/brodiffusion.exe"
[ -x "$BIN" ] || { echo "error: no brodiffusion.exe found" >&2; exit 2; }
mkdir -p .parity

# seed y0 x0 h w channels tile   — chosen to exercise origin, negatives,
# tile-straddling windows, multi-channel, and a tile size that does not divide
# the window.
CASES=(
  "12345      0      0   64  64  1  256"
  "12345     10      7   64  64  1   32"
  "12345   -100    -73   64  64  1   32"
  "999       -1     -1    8   8  1    4"
  "7      -1000  -1000   40  40  6   16"
  "42       250    250   64  64  5  256"
  "2024   65536 -65536   33  17  3   16"
)

status=0
for case in "${CASES[@]}"; do
  read -r SEED Y X H W C T <<<"$case"
  echo "== seed=$SEED (y0=$Y x0=$X) ${C}x${H}x${W} tile=$T =="

  SEED=$SEED Y=$Y X=$X H=$H W=$W C=$C T=$T python - <<'PY'
import ast, os, sys, numpy as np
from pathlib import Path
td = Path(os.environ.get("TERRAIN_TD_ROOT", "../terrain-diffusion")).resolve()
sys.path.insert(0, str(td))

# portable_rng imports cleanly (numpy + numba), so the RNG core -- PCG64 and the
# Marsaglia polar transform, i.e. the part that is actually hard to match -- is
# the REAL upstream implementation.
from terrain_diffusion.inference.portable_rng import fill_standard_normal

# world_pipeline.py, which holds _tile_seed and gaussian_noise_patch, also
# imports h5py / infinite_tensor / matplotlib at module scope. We deliberately
# do not depend on those, so lift the two functions we need straight out of the
# upstream SOURCE via AST rather than transcribing them by hand -- a hand
# transcription would just be testing our own code against our own reading of it.
src = (td / "terrain_diffusion" / "inference" / "world_pipeline.py").read_text()
tree = ast.parse(src)
want = {"_tile_seed", "gaussian_noise_patch"}
picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
if {n.name for n in picked} != want:
    raise SystemExit(f"could not lift {want - {n.name for n in picked}} from world_pipeline.py")
ns = {"np": np, "fill_standard_normal": fill_standard_normal}
exec(compile(ast.Module(body=picked, type_ignores=[]), "<upstream>", "exec"), ns)

e = os.environ
a = ns["gaussian_noise_patch"](int(e["SEED"]), int(e["Y"]), int(e["X"]),
                               int(e["H"]), int(e["W"]), channels=int(e["C"]),
                               tile_h=int(e["T"]), tile_w=int(e["T"]))
a.astype("<f4").tofile(".parity/terrain_rng_ref.f32")
PY

  "$BIN" terrain-rng --seed "$SEED" --y0 "$Y" --x0 "$X" --h "$H" --w "$W" \
                     --channels "$C" --tile "$T" --selfcheck \
                     --out .parity/terrain_rng_mine.f32 || status=1

  python - <<'PY' || status=1
import numpy as np, sys
ref  = np.fromfile('.parity/terrain_rng_ref.f32',  dtype='<f4')
mine = np.fromfile('.parity/terrain_rng_mine.f32', dtype='<f4')
if ref.shape != mine.shape:
    print(f"         SHAPE MISMATCH ref {ref.shape} mine {mine.shape}"); sys.exit(1)
# Bit-exact: compare the raw bit patterns, so a -0.0/+0.0 or NaN difference
# cannot slip through a float comparison.
exact = np.array_equal(ref.view(np.uint32), mine.view(np.uint32))
nbad = int((ref.view(np.uint32) != mine.view(np.uint32)).sum())
print(f"         {ref.size} values, {nbad} differing bit patterns  "
      f"ref mean {ref.mean():+.5f} std {ref.std():.5f}")
print("         BIT-EXACT OK" if exact else "         BIT-EXACT FAILED")
sys.exit(0 if exact else 1)
PY
done

[ "$status" -eq 0 ] && echo "ALL RNG PARITY OK" || echo "RNG PARITY FAILED"
exit "$status"
