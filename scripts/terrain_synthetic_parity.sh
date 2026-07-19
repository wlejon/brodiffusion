#!/usr/bin/env bash
# Parity gate for the synthetic climate map that conditions the coarse stage.
#
# Checks the two halves separately, because they fail in different ways and a
# single end-to-end number would not say which broke:
#
#   --raw   quantile-matched Perlin only. Catches a wrong FBm config, a swapped
#           coordinate axis, an off-by-one in the interpolation, a stale table.
#   full    the climate couplings on top (lapse rate, seasonality rebaselining,
#           precipitation-seasonality damping) plus the signed-sqrt elevation.
#           Catches sign and pivot errors in finalize().
#
# Windows cover the origin, negative world coordinates, a far offset, and a
# NON-SQUARE window — that last one specifically because upstream flattens a
# meshgrid and reshapes to the transposed extent, so a square window would hide
# an axis swap that a rectangular one exposes immediately.
#
# Seeds are deliberately nonzero: upstream reads `seed or random.randint(...)`,
# so seed 0 means "randomise" there and would make this gate nondeterministic.
# Our C++ treats 0 as a literal seed (documented in synthetic_map.cpp).
#
# Tolerance is not bit-exactness here, unlike the RNG gate. Upstream runs the
# climate couplings in numpy float32 while we run them in double and round once
# at the end; ours is the more accurate of the two, so demanding equality would
# be demanding we reproduce their rounding error. The bar below is tight enough
# that the negative controls in --selftest all breach it comfortably.
#
# Usage: scripts/terrain_synthetic_parity.sh [path/to/brodiffusion.exe]
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

BIN="${1:-build_cpu/Release/brodiffusion.exe}"
[ -x "$BIN" ] || BIN="build_cuda/Release/brodiffusion.exe"
[ -x "$BIN" ] || BIN="build/Release/brodiffusion.exe"
[ -x "$BIN" ] || { echo "error: no brodiffusion.exe found" >&2; exit 2; }
BIN="$REPO/$BIN"

STATS="$REPO/weights/terrain-diffusion-30m-bro/synthetic_map_stats.json"
[ -f "$STATS" ] || { echo "error: $(basename "$STATS") missing — run scripts/build-terrain-synthetic-stats.py" >&2; exit 2; }
mkdir -p "$REPO/.parity"
REF="$REPO/.parity/terrain_synth_ref.f32"
MINE="$REPO/.parity/terrain_synth_mine.f32"

# seed i1 j1 i2 j2
CASES=(
  "1234      0     0    64    64"
  "1234    -96   -96   -32   -32"
  "77    -1000   500  -936   564"
  "5         0     0    48    80"
)

status=0
for mode in raw full; do
for case in "${CASES[@]}"; do
  read -r SEED I1 J1 I2 J2 <<<"$case"
  echo "== $mode  seed=$SEED  ($I1,$J1)-($I2,$J2) =="

  SEED=$SEED I1=$I1 J1=$J1 I2=$I2 J2=$J2 MODE=$mode REF="$REF" python - <<'PY'
import os, sys
import numpy as np
from pathlib import Path

e = os.environ
ref_path = Path(e["REF"])                       # absolute: we chdir below
td = Path(e.get("TERRAIN_TD_ROOT", "../terrain-diffusion")).resolve()
sys.path.insert(0, str(td))
os.chdir(td)                                    # upstream resolves its stats cache off the CWD

from terrain_diffusion.inference.synthetic_map import make_synthetic_map_factory

# The shipped 30 m checkpoint's values, which must match both the quantile tables
# in weights/ and SyntheticMapConfig's defaults. NOT WorldPipeline's constructor
# signature defaults ([1.5, 3, 3, 3, 3]) — from_pretrained overrides those with
# the config that ships beside the weights.
f = make_synthetic_map_factory(frequency_mult=[1.0, 1.0, 1.0, 1.0, 1.0],
                               seed=int(e["SEED"]), drop_water_pct=0.5)
i1, j1, i2, j2 = (int(e[k]) for k in ("I1", "J1", "I2", "J2"))
a = np.asarray(f.sample_raw(i1, j1, i2, j2)) if e["MODE"] == "raw" \
    else f(i1, j1, i2, j2).numpy()
np.asarray(a, dtype="<f4").tofile(ref_path)
PY

  RAWFLAG=""
  if [ "$mode" = "raw" ]; then RAWFLAG="--raw"; fi
  rm -f "$MINE"
  "$BIN" terrain-synth --stats "$STATS" --seed "$SEED" \
         --i1 "$I1" --j1 "$J1" --i2 "$I2" --j2 "$J2" $RAWFLAG \
         --out "$MINE" >/dev/null || { echo "   CLI FAILED"; status=1; continue; }
  # Do not trust the exit code alone: a binary too old to know `terrain-synth`
  # prints its usage and exits 0, which would sail straight past `||` and only
  # surface later as a confusing missing-file traceback.
  WANT=$(( (I2 - I1) * (J2 - J1) * 5 * 4 ))
  GOT=$(wc -c < "$MINE" 2>/dev/null || echo 0)
  if [ "$GOT" -ne "$WANT" ]; then
    echo "   CLI WROTE $GOT BYTES, EXPECTED $WANT (stale binary?)"; status=1; continue
  fi

  REF="$REF" MINE="$MINE" python - <<'PY' || status=1
import os, sys
import numpy as np

ref  = np.fromfile(os.environ["REF"],  dtype="<f4").astype(np.float64)
mine = np.fromfile(os.environ["MINE"], dtype="<f4").astype(np.float64)
if ref.shape != mine.shape:
    print(f"   SHAPE MISMATCH ref {ref.shape} mine {mine.shape}"); sys.exit(1)

n = ref.size // 5
names = ["elev", "temp", "temp_std", "precip", "precip_std"]
worst, ok = 0.0, True
for c, nm in enumerate(names):
    r, m = ref[c*n:(c+1)*n], mine[c*n:(c+1)*n]
    denom = np.linalg.norm(r)
    rel = float(np.linalg.norm(r - m) / denom) if denom > 0 else float(np.abs(r - m).max())
    # Per-channel, not pooled: precipitation runs to thousands and temperature to
    # tens, so a pooled relative norm would let a badly broken temperature field
    # hide entirely behind precipitation's magnitude.
    worst = max(worst, rel)
    print(f"   {nm:11s} relL2 {rel:.3e}  maxabs {np.abs(r-m).max():.3e}  "
          f"ref range [{r.min():+.3f}, {r.max():+.3f}]")
    if rel > 1e-5:
        ok = False
print("   PARITY OK" if ok else "   PARITY FAILED")
sys.exit(0 if ok else 1)
PY
done
done

[ "$status" -eq 0 ] && echo "ALL SYNTHETIC PARITY OK" || echo "SYNTHETIC PARITY FAILED"
exit "$status"
