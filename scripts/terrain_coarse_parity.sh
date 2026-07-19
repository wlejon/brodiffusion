#!/usr/bin/env bash
# Parity gate for the coarse world stage — the first point in the port where all
# the individually-gated pieces run together and produce actual terrain.
#
# What this composes: the synthetic climate map, the tile-seeded Gaussian noise
# field, the EDM2 coarse UNet, DPM-Solver++, and the infinite-tensor evaluator.
# Each of those has its own gate; none of them bounds the error here, because
# every stage feeds the next and a 1e-4 discrepancy in the conditioning image is
# amplified through 20 solver steps. So this gate is not redundant with them —
# it is the only thing that measures the composition.
#
# Both sides run on CUDA — the path that ships. The concern that GPU
# nondeterminism would mask a real difference was measured rather than assumed:
# the reference's own CPU-vs-CUDA spread on this stage is 1.5e-4 relative, an
# order of magnitude under the bar, so the gate loses no discrimination. It does
# save about three orders of magnitude of wall clock; the CPU build takes >15
# minutes for a single 64x64 region against ~1s on GPU.
#
# Both forms are compared:
#   normalized  the 6-channel weighted mean — what you would render
#   raw         the 7-channel value*w / w form the next stage consumes
# The raw form matters on its own: a bug that scaled value and weight by the same
# factor would cancel exactly out of the normalized view and only surface once
# the latent stage read it.
#
# Windows are chosen to exercise tile blending rather than a single tile: the
# coarse tiling is 64 wide with stride 48, so a 64-cell request spans four
# windows and every interior cell is a blend of two or four of them. A run that
# only ever asked for one aligned tile would never test the weight channel.
#
# Usage: scripts/terrain_coarse_parity.sh [path/to/brodiffusion.exe]
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

BIN="${1:-build_cuda/Release/brodiffusion.exe}"
[ -x "$BIN" ] || { echo "error: no brodiffusion.exe at $BIN" >&2; exit 2; }
BIN="$REPO/$BIN"

UPSTREAM="$REPO/weights/terrain-diffusion-30m"
PORTED="$REPO/weights/terrain-diffusion-30m-bro"
[ -d "$UPSTREAM" ] || { echo "error: $UPSTREAM missing (run download-terrain-diffusion.sh)" >&2; exit 2; }
[ -d "$PORTED" ]   || { echo "error: $PORTED missing (run convert-terrain-diffusion.py)" >&2; exit 2; }
mkdir -p "$REPO/.parity"

# Two bars, because the two stages have genuinely different error floors.
#
# Coarse: 20 solver steps over a network whose own CUDA gate sits at ~1.5e-3,
# and the reference's device-to-device spread is 1.5e-4, so a few 1e-4 is the
# expected order. The transpose this gate caught on first run scored 0.7.
BAR_COARSE="2e-3"
#
# Latent: ~4e-3 on the elevation latent, and that is inherited rather than
# introduced. Measured, not assumed — reading the first of the two TrigFlow
# steps alone gives 3.810e-3 where both steps give 3.821e-3, so the second step
# contributes nothing and the discrepancy arrives with the coarse conditioning,
# amplified about 11x by the 254M base net. Consistency steps are contractive,
# which is why it does not compound.
#
# Note this is NOT an fp16 effect, though it looks like one: the port computes in
# fp16 on CUDA while the reference is fp32. Running the reference at fp16 for
# comparison makes it 100x WORSE (0.55 relative on the elevation latent), so
# upstream's own half-precision path is the degraded one and the port tracks its
# fp32 path closely. brotensor must be accumulating in fp32.
BAR_LATENT="2e-2"
#
# Elevation: 1.0e-3 to 3.1e-3 across ocean floor, coastline and low-relief land,
# which is TIGHTER than the residual it is built from. That is not luck — the
# low-frequency band carries most of the magnitude and comes from the latent
# map, so the residual's larger relative error is diluted. Worst observed
# absolute error is 18 m on a region with 3 km of relief.
BAR_ELEV="1e-2"

status=0
# Fed by the heredoc at the bottom of the loop rather than a pipe: a pipe would
# run the loop body in a subshell and `status` would never escape it, so the gate
# would report success no matter what failed.
while read -r SEED I1 J1 I2 J2 MODE; do
  [ -n "${SEED:-}" ] || continue

  REF="$REPO/.parity/coarse_ref_${SEED}_${I1}_${J1}_${MODE}.f32"
  MINE="$REPO/.parity/coarse_mine_${SEED}_${I1}_${J1}_${MODE}.f32"
  rm -f "$REF" "$MINE"

  # MODE selects both which tensor to read and which form of it.
  FLAGS=""; CH=6; BAR="$BAR_COARSE"
  case "$MODE" in
    full)        FLAGS="";                    CH=6 ;;
    raw)         FLAGS="--raw";               CH=7 ;;
    latent)      FLAGS="--latent";            CH=5; BAR="$BAR_LATENT" ;;
    latent-raw)  FLAGS="--latent --raw";      CH=6; BAR="$BAR_LATENT" ;;
    latent-init) FLAGS="--latent-init";       CH=5; BAR="$BAR_LATENT" ;;
    residual)     FLAGS="--residual";          CH=1; BAR="$BAR_LATENT" ;;
    residual-raw) FLAGS="--residual --raw";    CH=2; BAR="$BAR_LATENT" ;;
    elev)         FLAGS="--elev";              CH=1; BAR="$BAR_ELEV" ;;
    *) echo "   UNKNOWN MODE $MODE"; status=1; continue ;;
  esac

  echo "== $MODE  seed=$SEED  ($I1,$J1)-($I2,$J2) =="

  python "$REPO/scripts/terrain_coarse_ref.py" --weights "$UPSTREAM" --seed "$SEED" \
      --out "$REF" --i1 "$I1" --j1 "$J1" --i2 "$I2" --j2 "$J2" $FLAGS >/dev/null \
    || { echo "   REFERENCE FAILED"; status=1; continue; }

  "$BIN" terrain-coarse --weights "$PORTED" --seed "$SEED" \
      --out "$MINE" --i1 "$I1" --j1 "$J1" --i2 "$I2" --j2 "$J2" $FLAGS >/dev/null \
    || { echo "   CLI FAILED"; status=1; continue; }

  # A stale binary prints usage and exits 0, sailing past the || above; check the
  # byte count rather than trusting the exit code.
  WANT=$(( (I2 - I1) * (J2 - J1) * CH * 4 ))
  GOT=$(wc -c < "$MINE" 2>/dev/null || echo 0)
  if [ "$GOT" -ne "$WANT" ]; then
    echo "   CLI WROTE $GOT BYTES, EXPECTED $WANT (stale binary?)"; status=1; continue
  fi

  REF="$REF" MINE="$MINE" CH="$CH" BAR="$BAR" STAGE="$MODE" python - <<'PY' || status=1
import os, sys
import numpy as np

ch  = int(os.environ["CH"])
bar = float(os.environ["BAR"])
ref  = np.fromfile(os.environ["REF"],  dtype="<f4").astype(np.float64)
mine = np.fromfile(os.environ["MINE"], dtype="<f4").astype(np.float64)
if ref.shape != mine.shape:
    print(f"   SHAPE MISMATCH {ref.shape} vs {mine.shape}"); sys.exit(1)

# Coarse channels; the latent stage's five are unnamed latents, so fall back to
# an index when the count does not match.
names = ["elev", "elev_minus", "temp", "temp_std", "precip", "precip_std", "weight"]
stage = os.environ.get("STAGE", "")
if stage.startswith("latent"):
    names = [f"lat{i}" for i in range(5)] + ["weight"]
elif stage.startswith("residual"):
    names = ["residual", "weight"]
elif stage == "elev":
    names = ["elev_m"]
ref  = ref.reshape(ch, -1)
mine = mine.reshape(ch, -1)

bad = 0
for c in range(ch):
    r, m = ref[c], mine[c]
    denom = np.linalg.norm(r)
    rel = float(np.linalg.norm(m - r) / denom) if denom > 0 else float(np.linalg.norm(m - r))
    mx  = float(np.abs(m - r).max())
    flag = "" if rel <= bar else "   <-- OVER BAR"
    print(f"   {names[c]:11s} relL2 {rel:.3e}  maxabs {mx:.3e}  "
          f"ref range [{r.min():+.3f}, {r.max():+.3f}]{flag}")
    if rel > bar:
        bad = 1

print("   PARITY OK" if not bad else "   PARITY FAILED")
sys.exit(bad)
PY
# seed  i1  j1  i2  j2  mode
#
# Case 1 and 2 are the same region in both output forms. Cases 3 and 4 sit at
# negative and far-from-origin coordinates, where floor-vs-truncate division and
# any origin bias in the window lattice would show up — a gate that only ever
# read [0, 64) would pass with a pipeline that cannot leave the first quadrant.
done <<'CASES'
1234 0 0 64 64 full
1234 0 0 64 64 raw
77 -96 -96 -32 -32 full
5 -1000 500 -936 564 full
1234 0 0 64 64 latent-init
1234 0 0 64 64 latent
1234 0 0 64 64 latent-raw
77 -96 -96 -32 -32 latent
5 -1000 500 -936 564 latent
1234 0 0 512 512 residual
1234 0 0 512 512 residual-raw
77 -768 -768 -256 -256 residual
1234 0 0 512 512 elev
7 0 0 512 512 elev
7 -2048 3072 -1536 3584 elev
7 5120 -4096 5632 -3584 elev
CASES

[ "$status" -eq 0 ] && echo "ALL COARSE PARITY OK" || echo "COARSE PARITY FAILED"
exit "$status"
