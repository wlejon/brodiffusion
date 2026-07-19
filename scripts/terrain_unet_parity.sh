#!/usr/bin/env bash
# Parity gate for the terrain-diffusion EDM2 UNet (all three stages).
#
# Runs the PyTorch reference (real upstream EDMUnet2D + real checkpoint) to
# populate .parity/, then the C++ port on the SAME inputs, and reports
# cosine / rel-L2 / maxabsdiff per stage.
#
# The C++ side consumes the CONVERTED (pre-folded) weights; the reference uses
# the upstream ones. scripts/terrain_fold_parity.py separately proves that fold
# is exact, so a failure here is a forward-pass bug, not a weights bug.
#
# Usage: scripts/terrain_unet_parity.sh [path/to/brodiffusion.exe]
#   TERRAIN_CKPT  upstream dir  (default weights/terrain-diffusion-30m)
#   TERRAIN_BRO   converted dir (default <TERRAIN_CKPT>-bro)
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-build_cpu/Release/brodiffusion.exe}"
[ -x "$BIN" ] || BIN="build_cuda/Release/brodiffusion.exe"
[ -x "$BIN" ] || BIN="build/Release/brodiffusion.exe"
[ -x "$BIN" ] || { echo "error: no brodiffusion.exe found (tried build_cpu/build_cuda/build)" >&2; exit 2; }

CKPT="${TERRAIN_CKPT:-weights/terrain-diffusion-30m}"
BRO="${TERRAIN_BRO:-${CKPT}-bro}"
mkdir -p .parity

status=0
for STAGE in coarse base decoder; do
  echo "== reference ($STAGE) =="
  TERRAIN_STAGE="$STAGE" TERRAIN_CKPT="$CKPT" python scripts/terrain_unet_ref.py

  SIZE="$(python -c "import json;print(json.load(open('.parity/terrain_${STAGE}_meta.json'))['size'])")"
  COND_ARG=()
  [ -f ".parity/terrain_${STAGE}_cond.f32" ] && COND_ARG=(--cond ".parity/terrain_${STAGE}_cond.f32")

  echo "== C++ ($BIN, $STAGE) =="
  "$BIN" terrain-unet-fwd \
    --weights "$BRO" \
    --stage   "$STAGE" \
    --x       ".parity/terrain_${STAGE}_x.f32" \
    --noise   ".parity/terrain_${STAGE}_noise.f32" \
    "${COND_ARG[@]}" \
    --size    "$SIZE" \
    --out     ".parity/terrain_${STAGE}_mine_out.f32"

  STAGE="$STAGE" python - <<'PY' || status=1
import os, numpy as np
s = os.environ["STAGE"]
ref  = np.fromfile(f'.parity/terrain_{s}_ref_out.f32',  dtype='<f4')
mine = np.fromfile(f'.parity/terrain_{s}_mine_out.f32', dtype='<f4')
assert ref.shape == mine.shape, (ref.shape, mine.shape)
r, m = ref.astype(np.float64), mine.astype(np.float64)
cos = float(np.dot(r, m) / (np.linalg.norm(r) * np.linalg.norm(m)))
rel = float(np.linalg.norm(r - m) / (np.linalg.norm(r) + 1e-30))
mad = float(np.max(np.abs(r - m)))
print("%-8s cosine %.8f  relL2 %.3e  maxabsdiff %.3e" % (s, cos, rel, mad))
print("         ref std %.6f  mine std %.6f" % (ref.std(), mine.std()))
ok = cos > 0.99999 and rel < 2e-3
print("         PARITY OK" if ok else "         PARITY FAILED")
import sys; sys.exit(0 if ok else 1)
PY
done
[ "$status" -eq 0 ] && echo "ALL PARITY OK" || echo "PARITY FAILED"
exit "$status"
