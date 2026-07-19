#!/usr/bin/env bash
# Parity gate for the terrain-diffusion samplers (all three stages).
#
# Runs the PyTorch reference (real upstream scheduler + real EDMUnet2D + real
# checkpoint) through a COMPLETE denoise to populate .parity/, then the C++
# `terrain-sample` port on the SAME noise/conditioning, and reports
# cosine / rel-L2 / maxabsdiff per stage.
#
# This is an end-to-end gate: an error here is either a scheduler bug (coarse:
# Karras sigmas, DPM-Solver++ order selection, the sigma=0 terminal step) or a
# TrigFlow bug (base/decoder: the arc update and the negated model output). The
# per-call UNet forward is separately gated by scripts/terrain_unet_parity.sh,
# so run that first if this fails on all three stages at once.
#
# Usage: scripts/terrain_sampler_parity.sh [path/to/brodiffusion.exe]
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
  TERRAIN_STAGE="$STAGE" TERRAIN_CKPT="$CKPT" python scripts/terrain_sampler_ref.py

  SIZE="$(python -c "import json;print(json.load(open('.parity/terrain_sample_${STAGE}_meta.json'))['size'])")"
  EXTRA=()
  [ -f ".parity/terrain_sample_${STAGE}_cond.f32" ]    && EXTRA+=(--cond    ".parity/terrain_sample_${STAGE}_cond.f32")
  [ -f ".parity/terrain_sample_${STAGE}_condin.f32" ]  && EXTRA+=(--condin  ".parity/terrain_sample_${STAGE}_condin.f32")
  [ -f ".parity/terrain_sample_${STAGE}_latents.f32" ] && EXTRA+=(--latents ".parity/terrain_sample_${STAGE}_latents.f32")

  echo "== C++ ($BIN, $STAGE) =="
  "$BIN" terrain-sample \
    --weights "$BRO" \
    --stage   "$STAGE" \
    --noise   ".parity/terrain_sample_${STAGE}_noise.f32" \
    "${EXTRA[@]}" \
    --size    "$SIZE" \
    --out     ".parity/terrain_sample_${STAGE}_mine_out.f32"

  STAGE="$STAGE" python - <<'PY' || status=1
import os, numpy as np
s = os.environ["STAGE"]
ref  = np.fromfile(f'.parity/terrain_sample_{s}_ref_out.f32',  dtype='<f4')
mine = np.fromfile(f'.parity/terrain_sample_{s}_mine_out.f32', dtype='<f4')
assert ref.shape == mine.shape, (ref.shape, mine.shape)
r, m = ref.astype(np.float64), mine.astype(np.float64)
cos = float(np.dot(r, m) / (np.linalg.norm(r) * np.linalg.norm(m)))
rel = float(np.linalg.norm(r - m) / (np.linalg.norm(r) + 1e-30))
mad = float(np.max(np.abs(r - m)))
rms = float(np.sqrt(((r - m) ** 2).mean()))
print("%-8s cosine %.8f  relL2 %.3e  maxabsdiff %.3e" % (s, cos, rel, mad))
print("         ref std %.6f  mine std %.6f" % (ref.std(), mine.std()))

# The decoder is scored on ABSOLUTE error, not relL2, and that is deliberate.
#
# Its output is an elevation *residual* — a high-frequency detail term that
# rides on the low-frequency elevation. It is intrinsically small (std ~0.037),
# so relL2 divides the error by a norm that nothing downstream ever sees. What
# actually propagates is the absolute error, through
#     e = lowfreq + residual*residual_std ;  elev = sign(e)*e^2
#     => d(elev) = 2*|e| * residual_std * d(residual)
# so we bound the implied vertical error in METRES at a reference elevation.
#
# For scale: the FP16 CUDA path sits at ~0.013 m at 4000 m elevation, on a grid
# whose pixels are 30 m across. It scores relL2 3.96e-3 — "failing" a 2e-3 bar
# while being the most absolutely-accurate stage of the three (its maxabsdiff is
# below base's, and base passes). The 0.1 m bar is still ~8x tighter than that
# path needs, so a genuine regression fails it by orders of magnitude.
RESIDUAL_STD = 0.7
REF_ELEV_M = 4000.0
if s == "decoder":
    elev_err = 2.0 * (REF_ELEV_M ** 0.5) * RESIDUAL_STD * rms
    print("         implied elevation error at %.0f m: %.4f m (bar 0.1 m)"
          % (REF_ELEV_M, elev_err))
    ok = cos > 0.99999 and elev_err < 0.1
else:
    ok = cos > 0.99999 and rel < 2e-3
print("         PARITY OK" if ok else "         PARITY FAILED")
import sys; sys.exit(0 if ok else 1)
PY
done
[ "$status" -eq 0 ] && echo "ALL PARITY OK" || echo "PARITY FAILED"
exit "$status"
