#!/usr/bin/env bash
# Parity gate for the ARDY denoiser backbone (TransformerEncoderBlock). Runs the
# PyTorch reference (real g152 denoiser weights) to populate .parity/, then the
# C++ subcommand on the SAME inputs, and reports cosine / rel-L2 / maxabsdiff for
# the block output. Runs both the root (out 20) and body (out 128) stages.
#
# Usage: scripts/ardy_backbone_parity.sh [path/to/brodiffusion.exe]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-build_cpu/Release/brodiffusion.exe}"
[ -x "$BIN" ] || BIN="build_cuda/Release/brodiffusion.exe"
[ -x "$BIN" ] || BIN="build/Release/brodiffusion.exe"
CKPT="${ARDY_CKPT:-weights/ardy-g152}"
T=13

status=0
for STAGE in root body; do
  echo "== reference ($STAGE) =="
  ARDY_STAGE="$STAGE" python scripts/ardy_backbone_ref.py

  echo "== C++ ($BIN, $STAGE) =="
  "$BIN" ardy-backbone-fwd \
    --weights "$CKPT/denoiser.safetensors" \
    --stage "$STAGE" \
    --x    ".parity/ardy_bb_${STAGE}_x.f32" \
    --text ".parity/ardy_bb_${STAGE}_text.f32" \
    --T "$T" --timestep 3 --heading 0.6 \
    --out ".parity/ardy_bb_${STAGE}_mine_out.f32"

  STAGE="$STAGE" python - <<'PY' || status=1
import os, numpy as np
s = os.environ["STAGE"]
ref  = np.fromfile(f'.parity/ardy_bb_{s}_ref_out.f32',  dtype='<f4')
mine = np.fromfile(f'.parity/ardy_bb_{s}_mine_out.f32', dtype='<f4')
assert ref.shape == mine.shape, (ref.shape, mine.shape)
cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
rel = float(np.linalg.norm(ref - mine) / (np.linalg.norm(ref) + 1e-30))
mad = float(np.max(np.abs(ref - mine)))
print("%s  cosine %.8f  relL2 %.3e  maxabsdiff %.3e" % (s, cos, rel, mad))
print("   ref std %.5f  mine std %.5f" % (ref.std(), mine.std()))
ok = cos > 0.99999 and rel < 2e-3
print("   PARITY OK" if ok else "   PARITY FAILED")
import sys; sys.exit(0 if ok else 1)
PY
done
[ "$status" -eq 0 ] && echo "ALL PARITY OK" || echo "PARITY FAILED"
exit "$status"
