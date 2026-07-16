#!/usr/bin/env bash
# Parity gate for the ARDY two-stage denoiser HISTORY path (autoregressive
# conditioning). Runs the PyTorch reference over a window of NUM_HISTORY clean
# history tokens + 13 generation tokens (real g152), then the C++ subcommand with
# --history-tok on the SAME input, and reports cosine / rel-L2 / maxabsdiff for
# the full predicted hybrid (history rows carried through + generation preds).
#
# Usage: scripts/ardy_denoiser_hist_parity.sh [path/to/brodiffusion.exe]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-build_cpu/Release/brodiffusion.exe}"
[ -x "$BIN" ] || BIN="build_cuda/Release/brodiffusion.exe"
[ -x "$BIN" ] || BIN="build/Release/brodiffusion.exe"
CKPT="${ARDY_CKPT:-weights/ardy-g152}"
HIST="${ARDY_HIST_TOK:-3}"
T_TOK=$((HIST + 13))

echo "== reference (history=$HIST) =="
ARDY_HIST_TOK="$HIST" python scripts/ardy_denoiser_hist_ref.py

echo "== C++ ($BIN) =="
"$BIN" ardy-denoiser-fwd \
  --weights "$CKPT/denoiser.safetensors" \
  --mean "$CKPT/stats/motion/mean.npy" \
  --std  "$CKPT/stats/motion/std.npy" \
  --hybrid .parity/ardy_dnh_hybrid.f32 \
  --text   .parity/ardy_dnh_text.f32 \
  --T-tok "$T_TOK" --history-tok "$HIST" --timestep 3 --heading 0.6 \
  --out .parity/ardy_dnh_mine_out.f32

python - <<'PY'
import numpy as np
ref  = np.fromfile('.parity/ardy_dnh_ref_out.f32',  dtype='<f4')
mine = np.fromfile('.parity/ardy_dnh_mine_out.f32', dtype='<f4')
assert ref.shape == mine.shape, (ref.shape, mine.shape)
cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
rel = float(np.linalg.norm(ref - mine) / (np.linalg.norm(ref) + 1e-30))
mad = float(np.max(np.abs(ref - mine)))
print("pred-hybrid  cosine %.8f  relL2 %.3e  maxabsdiff %.3e" % (cos, rel, mad))
print("ref std %.5f  mine std %.5f" % (ref.std(), mine.std()))
ok = cos > 0.99999 and rel < 2e-3
print("PARITY OK" if ok else "PARITY FAILED")
import sys; sys.exit(0 if ok else 1)
PY
