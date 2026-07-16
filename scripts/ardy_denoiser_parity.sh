#!/usr/bin/env bash
# Parity gate for the ARDY two-stage denoiser (x0-pred, generation window). Runs
# the PyTorch reference (real g152 AutoLatentTwostageDenoiser + motion rep) to
# populate .parity/, then the C++ subcommand on the SAME hybrid/text input, and
# reports cosine / rel-L2 / maxabsdiff for the predicted clean hybrid.
#
# Usage: scripts/ardy_denoiser_parity.sh [path/to/brodiffusion.exe]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-build_cpu/Release/brodiffusion.exe}"
[ -x "$BIN" ] || BIN="build_cuda/Release/brodiffusion.exe"
[ -x "$BIN" ] || BIN="build/Release/brodiffusion.exe"
CKPT="${ARDY_CKPT:-weights/ardy-g152}"
T_TOK=13

echo "== reference =="
python scripts/ardy_denoiser_ref.py

echo "== C++ ($BIN) =="
"$BIN" ardy-denoiser-fwd \
  --weights "$CKPT/denoiser.safetensors" \
  --mean "$CKPT/stats/motion/mean.npy" \
  --std  "$CKPT/stats/motion/std.npy" \
  --hybrid .parity/ardy_dn_hybrid.f32 \
  --text   .parity/ardy_dn_text.f32 \
  --T-tok "$T_TOK" --timestep 3 --heading 0.6 \
  --out .parity/ardy_dn_mine_out.f32

python - <<'PY'
import numpy as np
ref  = np.fromfile('.parity/ardy_dn_ref_out.f32',  dtype='<f4')
mine = np.fromfile('.parity/ardy_dn_mine_out.f32', dtype='<f4')
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
