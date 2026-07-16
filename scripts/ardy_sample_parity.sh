#!/usr/bin/env bash
# Parity gate for the ARDY spaced-DDIM text-to-motion window sampler. Runs the
# PyTorch reference (real g152 denoiser + real Diffusion/DDIMSampler, full
# generation loop) to populate .parity/, then the C++ `ardy-sample` subcommand on
# the SAME noise seed / text, and reports cosine / rel-L2 / maxabsdiff of the
# final denoised hybrid.
#
# Usage: scripts/ardy_sample_parity.sh [path/to/brodiffusion.exe]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-build_cpu/Release/brodiffusion.exe}"
[ -x "$BIN" ] || BIN="build_cuda/Release/brodiffusion.exe"
[ -x "$BIN" ] || BIN="build/Release/brodiffusion.exe"
CKPT="${ARDY_CKPT:-weights/ardy-g152}"
T_TOK=13
STEPS=10
CFG=2.5
HEADING=0.6

echo "== reference =="
python scripts/ardy_sample_ref.py

echo "== C++ ($BIN) =="
"$BIN" ardy-sample \
  --weights "$CKPT/denoiser.safetensors" \
  --mean "$CKPT/stats/motion/mean.npy" \
  --std  "$CKPT/stats/motion/std.npy" \
  --x-init .parity/ardy_smp_xinit.f32 \
  --text   .parity/ardy_smp_text.f32 \
  --T-tok "$T_TOK" --steps "$STEPS" --cfg "$CFG" --heading "$HEADING" \
  --out .parity/ardy_smp_mine_out.f32

python - <<'PY'
import numpy as np
ref  = np.fromfile('.parity/ardy_smp_ref_out.f32',  dtype='<f4')
mine = np.fromfile('.parity/ardy_smp_mine_out.f32', dtype='<f4')
assert ref.shape == mine.shape, (ref.shape, mine.shape)
cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
rel = float(np.linalg.norm(ref - mine) / (np.linalg.norm(ref) + 1e-30))
mad = float(np.max(np.abs(ref - mine)))
print("final-hybrid  cosine %.8f  relL2 %.3e  maxabsdiff %.3e" % (cos, rel, mad))
print("ref std %.5f  mine std %.5f" % (ref.std(), mine.std()))
ok = cos > 0.99999 and rel < 2e-3
print("PARITY OK" if ok else "PARITY FAILED")
import sys; sys.exit(0 if ok else 1)
PY
