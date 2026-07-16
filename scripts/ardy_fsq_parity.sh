#!/usr/bin/env bash
# Parity gate for the ARDY FSQ motion decoder (detokenize). Runs the PyTorch
# reference (real g152 tokenizer weights) to populate .parity/, then the C++
# subcommand on the SAME tokens + local root, and reports cosine / rel-L2 /
# maxabsdiff for the decoded pose features.
#
# Usage: scripts/ardy_fsq_parity.sh [path/to/brodiffusion.exe]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-build_cpu/Release/brodiffusion.exe}"
[ -x "$BIN" ] || BIN="build_cuda/Release/brodiffusion.exe"
[ -x "$BIN" ] || BIN="build/Release/brodiffusion.exe"
CKPT="${ARDY_CKPT:-weights/ardy-g152}"

echo "== reference =="
python scripts/ardy_fsq_ref.py

T_TOK=13

echo "== C++ ($BIN) =="
"$BIN" ardy-fsq-detok \
  --weights "$CKPT/tokenizer.safetensors" \
  --pq-mean "$CKPT/stats/post_quantization/mean.npy" \
  --pq-std  "$CKPT/stats/post_quantization/std.npy" \
  --tokens     .parity/ardy_fsq_tokens.f32 \
  --local-root .parity/ardy_fsq_localroot.f32 \
  --T-tok "$T_TOK" \
  --out .parity/ardy_fsq_mine_out.f32

python - <<'PY'
import numpy as np
ref  = np.fromfile('.parity/ardy_fsq_ref_out.f32',  dtype='<f4')
mine = np.fromfile('.parity/ardy_fsq_mine_out.f32', dtype='<f4')
assert ref.shape == mine.shape, (ref.shape, mine.shape)
cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
rel = float(np.linalg.norm(ref - mine) / (np.linalg.norm(ref) + 1e-30))
mad = float(np.max(np.abs(ref - mine)))
print("decoded-pose  cosine %.8f  relL2 %.3e  maxabsdiff %.3e" % (cos, rel, mad))
print("ref  std %.5f  mine std %.5f" % (ref.std(), mine.std()))
ok = cos > 0.99999 and rel < 2e-3
print("PARITY OK" if ok else "PARITY FAILED")
import sys; sys.exit(0 if ok else 1)
PY
