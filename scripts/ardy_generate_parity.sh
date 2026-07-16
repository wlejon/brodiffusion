#!/usr/bin/env bash
# Parity gate for the ARDY autoregressive text-to-motion rollout (hybrid level).
# Runs the PyTorch reference — the real Ardy.__call__ over several gen_horizon
# windows (history-conditioned sampling + per-window recenter/requantize + global
# translation) — capturing the world-frame hybrid before FSQ detokenization plus
# the per-window noise, then the C++ `ardy-generate` on the SAME noise/text, and
# reports cosine / rel-L2 / maxabsdiff for the full hybrid sequence.
#
# Usage: scripts/ardy_generate_parity.sh [path/to/brodiffusion.exe]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-build_cpu/Release/brodiffusion.exe}"
[ -x "$BIN" ] || BIN="build_cuda/Release/brodiffusion.exe"
[ -x "$BIN" ] || BIN="build/Release/brodiffusion.exe"
CKPT="${ARDY_CKPT:-weights/ardy-g152}"
FRAMES="${ARDY_GEN_FRAMES:-104}"

echo "== reference (frames=$FRAMES) =="
ARDY_GEN_FRAMES="$FRAMES" python scripts/ardy_generate_ref.py

echo "== C++ ($BIN) =="
"$BIN" ardy-generate \
  --denoiser  "$CKPT/denoiser.safetensors" \
  --tokenizer "$CKPT/tokenizer.safetensors" \
  --mean "$CKPT/stats/motion/mean.npy" \
  --std  "$CKPT/stats/motion/std.npy" \
  --pq-mean "$CKPT/stats/post_quantization/mean.npy" \
  --pq-std  "$CKPT/stats/post_quantization/std.npy" \
  --text  .parity/ardy_gen_text.f32 \
  --noise .parity/ardy_gen_noise.f32 \
  --frames "$FRAMES" --steps 10 --cfg 2.5 --heading 0.6 \
  --out .parity/ardy_gen_mine_hyb.f32

python - <<'PY'
import numpy as np
ref  = np.fromfile('.parity/ardy_gen_ref_hyb.f32',  dtype='<f4')
mine = np.fromfile('.parity/ardy_gen_mine_hyb.f32', dtype='<f4')
assert ref.shape == mine.shape, (ref.shape, mine.shape)
cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
rel = float(np.linalg.norm(ref - mine) / (np.linalg.norm(ref) + 1e-30))
mad = float(np.max(np.abs(ref - mine)))
print("world-hybrid  cosine %.8f  relL2 %.3e  maxabsdiff %.3e" % (cos, rel, mad))
print("ref std %.5f  mine std %.5f" % (ref.std(), mine.std()))
ok = cos > 0.99999 and rel < 2e-3
print("PARITY OK" if ok else "PARITY FAILED")
import sys; sys.exit(0 if ok else 1)
PY
