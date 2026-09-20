#!/usr/bin/env bash
# End-to-end parity gate for Qwen-Image 2.1 text-to-image: run
# qwenimage21_pipeline_ref.py (diffusers), then run brodiffusion's CLI on the
# SAME prompt, resolution, step count and — crucially — the SAME initial noise
# field, and compare the final latents.
#
#   scripts/qwenimage21_pipeline_parity.sh [size] [steps]
#
# The initial latent comes from the reference via --latent-in, so what is left
# in the difference is the text encoder, the DiT and the scheduler, which is
# exactly what this is meant to measure. The comparison is on the LATENT, not
# the image: the VAE is a separate component with its own parity script, and a
# pixel diff would fold its error in.
#
# Expected: cosine > 0.99. Both sides run the DiT in BF16, and 8-40 Euler steps
# compound that rounding — a rectified-flow trajectory is not contractive, so
# two BF16 implementations drift apart even when every op is correct. The
# brodiffusion side additionally runs the text encoder at INT8 by default
# (see the pipeline wiring: the BF16 8B backbone does not fit beside the DiT).
#
# Env: PYTHON (interpreter with torch + diffusers main), QI21_QUANTIZE=1 to run
# the C++ DiT at INT8 too (--quantize-unet; needed if the BF16 DiT plus the
# text encoder overrun the card).
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT=weights/qwen-image-2.1
BIN=build_cuda/Release/brodiffusion.exe
PY=${PYTHON:-/c/Users/jonny/scoop/apps/python/current/python}
OUT=.parity

SIZE="${1:-512}"
STEPS="${2:-8}"
export QI21_PIPE_SIZE="$SIZE"
export QI21_PIPE_STEPS="$STEPS"

"$PY" scripts/qwenimage21_pipeline_ref.py "$ROOT" "$OUT"

# Read back exactly what the reference ran, so the two sides cannot drift.
SIZE=$(sed -n 1p "$OUT/qi21_pipe_meta.txt")
STEPS=$(sed -n 2p "$OUT/qi21_pipe_meta.txt")
CFG=$(sed -n 3p "$OUT/qi21_pipe_meta.txt")
PROMPT=$(sed -n 5p "$OUT/qi21_pipe_meta.txt")

QFLAG=()
if [ "${QI21_QUANTIZE:-0}" != "0" ]; then
  QFLAG=(--quantize-unet)
fi

BRODIFFUSION_TIME=1 "$BIN" txt2img \
  --model "$ROOT" \
  --prompt "$PROMPT" \
  --width "$SIZE" --height "$SIZE" \
  --steps "$STEPS" --cfg "$CFG" \
  --latent-in  "$OUT/qi21_pipe_init.f32" \
  --latent-out "$OUT/qi21_pipe_mine.f32" \
  --out "$OUT/qi21_pipe_mine.png" \
  "${QFLAG[@]}"

"$PY" - "$SIZE" "$STEPS" <<'PY'
import sys
import numpy as np

size, steps = int(sys.argv[1]), int(sys.argv[2])
Z, H, W = 64, size // 16, size // 16
ref = np.fromfile('.parity/qi21_pipe_ref.f32', dtype='<f4')
mine = np.fromfile('.parity/qi21_pipe_mine.f32', dtype='<f4')
print('latent      : (%d, %d, %d)  %d values' % (Z, H, W, Z * H * W))
print('rows        : ref %d  mine %d' % (ref.size, mine.size))
assert ref.size == Z * H * W, ref.size
assert ref.shape == mine.shape, (ref.shape, mine.shape)

cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
rel = float(np.linalg.norm(ref - mine) / np.linalg.norm(ref))
print('steps       : %d' % steps)
print('cosine      : %.6f' % cos)
print('rel L2 err  : %.6f' % rel)
print('ref std %.5f  mine std %.5f  maxabsdiff %.5f'
      % (ref.std(), mine.std(), float(np.max(np.abs(ref - mine)))))

# Per-channel cosine: a systematic error (a mis-applied scale, a swapped
# channel mapping) shows up as one or a few bad planes rather than a uniform
# drift, which compounding BF16 rounding never does.
r3, m3 = ref.reshape(Z, -1), mine.reshape(Z, -1)
cc = np.array([float(np.dot(r3[c], m3[c]) /
                     (np.linalg.norm(r3[c]) * np.linalg.norm(m3[c])))
               for c in range(Z)])
order = np.argsort(cc)
print('per-channel cosine: min %.6f (ch %d)  median %.6f  max %.6f'
      % (cc[order[0]], order[0], float(np.median(cc)), cc[order[-1]]))
print('worst 5 channels  :',
      ', '.join('%d:%.4f' % (c, cc[c]) for c in order[:5]))

BAR = 0.99
print('PASS' if cos > BAR else 'FAIL (cosine <= %.4f)' % BAR)
raise SystemExit(0 if cos > BAR else 1)
PY
