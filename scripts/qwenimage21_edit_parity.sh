#!/usr/bin/env bash
# End-to-end parity gate for Qwen-Image 2.1 IMAGE-CONDITIONED generation: run
# qwenimage21_edit_ref.py (diffusers) with one or more condition images, then
# run brodiffusion's CLI on the same prompt, the same resized condition
# images, the same canvas and step count, and — crucially — the same initial
# noise field, and compare the final latents.
#
#   scripts/qwenimage21_edit_parity.sh <image.png>[,<image2.png>] [res] [steps]
#
# What is deliberately removed from the comparison, so the number measures the
# model rather than the plumbing: the RNG (the reference's noise arrives via
# --latent-in) and the resampler (the reference writes the resized condition
# images and the C++ side reads exactly those). What remains is the vision
# tower, the text backbone, the autoencoder's encode, and the DiT running a
# joint sequence whose prefix now carries image tokens.
#
# Expected: cosine > 0.99 with a BF16 DiT. Both sides run the DiT in reduced
# precision and 8 Euler steps compound that rounding — a rectified-flow
# trajectory is not contractive. INT8 (QI21_QUANTIZE=1) lands lower; that is
# the quantisation error, not an implementation gap, and it is the setting the
# model actually fits a 24 GB card in alongside the text encoder.
#
# Env: PYTHON (interpreter with torch + diffusers main), QI21_QUANTIZE=1 for
# the INT8 C++ run, QI21_EDIT_* forwarded to the reference (prompt, steps,
# cfg, seed, dtype).
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT=weights/qwen-image-2.1
BIN=build_cuda/Release/brodiffusion.exe
PY=${PYTHON:-/c/Users/jonny/scoop/apps/python/current/python}
OUT=.parity

if [ $# -lt 1 ]; then
  echo "usage: $0 <image.png>[,<image2.png>] [res] [steps]" >&2
  exit 2
fi
IMAGES="$1"
export QI21_EDIT_RES="${2:-${QI21_EDIT_RES:-512}}"
export QI21_EDIT_STEPS="${3:-${QI21_EDIT_STEPS:-8}}"

"$PY" scripts/qwenimage21_edit_ref.py "$IMAGES" "$ROOT" "$OUT"

# Read back exactly what the reference ran.
WIDTH=$(sed -n 1p "$OUT/qi21_edit_meta.txt")
HEIGHT=$(sed -n 2p "$OUT/qi21_edit_meta.txt")
STEPS=$(sed -n 3p "$OUT/qi21_edit_meta.txt")
CFG=$(sed -n 4p "$OUT/qi21_edit_meta.txt")
NIMG=$(sed -n 6p "$OUT/qi21_edit_meta.txt")
PROMPT=$(sed -n 7p "$OUT/qi21_edit_meta.txt")

IMGFLAG=()
i=0
while [ "$i" -lt "$NIMG" ]; do
  IMGFLAG+=(--image "$OUT/qi21_edit_cond_${i}.png")
  i=$((i + 1))
done

QFLAG=()
if [ "${QI21_QUANTIZE:-0}" != "0" ]; then
  QFLAG=(--quantize-unet)
fi

BRODIFFUSION_TIME=1 "$BIN" txt2img \
  --model "$ROOT" \
  --prompt "$PROMPT" \
  "${IMGFLAG[@]}" \
  --output-resolution "$QI21_EDIT_RES" \
  --width "$WIDTH" --height "$HEIGHT" \
  --steps "$STEPS" --cfg "$CFG" \
  --latent-in  "$OUT/qi21_edit_init.f32" \
  --latent-out "$OUT/qi21_edit_mine.f32" \
  --out "$OUT/qi21_edit_mine.png" \
  "${QFLAG[@]}"

"$PY" - "$WIDTH" "$HEIGHT" "$STEPS" "$NIMG" <<'PY'
import sys
import numpy as np

width, height, steps, nimg = (int(a) for a in sys.argv[1:5])
Z, H, W = 64, height // 16, width // 16
ref = np.fromfile('.parity/qi21_edit_ref.f32', dtype='<f4')
mine = np.fromfile('.parity/qi21_edit_mine.f32', dtype='<f4')
print('condition images : %d' % nimg)
print('canvas      : %dx%d  latent (%d, %d, %d)' % (width, height, Z, H, W))
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

# Per-channel cosine: a systematic error — a condition image landing at the
# wrong prefix offset, a mis-ordered latent flatten — concentrates in a few
# planes, which compounding rounding never does.
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
