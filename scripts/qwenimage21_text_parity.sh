#!/usr/bin/env bash
# Parity gate for the Qwen-Image 2.1 text-conditioning pathway: run
# qwenimage21_text_ref.py to populate .parity/, then run the C++ path (CLI
# `qi21-text-fwd`) on the SAME prompt and report cosine / rel-L2 per token.
#
#   scripts/qwenimage21_text_parity.sh [fp32|bf16] [prompt]
#
# With QI21_TEXT_IMAGE=<png>[,<png>...] the image-conditioned template runs
# instead: the reference encodes prompt + condition images through the vision
# tower and writes the RESIZED images it used, and the C++ side reads exactly
# those, so a resampler difference cannot show up as an encoder difference.
# The image_pad_mask is compared alongside the ids — it says which rows the
# DiT will fill with condition latents, and a one-row shift there would move
# the whole joint sequence.
#
# The token ids are compared first and must be IDENTICAL — a tokenizer
# disagreement moves every row, and a hidden-state number computed on top of
# one is meaningless.
#
# Mode `fp32` (default) runs the reference in float32 on the CPU. That is the
# bar that means something: this encoder's `<|im_start|>` rows carry massive
# activations that survive as a small difference of large numbers, so in
# bfloat16 that row is mostly rounding — the reference's own bf16 run scores
# only ~0.90 cosine against its own fp32 run on it. Mode `bf16` reproduces the
# usual bf16/CUDA reference and is kept for speed, with a correspondingly
# looser bar.
#
# Env: PYTHON (interpreter with torch + diffusers main), QI21_QUANTIZE=1 to run
# the C++ side with the INT8 (W8A16) text backbone instead of FP16,
# QI21_TEXT_IMAGE=<png>[,...] for the image-conditioned template and
# QI21_TEXT_RES (default 512) for the resolution they are sized to.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT=weights/qwen-image-2.1
BIN=build_cuda/Release/brodiffusion.exe
PY=${PYTHON:-/c/Users/jonny/scoop/apps/python/current/python}

MODE="${1:-fp32}"
PROMPT_ARG="${2:-}"
case "$MODE" in
  fp32) export QI21_TEXT_REF_DTYPE=float32  QI21_TEXT_REF_DEVICE=cpu;  BAR=0.999 ;;
  bf16) export QI21_TEXT_REF_DTYPE=bfloat16 QI21_TEXT_REF_DEVICE=cuda; BAR=0.995 ;;
  *) echo "usage: $0 [fp32|bf16] [prompt]" >&2; exit 2 ;;
esac

# Image-conditioned prompts get their own bars. A condition image's rows carry
# activations an order of magnitude larger than a text row's, and they are
# produced by a 27-layer vision tower feeding a 36-layer backbone, so reduced
# precision has far more room to accumulate. The measured numbers for one
# 512x512 image (256 slots) say what these are worth: brodiffusion at FP16
# scores 0.9976 against the fp32 reference and 0.9729 against the bf16 one —
# i.e. the reference's own bf16 run is FARTHER from fp32 than brodiffusion is,
# the same relation the text-only path shows and for the same reason. The per-
# token breakdown below is where a real disagreement would show: a logic error
# moves the image slots' MEDIAN, which sits at 0.99988 in fp32.
if [ -n "${QI21_TEXT_IMAGE:-}" ]; then
  case "$MODE" in
    fp32) BAR=0.995 ;;
    bf16) BAR=0.96  ;;
  esac
fi

export QI21_TEXT_REF_IMAGE="${QI21_TEXT_IMAGE:-}"
export QI21_TEXT_REF_RES="${QI21_TEXT_RES:-512}"

"$PY" scripts/qwenimage21_text_ref.py "$ROOT" .parity ${PROMPT_ARG:+"$PROMPT_ARG"}
PROMPT="$(cat .parity/qi21_text_prompt.txt)"

# Hand the C++ side the reference's own resized copies, one --image each.
IMGFLAG=()
if [ -n "${QI21_TEXT_IMAGE:-}" ]; then
  n=0
  IFS=',' read -ra _paths <<< "$QI21_TEXT_IMAGE"
  for _ in "${_paths[@]}"; do
    IMGFLAG+=(--image ".parity/qi21_text_cond_${n}.png")
    n=$((n + 1))
  done
fi

QFLAG=()
if [ "${QI21_QUANTIZE:-0}" != "0" ]; then
  QFLAG=(--quantize)
  # INT8 weight-only against an fp32 reference is a different comparison: the
  # measured ~0.997 is the quantisation error, not an implementation gap (it
  # is still tighter than the bf16 reference's own 0.9961 against fp32).
  BAR=0.99
fi

"$BIN" qi21-text-fwd \
  --weights-dir "$ROOT" \
  --prompt "$PROMPT" \
  --out .parity/qi21_text_mine.f32 \
  --mask-out .parity/qi21_text_mine_mask.f32 \
  --ids-out .parity/qi21_text_mine_ids.i32 \
  --pad-out .parity/qi21_text_mine_pad.i32 \
  "${IMGFLAG[@]}" \
  "${QFLAG[@]}"

"$PY" - "$MODE" "$BAR" <<'PY'
import sys
import numpy as np

mode, bar = sys.argv[1], float(sys.argv[2])
D = 4096
ref  = np.fromfile('.parity/qi21_text_ref.f32',  dtype='<f4').reshape(-1, D)
mine = np.fromfile('.parity/qi21_text_mine.f32', dtype='<f4').reshape(-1, D)
rids = np.fromfile('.parity/qi21_text_ref_ids.i32',  dtype='<i4')
mids = np.fromfile('.parity/qi21_text_mine_ids.i32', dtype='<i4')
rmask = np.fromfile('.parity/qi21_text_ref_mask.f32',  dtype='<f4')
mmask = np.fromfile('.parity/qi21_text_mine_mask.f32', dtype='<f4')

print('mode        : %s (bar cosine > %.4f)' % (mode, bar))
print('rows        : ref %d  mine %d' % (ref.shape[0], mine.shape[0]))
ids_ok = rids.shape == mids.shape and bool(np.array_equal(rids, mids))
print('token ids   :', 'IDENTICAL' if ids_ok else 'MISMATCH')
if not ids_ok:
    print('  ref ', rids.tolist())
    print('  mine', mids.tolist())
    raise SystemExit(1)
print('mask        :', bool(np.array_equal(rmask, mmask)),
      '| all ones ref', bool((rmask == 1).all()), 'mine', bool((mmask == 1).all()))

# image_pad_mask: which rows the DiT fills with condition latents. It has to
# match exactly — one row of drift moves every later token of the joint
# sequence, and the embeddings would still look "close" while describing a
# different picture.
rpad = np.fromfile('.parity/qi21_text_ref_pad.i32',  dtype='<i4')
mpad = np.fromfile('.parity/qi21_text_mine_pad.i32', dtype='<i4')
pad_ok = rpad.shape == mpad.shape and bool(np.array_equal(rpad, mpad))
print('image slots : ref %d  mine %d  %s'
      % (int(rpad.sum()), int(mpad.sum()),
         'IDENTICAL' if pad_ok else 'MISMATCH'))
if not pad_ok:
    raise SystemExit(1)
assert ref.shape == mine.shape, (ref.shape, mine.shape)

r, m = ref.ravel(), mine.ravel()
cos = float(np.dot(r, m) / (np.linalg.norm(r) * np.linalg.norm(m)))
rel = float(np.linalg.norm(r - m) / np.linalg.norm(r))
print('cosine      : %.6f' % cos)
print('rel L2 err  : %.6f' % rel)
print('ref std %.5f  mine std %.5f  maxabsdiff %.5f'
      % (r.std(), m.std(), float(np.max(np.abs(r - m)))))

worst, worst_t = 1.0, -1
# An image-conditioned prompt runs to hundreds of rows, most of them one
# image's slots; print every row only when the sequence is short enough to
# read, and always report the worst.
percos = np.zeros(ref.shape[0])
for t in range(ref.shape[0]):
    a, b = ref[t], mine[t]
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    e = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    percos[t] = c
    if ref.shape[0] <= 64:
        print('  tok %3d id %6d  cos %.6f  relL2 %.6f' % (t, int(rids[t]), c, e))
    if c < worst:
        worst, worst_t = c, t
if ref.shape[0] > 64:
    for t in np.argsort(percos)[:8]:
        print('  tok %3d id %6d  cos %.6f  %s'
              % (t, int(rids[t]), percos[t],
                 'image slot' if rpad[t] else 'text'))
if int(rpad.sum()) > 0:
    img_rows, txt_rows = percos[rpad == 1], percos[rpad == 0]
    print('per-token cosine: image slots min %.6f median %.6f | text min '
          '%.6f median %.6f'
          % (img_rows.min(), float(np.median(img_rows)),
             txt_rows.min(), float(np.median(txt_rows))))
print('worst token : %d cosine %.6f' % (worst_t, worst))
print('PASS' if cos > bar else 'FAIL (cosine <= %.4f)' % bar)
raise SystemExit(0 if cos > bar else 1)
PY
