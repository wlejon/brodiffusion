#!/usr/bin/env bash
# Parity gate for the Qwen-Image 2.1 text-conditioning pathway: run
# qwenimage21_text_ref.py to populate .parity/, then run the C++ path (CLI
# `qi21-text-fwd`) on the SAME prompt and report cosine / rel-L2 per token.
#
#   scripts/qwenimage21_text_parity.sh [fp32|bf16] [prompt]
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
# the C++ side with the INT8 (W8A16) text backbone instead of FP16.
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

"$PY" scripts/qwenimage21_text_ref.py "$ROOT" .parity ${PROMPT_ARG:+"$PROMPT_ARG"}
PROMPT="$(cat .parity/qi21_text_prompt.txt)"

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
assert ref.shape == mine.shape, (ref.shape, mine.shape)

r, m = ref.ravel(), mine.ravel()
cos = float(np.dot(r, m) / (np.linalg.norm(r) * np.linalg.norm(m)))
rel = float(np.linalg.norm(r - m) / np.linalg.norm(r))
print('cosine      : %.6f' % cos)
print('rel L2 err  : %.6f' % rel)
print('ref std %.5f  mine std %.5f  maxabsdiff %.5f'
      % (r.std(), m.std(), float(np.max(np.abs(r - m)))))

worst, worst_t = 1.0, -1
for t in range(ref.shape[0]):
    a, b = ref[t], mine[t]
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    e = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    print('  tok %3d id %6d  cos %.6f  relL2 %.6f' % (t, int(rids[t]), c, e))
    if c < worst:
        worst, worst_t = c, t
print('worst token : %d cosine %.6f' % (worst_t, worst))
print('PASS' if cos > bar else 'FAIL (cosine <= %.4f)' % bar)
raise SystemExit(0 if cos > bar else 1)
PY
