#!/usr/bin/env bash
# Parity gate for the Krea 2 text-conditioning pathway: run krea2_text_ref.py to
# populate .parity/, then run the C++ path (CLI `krea2-text-fwd`) on the SAME
# prompt and report cosine / rel-L2 over the VALID (mask==1) token rows only.
#
#   scripts/krea2_text_parity.sh                  text prompt (the default)
#   scripts/krea2_text_parity.sh --image PATH     image prompt
#
# In image mode the reference resizes the image to 512x512 and writes it back
# out; the C++ side reads THAT file, so no resampler sits between the two and
# the only thing under test is the encoder. Both sides tap the same 12 decoder
# layers of the same Qwen3-VL-4B backbone; the image just substitutes a
# "<|vision_start|><|image_pad|>...<|vision_end|>" run for the prompt's content
# tokens. Per-token cosine is reported as well as the flat one, because the
# flat figure is dominated by the high-norm rows and a splice bug shows up as a
# few bad tokens rather than a uniformly worse number.
#
# The reference runs BF16 by default, matching the shipped pipeline. On an
# image prompt that is the limiting term, not the C++ path: the vision tower's
# massive-activation channels turn the reference's own bf16 rounding into
# ~0.9994 flat / ~0.83 worst-token against an fp32 run of the same weights, so
# a per-token min in that neighbourhood is the floor and not a defect. Set
# KREA2_TEXT_REF_DTYPE=float32 to move the floor out of the way and measure the
# implementation; there the C++ path scores ~0.99998 flat / ~0.98 worst-token.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT=weights/krea-2-raw
TOK=weights/krea-2-raw/tokenizer
BIN=build_cuda/Release/brodiffusion.exe

IMAGE=""
if [ "${1:-}" = "--image" ]; then
  IMAGE="${2:?--image needs a path}"
fi

if [ -n "$IMAGE" ]; then
  KREA2_TEXT_REF_IMAGE="$IMAGE" python scripts/krea2_text_ref.py "$ROOT" .parity
else
  python scripts/krea2_text_ref.py "$ROOT" .parity
fi
PROMPT="$(cat .parity/krea2_text_prompt.txt)"

if [ -n "$IMAGE" ]; then
  "$BIN" krea2-text-fwd \
    --weights-dir "$ROOT" \
    --tokenizer-dir "$TOK" \
    --prompt "$PROMPT" \
    --image .parity/krea2_text_cond.png \
    --out .parity/krea2_text_mine_embeds.f32 \
    --mask-out .parity/krea2_text_mine_mask.f32
else
  "$BIN" krea2-text-fwd \
    --weights-dir "$ROOT" \
    --tokenizer-dir "$TOK" \
    --prompt "$PROMPT" \
    --out .parity/krea2_text_mine_embeds.f32 \
    --mask-out .parity/krea2_text_mine_mask.f32
fi

python - <<'PY'
import numpy as np
D = 2560
L = 12
ref  = np.fromfile('.parity/krea2_text_ref_embeds.f32',  dtype='<f4').reshape(512, L, D)
mine = np.fromfile('.parity/krea2_text_mine_embeds.f32', dtype='<f4').reshape(512, L, D)
rmask = np.fromfile('.parity/krea2_text_ref_mask.f32',  dtype='<f4')
mmask = np.fromfile('.parity/krea2_text_mine_mask.f32', dtype='<f4')

print('mask match  :', bool(np.array_equal(rmask, mmask)),
      '| ref valid', int(rmask.sum()), 'mine valid', int(mmask.sum()))

sel = rmask > 0.5
r = ref[sel].ravel()
m = mine[sel].ravel()
cos = float(np.dot(r, m) / (np.linalg.norm(r) * np.linalg.norm(m)))
rel = float(np.linalg.norm(r - m) / np.linalg.norm(r))
print('cosine       %.6f' % cos)
print('rel L2 err   %.6f' % rel)
print('ref  std %.5f  mine std %.5f  maxabsdiff %.5f'
      % (r.std(), m.std(), float(np.max(np.abs(r - m)))))

# Per-TOKEN cosine over all 12 taps at once. The flat figure above is
# dominated by whichever rows carry the largest norm; a splice bug lands on a
# handful of tokens and barely moves it, so the worst token is the number that
# actually gates this pathway.
rows = np.where(sel)[0]
per = np.empty(len(rows))
for i, t in enumerate(rows):
    rr = ref[t].ravel(); mm = mine[t].ravel()
    per[i] = float(np.dot(rr, mm) / (np.linalg.norm(rr) * np.linalg.norm(mm)))
order = np.argsort(per)
print('per-token cosine: min %.6f  p01 %.6f  median %.6f  mean %.6f'
      % (per.min(), np.percentile(per, 1), np.median(per), per.mean()))
print('  worst tokens:',
      ', '.join('%d:%.5f' % (rows[j], per[j]) for j in order[:8]))
print('  tokens below 0.999: %d / %d' % (int((per < 0.999).sum()), len(per)))
print('  tokens below 0.996: %d / %d' % (int((per < 0.996).sum()), len(per)))

# Where a bad token sits, and how big it is. A divergence that tracks token
# NORM is an arithmetic/precision story; one that tracks POSITION in the image
# grid is a layout or m-RoPE story; one confined to the first taps is a
# DeepStack story, since DeepStack only injects into the earliest layers.
norms = np.array([np.linalg.norm(ref[t].ravel()) for t in rows])
lo = per < np.percentile(per, 10)
print('  worst-decile tokens: mean ref norm %.3f vs rest %.3f'
      % (norms[lo].mean(), norms[~lo].mean()))
if len(rows) > 200:
    # Image mode: content row 0 is <|vision_start|>, rows 1..N the merged
    # image tokens row-major over a sqrt(N) grid, then <|vision_end|>.
    n_img = int(rows.max()) - 1
    side = int(round(n_img ** 0.5))
    if side * side == n_img:
        img_rows = [(j, rows[j]) for j in order[:12] if 1 <= rows[j] <= n_img]
        print('  worst image tokens at (y,x) on the %dx%d grid:' % (side, side),
              ', '.join('(%d,%d):%.3f' % ((rows[j] - 1) // side,
                                          (rows[j] - 1) % side, per[j])
                        for j, _ in [(j, 0) for j, _ in img_rows]))

# Per-tap, per-token: is the error born early (DeepStack / vision features) or
# does it accumulate with depth?
for l in (0, L // 2, L - 1):
    pt = np.empty(len(rows))
    for i, t in enumerate(rows):
        rr = ref[t, l]; mm = mine[t, l]
        pt[i] = float(np.dot(rr, mm) / (np.linalg.norm(rr) * np.linalg.norm(mm)))
    print('  tap %2d per-token: min %.5f  median %.5f  below 0.996: %d'
          % (l, pt.min(), np.median(pt), int((pt < 0.996).sum())))

# Per-tap cosine (bisect which layer drifts, if any).
for l in range(L):
    rr = ref[sel, l, :].ravel(); mm = mine[sel, l, :].ravel()
    c = float(np.dot(rr, mm) / (np.linalg.norm(rr) * np.linalg.norm(mm)))
    print('  tap %2d cosine %.6f' % (l, c))
PY
