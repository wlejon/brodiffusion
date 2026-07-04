#!/usr/bin/env bash
# Parity gate for the Krea 2 text-conditioning pathway: run krea2_text_ref.py to
# populate .parity/, then run the C++ path (CLI `krea2-text-fwd`) on the SAME
# prompt and report cosine / rel-L2 over the VALID (mask==1) token rows only.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT=weights/krea-2-raw
TOK=weights/krea-2-raw/tokenizer
BIN=build_cuda/Release/brodiffusion.exe

python scripts/krea2_text_ref.py "$ROOT" .parity
PROMPT="$(cat .parity/krea2_text_prompt.txt)"

"$BIN" krea2-text-fwd \
  --weights-dir "$ROOT" \
  --tokenizer-dir "$TOK" \
  --prompt "$PROMPT" \
  --out .parity/krea2_text_mine_embeds.f32 \
  --mask-out .parity/krea2_text_mine_mask.f32

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

# Per-tap cosine (bisect which layer drifts, if any).
for l in range(L):
    rr = ref[sel, l, :].ravel(); mm = mine[sel, l, :].ravel()
    c = float(np.dot(rr, mm) / (np.linalg.norm(rr) * np.linalg.norm(mm)))
    print('  tap %2d cosine %.6f' % (l, c))
PY
