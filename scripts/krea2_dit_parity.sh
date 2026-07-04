#!/usr/bin/env bash
# Parity gate for the Krea 2 image DiT. Default mode `synth` builds a small
# random-weight Krea2Transformer2DModel and compares FP32-vs-FP32 (exact
# architecture check, forced onto the CPU backend so brodiffusion runs FP32).
# Mode `real` runs the full 12.9B model on CPU float32 both sides (slow).
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-synth}"
BIN=build_cuda/Release/brodiffusion.exe

python scripts/krea2_dit_ref.py "$MODE" .parity
read -r HP WP SEQ < .parity/krea2_dit_dims.txt

if [ "$MODE" = "synth" ]; then
  WDIR=.parity/krea2_synth_transformer
else
  WDIR=weights/krea-2-raw/transformer
fi

# brodiffusion runs FP32 only on the CPU backend; force it so the compute dtype
# matches the FP32 reference exactly.
BROTENSOR_DEFAULT_DEVICE=CPU "$BIN" krea2-fwd \
  --weights-dir "$WDIR" \
  --latent .parity/krea2_dit_latent.f32 \
  --embeds .parity/krea2_dit_embeds.f32 \
  --mask   .parity/krea2_dit_mask.f32 \
  --out    .parity/krea2_dit_mine.f32 \
  --t 0.7 --hp "$HP" --wp "$WP" --seq "$SEQ"

python - <<'PY'
import numpy as np
ref  = np.fromfile('.parity/krea2_dit_velocity.f32', dtype='<f4')
mine = np.fromfile('.parity/krea2_dit_mine.f32',     dtype='<f4')
assert ref.shape == mine.shape, (ref.shape, mine.shape)
cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
rel = float(np.linalg.norm(ref - mine) / np.linalg.norm(ref))
print('cosine       %.8f' % cos)
print('rel L2 err   %.8f' % rel)
print('ref  std %.5f  mine std %.5f  maxabsdiff %.6f'
      % (ref.std(), mine.std(), float(np.max(np.abs(ref - mine)))))
PY
