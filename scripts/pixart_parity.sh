#!/usr/bin/env bash
# Parity gate for the PixArt denoiser: run one forward through the C++ path on the
# SAME fixed inputs as scripts/pixart_ref.py and report cosine / rel-L2 vs the
# diffusers reference. Assumes scripts/pixart_ref.py has already populated .parity/.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build_cuda/Release/brodiffusion.exe
TF=weights/pixart-sigma/transformer/diffusion_pytorch_model.safetensors

"$BIN" pixart-fwd \
  --weights "$TF" \
  --latent .parity/ref_latent.f32 --ctx .parity/ref_ctx.f32 \
  --out .parity/mine_eps.f32 \
  --t 500 --H 16 --W 16 --L 4

python - <<'PY'
import numpy as np
ref  = np.fromfile('.parity/ref_eps.f32',  dtype='<f4')
mine = np.fromfile('.parity/mine_eps.f32', dtype='<f4')
cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
rel = float(np.linalg.norm(ref - mine) / np.linalg.norm(ref))
print('cosine       %.6f' % cos)
print('rel L2 err   %.6f' % rel)
print('ref  std %.5f  mine std %.5f  maxabsdiff %.5f'
      % (ref.std(), mine.std(), float(np.max(np.abs(ref - mine)))))
PY
