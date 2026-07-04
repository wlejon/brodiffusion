#!/usr/bin/env bash
# Parity gate for the Qwen-Image VAE decoder (Krea 2): run one decode through
# the C++ path on the SAME latent as scripts/krea2_vae_ref.py and report
# cosine / rel-L2 vs the diffusers reference. Assumes krea2_vae_ref.py has
# already populated .parity/.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build_cuda/Release/brodiffusion.exe
VAE=weights/krea-2-raw/vae/diffusion_pytorch_model.safetensors

"$BIN" krea2-vae-fwd \
  --weights "$VAE" \
  --latent .parity/krea2_ref_latent.f32 \
  --out .parity/krea2_mine_image.f32 \
  --H 8 --W 8

python - <<'PY'
import numpy as np
ref  = np.fromfile('.parity/krea2_ref_image.f32',  dtype='<f4')
mine = np.fromfile('.parity/krea2_mine_image.f32', dtype='<f4')
cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
rel = float(np.linalg.norm(ref - mine) / np.linalg.norm(ref))
print('cosine       %.6f' % cos)
print('rel L2 err   %.6f' % rel)
print('ref  std %.5f  mine std %.5f  maxabsdiff %.5f'
      % (ref.std(), mine.std(), float(np.max(np.abs(ref - mine)))))
PY
