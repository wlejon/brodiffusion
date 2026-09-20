#!/usr/bin/env bash
# Parity gate for the Qwen-Image 2.1 VAE: decode and encode through the C++
# path on the SAME inputs as scripts/qwenimage21_vae_ref.py and report
# cosine / rel-L2 vs the diffusers-main reference. Assumes the ref script has
# already populated .parity/.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build_cuda/Release/brodiffusion.exe
VAE=${QI21_VAE:-D:/projects/Qwen-Image-2.1/vae/diffusion_pytorch_model.safetensors}

"$BIN" qi21-vae-fwd --weights "$VAE" \
  --latent .parity/qi21_ref_latent.f32 --out .parity/qi21_mine_image.f32 --H 8 --W 8
"$BIN" qi21-vae-fwd --weights "$VAE" \
  --image .parity/qi21_ref_input.f32 --out .parity/qi21_mine_encoded.f32 --H 128 --W 128

PY=${PYTHON:-/c/Users/jonny/scoop/apps/python/current/python}
"$PY" - <<'PY'
import numpy as np
def cmp(name, ref_path, mine_path):
    ref  = np.fromfile(ref_path,  dtype='<f4')
    mine = np.fromfile(mine_path, dtype='<f4')
    assert ref.size == mine.size, (name, ref.size, mine.size)
    cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
    rel = float(np.linalg.norm(ref - mine) / np.linalg.norm(ref))
    print('%-8s cosine %.6f  rel L2 %.6f  ref std %.5f  mine std %.5f  maxabs %.5f'
          % (name, cos, rel, ref.std(), mine.std(), float(np.max(np.abs(ref - mine)))))
cmp('decode', '.parity/qi21_ref_image.f32',   '.parity/qi21_mine_image.f32')
cmp('encode', '.parity/qi21_ref_encoded.f32', '.parity/qi21_mine_encoded.f32')
PY
