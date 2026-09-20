#!/usr/bin/env bash
# Parity gate for the Qwen-Image 2.1 image DiT.
#
# Default mode `synth` builds a small random-weight
# QwenImage21Transformer2DModel and compares FP32-vs-FP32 (exact architecture
# check, forced onto the CPU backend so brodiffusion runs FP32). Both the
# "extract" prefill step and the "cached" decode step are compared.
#
# Mode `real` runs the actual 7.1B transformer: the reference in BF16 on CUDA
# (set QI21_REF_DEVICE=cpu for FP32 on CPU) and brodiffusion in BF16, then
# again with INT8 weight-only quantisation.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-synth}"
BIN=build_cuda/Release/brodiffusion.exe
PY=${PYTHON:-/c/Users/jonny/scoop/apps/python/current/python}

"$PY" scripts/qwenimage21_dit_ref.py "$MODE" .parity
read -r HP WP SEQ T1 T2 < .parity/qi21_dit_dims.txt

compare() {
  "$PY" - "$1" "$2" "$3" <<'PY'
import sys
import numpy as np
tag, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
ref  = np.fromfile(a, dtype='<f4')
mine = np.fromfile(b, dtype='<f4')
assert ref.shape == mine.shape, (tag, ref.shape, mine.shape)
cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
rel = float(np.linalg.norm(ref - mine) / np.linalg.norm(ref))
print('%-18s cosine %.8f   relL2 %.8f   ref std %.5f  mine std %.5f  maxabs %.6f'
      % (tag, cos, rel, ref.std(), mine.std(), float(np.max(np.abs(ref - mine)))))
PY
}

run_bd() {
  # $1 = output tag, remaining args are extra flags for qi21-fwd
  local tag="$1"; shift
  "$BIN" qi21-fwd \
    --weights-dir "$WDIR" \
    --config "$WDIR/config.json" \
    --latent .parity/qi21_dit_latent.f32 \
    --embeds .parity/qi21_dit_embeds.f32 \
    --out  ".parity/qi21_dit_mine_${tag}.f32" \
    --out2 ".parity/qi21_dit_mine2_${tag}.f32" \
    --t "$T1" --t2 "$T2" --steps 2 \
    --hp "$HP" --wp "$WP" --seq "$SEQ" "$@" > /dev/null
  compare "${tag} step0"      .parity/qi21_dit_velocity.f32  ".parity/qi21_dit_mine_${tag}.f32"
  compare "${tag} step1/cache" .parity/qi21_dit_velocity2.f32 ".parity/qi21_dit_mine2_${tag}.f32"
}

if [ "$MODE" = "synth" ]; then
  WDIR=.parity/qi21_synth_transformer
  # brodiffusion runs FP32 only on the CPU backend; force it so the compute
  # dtype matches the FP32 reference exactly.
  export BROTENSOR_DEFAULT_DEVICE=CPU
  run_bd fp32
  # The cached decode must reproduce a cache-free full prefill exactly.
  run_bd nocache --no-cache
else
  WDIR=${QI21_DIR:-weights/qwen-image-2.1}/transformer
  run_bd bf16
  run_bd int8 --quantize
fi
