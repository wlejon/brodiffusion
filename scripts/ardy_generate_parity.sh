#!/usr/bin/env bash
# Parity gate for the ARDY autoregressive text-to-motion rollout (hybrid level).
# Runs the PyTorch reference — the real Ardy.__call__ over several gen_horizon
# windows (history-conditioned sampling + per-window recenter/requantize + global
# translation) — capturing the world-frame hybrid before FSQ detokenization plus
# the per-window noise, then the C++ `ardy-generate` on the SAME noise/text, and
# reports cosine / rel-L2 / maxabsdiff for the full hybrid sequence.
#
# Usage: scripts/ardy_generate_parity.sh [path/to/brodiffusion.exe]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-build_cpu/Release/brodiffusion.exe}"
[ -x "$BIN" ] || BIN="build_cuda/Release/brodiffusion.exe"
[ -x "$BIN" ] || BIN="build/Release/brodiffusion.exe"
CKPT="${ARDY_CKPT:-weights/ardy-g152}"
FRAMES="${ARDY_GEN_FRAMES:-104}"

echo "== reference (frames=$FRAMES) =="
ARDY_GEN_FRAMES="$FRAMES" python scripts/ardy_generate_ref.py

echo "== C++ ($BIN) =="
"$BIN" ardy-generate \
  --denoiser  "$CKPT/denoiser.safetensors" \
  --tokenizer "$CKPT/tokenizer.safetensors" \
  --mean "$CKPT/stats/motion/mean.npy" \
  --std  "$CKPT/stats/motion/std.npy" \
  --pq-mean "$CKPT/stats/post_quantization/mean.npy" \
  --pq-std  "$CKPT/stats/post_quantization/std.npy" \
  --text  .parity/ardy_gen_text.f32 \
  --noise .parity/ardy_gen_noise.f32 \
  --frames "$FRAMES" --steps 10 --cfg 2.5 --heading 0.6 \
  --out .parity/ardy_gen_mine_hyb.f32 \
  --out-motion .parity/ardy_gen_mine_motion.f32

# Strict detokenize check on the REFERENCE hybrid: isolates the FSQ-decode +
# local-root conditioning from the rollout (both sides detokenize identical
# tokens), so it is not perturbed by an FSQ-grid boundary flip in the rollout.
echo "== C++ detok (ref hybrid) =="
"$BIN" ardy-detok-motion \
  --denoiser  "$CKPT/denoiser.safetensors" \
  --tokenizer "$CKPT/tokenizer.safetensors" \
  --mean "$CKPT/stats/motion/mean.npy" \
  --std  "$CKPT/stats/motion/std.npy" \
  --pq-mean "$CKPT/stats/post_quantization/mean.npy" \
  --pq-std  "$CKPT/stats/post_quantization/std.npy" \
  --hybrid .parity/ardy_gen_ref_hyb.f32 \
  --T-tok $(( (FRAMES + 51) / 52 * 13 )) \
  --out .parity/ardy_gen_detok_motion.f32

python - <<'PY'
import numpy as np
import sys


def stats(ref_path, mine_path):
    ref = np.fromfile(ref_path, dtype='<f4')
    mine = np.fromfile(mine_path, dtype='<f4')
    assert ref.shape == mine.shape, (ref.shape, mine.shape)
    cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
    rel = float(np.linalg.norm(ref - mine) / (np.linalg.norm(ref) + 1e-30))
    mad = float(np.max(np.abs(ref - mine)))
    return cos, rel, mad, ref.std(), mine.std()


def report(label, tup):
    cos, rel, mad, rs, ms = tup
    print("%-16s cosine %.8f  relL2 %.3e  maxabsdiff %.3e  (ref std %.5f mine %.5f)"
          % (label, cos, rel, mad, rs, ms))


# Gate 1: the autoregressive rollout, at the hybrid level (strict).
h = stats('.parity/ardy_gen_ref_hyb.f32', '.parity/ardy_gen_mine_hyb.f32')
report("world-hybrid", h)
# Gate 2: the FSQ detokenize -> explicit motion on identical (ref) tokens (strict).
d = stats('.parity/ardy_gen_ref_motion.f32', '.parity/ardy_gen_detok_motion.f32')
report("explicit(detok)", d)
# Informational: full text->motion (mine rollout + mine detok). Inherits any
# FSQ-grid boundary flip from the rollout, amplified through the causal decoder.
e = stats('.parity/ardy_gen_ref_motion.f32', '.parity/ardy_gen_mine_motion.f32')
report("explicit(e2e)", e)

ok = (h[0] > 0.99999 and h[1] < 2e-3) and (d[0] > 0.99999 and d[1] < 2e-3)
print("PARITY OK" if ok else "PARITY FAILED")
sys.exit(0 if ok else 1)
PY
