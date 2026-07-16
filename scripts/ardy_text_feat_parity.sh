#!/usr/bin/env bash
# Parity gate for ARDY text conditioning (brodiffusion::ardy::TextConditioner).
#
# TOKENS (default): the reference reproduces ardy's prepare_for_tokenization +
# tokenize with the authoritative HF tokenizer; the C++ `ardy-text-feat`
# (--dump-ids/--dump-mask, no model forward) builds the same. Ids must match
# EXACTLY (integer equality) and the pool mask must match exactly.
#
# FULL (ARDY_TEXT_FULL=1): additionally diff the (4096) pooled embedding — the
# reference runs the bidirectional LLM2Vec forward on the real merged checkpoint
# (fp32 CPU) + masked mean; the C++ runs the real encoder via encode_pooled,
# forced onto CPU for fp32 parity. Gate: cosine > 0.9999, relL2 < 1e-3.
#
# Usage: scripts/ardy_text_feat_parity.sh [path/to/brodiffusion.exe]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-build_cpu/Release/brodiffusion.exe}"
[ -x "$BIN" ] || BIN="build_cuda/Release/brodiffusion.exe"
[ -x "$BIN" ] || BIN="build/Release/brodiffusion.exe"

WEIGHTS="${ARDY_TEXT_WEIGHTS:-D:/projects/brolm/weights/llm2vec-llama3-8b}"
PROMPT="${ARDY_TEXT_PROMPT:-a person walks forward and waves}"
TOKJSON="$WEIGHTS/tokenizer.json"
CONFIG="$WEIGHTS/config.json"
ENCODER="$WEIGHTS/model.safetensors"

echo "== reference =="
ARDY_TEXT_WEIGHTS="$WEIGHTS" ARDY_TEXT_PROMPT="$PROMPT" python scripts/ardy_text_feat_ref.py

echo "== C++ tokens ($BIN) =="
"$BIN" ardy-text-feat \
  --tokenizer-json "$TOKJSON" \
  --text "$PROMPT" \
  --dump-ids  .parity/ardy_text_ids_mine.i32 \
  --dump-mask .parity/ardy_text_mask_mine.f32

FULL_FLAG=0
if [ "${ARDY_TEXT_FULL:-}" != "" ] && [ "${ARDY_TEXT_FULL:-0}" != "0" ]; then
  FULL_FLAG=1
  echo "== C++ pooled embedding (CPU fp32) =="
  BROTENSOR_DEFAULT_DEVICE=CPU "$BIN" ardy-text-feat \
    --tokenizer-json "$TOKJSON" \
    --config "$CONFIG" \
    --encoder "$ENCODER" \
    --text "$PROMPT" \
    --out .parity/ardy_text_feat_mine.f32
fi

FULL="$FULL_FLAG" python - <<'PY'
import os
import sys

import numpy as np

ref_ids = np.fromfile('.parity/ardy_text_ids_ref.i32', dtype='<i4')
mine_ids = np.fromfile('.parity/ardy_text_ids_mine.i32', dtype='<i4')
ref_mask = np.fromfile('.parity/ardy_text_mask_ref.f32', dtype='<f4')
mine_mask = np.fromfile('.parity/ardy_text_mask_mine.f32', dtype='<f4')

ok = True
if ref_ids.shape != mine_ids.shape or not np.array_equal(ref_ids, mine_ids):
    ok = False
    print("IDS MISMATCH")
    print("  ref :", ref_ids.tolist())
    print("  mine:", mine_ids.tolist())
else:
    print("ids   OK  (L=%d, exact match)" % len(ref_ids))

if ref_mask.shape != mine_mask.shape or not np.array_equal(ref_mask, mine_mask):
    ok = False
    print("MASK MISMATCH")
    print("  ref :", ref_mask.astype(int).tolist())
    print("  mine:", mine_mask.astype(int).tolist())
else:
    print("mask  OK  (pooled=%d, exact match)" % int(ref_mask.sum()))

if os.environ.get("FULL", "0") == "1":
    ref = np.fromfile('.parity/ardy_text_feat_ref.f32', dtype='<f4')
    mine = np.fromfile('.parity/ardy_text_feat_mine.f32', dtype='<f4')
    assert ref.shape == mine.shape, (ref.shape, mine.shape)
    cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
    rel = float(np.linalg.norm(ref - mine) / (np.linalg.norm(ref) + 1e-30))
    mad = float(np.max(np.abs(ref - mine)))
    print("feat  cosine %.8f  relL2 %.3e  maxabsdiff %.3e" % (cos, rel, mad))
    if not (cos > 0.9999 and rel < 1e-3):
        ok = False

print("PARITY OK" if ok else "PARITY FAILED")
sys.exit(0 if ok else 1)
PY
