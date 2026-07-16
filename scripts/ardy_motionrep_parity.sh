#!/usr/bin/env bash
# Parity gate for the ARDY motion-rep codec (G1 skeleton + FK + cont6d +
# forward/inverse). Runs the PyTorch reference to populate .parity/, then the
# C++ subcommand on the SAME inputs, and reports cosine / rel-L2 / maxabsdiff
# for features, posed joints, and recovered local rotations.
#
# Usage: scripts/ardy_motionrep_parity.sh [path/to/brodiffusion.exe]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-build_cuda/Release/brodiffusion.exe}"
[ -x "$BIN" ] || BIN="build/Release/brodiffusion.exe"
[ -x "$BIN" ] || BIN="build_cpu/brodiffusion"

echo "== reference =="
python scripts/ardy_motionrep_ref.py

# T is fixed in the ref script (6); keep in sync.
T=6

echo "== C++ ($BIN) =="
"$BIN" ardy-motionrep-fwd \
  --local-rots .parity/ardy_mr_local_rots.f64 \
  --root-pos   .parity/ardy_mr_root_pos.f64 \
  --T "$T" \
  --out-features .parity/ardy_mr_mine_features.f64 \
  --out-posed    .parity/ardy_mr_mine_posed.f64 \
  --out-local    .parity/ardy_mr_mine_local.f64

python - <<'PY'
import numpy as np

def cmp(name, ref_p, mine_p):
    ref  = np.fromfile(ref_p,  dtype='<f8')
    mine = np.fromfile(mine_p, dtype='<f8')
    assert ref.shape == mine.shape, (name, ref.shape, mine.shape)
    cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine)))
    rel = float(np.linalg.norm(ref - mine) / (np.linalg.norm(ref) + 1e-30))
    mad = float(np.max(np.abs(ref - mine)))
    ok = "OK " if (cos > 0.999999 and mad < 1e-9) else "!! "
    print("%s%-9s cosine %.9f  relL2 %.3e  maxabsdiff %.3e" % (ok, name, cos, rel, mad))
    return ok == "OK "

allok = True
allok &= cmp("features", '.parity/ardy_mr_ref_features.f64', '.parity/ardy_mr_mine_features.f64')
allok &= cmp("posed",    '.parity/ardy_mr_ref_posed.f64',    '.parity/ardy_mr_mine_posed.f64')
allok &= cmp("local",    '.parity/ardy_mr_ref_local.f64',    '.parity/ardy_mr_mine_local.f64')
print("PARITY OK" if allok else "PARITY FAILED")
import sys; sys.exit(0 if allok else 1)
PY
