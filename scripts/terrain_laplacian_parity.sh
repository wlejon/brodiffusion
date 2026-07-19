#!/usr/bin/env bash
# Parity gate for the Laplacian resampling primitives, against torchvision itself.
#
# WHY THIS EXISTS SEPARATELY. The composed elevation gate cannot protect these.
# Measured, not supposed: replacing the antialiased triangle filter with plain
# bilinear — a real algorithmic error, and the exact one laplacian.h warns about
# — moves the elevation output from 2.058e-3 to 3.063e-3 relative, which sails
# under that gate's 1e-2 bar on every case. The reason is structural: the low
# band is Gaussian-blurred at sigma=5 immediately afterwards, and the blur
# removes most of what the resampling choice affects. So the composition is
# genuinely insensitive and the primitives need gating at their own level, where
# the same substitution shows up at ~1e-1.
#
# The bar here is 1e-6 relative, essentially exact. These are deterministic
# host-side double-precision kernels compared against torchvision's own output;
# the only gap should be the f32 round on the way out of the CLI. Anything
# larger is an algorithm difference, not arithmetic.
#
# The probe input is deliberately neither symmetric nor separable: a symmetric
# input would hide a transposed axis, and a separable one would hide a swapped
# horizontal/vertical pass.
#
# Usage: scripts/terrain_laplacian_parity.sh [path/to/brodiffusion.exe]
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

BIN="${1:-build_cuda/Release/brodiffusion.exe}"
[ -x "$BIN" ] || BIN="build_cpu/Release/brodiffusion.exe"
[ -x "$BIN" ] || { echo "error: no brodiffusion.exe found" >&2; exit 2; }
BIN="$REPO/$BIN"
mkdir -p "$REPO/.parity"

BAR="1e-6"
status=0

while read -r OP IH IW OH OW; do
  [ -n "${OP:-}" ] || continue

  MINE="$REPO/.parity/lap_mine_${OP}_${IH}x${IW}_${OH}x${OW}.f32"
  REF="$REPO/.parity/lap_ref_${OP}_${IH}x${IW}_${OH}x${OW}.f32"
  rm -f "$MINE" "$REF"

  "$BIN" terrain-laplacian --op "$OP" --ih "$IH" --iw "$IW" --oh "$OH" --ow "$OW" \
      --out "$MINE" >/dev/null \
    || { echo "$OP $IH x $IW -> $OH x $OW: CLI FAILED"; status=1; continue; }
  [ -s "$MINE" ] || { echo "$OP: CLI WROTE NOTHING (stale binary?)"; status=1; continue; }

  MINE="$MINE" REF="$REF" OP="$OP" IH="$IH" IW="$IW" OH="$OH" OW="$OW" BAR="$BAR" \
    python - <<'PY' || status=1
import os, sys
import numpy as np
import torch
import torchvision.transforms.functional as TF

BIL = TF.InterpolationMode.BILINEAR
op = os.environ["OP"]
ih, iw = int(os.environ["IH"]), int(os.environ["IW"])
oh, ow = int(os.environ["OH"]), int(os.environ["OW"])
bar = float(os.environ["BAR"])

# Must match run_terrain_laplacian's generator exactly.
y, x = np.meshgrid(np.arange(ih), np.arange(iw), indexing="ij")
src = (np.sin(0.13 * x + 0.29 * y) + 0.4 * np.cos(0.07 * x * y + 1.0)
       + 0.001 * (x * x - y)).astype(np.float64)
t = torch.from_numpy(src)[None, None]


def pad_lin(a):
    top = 2 * a[..., 0:1, :] - a[..., 1:2, :]
    bot = 2 * a[..., -1:, :] - a[..., -2:-1, :]
    a = torch.cat([top, a, bot], dim=-2)
    left = 2 * a[..., :, 0:1] - a[..., :, 1:2]
    right = 2 * a[..., :, -1:] - a[..., :, -2:-1]
    return torch.cat([left, a, right], dim=-1)


def resize_extrap(a, size):
    th, tw = size
    h, w = a.shape[-2:]
    sh, sw = th / h, tw / w
    big = TF.resize(pad_lin(a), (int(round(th + 2 * sh)), int(round(tw + 2 * sw))),
                    interpolation=BIL)
    ph, pw = int(round(sh)), int(round(sw))
    return big[..., ph:ph + th, pw:pw + tw]


if op == "resize":
    ref = TF.resize(t, [oh, ow], interpolation=BIL)
elif op == "blur":
    ref = TF.gaussian_blur(t, kernel_size=11, sigma=5.0)
elif op == "extrap":
    ref = resize_extrap(t, (oh, ow))
elif op == "denoise":
    ly, lx = np.meshgrid(np.arange(oh), np.arange(ow), indexing="ij")
    low = torch.from_numpy((np.cos(0.31 * lx - 0.17 * ly) + 0.2 * lx).astype(np.float64))[None, None]
    decoded = t + resize_extrap(low, (ih, iw))            # laplacian_decode(extrapolate=True)
    small = TF.resize(decoded, [oh, ow], interpolation=BIL)
    new_low = TF.gaussian_blur(small, kernel_size=11, sigma=5.0)
    ref = t + TF.resize(new_low, [ih, iw], interpolation=BIL)  # decode(extrapolate=False)
else:
    print(f"unknown op {op}"); sys.exit(1)

ref = ref.numpy()[0, 0].astype(np.float64)
mine = np.fromfile(os.environ["MINE"], dtype="<f4").astype(np.float64)
if mine.size != ref.size:
    print(f"   {op:8s} SIZE MISMATCH {mine.size} vs {ref.size}"); sys.exit(1)
mine = mine.reshape(ref.shape)

denom = np.linalg.norm(ref)
rel = float(np.linalg.norm(mine - ref) / denom) if denom > 0 else float(np.linalg.norm(mine - ref))
flag = "" if rel <= bar else "   <-- OVER BAR"
print(f"   {op:8s} {ih}x{iw} -> {oh}x{ow}  relL2 {rel:.3e}  "
      f"maxabs {np.abs(mine - ref).max():.3e}{flag}")
sys.exit(0 if rel <= bar else 1)
PY
# op  ih  iw  oh  ow
#
# Both resize directions, and non-square and non-integer-ratio shapes: the
# integer-ratio square case is the one where several wrong implementations
# happen to agree with the right one.
done <<'CASES'
resize 64 64 8 8
resize 80 48 10 6
resize 512 512 64 64
resize 8 8 64 64
resize 6 10 48 80
resize 37 53 11 17
blur 64 64 0 0
blur 33 51 0 0
extrap 8 8 64 64
extrap 10 6 80 48
denoise 64 64 8 8
denoise 128 128 16 16
CASES

[ "$status" -eq 0 ] && echo "ALL LAPLACIAN PARITY OK" || echo "LAPLACIAN PARITY FAILED"
exit "$status"
