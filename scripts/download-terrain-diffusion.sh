#!/usr/bin/env bash
# Download the REAL terrain-diffusion pipeline checkpoint — ungated, plain curl,
# no HF CLI or auth token.
#
# terrain-diffusion (Alexander Goslin, MIT, arXiv 2512.08309) is a learned
# replacement for Perlin noise: infinite, deterministic, randomly-accessible
# terrain with elevation *and* climate. Three EDM2 magnitude-preserving UNets:
#
#   config.json                                       WorldPipeline constants
#   coarse_model/{config.json,*.safetensors}     2.8M  7.7 km/cell, 6ch elev+climate
#   base_model/{config.json,*.safetensors}       254M  latent stage
#   decoder_model/{config.json,*.safetensors}     28M  512x512 elevation residual
#
# ~1.14 GB fp32 total.
#
# Two variants exist; 30m is the one to use for playable worlds (finer local
# control), 90m is more expansive and often too expansive.
#
# Usage: scripts/download-terrain-diffusion.sh [--out-dir DIR] [--90m] [--force]
#   --out-dir DIR   default: weights/terrain-diffusion-30m
#   --90m           fetch the 90 m/px model instead of 30 m/px
set -euo pipefail

OUT_DIR=""
FORCE=0
RES="30m"
while [ $# -gt 0 ]; do
    case "$1" in
        --out-dir) OUT_DIR="${2:?}"; shift 2 ;;
        --90m)     RES="90m"; shift ;;
        --30m)     RES="30m"; shift ;;
        --force)   FORCE=1; shift ;;
        *) echo "error: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/terrain-diffusion-$RES"
mkdir -p "$OUT_DIR"; OUT_DIR="$(cd "$OUT_DIR" && pwd)"

REPO="xandergos/terrain-diffusion-$RES"

fetch() {
    local rel="$1" dest="$OUT_DIR/$1"
    local url="https://huggingface.co/$REPO/resolve/main/$rel"
    if [ "$FORCE" -eq 0 ] && [ -s "$dest" ]; then echo "==> $rel (cached)"; return 0; fi
    echo "==> $rel"
    mkdir -p "$(dirname "$dest")"
    local code
    code="$(curl -sL --retry 5 --retry-delay 3 -o "$dest.part" -w '%{http_code}' "$url")" \
        || { echo "   curl failed for $url" >&2; rm -f "$dest.part"; return 2; }
    [ "$code" = "200" ] && { mv "$dest.part" "$dest"; return 0; }
    rm -f "$dest.part"; echo "   HTTP $code for $url" >&2; return 2
}

FILES=(
    config.json
    coarse_model/config.json  coarse_model/diffusion_pytorch_model.safetensors
    base_model/config.json    base_model/diffusion_pytorch_model.safetensors
    decoder_model/config.json decoder_model/diffusion_pytorch_model.safetensors
)
for f in "${FILES[@]}"; do fetch "$f"; done

echo; echo "Done -> $OUT_DIR"
du -sh "$OUT_DIR" 2>/dev/null || true
echo
echo "Next: scripts/convert-terrain-diffusion.py --src \"$OUT_DIR\""
