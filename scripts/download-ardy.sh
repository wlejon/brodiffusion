#!/usr/bin/env bash
# Download the REAL ARDY-G1-RP-25FPS-Horizon52 (g152) checkpoint — ungated,
# plain curl, no HF CLI or auth token.
#
#   config.yaml            model + motion-rep config
#   tokenizer.safetensors  FSQ motion autoencoder (encoder + decoder)   143 MB
#   denoiser.safetensors   two-stage diffusion denoiser                 631 MB
#   stats/{motion,pre_quantization,post_quantization}/{mean,std}.npy
#
# Usage: scripts/download-ardy.sh [--out-dir DIR] [--force]
#   --out-dir DIR   default: weights/ardy-g152
set -euo pipefail

OUT_DIR=""
FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --out-dir) OUT_DIR="${2:?}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        *) echo "error: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/ardy-g152"
mkdir -p "$OUT_DIR"; OUT_DIR="$(cd "$OUT_DIR" && pwd)"

REPO="nvidia/ARDY-G1-RP-25FPS-Horizon52"

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
    config.yaml
    tokenizer.safetensors
    denoiser.safetensors
    stats/motion/mean.npy stats/motion/std.npy
    stats/pre_quantization/mean.npy stats/pre_quantization/std.npy
    stats/post_quantization/mean.npy stats/post_quantization/std.npy
)
for f in "${FILES[@]}"; do fetch "$f"; done

echo; echo "Done -> $OUT_DIR"
du -sh "$OUT_DIR" 2>/dev/null || true
