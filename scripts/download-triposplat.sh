#!/usr/bin/env bash
# Download the brodiffusion-owned half of TripoSplat's weights.
#
# TripoSplat (VAST-AI/TripoSplat) is a single-image -> 3D Gaussian-Splat model
# composed from several backbones. The two image *encoders* (DINOv3 + BiRefNet)
# live in brovisionml and have their own download script; the generative core
# lives here in brodiffusion:
#
#   - the Flux2 VAE image encoder (conditioning feature2),
#   - the LatentSeqMMFlowModel flow-matching DiT,
#   - the OctreeGaussian VAE decoder (latent -> 3D Gaussians).
#
# Like the rest of the stack these load directly from HF `safetensors` (no
# offline conversion) — this script just fetches the right files from the HF
# `resolve` endpoint with `curl`. No Python, no HF CLI dependency.
#
# Usage:
#   scripts/download-triposplat.sh [component] [--repo R] [--out-dir D] [--force]
#
#   component        all (default) | vae | dit | decoder
#   --repo R         override the HuggingFace repo id (default VAST-AI/TripoSplat)
#   --out-dir D      override the output directory (default weights/triposplat)
#   --force          re-download even if the file already exists
#
# Components (all ship as a single un-sharded safetensors with no config.json —
# the loaders carry the architecture as a preset, since the upstream repo
# bundles raw weights only):
#
#   vae        vae/flux2-vae.safetensors                  (~336 MB) — Flux.2 VAE
#              (32-ch latent). Only the encoder half is used: image -> 128-ch
#              pixel-shuffled, bn-normalized conditioning tokens (feature2).
#   dit        diffusion_models/triposplat_fp16.safetensors  (~741 MB) —
#              LatentSeqMMFlowModel: the flow-matching transformer (1024 hidden,
#              16 heads, content-dependent 3D RoPE, QK-RMSNorm, adaLN-single).
#   decoder    vae/triposplat_vae_decoder_fp16.safetensors   (~576 MB) —
#              OctreeGaussianDecoder: octree structure predictor + elastic
#              Gaussian head (latent -> up to 262144 3D Gaussians).
#
# Auth: the repo is public and needs no token. For rate-limited or gated repos,
# export HF_TOKEN=hf_... and it is sent as a bearer token.
#
# Output: <repo>/weights/triposplat/   (subpaths preserved from the HF repo, so
# this tree mirrors brovisionml's half — clip_vision/, background_removal/,
# vae/, diffusion_models/ all under one weights/triposplat/.)

set -euo pipefail

# --- arg parsing ------------------------------------------------------------
COMPONENT="all"
REPO=""
OUT_DIR=""
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        all|vae|dit|decoder) COMPONENT="$1"; shift ;;
        --repo)    REPO="${2:?--repo needs a value}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        -h|--help)
            sed -n '2,46p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

[ -n "$REPO" ] || REPO="VAST-AI/TripoSplat"
[ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/triposplat"

# --- per-component file lists -----------------------------------------------
# Subpaths are kept exactly as they are in the HF repo.
FILES=()
case "$COMPONENT" in
    vae)     FILES=( "vae/flux2-vae.safetensors" ) ;;
    dit)     FILES=( "diffusion_models/triposplat_fp16.safetensors" ) ;;
    decoder) FILES=( "vae/triposplat_vae_decoder_fp16.safetensors" ) ;;
    all)     FILES=( "vae/flux2-vae.safetensors"
                     "diffusion_models/triposplat_fp16.safetensors"
                     "vae/triposplat_vae_decoder_fp16.safetensors" ) ;;
esac

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "Component: $COMPONENT"
echo "Repo:      $REPO"
echo "Target:    $OUT_DIR"
[ -n "${HF_TOKEN:-}" ] && echo "Auth:      HF_TOKEN (bearer)"
echo

# --- download helper --------------------------------------------------------
# fetch <relative-path> <dest-file> [repo] -> 0 on success, 1 on a 404,
# 2 on any other error. `repo` defaults to $REPO.
fetch() {
    local rel="$1" dest="$2" repo="${3:-$REPO}"
    local url="https://huggingface.co/$repo/resolve/main/$rel"
    local auth=()
    [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")

    mkdir -p "$(dirname "$dest")"
    local code
    code="$(curl -sL --retry 3 --retry-delay 2 \
                 "${auth[@]}" \
                 -o "$dest.part" -w '%{http_code}' "$url")" || {
        echo "    curl failed for $url" >&2
        rm -f "$dest.part"
        return 2
    }
    if [ "$code" = "200" ]; then
        mv "$dest.part" "$dest"
        return 0
    fi
    rm -f "$dest.part"
    if [ "$code" = "404" ]; then return 1; fi
    echo "    HTTP $code for $url" >&2
    return 2
}

for f in "${FILES[@]}"; do
    dest="$OUT_DIR/$f"
    if [ "$FORCE" -eq 0 ] && [ -s "$dest" ]; then
        echo "==> $f  (cached, skipping)"
        continue
    fi
    echo "==> $f"
    if ! fetch "$f" "$dest"; then
        echo "error: download failed for $f" >&2
        exit 1
    fi
done

echo
echo "Done. Files in $OUT_DIR :"
find "$OUT_DIR" -type f | sort | while read -r p; do
    sz="$(wc -c < "$p" | tr -d ' ')"
    printf '  %12s  %s\n' "$sz" "${p#$OUT_DIR/}"
done
