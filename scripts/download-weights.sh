#!/usr/bin/env bash
# Download model weights for brodiffusion end-to-end testing.
#
# Bash port of download-weights.ps1, for macOS / Linux. Unlike the PowerShell
# script it does NOT depend on the Hugging Face CLI — it fetches files straight
# from the HF `resolve` endpoint with `curl`, so it works with any vintage of
# huggingface_hub (or none at all).
#
# Usage:
#   scripts/download-weights.sh [model] [--repo R] [--out-dir D] [--force]
#
#   model            sd15 (default) | lcm-dreamshaper | clip-vit-l-14 |
#                    flux-schnell | sana-600m | sana-1.6b |
#                    sana-sprint-0.6b | sana-sprint-1.6b | pixart-sigma
#   --repo R         override the Hugging Face repo id
#   --out-dir D      override the output directory
#   --force          re-download even if the file already exists
#
# Models:
#   sd15             SD1.5 diffusers components from
#                    stable-diffusion-v1-5/stable-diffusion-v1-5. FP16 component
#                    files (the full v1-5-pruned-emaonly.safetensors uses the
#                    original LDM tensor names which the loaders don't translate).
#   lcm-dreamshaper  LCM-distilled Dreamshaper-7 from SimianLuo/LCM_Dreamshaper_v7.
#                    The repo does NOT ship fp16-suffixed variants of every file;
#                    this script tries the `.fp16.safetensors` name first and
#                    falls back to plain `.safetensors`. The upload helper
#                    (`upload_fp16_checked`) accepts F32 and converts host-side.
#   clip-vit-l-14    OpenAI CLIP ViT-L/14 (vision + text + both projections).
#   flux-schnell     Flux.1-schnell diffusers components from
#                    black-forest-labs/FLUX.1-schnell. The transformer and the
#                    T5-XXL text encoder ship SHARDED — the script fetches each
#                    `*.index.json` first, extracts the shard filenames from it,
#                    and downloads each shard. NOTE: these weights are large
#                    (~24 GB transformer + ~10 GB T5).
#   t5-xxl           Just the T5-XXL text encoder (text_encoder_2) + tokenizer
#                    from the same Flux.1-schnell repo. ~9.5 GB. The standalone
#                    target for working on the T5 encoder.
#   controlnet-canny / controlnet-depth / controlnet-openpose
#                    lllyasviel/sd-controlnet-{canny,depth,openpose}. The FP16
#                    diffusion_pytorch_model + the model card's example
#                    control image (so a clean checkout has something concrete
#                    to feed --control-image). ~1.4 GB each.
#   sana-600m        NVIDIA Sana 0.6B 1024px from
#                    Efficient-Large-Model/Sana_600M_1024px_diffusers (FP16).
#                    SanaTransformer2DModel DiT + DC-AE VAE + sharded Gemma-2
#                    text encoder. 0.6B is FP16-stable.
#   sana-1.6b        NVIDIA Sana 1.6B 1024px. The DiT is pulled as the BF16
#                    variant from Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers
#                    because the 1.6B transformer is numerically UNSTABLE in FP16
#                    (the shipped .fp16 weights decode to confetti — confirmed in
#                    both diffusers and brodiffusion); everything else (VAE,
#                    Gemma-2 encoder, tokenizer, scheduler) is the FP16 set from
#                    Efficient-Large-Model/Sana_1600M_1024px_diffusers, identical
#                    to the 0.6B components.
#   sana-sprint-0.6b / sana-sprint-1.6b
#                    NVIDIA Sana-Sprint (few-step, guidance-distilled) from
#                    Efficient-Large-Model/Sana_Sprint_{0.6B,1.6B}_1024px_diffusers.
#                    Same SanaTransformer2DModel/DC-AE/Gemma-2 components plus a
#                    guidance embedding and an SCMScheduler (TrigFlow, 1-4 steps).
#                    Single (unsharded, non-variant) transformer safetensors.
#   pixart-sigma     PixArt-Sigma-XL-2-1024-MS from PixArt-alpha. A ~0.6B DiT
#                    (PixArtTransformer2DModel: AdaLN-single, softmax self +
#                    text cross-attention; epsilon / discrete linear-beta;
#                    DPM-Solver++) with the SDXL VAE (scale 0.13025) and a
#                    T5-XXL text encoder. The ~19 GB T5 encoder is the SAME one
#                    as Flux's and is NOT fetched here — run `download-weights.sh
#                    t5-xxl` once and the PixArt pipeline reuses weights/t5-xxl.
#                    This entry pulls only the PixArt-specific parts: the FP32
#                    transformer (~2.4 GB), the FP16 SDXL VAE (~0.3 GB), the T5
#                    SentencePiece tokenizer (spiece.model), and the config /
#                    scheduler JSON. text_encoder/config.json is informational.
#
# Auth: the SD1.5 mirror is public and needs no token. For rate-limited repos,
# export HF_TOKEN=hf_... and it will be sent as a bearer token.
#
# Output: <repo>/weights/<model>/
#   text_encoder/model[.fp16].safetensors
#   unet/diffusion_pytorch_model[.fp16].safetensors
#   vae/diffusion_pytorch_model[.fp16].safetensors
#   tokenizer/vocab.json
#   tokenizer/merges.txt

set -euo pipefail

# --- arg parsing ------------------------------------------------------------
MODEL="sd15"
REPO=""
OUT_DIR=""
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        sd15|lcm-dreamshaper|clip-vit-l-14|flux-schnell|t5-xxl|controlnet-canny|controlnet-depth|controlnet-openpose) MODEL="$1"; shift ;;
        sana-600m|sana-1.6b|sana-sprint-0.6b|sana-sprint-1.6b) MODEL="$1"; shift ;;
        pixart-sigma) MODEL="$1"; shift ;;
        --repo)    REPO="${2:?--repo needs a value}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        -h|--help)
            sed -n '2,82p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- per-model file lists ---------------------------------------------------
# Files ending in `.fp16.safetensors` are retried without the `.fp16` infix
# if the primary download 404s — that single rule covers every fallback the
# PowerShell script spelled out explicitly.
case "$MODEL" in
    sd15)
        [ -n "$REPO" ]    || REPO="stable-diffusion-v1-5/stable-diffusion-v1-5"
        [ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/sd15"
        FILES=(
            "model_index.json"
            "scheduler/scheduler_config.json"
            "text_encoder/config.json"
            "text_encoder/model.fp16.safetensors"
            "unet/config.json"
            "unet/diffusion_pytorch_model.fp16.safetensors"
            "vae/config.json"
            "vae/diffusion_pytorch_model.fp16.safetensors"
            "tokenizer/vocab.json"
            "tokenizer/merges.txt"
        )
        ;;
    lcm-dreamshaper)
        [ -n "$REPO" ]    || REPO="SimianLuo/LCM_Dreamshaper_v7"
        [ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/lcm-dreamshaper"
        FILES=(
            "model_index.json"
            "scheduler/scheduler_config.json"
            "text_encoder/config.json"
            "text_encoder/model.fp16.safetensors"
            "unet/config.json"
            "unet/diffusion_pytorch_model.fp16.safetensors"
            "vae/config.json"
            "vae/diffusion_pytorch_model.fp16.safetensors"
            "tokenizer/vocab.json"
            "tokenizer/merges.txt"
        )
        ;;
    clip-vit-l-14)
        [ -n "$REPO" ]    || REPO="openai/clip-vit-large-patch14"
        [ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/clip-vit-l-14"
        FILES=( "model.safetensors" )
        ;;
    t5-xxl)
        # Just the T5-XXL text encoder + its tokenizer — the standalone target
        # for working on the T5 encoder without the ~24 GB Flux transformer.
        # ~9.5 GB (FP16).
        #
        # The Flux.1-schnell repo (which ships T5-XXL as `text_encoder_2`) is
        # now gated, so the weights come from comfyanonymous/flux_text_encoders
        # instead — the standard community T5-XXL for Flux: a single,
        # un-sharded `t5xxl_fp16.safetensors` whose tensor names
        # (`shared.weight`, `encoder.block.*`, ...) match what t5::TextEncoder
        # expects. The tokenizer.json is the plain T5 SentencePiece Unigram
        # model (identical across t5 / t5-v1.1), pulled from google-t5/t5-base;
        # config.json is informational (T5Config defaults already match
        # t5-v1_1-xxl). FILES entries may carry a `repo|path` override.
        [ -n "$REPO" ]    || REPO="comfyanonymous/flux_text_encoders"
        [ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/t5-xxl"
        FILES=(
            "t5xxl_fp16.safetensors"
            "google-t5/t5-base|tokenizer.json"
            "google/t5-v1_1-xxl|config.json"
        )
        ;;
    controlnet-canny|controlnet-depth|controlnet-openpose)
        # ControlNet weights for SD1.5 from lllyasviel. The diffusion_pytorch_model
        # is the residual-producing network; --control-image must be a prepared
        # control map (an edge image for canny, a depth map for depth, a stick-
        # figure pose render for openpose) — brodiffusion does NOT extract these
        # from a natural image. The model card's example image is bundled so
        # --control-image has something to point at out of the box.
        case "$MODEL" in
            controlnet-canny)    cn_repo="lllyasviel/sd-controlnet-canny"
                                 cn_example="images/bird_canny.png" ;;
            controlnet-depth)    cn_repo="lllyasviel/sd-controlnet-depth"
                                 cn_example="images/stormtrooper_depth.png" ;;
            controlnet-openpose) cn_repo="lllyasviel/sd-controlnet-openpose"
                                 cn_example="images/pose.png" ;;
        esac
        [ -n "$REPO" ]    || REPO="$cn_repo"
        [ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/$MODEL"
        FILES=(
            "config.json"
            "diffusion_pytorch_model.fp16.safetensors"
            "$cn_example"
        )
        ;;
    flux-schnell)
        # Flux.1-schnell, diffusers format. The transformer + the T5-XXL text
        # encoder ship sharded — only the `.index.json` weight-maps are listed
        # here; the shard files themselves are discovered from those indexes
        # below. NOTE: large download (~24 GB transformer + ~10 GB T5).
        [ -n "$REPO" ]    || REPO="black-forest-labs/FLUX.1-schnell"
        [ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/flux-schnell"
        FILES=(
            "model_index.json"
            "scheduler/scheduler_config.json"
            "transformer/config.json"
            "transformer/diffusion_pytorch_model.safetensors.index.json"
            "vae/config.json"
            "vae/diffusion_pytorch_model.safetensors"
            "text_encoder/config.json"
            "text_encoder/model.safetensors"
            "text_encoder_2/config.json"
            "text_encoder_2/model.safetensors.index.json"
            "tokenizer/vocab.json"
            "tokenizer/merges.txt"
            "tokenizer_2/tokenizer.json"
        )
        ;;
    sana-600m)
        # Sana 0.6B, diffusers format, FP16. The Gemma-2 text encoder ships
        # SHARDED — only the `*.index.fp16.json` weight-map is listed; the shard
        # files are discovered from it below (same mechanism as flux-schnell, but
        # the index lives under text_encoder/ with an `.fp16` infix).
        [ -n "$REPO" ]    || REPO="Efficient-Large-Model/Sana_600M_1024px_diffusers"
        [ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/sana-600m"
        FILES=(
            "model_index.json"
            "scheduler/scheduler_config.json"
            "transformer/config.json"
            "transformer/diffusion_pytorch_model.fp16.safetensors"
            "vae/config.json"
            "vae/diffusion_pytorch_model.fp16.safetensors"
            "text_encoder/config.json"
            "text_encoder/model.safetensors.index.fp16.json"
            "tokenizer/special_tokens_map.json"
            "tokenizer/tokenizer.json"
            "tokenizer/tokenizer.model"
            "tokenizer/tokenizer_config.json"
        )
        ;;
    sana-1.6b)
        # Sana 1.6B. The DiT MUST be the BF16 variant: the 1.6B transformer
        # overflows/loses precision in FP16, and the shipped `.fp16` weights
        # decode to colorful confetti (verified in both diffusers and
        # brodiffusion). So the transformer (config + weights) is pulled from the
        # BF16 repo via per-file `repo|path` overrides; the VAE, Gemma-2 encoder,
        # tokenizer and scheduler are the FP16 set from the main 1.6B repo (those
        # components are FP16-stable and byte-identical to the 0.6B ones).
        [ -n "$REPO" ]    || REPO="Efficient-Large-Model/Sana_1600M_1024px_diffusers"
        [ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/sana-1.6b"
        sana_bf16="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
        FILES=(
            "model_index.json"
            "scheduler/scheduler_config.json"
            "${sana_bf16}|transformer/config.json"
            "${sana_bf16}|transformer/diffusion_pytorch_model.bf16.safetensors"
            "vae/config.json"
            "vae/diffusion_pytorch_model.fp16.safetensors"
            "text_encoder/config.json"
            "text_encoder/model.safetensors.index.fp16.json"
            "tokenizer/special_tokens_map.json"
            "tokenizer/tokenizer.json"
            "tokenizer/tokenizer.model"
            "tokenizer/tokenizer_config.json"
        )
        ;;
    sana-sprint-0.6b|sana-sprint-1.6b)
        # Sana-Sprint (few-step, guidance-distilled). Same component classes as
        # standard Sana (SanaTransformer2DModel + DC-AE + Gemma-2) but with a
        # guidance embedding in the DiT and an SCMScheduler (TrigFlow, 1-4 steps,
        # no CFG). The weights ship without fp16/bf16 filename variants — one
        # file per component (transformer is a single, unsharded safetensors;
        # the text encoder is sharded via `model.safetensors.index.json`). Run
        # the DiT in BF16.
        case "$MODEL" in
            sana-sprint-0.6b) sprint_repo="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers" ;;
            sana-sprint-1.6b) sprint_repo="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers" ;;
        esac
        [ -n "$REPO" ]    || REPO="$sprint_repo"
        [ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/$MODEL"
        FILES=(
            "model_index.json"
            "scheduler/scheduler_config.json"
            "transformer/config.json"
            "transformer/diffusion_pytorch_model.safetensors"
            "vae/config.json"
            "vae/diffusion_pytorch_model.safetensors"
            "text_encoder/config.json"
            "text_encoder/model.safetensors.index.json"
            "tokenizer/special_tokens_map.json"
            "tokenizer/tokenizer.json"
            "tokenizer/tokenizer.model"
            "tokenizer/tokenizer_config.json"
        )
        ;;
    pixart-sigma)
        # PixArt-Sigma-XL-2-1024-MS. PixArt-specific parts only — the ~19 GB
        # T5-XXL text encoder (text_encoder/model-*.safetensors) is deliberately
        # NOT listed: it is byte-identical to Flux's T5-XXL, so the pipeline reuses
        # weights/t5-xxl (fetch that once with `download-weights.sh t5-xxl`). The
        # transformer is FP32 (no .fp16 variant); the VAE is the FP16 SDXL VAE.
        [ -n "$REPO" ]    || REPO="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
        [ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/weights/pixart-sigma"
        FILES=(
            "model_index.json"
            "scheduler/scheduler_config.json"
            "transformer/config.json"
            "transformer/diffusion_pytorch_model.safetensors"
            "vae/config.json"
            "vae/diffusion_pytorch_model.safetensors"
            "text_encoder/config.json"
            "tokenizer/special_tokens_map.json"
            "tokenizer/spiece.model"
            "tokenizer/tokenizer_config.json"
        )
        ;;
esac

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "Model:   $MODEL"
echo "Repo:    $REPO"
echo "Target:  $OUT_DIR"
[ -n "${HF_TOKEN:-}" ] && echo "Auth:    HF_TOKEN (bearer)"
echo

# --- download helper --------------------------------------------------------
# fetch <relative-path> <dest-file> [repo] -> 0 on success, 1 on a 404,
# exits on error. `repo` defaults to the model's $REPO.
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

# A FILES entry may be a bare path (fetched from $REPO) or `repo|path` to
# pull that one file from a different repo — used by t5-xxl, whose weights,
# tokenizer, and config live in three separate public repos.
for entry in "${FILES[@]}"; do
    f="$entry"
    ent_repo="$REPO"
    case "$entry" in
        *"|"*) ent_repo="${entry%%|*}"; f="${entry#*|}" ;;
    esac
    dest="$OUT_DIR/$f"
    if [ "$FORCE" -eq 0 ] && [ -s "$dest" ]; then
        echo "==> $f  (cached, skipping)"
        continue
    fi
    echo "==> $f${ent_repo:+  [$ent_repo]}"
    if fetch "$f" "$dest" "$ent_repo"; then
        :
    else
        rc=$?
        if [ "$rc" -eq 1 ] && [[ "$f" == *.fp16.safetensors ]]; then
            alt="${f/.fp16.safetensors/.safetensors}"
            echo "    fp16 variant not found, trying fallback: $alt"
            if ! fetch "$alt" "$OUT_DIR/$alt" "$ent_repo"; then
                echo "error: download failed for both $f and $alt" >&2
                exit 1
            fi
        else
            echo "error: download failed for $f" >&2
            exit 1
        fi
    fi
done

# --- sharded components -----------------------------------------------------
# For models whose components ship sharded (flux-schnell), the loop above
# fetched the `*.index.json` weight-maps. Extract the shard filenames from
# each index and fetch every shard.
for idx in "$OUT_DIR"/transformer/*.index.json \
           "$OUT_DIR"/text_encoder/*.index.json \
           "$OUT_DIR"/text_encoder/*.index.*.json \
           "$OUT_DIR"/text_encoder_2/*.index.json; do
    [ -e "$idx" ] || continue
    comp_dir="$(dirname "$idx")"
    comp_rel="$(basename "$comp_dir")"
    echo
    echo "Expanding shards from ${idx#$OUT_DIR/}"
    shards="$(grep -oE '[A-Za-z0-9_.-]+-[0-9]+-of-[0-9]+\.safetensors' "$idx" \
              | sort -u)"
    for s in $shards; do
        dest="$OUT_DIR/$comp_rel/$s"
        if [ "$FORCE" -eq 0 ] && [ -s "$dest" ]; then
            echo "==> $comp_rel/$s  (cached, skipping)"
            continue
        fi
        echo "==> $comp_rel/$s"
        if ! fetch "$comp_rel/$s" "$dest"; then
            echo "error: download failed for shard $comp_rel/$s" >&2
            exit 1
        fi
    done
done

echo
echo "Done. Files in $OUT_DIR :"
find "$OUT_DIR" -type f | sort | while read -r p; do
    sz="$(wc -c < "$p" | tr -d ' ')"
    printf '  %12s  %s\n' "$sz" "${p#$OUT_DIR/}"
done
