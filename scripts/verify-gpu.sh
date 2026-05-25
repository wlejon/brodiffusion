#!/usr/bin/env bash
# GPU verification for brodiffusion against real weights.
#
# Exercises every public pipeline path on the CUDA build with downloaded SD1.5
# weights — txt2img (baseline), img2img at three strengths, inpaint with a
# generated center-square mask, and ControlNet. PNGs land in <out-dir>/ for
# eyeball inspection; this is the "run a few and look" verification, not an
# automated pixel diff.
#
# Usage:
#   scripts/verify-gpu.sh [--exe <brodiffusion.exe>] [--out-dir <dir>]
#                         [--prompt <text>] [--steps N] [--seed N]
#                         [--cn canny|depth|openpose|none]
#                         [--skip txt2img,img2img,inpaint,controlnet]
#
# Defaults:
#   --exe       build_cuda/Release/brodiffusion.exe
#   --out-dir   out/verify-gpu
#   --prompt    "a photo of an astronaut riding a horse on the moon"
#   --steps     20  (DDIM)
#   --seed      42
#   --cn        canny
#
# Prerequisites:
#   * scripts/download-weights.sh sd15
#   * scripts/download-weights.sh controlnet-canny   (or whatever --cn picks)
#   * A CUDA build at the --exe path
#
# Exit codes: 0 = every requested step produced an output; non-zero = first
# failure. The script does NOT diff outputs — it's a smoke check that the GPU
# path runs end-to-end and that the resulting PNGs are well-formed (non-empty).

set -euo pipefail

EXE=""
OUT_DIR=""
PROMPT="a photo of an astronaut riding a horse on the moon"
STEPS=20
SEED=42
CN="canny"
SKIP=""

while [ $# -gt 0 ]; do
    case "$1" in
        --exe)     EXE="${2:?--exe needs a value}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --prompt)  PROMPT="${2:?--prompt needs a value}"; shift 2 ;;
        --steps)   STEPS="${2:?--steps needs a value}"; shift 2 ;;
        --seed)    SEED="${2:?--seed needs a value}"; shift 2 ;;
        --cn)      CN="${2:?--cn needs a value}"; shift 2 ;;
        --skip)    SKIP="${2:?--skip needs a value}"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument '$1' (try --help)" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -n "$EXE" ]     || EXE="$REPO_ROOT/build_cuda/Release/brodiffusion.exe"
[ -n "$OUT_DIR" ] || OUT_DIR="$REPO_ROOT/out/verify-gpu"

SD15_DIR="$REPO_ROOT/weights/sd15"
INIT_PNG="$SD15_DIR/grid/grid.png"

if [ ! -x "$EXE" ] && [ ! -f "$EXE" ]; then
    echo "error: brodiffusion exe not found at $EXE" >&2
    echo "       run: cmake --build build_cuda --config Release" >&2
    exit 1
fi
if [ ! -d "$SD15_DIR/unet" ]; then
    echo "error: SD1.5 weights missing under $SD15_DIR" >&2
    echo "       run: scripts/download-weights.sh sd15" >&2
    exit 1
fi
if [ ! -f "$INIT_PNG" ]; then
    echo "error: init image not found at $INIT_PNG" >&2
    echo "       any 512x512 PNG works; pass --init manually if needed" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

skipped() { case ",$SKIP," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
run()     { echo; echo "==> $*"; "$@"; }
check()   {
    local f="$1"
    if [ ! -s "$f" ]; then
        echo "error: expected output $f is missing or empty" >&2
        exit 1
    fi
    printf '    ok %s (%s bytes)\n' "$f" "$(wc -c < "$f" | tr -d ' ')"
}

echo "brodiffusion GPU verify"
echo "  exe:      $EXE"
echo "  weights:  $SD15_DIR"
echo "  out:      $OUT_DIR"
echo "  prompt:   $PROMPT"
echo "  steps:    $STEPS  seed: $SEED"

# 1. txt2img baseline -------------------------------------------------------
if ! skipped txt2img; then
    out="$OUT_DIR/01_txt2img.png"
    run "$EXE" txt2img \
        --model "$SD15_DIR" \
        --prompt "$PROMPT" \
        --out "$out" \
        --steps "$STEPS" --seed "$SEED"
    check "$out"
fi

# 2. img2img at three strengths --------------------------------------------
if ! skipped img2img; then
    for strength in 0.3 0.6 0.9; do
        out="$OUT_DIR/02_img2img_s${strength}.png"
        run "$EXE" img2img \
            --model "$SD15_DIR" \
            --init "$INIT_PNG" \
            --strength "$strength" \
            --prompt "$PROMPT" \
            --out "$out" \
            --steps "$STEPS" --seed "$SEED"
        check "$out"
    done
fi

# 3. inpaint with a generated center-square mask ---------------------------
if ! skipped inpaint; then
    mask="$OUT_DIR/03_mask.png"
    run "$EXE" make-mask --out "$mask" --width 512 --height 512
    check "$mask"

    out="$OUT_DIR/03_inpaint.png"
    run "$EXE" inpaint \
        --model "$SD15_DIR" \
        --init "$INIT_PNG" \
        --mask "$mask" \
        --strength 0.85 \
        --prompt "$PROMPT" \
        --out "$out" \
        --steps "$STEPS" --seed "$SEED"
    check "$out"
fi

# 4. ControlNet ------------------------------------------------------------
if ! skipped controlnet && [ "$CN" != "none" ]; then
    case "$CN" in
        canny)    cn_dir="$REPO_ROOT/weights/controlnet-canny"
                  cn_img="$cn_dir/images/bird_canny.png"
                  cn_prompt="a bird perched on a branch" ;;
        depth)    cn_dir="$REPO_ROOT/weights/controlnet-depth"
                  cn_img="$cn_dir/images/stormtrooper_depth.png"
                  cn_prompt="a stormtrooper standing in a forest" ;;
        openpose) cn_dir="$REPO_ROOT/weights/controlnet-openpose"
                  cn_img="$cn_dir/images/pose.png"
                  cn_prompt="a person dancing in a sunlit field" ;;
        *) echo "error: --cn must be canny|depth|openpose|none" >&2; exit 2 ;;
    esac
    cn_weights="$cn_dir/diffusion_pytorch_model.fp16.safetensors"
    # Some ControlNet repos don't ship the .fp16 variant; the loader accepts
    # either, so fall back to the plain name.
    [ -f "$cn_weights" ] || cn_weights="$cn_dir/diffusion_pytorch_model.safetensors"
    if [ ! -f "$cn_weights" ] || [ ! -f "$cn_img" ]; then
        echo "warning: ControlNet weights or example image missing under $cn_dir" >&2
        echo "         run: scripts/download-weights.sh controlnet-$CN" >&2
        echo "         skipping ControlNet step" >&2
    else
        out="$OUT_DIR/04_controlnet_${CN}.png"
        run "$EXE" txt2img \
            --model "$SD15_DIR" \
            --control "$cn_weights" \
            --control-image "$cn_img" \
            --control-scale 1.0 \
            --prompt "$cn_prompt" \
            --out "$out" \
            --steps "$STEPS" --seed "$SEED"
        check "$out"
    fi
fi

echo
echo "verify-gpu: all requested steps produced non-empty PNGs in $OUT_DIR"
ls -lh "$OUT_DIR" | tail -n +2
