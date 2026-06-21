#!/usr/bin/env bash
# Baseline wall-clock for the FP32 PixArt path (committed binary), for an A/B
# against the mixed-precision change. Full release settings: 1024px, 20 steps.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build_cuda/Release/brodiffusion.exe
export BRODIFFUSION_TIME=1

"$BIN" txt2img \
  --model weights/pixart-sigma \
  --prompt "a red fox sitting in a snowy forest, photorealistic" \
  --steps 20 --cfg 4.5 --width 1024 --height 1024 --seed 42 \
  --out pixart_baseline_fp32.png
