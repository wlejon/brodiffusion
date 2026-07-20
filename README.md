# brodiffusion

[![CI](https://github.com/wlejon/brodiffusion/actions/workflows/ci.yml/badge.svg)](https://github.com/wlejon/brodiffusion/actions/workflows/ci.yml)
[![CodeQL](https://github.com/wlejon/brodiffusion/actions/workflows/codeql.yml/badge.svg)](https://github.com/wlejon/brodiffusion/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Diffusion-model inference, written from scratch in C++20 — text-to-image
through the CLI, plus image-to-3D, text-to-motion, and terrain generation as
library modules.
Part of the [bro](https://github.com/wlejon/bro) stack, and usable
standalone as a CLI or library — built on [brotensor](https://github.com/wlejon/brotensor)
for tensors and compute kernels (CPU / CUDA / Metal), [bromath](https://github.com/wlejon/bromath)
for scalar/RNG/color math, [brolm](https://github.com/wlejon/brolm) for
tokenizers and text encoders, and [broimage](https://github.com/wlejon/broimage)
for image decode/encode and host preprocessing.

Every model family below is ported weight-for-weight from its Hugging Face
`diffusers` implementation and checked against it component-by-component
(text encoder, denoiser, VAE, scheduler) — see `scripts/*_parity.sh` and the
`diffusers`-referenced cases in `tests/`. Matches are validated within
FP16/BF16 tolerance.

Runs **CPU-by-default and on a GPU when one is available** — FP32 on the CPU
backend, FP16/BF16 on CUDA or Metal — with the device chosen at runtime, no
rebuild required.

## Model families

All of these run end-to-end through `brodiffusion txt2img --model <dir>`
against a downloaded diffusers model directory — the loader reads
`model_index.json` and auto-detects which family it's looking at.

| Family | Text encoder | Denoiser | Scheduler | Notes |
|---|---|---|---|---|
| Stable Diffusion 1.5 | CLIP | U-Net | DDIM / LCM | img2img, inpaint, ControlNet, LoRA, optional INT8 U-Net |
| Flux.1 | CLIP-pooled + T5-XXL | DiT, double/single-stream joint attention | flow-match Euler | `schnell` defaults to 4 steps |
| Sana / Sana-Sprint | Gemma-2 | linear-attention DiT | flow-match Euler / SCM (TrigFlow) | native 1024px; Sprint is the 2-step guidance-distilled variant |
| PixArt-Sigma | T5-XXL | DiT (ada-norm-single) | DPM-Solver++ (2M) | native 1024px |
| Krea 2 (Raw / Turbo) | Qwen3-VL, 12 tapped decoder layers | single-stream flow-matching DiT | flow-match Euler | Turbo is the distilled, no-CFG, 8-step checkpoint; LoRA as runtime adapters (INT8-safe, live rescale), optional INT8 DiT |

## Other modalities

Three further ports live here as **libraries only** — none is wired into the
CLI; drive them through their headers. Each is validated the same way as the
image families, against a reference implementation (`scripts/*_parity.sh`).

### TripoSplat — single image → 3D Gaussian splats

The generative core of
[VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat): a Flux.2 VAE
image encoder, a flow-matching DiT, and an `OctreeGaussianDecoder` producing up
to 262144 splats. Headers under `brodiffusion/triposplat/`. brodiffusion owns
just this generative core; the image encoders (DINOv3 + BiRefNet) live in
brovisionml and the renderer / `GaussianSplatCloud` container lives in bromesh.

### ARDY — text → humanoid motion

[nvidia/ARDY-G1-RP](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon52),
autoregressive text-to-motion for the Unitree G1. Headers under
`brodiffusion/ardy/`: a G1 skeleton + forward kinematics + motion-rep codec
(`motion_rep.h`), an FSQ motion-tokenizer decoder, a two-stage x0-prediction
denoiser, LLM2Vec text conditioning (via brolm), and `ArdyMotionGenerator`,
which rolls arbitrary-length sequences out window-by-window with CFG and
detokenizes them to explicit 414-dim motion features.

### terrain-diffusion — infinite procedural worlds

[xandergos/terrain-diffusion](https://huggingface.co/xandergos/terrain-diffusion-30m)
(arXiv 2512.08309), a learned replacement for Perlin noise: infinite,
deterministic, randomly-accessible terrain with elevation *and* climate.
Headers under `brodiffusion/terrain/`. Three EDM2 magnitude-preserving UNets
(`mp_unet.h`) run as a lazily-materialized, tile-cached DAG (`infinite_tensor.h`)
so any region is a pure function of `(seed, position)` — a region generates
identically whether or not its neighbours ever existed:

| Stage | Cell size | Output |
|---|---|---|
| coarse | 7.7 km | 6ch elevation + climate, conditioned on a synthetic Perlin climate map |
| latent | `native_resolution * 8` m | 5ch latents, two TrigFlow steps |
| decoder | `native_resolution` m (30 m default) | elevation Laplacian residual |

`WorldPipeline::elevation(i1, j1, i2, j2)` is the product: elevation in metres
at native resolution, reconstructed from the decoder's high-pass band and the
latent map's low-frequency band.

## Build

brotensor always builds its CPU backend; CUDA and Metal are additive and
mutually exclusive. brodiffusion forwards the choice:

```bash
# CPU-only — the default. Runs the full inference path in FP32.
cmake -B build
cmake --build build --config Release

# CUDA (NVIDIA, any OS) — adds the FP16/BF16 GPU path + fused kernels.
cmake -B build -DBROTENSOR_WITH_CUDA=ON
cmake --build build --config Release

# Metal (Apple)
cmake -B build -DBROTENSOR_WITH_METAL=ON
cmake --build build --config Release
```

| Option | Default | Effect |
|---|---|---|
| `BROTENSOR_WITH_CUDA` | `OFF` | Build the CUDA GPU backend |
| `BROTENSOR_WITH_METAL` | `OFF` | Build the Metal GPU backend |
| `BRODIFFUSION_CLI` | on when top-level | Build the `brodiffusion` CLI |
| `BRODIFFUSION_KREA2_CAPI` | on when top-level | Build `krea2_capi`, a flat C API DLL over the Krea 2 components for research bindings |
| `BRODIFFUSION_TESTS` | `ON` | Build the test suite (only runs when brodiffusion is the top-level project) |
| `BRODIFFUSION_INSTALL` | `OFF` | Generate `install` / `find_package` targets |

INT8 (`--quantize-unet`) weight quantization is GPU-only; it's ignored on
the CPU backend.

## Layout

Standalone sibling of [bro](https://github.com/wlejon/bro), [bromath](https://github.com/wlejon/bromath), and
[brotensor](https://github.com/wlejon/brotensor). The build auto-detects siblings at `../<name>`:

```
projects/
├── bromath/          # ../bromath    (header-only math)
├── brotensor/        # ../brotensor  (tensor + compute, CPU + GPU)
├── brolm/            # ../brolm      (tokenizers + text encoders)
├── broimage/         # ../broimage   (image decode/encode + host preproc)
└── brodiffusion/     # this repo
```

If a sibling isn't found at `../<name>`, CMake falls back to the matching
`third_party/` submodule. Override with `-DBROMATH_DIR=...` /
`-DBROTENSOR_DIR=...` / `-DBROLM_DIR=...` / `-DBROIMAGE_DIR=...`.

## Weights

Model weights are not bundled — `scripts/download-weights.sh` (macOS/Linux,
plain `curl`) or `scripts/download-weights.ps1` (Windows, the `hf` CLI) fetch
them into `weights/<model>/` as a diffusers-format directory (component
subfolders + `model_index.json`), loadable straight from `--model`:

```bash
scripts/download-weights.sh sd15                # SD1.5 components
scripts/download-weights.sh lcm-dreamshaper     # LCM-distilled Dreamshaper-7
scripts/download-weights.sh clip-vit-l-14       # OpenAI CLIP ViT-L/14
scripts/download-weights.sh flux-schnell        # Flux.1-schnell (sharded; ~34 GB)
scripts/download-weights.sh t5-xxl              # just the T5-XXL text encoder (~9.5 GB)
scripts/download-weights.sh controlnet-canny    # SD1.5 ControlNet (also -depth / -openpose)
scripts/download-weights.sh sana-600m           # Sana 0.6B, 1024px
scripts/download-weights.sh sana-1.6b           # Sana 1.6B, 1024px
scripts/download-weights.sh sana-sprint-0.6b    # Sana-Sprint (2-step, guidance-distilled)
scripts/download-weights.sh sana-sprint-1.6b
scripts/download-weights.sh pixart-sigma        # PixArt-Sigma-XL-2-1024-MS
scripts/download-weights.sh krea-2-raw          # Krea 2 (real CFG, 28 steps)
scripts/download-weights.sh krea-2-turbo        # Krea 2 Turbo (distilled, no CFG, 8 steps)
```

```powershell
pwsh scripts/download-weights.ps1 -Model sd15
pwsh scripts/download-weights.ps1 -Model lcm-dreamshaper
pwsh scripts/download-weights.ps1 -Model clip-vit-l-14
```

For rate-limited repos, export `HF_TOKEN=hf_...` before running the `.sh`.
The `weights/` directory is gitignored.

The library-only modules have their own scripts. TripoSplat (the image-encoder
half is fetched by brovisionml):

```bash
scripts/download-triposplat.sh all        # vae + dit + decoder (~1.6 GB)
scripts/download-triposplat.sh vae        # Flux.2 VAE only         (~336 MB)
scripts/download-triposplat.sh dit        # flow-matching DiT       (~741 MB)
scripts/download-triposplat.sh decoder    # OctreeGaussian decoder  (~576 MB)
```

ARDY, into `weights/ardy-g152/` (FSQ tokenizer + denoiser + normalisation
stats, ~774 MB). The text frontend additionally needs an LLM2Vec encoder,
loaded through brolm:

```bash
scripts/download-ardy.sh [--out-dir DIR] [--force]
```

terrain-diffusion, into `weights/terrain-diffusion-30m/` (~1.14 GB fp32).
30 m/cell is the one to use for playable worlds; `--90m` is more expansive:

```bash
scripts/download-terrain-diffusion.sh [--90m] [--out-dir DIR]
scripts/convert-terrain-diffusion.py [--src DIR] [--dst DIR]
```

`--src` defaults to `weights/terrain-diffusion-30m` and `--dst` to
`<src>-bro`, so with no arguments the two commands chain.

The conversion is required, not a repack: EDM2's `MPConv` force-normalizes
every weight at inference time, so the converter pre-folds that normalisation
and gain into flat `{coarse,base,decoder}.safetensors` alongside a resolved
`config.json`. It also needs `synthetic_map_stats.json` — the quantile tables
that match the synthetic climate fields onto Earth's real marginals, built
once offline by `scripts/build-terrain-synthetic-stats.py` (upstream derives
them at runtime from a WorldClim raster download).

## CLI

The simplest invocation points `--model` at a downloaded diffusers model
directory; the loader detects which family it is and loads every component:

```bash
brodiffusion --version

brodiffusion txt2img --model <dir> --prompt "a cat astronaut" --out cat.png \
                     [--negative <text>] [--steps N] [--cfg F] \
                     [--width N] [--height N] [--seed N]
```

Each family carries its own reference defaults for resolution / steps /
guidance (e.g. Sana and PixArt-Sigma default to native 1024px; Krea 2 Turbo
defaults to 8 steps with no CFG) — pass `--steps` / `--cfg` / `--width` /
`--height` to override.

SD1.5 also accepts explicit per-component files instead of `--model`:

```bash
brodiffusion txt2img --text  <text_encoder.safetensors> \
                     --unet  <unet.safetensors> \
                     --vae   <vae.safetensors> \
                     --vocab <vocab.json> --merges <merges.txt> \
                     --prompt "a cat astronaut" --out cat.png \
                     [--negative <text>] [--steps N] [--cfg F] \
                     [--width N] [--height N] [--seed N] \
                     [--scheduler ddim|lcm] [--noise internal|torch] \
                     [--latent-in <f32>] [--latent-out <f32>] \
                     [--lora <path>[:<scale>]]... [--lcm-lora <path>] \
                     [--quantize-unet] \
                     [--control <weights> --control-image <png> \
                      [--control-scale F] [--control-window S:E]]...

brodiffusion img2img  --init <png> [--strength F] [--vae-sample] ...   # SD1.5 only
brodiffusion inpaint  --init <png> --mask <png> [--strength F] ...     # SD1.5 only
brodiffusion make-mask --out <png> [--width N] [--height N]            # center-square mask
brodiffusion t5       --weights <st> --tokenizer <json> --prompt <text> \
                      [--max-length N] [--quantize]                    # T5-XXL encoder check

brodiffusion bench    --text <st> --unet <st> --vae <st> \
                      --vocab <vocab.json> --merges <merges.txt> \
                      [--steps N] [--iters N] [--warmup N] \
                      [--scheduler ddim|lcm] [--lora <path>[:<scale>]]...
```

Notable flags:

- `--lora <path>[:<scale>]` merges a LoRA into the loaded weights before
  generation (repeatable, scale defaults to 1.0, may be negative). Both
  kohya-ss/A1111 and diffusers/PEFT key conventions are auto-detected.
  `--lcm-lora <path>` is sugar for running an LCM-LoRA on a vanilla SD1.5 U-Net.
- `--control <weights> --control-image <png>` registers a ControlNet
  (SD1.5 only, stackable); `--control-window S:E` restricts one to a
  fraction of the schedule (default `0:1`).
- `img2img` / `inpaint` re-use the txt2img flags; `--init` encodes a source
  image with the VAE encoder and noises it per `--strength`, `inpaint` adds a
  binary `--mask` (white = inpaint, black = keep).
- `--noise torch` reproduces a PyTorch reference run's starting latent for a
  given `--seed`; `--latent-in` / `--latent-out` load / dump the raw FP32
  latent for cross-implementation diffing.

`--out` writes an RGB **PNG** via broimage. The library itself stays
codec-agnostic: `Pipeline::generate()` returns an RGB host buffer
(`3 * H * W` FP32, NCHW, in `[-1, 1]`) and the consumer encodes as it sees fit.

## Tests

```bash
cmake -B build
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

## License

[MIT](LICENSE)
