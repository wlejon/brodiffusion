# brodiffusion

Diffusion-model inference for the bro stack. Pure C++20, built on
[brotensor](https://github.com/wlejon/brotensor) (tensor + compute kernels),
[bromath](https://github.com/wlejon/bromath) (scalar / RNG / color helpers),
[brolm](https://github.com/wlejon/brolm) (tokenizers + text encoders —
brodiffusion's text frontend), and
[broimage](https://github.com/wlejon/broimage) (image decode/encode + host
preproc, used by img2img / inpaint priming and PNG output). Runs
**CPU-by-default and on a GPU when one is available** — FP32 on the CPU
backend, FP16 on a GPU — with the device chosen at runtime.

## Status

Functional text-to-image inference on the CPU (FP32), CUDA (FP16), and Metal
(FP16) backends. Two image model families:

- **Stable Diffusion 1.5** — U-Net + CLIP text encoder, DDIM / LCM schedulers,
  txt2img / img2img / inpaint, ControlNet, LoRA, and optional INT8 (W8A16)
  U-Net weights.
- **Flux.1** — DiT (`FluxTransformer2DModel`) + CLIP-pooled + T5-XXL text
  encoders, rectified-flow (flow-match Euler) scheduler, txt2img.

A third path, **TripoSplat** (single-image → 3D Gaussian splats, the generative
core of [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat)), is
implemented as a library module — a Flux.2 VAE image encoder, a flow-matching
DiT (`LatentSeqMMFlowModel`), the rectified-flow Euler CFG sampler, and the
`OctreeGaussianDecoder` that turns the clean latent into an explicit 3D Gaussian
cloud (up to 262144 splats). brodiffusion owns this generative core only; the
image encoders (DINOv3 + BiRefNet) live in brovisionml and the renderer /
`GaussianSplatCloud` container lives in bromesh, with `bro` compositing the
three. It is not yet wired into the CLI — drive it through the headers under
`brodiffusion/triposplat/`.

| Header | Purpose |
|---|---|
| `brodiffusion/denoiser.h` | model-agnostic noise / velocity-prediction backbone the pipeline owns (`Denoiser`, `Conditioning`, `AttentionTrace`) |
| `brodiffusion/unet.h` | SD1.5 U-Net — ResBlock + cross-attention + down/up; optional INT8 weights |
| `brodiffusion/dit/flux.h` | Flux.1 DiT denoiser — double/single-stream joint-attention blocks, axial RoPE, velocity prediction |
| `brodiffusion/vae.h` | VAE decoder (latents → RGB) **and** encoder (RGB → latents, for img2img / inpaint) |
| `brodiffusion/controlnet.h` | SD1.5 ControlNet — encoder/mid half + zero-convs producing residuals onto the U-Net skips |
| `brodiffusion/scheduler.h` | DDIM sampler (eta = 0, deterministic) |
| `brodiffusion/lcm_scheduler.h` | LCM (latent-consistency) sampler for distilled checkpoints |
| `brodiffusion/flow_match_scheduler.h` | rectified-flow Euler sampler for Flux / SD3 (continuous sigma schedule, velocity update) |
| `brodiffusion/lora.h` | LoRA merging — kohya-ss/A1111 and diffusers/PEFT key conventions |
| `brodiffusion/model_config.h` | parse a diffusers model directory's JSON configs (`model_index.json` + component configs) into brodiffusion's arch structs; auto-detects SD1.5 vs Flux |
| `brodiffusion/pipeline.h` | high-level pipeline (`generate`, `from_model_dir`) + step-wise (`prime`/`step_once`/`decode`) API + cross-attention trace / logit-bias steering |
| `brodiffusion/fused_resblock.h`, `fused_transformer.h` | SD1.5-tuned fused kernels (CUDA + Metal, with CPU fallbacks) |
| `brodiffusion/optim.h` | mixed-precision Adam (FP16 weights, FP32 master copy) for fine-tuning |
| `brodiffusion/triposplat/vae_encoder.h` | TripoSplat — Flux.2 VAE image encoder (image → 128-d conditioning tokens, `feature2`) |
| `brodiffusion/triposplat/flow_model.h` | TripoSplat — flow-matching DiT (`LatentSeqMMFlowModel`): content-dependent 3D RoPE, QK-RMSNorm, adaLN-single |
| `brodiffusion/triposplat/sampler.h` | TripoSplat — rectified-flow Euler CFG sampler integrating the flow DiT to a clean latent |
| `brodiffusion/triposplat/octree_decoder.h` | TripoSplat — `OctreeGaussianDecoder`: octree structure predictor + elastic Gaussian head (latent → 3D Gaussian cloud) |

## Build

brotensor always builds its CPU backend; CUDA and Metal are additive and
mutually exclusive. brodiffusion forwards the choice:

```bash
# CPU-only — the default. Runs the full inference path in FP32.
cmake -B build
cmake --build build --config Release

# CUDA (NVIDIA, any OS) — adds the FP16 GPU path + SD1.5-tuned fused kernels.
cmake -B build -DBROTENSOR_WITH_CUDA=ON
cmake --build build --config Release

# Metal (Apple)
cmake -B build -DBROTENSOR_WITH_METAL=ON
cmake --build build --config Release
```

brodiffusion runs the full diffusion pipeline on whichever backend brotensor
resolves at runtime: a CPU-only build generates images in FP32, and a CUDA or
Metal build uses the FP16 GPU kernels when a device is present. brodiffusion's
fused ops dispatch the same way — the CUDA or Metal kernel when the input is
device-resident, an FP32 fallback (`src/fused.cpp`) on CPU. INT8
(`--quantize-unet`) weight quantization remains GPU-only; it is ignored on the
CPU backend.

CMake options:

| Option | Default | Effect |
|---|---|---|
| `BROTENSOR_WITH_CUDA` | `OFF` | Build the CUDA GPU backend (FP16 + fused CUDA kernels) |
| `BROTENSOR_WITH_METAL` | `OFF` | Build the Metal GPU backend |
| `BRODIFFUSION_CLI` | on when top-level | Build the `brodiffusion` CLI |
| `BRODIFFUSION_TESTS` | `ON` | Build the test suite (only runs when brodiffusion is the top-level project) |
| `BRODIFFUSION_INSTALL` | `OFF` | Generate `install` / `find_package` targets |


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

CMake first looks for siblings at `../bromath`, `../brotensor`, `../brolm`,
and `../broimage`; if not found, it falls back to the matching `third_party/`
submodules. Override paths with `-DBROMATH_DIR=...` / `-DBROTENSOR_DIR=...` /
`-DBROLM_DIR=...` / `-DBROIMAGE_DIR=...`.

## Weights

Model weights are not bundled. A download script fetches them into
`weights/<model>/` — `scripts/download-weights.sh` on macOS / Linux,
`scripts/download-weights.ps1` on Windows:

```bash
# macOS / Linux — fetches straight from Hugging Face with curl (no hf CLI).
scripts/download-weights.sh sd15                # SD1.5 components
scripts/download-weights.sh lcm-dreamshaper     # LCM-distilled Dreamshaper-7
scripts/download-weights.sh clip-vit-l-14       # OpenAI CLIP ViT-L/14
scripts/download-weights.sh flux-schnell        # Flux.1-schnell (sharded; ~34 GB)
scripts/download-weights.sh t5-xxl              # just the T5-XXL text encoder (~9.5 GB)
scripts/download-weights.sh controlnet-canny    # SD1.5 ControlNet (also -depth / -openpose)
```

```powershell
# Windows — uses the Hugging Face CLI (hf).
pwsh scripts/download-weights.ps1 -Model sd15
pwsh scripts/download-weights.ps1 -Model lcm-dreamshaper
pwsh scripts/download-weights.ps1 -Model clip-vit-l-14
```

TripoSplat's generative-core weights have their own script (the image-encoder
half is fetched by brovisionml):

```bash
# Fetches into weights/triposplat/. Default component is `all`.
scripts/download-triposplat.sh all        # vae + dit + decoder (~1.6 GB)
scripts/download-triposplat.sh vae        # Flux.2 VAE only         (~336 MB)
scripts/download-triposplat.sh dit        # flow-matching DiT       (~741 MB)
scripts/download-triposplat.sh decoder    # OctreeGaussian decoder  (~576 MB)
```

For rate-limited repos, export `HF_TOKEN=hf_...` before running the `.sh`.

SD1.5-family models download diffusers-format component files — `text_encoder/`,
`unet/`, `vae/`, the `tokenizer/` `vocab.json` + `merges.txt`, and each
component's `config.json` plus the root `model_index.json` (so the directory
loads directly via `--model`). Flux additionally fetches the sharded
`transformer/` and T5-XXL `text_encoder_2/` (discovered from each
`*.index.json`). The ControlNet targets fetch the residual network plus the
model card's example control image. The `weights/` directory is gitignored.

## CLI

The simplest invocation points `--model` at a downloaded diffusers model
directory; the loader reads `model_index.json`, detects SD1.5 vs Flux, and
loads every component + tokenizer:

```bash
brodiffusion --version

# Whole model directory (SD1.5 or Flux, auto-detected).
brodiffusion txt2img --model <dir> --prompt "a cat astronaut" --out cat.png \
                     [--negative <text>] [--steps N] [--cfg F] \
                     [--width N] [--height N] [--seed N]

# Explicit per-component SD1.5 files.
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

- `--model <dir>` loads a diffusers model directory (`model_index.json` +
  component subdirs). Detects SD1.5 vs Flux automatically; the explicit
  `--text` / `--unet` / `--vae` / `--vocab` / `--merges` flags are then unused.
  Flux defaults `--steps` to 4 (flux-schnell).
- `--scheduler lcm` selects the LCM (Latent Consistency Model) scheduler;
  requires an LCM-distilled U-Net checkpoint (e.g. `SimianLuo/LCM_Dreamshaper_v7`).
  Default `--steps` becomes 4 and the unconditional pass is skipped.
- `--lora <path>[:<scale>]` merges a LoRA file into the loaded weights before
  generation. Repeatable; scale defaults to 1.0 and may be negative. Both
  kohya-ss/A1111 and diffusers/PEFT key conventions are auto-detected.
- `--lcm-lora <path>` is sugar for running an LCM-LoRA on a vanilla SD1.5 U-Net.
- `--quantize-unet` quantizes U-Net weights to INT8 (W8A16); GPU-only.
- `--control <weights> --control-image <png>` registers a ControlNet (SD1.5
  only). Repeat the group to stack multiple nets — residuals are summed
  position-wise, each weighted by its `--control-scale` (default 1.0).
  `--control-window S:E` restricts a net to a half-open fraction of the
  schedule (default `0:1` = full).
- `img2img` / `inpaint` re-use the txt2img flags; `--init` encodes a source
  image with the VAE encoder and noises it per `--strength`. `inpaint` adds a
  binary `--mask` (white = inpaint, black = keep); `make-mask` writes a
  center-square mask to feed it.
- `--noise torch` makes `--seed` reproduce a PyTorch reference run's starting
  latent; `--latent-in` / `--latent-out` load / dump the raw float32 latent for
  cross-implementation diffing.

`--out` writes an RGB **PNG** via broimage. The library itself stays
codec-agnostic: `Pipeline::generate()` returns an RGB host buffer (`3 * H * W`
FP32 values, NCHW, in `[-1, 1]`) and the consumer encodes as it sees fit.

## Tests

```bash
cmake -B build
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

## License

[MIT](LICENSE)