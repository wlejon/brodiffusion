# brodiffusion

Diffusion-model inference for the bro stack. Pure C++20, built on
[brotensor](https://github.com/wlejon/brotensor) (tensor + compute kernels) and [bromath](https://github.com/wlejon/bromath)
(scalar / RNG / color helpers). Runs **CPU-by-default and on a GPU when one is
available** — FP32 on the CPU backend, FP16 on a GPU — with the device chosen
at runtime.

Target model: **Stable Diffusion 1.5** text-to-image. CLIP text encoder,
safetensors weight loading, DDIM / LCM schedulers, classifier-free guidance,
LoRA merging, and VAE decode.

QuickJS / JS bindings live in [bro](https://github.com/wlejon/bro) once the library is integrated.
This repo stays a pure C++ library + CLI.

## Status

Functional SD1.5 text-to-image inference on the CPU (FP32), CUDA (FP16), and
Metal (FP16) backends. The library builds CPU-only by default; a 12-executable
test suite runs under `ctest`.

| Header | Purpose |
|---|---|
| `brodiffusion/safetensors.h` | mmap'd safetensors reader + writer; FP16/FP32 tensor views |
| `brodiffusion/tokenizer.h` | CLIP BPE tokenizer (`vocab.json` + `merges.txt`) |
| `brodiffusion/clip.h` | CLIP ViT-L/14 text encoder (SD1.5 conditioning) |
| `brodiffusion/clip_image.h` | CLIP ViT-L/14 vision encoder |
| `brodiffusion/clip_score.h` | CLIP image/text similarity scoring |
| `brodiffusion/unet.h` | SD1.5 U-Net — ResBlock + cross-attention + down/up; optional INT8 weights |
| `brodiffusion/vae.h` | VAE decoder (latents → RGB) |
| `brodiffusion/scheduler.h` | DDIM sampler (eta = 0, deterministic) |
| `brodiffusion/lcm_scheduler.h` | LCM (latent-consistency) sampler for distilled checkpoints |
| `brodiffusion/lora.h` | LoRA merging — kohya-ss/A1111 and diffusers/PEFT key conventions |
| `brodiffusion/pipeline.h` | high-level txt2img pipeline + step-wise (`prime`/`step_once`/`decode`) API |
| `brodiffusion/fused_resblock.h`, `fused_transformer.h` | SD1.5-tuned fused kernels (CUDA + Metal, with CPU fallbacks) |
| `brodiffusion/optim.h` | mixed-precision Adam (FP16 weights, FP32 master copy) for fine-tuning |

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
└── brodiffusion/     # this repo
```

CMake first looks for siblings at `../bromath` and `../brotensor`; if not
found, it falls back to `third_party/bromath` and `third_party/brotensor`
submodules. Override paths with `-DBROMATH_DIR=...` / `-DBROTENSOR_DIR=...`.

## Weights

Model weights are not bundled. `scripts/download-weights.ps1` fetches them via
the Hugging Face CLI (`hf`) into `weights/<model>/`:

```powershell
pwsh scripts/download-weights.ps1 -Model sd15             # SD1.5 components
pwsh scripts/download-weights.ps1 -Model lcm-dreamshaper  # LCM-distilled Dreamshaper-7
pwsh scripts/download-weights.ps1 -Model clip-vit-l-14    # OpenAI CLIP ViT-L/14
```

Each model downloads diffusers-format component files — `text_encoder/`,
`unet/`, `vae/`, and the `tokenizer/` `vocab.json` + `merges.txt`. The
`weights/` directory is gitignored.

## CLI

```bash
brodiffusion --version

brodiffusion txt2img --text  <text_encoder.safetensors> \
                     --unet  <unet.safetensors> \
                     --vae   <vae.safetensors> \
                     --vocab <vocab.json> --merges <merges.txt> \
                     --prompt "a cat astronaut" --out cat.ppm \
                     [--negative <text>] [--steps N] [--cfg F] \
                     [--width N] [--height N] [--seed N] \
                     [--scheduler ddim|lcm] \
                     [--lora <path>[:<scale>]]... [--lcm-lora <path>] \
                     [--quantize-unet]

brodiffusion bench   --text <st> --unet <st> --vae <st> \
                     --vocab <vocab.json> --merges <merges.txt> \
                     [--steps N] [--iters N] [--warmup N] [--scheduler ddim|lcm]
```

- `--scheduler lcm` selects the LCM (Latent Consistency Model) scheduler;
  requires an LCM-distilled U-Net checkpoint (e.g. `SimianLuo/LCM_Dreamshaper_v7`).
  Default `--steps` becomes 4 and the unconditional pass is skipped.
- `--lora <path>[:<scale>]` merges a LoRA file into the loaded weights before
  generation. Repeatable; scale defaults to 1.0 and may be negative. Both
  kohya-ss/A1111 and diffusers/PEFT key conventions are auto-detected.
- `--lcm-lora <path>` is sugar for running an LCM-LoRA on a vanilla SD1.5 U-Net.
- `--quantize-unet` quantizes U-Net weights to INT8 (W8A16); GPU-only.

`--out` writes an uncompressed PPM (`P6`) — a development convenience for
sanity-checking generation. The library itself does not encode images:
`Pipeline::generate()` returns an RGB host buffer (`3 * H * W` FP32 values,
NCHW, in `[-1, 1]`) and the consumer encodes as it sees fit.

## Tests

```bash
cmake -B build
cmake --build build --config Release
cd build && ctest -C Release --output-on-failure
```

