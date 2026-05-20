# brodiffusion

Diffusion-model inference pipeline for the bro stack. Pure C++20, built on
brotensor; bromath for scalar/RNG/color utilities. Runs CPU-by-default and
on a GPU when one is available — FP32 on the CPU backend, FP16 on a GPU —
with the device chosen at runtime. Initial target: **Stable Diffusion 1.5**
text-to-image. T5 + CLIP text encoders, safetensors weight loading,
sampler/scheduler loop, VAE decode.

brotensor owns the compute kernels. brodiffusion is the scaffolding around them:
graph wiring (U-Net / VAE / text encoders), weight loading, tokenizers,
schedulers, and conditioning. Output stops at RGB pixels in host memory —
encoding (PNG/JPEG) is the consumer's job (bro's image-api on integration).

QuickJS bindings will live in [bro](../bro) once the library is ready to
integrate. This repo stays a pure C++ library + CLI.

## Status

Skeleton only. No model code yet. CMake build wires up bromath + brotensor and
produces an empty `brodiffusion` static library plus a stub `brodiffusion` CLI.

Planned modules (not yet present):

| Header (planned) | Purpose |
|---|---|
| `brodiffusion/safetensors.h` | mmap'd safetensors reader, FP16/FP32 views over `brotensor::GpuTensor` |
| `brodiffusion/tokenizer.h` | CLIP BPE + T5 SentencePiece tokenizers |
| `brodiffusion/clip.h` | CLIP ViT-L/14 text encoder (SD1.5) |
| `brodiffusion/t5.h` | T5 encoder (later models) |
| `brodiffusion/unet.h` | SD1.5 U-Net (ResBlock + cross-attn + up/down) |
| `brodiffusion/vae.h` | VAE decoder (latents → RGB) |
| `brodiffusion/scheduler.h` | DDIM / Euler / Euler-A samplers |
| `brodiffusion/pipeline.h` | high-level txt2img pipeline |
| `brodiffusion/image_buffer.h` | plain RGB8 host buffer (no encoders) |

First pass is inference-only; training is in scope for later. Backward
passes for the diffusion-flavored ops (conv2d, GroupNorm, cross-attention,
resample) live in brotensor, and a mixed-precision Adam helper for
FP16 weights with an FP32 master copy lives in `brodiffusion/optim.h`.

## Build

brotensor always builds its CPU backend; CUDA and Metal are additive and
mutually exclusive. brodiffusion forwards the choice:

```bash
# CPU-only — the default. Runs the full inference path in FP32.
cmake -B build
cmake --build build --config Release

# CUDA (NVIDIA, any OS) — adds the FP16 GPU path.
cmake -B build -DBROTENSOR_WITH_CUDA=ON
cmake --build build --config Release

# Metal (Apple)
cmake -B build -DBROTENSOR_WITH_METAL=ON
cmake --build build --config Release
```

brodiffusion runs the full diffusion pipeline on whichever backend brotensor
resolves at runtime: a CPU-only build generates images in FP32, and a CUDA
build uses the FP16 GPU kernels when a device is present. brodiffusion's
fused ops dispatch the same way — GPU kernel when the input is device-
resident, FP32 fallback on CPU. INT8 (W8A16) weight quantization
(`--quantize-unet`) remains GPU-only; it is ignored on the CPU backend.

## Layout

Standalone sibling of [bro](../bro), [bromath](../bromath), and
[brotensor](../brotensor). The build auto-detects:

```
D:/projects/
├── bromath/          # ../bromath  (header-only math)
├── brotensor/        # ../brotensor (tensor ops, CPU + GPU)
└── brodiffusion/     # this repo
```

CMake first looks for siblings at `../bromath` and `../brotensor`; if not
found, falls back to `third_party/bromath` and `third_party/brotensor`
submodules (not yet wired). Override paths with `-DBROMATH_DIR=...` /
`-DBROTENSOR_DIR=...`. See `bro/docs/multi-repo-workflow.md` for the broader
picture.

## CLI

```bash
./build/Release/brodiffusion --version
./build/Release/brodiffusion txt2img --model sd15.safetensors \
    --prompt "a cat astronaut" --steps 30 --cfg 7.5 --out cat.ppm
```

`txt2img` currently exits with "not implemented yet". `--out` writes
uncompressed PPM (`P6`) as a development convenience — the library itself
does not encode images. Consumers (bro via image-api, or anything else) take
the RGB8 host buffer returned by the pipeline and encode as they see fit.

## Out of scope

- **GPU kernels** — live in brotensor.
- **Geometric / scalar math** — lives in bromath.
- **Image encoding (PNG/JPEG)** — consumer's responsibility; bro's image-api
  handles it on integration. The library returns RGB8 host buffers.
- **QuickJS / JS bindings** — live in bro.
