# brodiffusion

Diffusion-model inference pipeline for the bro stack. Pure C++20, GPU-only via
brotensor; bromath for scalar/RNG/color utilities. Initial target:
**Stable Diffusion 1.5** text-to-image. T5 + CLIP text encoders, safetensors
weight loading, sampler/scheduler loop, VAE decode.

brotensor owns the GPU kernels. brodiffusion is the scaffolding around them:
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

Inference-only for now. Training would gate on brotensor adding backward
passes for the diffusion-flavored ops (conv2d, GroupNorm, cross-attention,
resample) which currently ship FP16 forward only.

## Build

brotensor requires exactly one GPU backend at configure time.

```bash
# CUDA (NVIDIA, any OS)
cmake -B build -DBROTENSOR_WITH_CUDA=ON
cmake --build build --config Release

# Metal (Apple)
cmake -B build -DBROTENSOR_WITH_METAL=ON
cmake --build build --config Release
```

## Layout

Standalone sibling of [bro](../bro), [bromath](../bromath), and
[brotensor](../brotensor). The build auto-detects:

```
D:/projects/
├── bromath/          # ../bromath  (header-only math)
├── brotensor/        # ../brotensor (GPU ops)
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
- **Training** — inference only for now.
