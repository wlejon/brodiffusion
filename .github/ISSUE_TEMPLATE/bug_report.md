---
name: Bug report
about: A model loads wrongly or its output differs from diffusers, a crash, or a test fails
labels: bug
---

**The model:** the checkpoint (Hugging Face repo id) and the call: the CLI
command line, or the prompt, steps, guidance, size, seed, scheduler, and any
LoRA or ControlNet.

**What the reference produces** (diffusers, or the upstream implementation for
TripoSplat, ARDY and terrain-diffusion, with the same weights and settings;
`--noise torch` makes the starting latent match a PyTorch run):

**What brodiffusion produced instead** (the output, the error, a crash, or the
failing `ctest --output-on-failure` output — paste it):

```
```

**Environment:**
- OS:
- brotensor backend (CPU / CUDA / Metal / Vulkan), GPU and driver version:
- Compiler / toolchain (MSVC / GCC / Clang):
- Called from the CLI, C++ or JavaScript (`bro.diffusion`):
- brodiffusion commit, and sibling commits if built from siblings:
