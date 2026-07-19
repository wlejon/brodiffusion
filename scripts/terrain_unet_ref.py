#!/usr/bin/env python3
"""PyTorch reference for the terrain-diffusion EDM2 UNet forward pass.

Runs the REAL upstream EDMUnet2D with the REAL downloaded checkpoint and dumps
inputs + output as raw little-endian FP32 to .parity/, for the C++ port to
consume via `brodiffusion terrain-unet-fwd`.

Driven by env vars so the .sh wrapper can loop over stages:
  TERRAIN_STAGE   coarse | base | decoder   (default: decoder)
  TERRAIN_CKPT    upstream checkpoint dir   (default: weights/terrain-diffusion-30m)
  TERRAIN_SIZE    spatial size of the test forward (default: 64; coarse clamps to 16)
  TERRAIN_TD_ROOT terrain-diffusion source  (default: ../terrain-diffusion)

Files written (STAGE = the stage name):
  .parity/terrain_<STAGE>_x.f32       (1, C_in, S, S)   input sample
  .parity/terrain_<STAGE>_noise.f32   (1,)              noise label t
  .parity/terrain_<STAGE>_cond.f32    flattened conditioning, omitted if none
  .parity/terrain_<STAGE>_ref_out.f32 (1, C_out, S, S)  reference output
  .parity/terrain_<STAGE>_meta.json   shapes + config echo
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

SUBDIR = {"coarse": "coarse_model", "base": "base_model", "decoder": "decoder_model"}


def dump(path: Path, t: torch.Tensor) -> None:
    a = t.detach().to(torch.float32).cpu().contiguous().numpy().astype("<f4", copy=False)
    path.write_bytes(a.tobytes())


def main() -> int:
    stage = os.environ.get("TERRAIN_STAGE", "decoder")
    ckpt = Path(os.environ.get("TERRAIN_CKPT", "weights/terrain-diffusion-30m"))
    size = int(os.environ.get("TERRAIN_SIZE", "64"))
    td_root = Path(os.environ.get("TERRAIN_TD_ROOT", "../terrain-diffusion")).resolve()

    if stage not in SUBDIR:
        raise SystemExit(f"error: TERRAIN_STAGE must be one of {sorted(SUBDIR)}, got {stage!r}")
    if not (td_root / "terrain_diffusion").is_dir():
        raise SystemExit(f"error: {td_root} has no terrain_diffusion/ package")

    sys.path.insert(0, str(td_root))
    from terrain_diffusion.models.edm_unet import EDMUnet2D  # noqa: E402

    sub = SUBDIR[stage]
    cfg = {k: v for k, v in json.loads((ckpt / sub / "config.json").read_text()).items()
           if not k.startswith("_")}
    sd = load_file(str(ckpt / sub / "diffusion_pytorch_model.safetensors"))

    # Coarse is a single-level 16x16 net — don't run it wider than it was built for.
    if len(cfg.get("model_channel_mults") or [1, 2, 3, 4]) == 1:
        size = min(size, 16)

    model = EDMUnet2D.from_config(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise SystemExit(f"{stage}: unexpected keys in checkpoint: {unexpected}")
    model.eval()  # critical: train mode mutates weights inside MPConv.forward

    gen = torch.Generator().manual_seed(1234)
    x = torch.randn(1, cfg["in_channels"], size, size, generator=gen)
    noise = torch.full((1,), 1.2)  # t = atan(sigma/sigma_data), on the TrigFlow arc

    conds, cond_flat = [], []
    for kind, dim, _w in (cfg.get("conditional_inputs") or []):
        if kind == "float":
            c = torch.randn(1, generator=gen)
        elif kind == "tensor":
            c = torch.randn(1, dim, generator=gen)
        else:
            raise SystemExit(f"{stage}: unsupported conditional kind {kind!r}")
        conds.append(c)
        cond_flat.append(c.reshape(-1))

    with torch.no_grad():
        out = model(x, noise_labels=noise, conditional_inputs=conds)

    outdir = Path(".parity")
    outdir.mkdir(exist_ok=True)
    dump(outdir / f"terrain_{stage}_x.f32", x)
    dump(outdir / f"terrain_{stage}_noise.f32", noise)
    if cond_flat:
        dump(outdir / f"terrain_{stage}_cond.f32", torch.cat(cond_flat))
    dump(outdir / f"terrain_{stage}_ref_out.f32", out)

    meta = {
        "stage": stage,
        "size": size,
        "in_channels": cfg["in_channels"],
        "out_channels": cfg.get("out_channels") or cfg["in_channels"],
        "x_shape": list(x.shape),
        "out_shape": list(out.shape),
        "cond_numel": int(sum(c.numel() for c in cond_flat)),
        "noise_label": 1.2,
    }
    (outdir / f"terrain_{stage}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    o = out.detach().numpy()
    print(f"{stage}: x {tuple(x.shape)} -> out {tuple(out.shape)}  "
          f"mean {o.mean():+.6f}  std {o.std():.6f}  "
          f"min {o.min():+.4f}  max {o.max():+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
