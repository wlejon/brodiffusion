#!/usr/bin/env python3
"""Convert an upstream terrain-diffusion checkpoint to brodiffusion's format.

Upstream `xandergos/terrain-diffusion-{30m,90m}` ships a diffusers WorldPipeline:

  config.json                                    WorldPipeline constants
  {coarse,base,decoder}_model/config.json        EDMUnet2D constructor args
  {coarse,base,decoder}_model/diffusion_pytorch_model.safetensors

brodiffusion's loader expects:

  config.json                                    resolved pipeline + model configs
  {coarse,base,decoder}.safetensors              flat name -> PRE-FOLDED tensor


Why this conversion is more than a repack
-----------------------------------------
terrain-diffusion uses the EDM2 (Karras et al. 2024) "magnitude-preserving"
UNet. It has no normalization layers and no biases; instead every MPConv
force-normalizes its own weight at *inference* time and scales by a gain:

    # mp_layers.py, MPConv.forward
    w = normalize(w)                            # w / (eps + RMS(w)), eps=1e-4
    w = w * (gain / np.sqrt(w[0].numel()))      # gain / sqrt(fan_in)
    return F.conv2d(x, w, padding=k//2, groups=groups)

and `normalize()` reduces over *all* dims (dim=None), so the whole thing is a
single global scalar per weight tensor:

    normalize(w) = w / (eps + ||w||_F / sqrt(w.numel()))
                 = w / (eps + RMS(w))

All of it is static at inference. Folding it in at conversion time collapses
every MPConv to a plain `conv2d` with no runtime normalize and no bias:

    w_folded = w * (gain / sqrt(fan_in)) / (1e-4 + RMS(w))

`gain` is 1.0 everywhere except two learned scalars, which fold in the same way
because they only ever scale a layer's output:

    out_conv(x, gain=self.out_gain)                     # EDMUnet2D.forward
    c = self.emb_linear(emb, gain=self.emb_gain) + 1    # UNetBlock.forward

Both are `nn.Parameter(torch.zeros([]))` — note they init to 0, so a freshly
initialized model emits zeros; only trained checkpoints are meaningful here.

What is NOT folded (these are activation-dependent, not weight-static):
  * `normalize(x, dim=1)`  — pixel norm on the encoder skip path
  * `normalize(y, dim=2)`  — per-head pixel norm on attention QKV
  * mp_silu / mp_sum / mp_concat — plain arithmetic with compile-time constants

Buffers that must survive
-------------------------
The coarse model's 5 `['float', 64, 0.2]` conditional inputs each build an
`nn.Sequential(MPFourier(64), MPConv(64, emb, []))`. MPFourier's `freqs` and
`phases` are RANDOM buffers baked at init — they are not recomputable and must
be carried through verbatim. (MPPositionalEmbedding's `freqs`, used for the
noise embedding under fourier_scale='pos', IS deterministic, but we carry it
anyway so the C++ side never has to agree on a formula.)

Dropped: the `logvar_fourier` / `logvar_linear` head is training-only
(`return_logvar=False` at inference).

Usage:
  scripts/convert-terrain-diffusion.py [--src DIR] [--dst DIR] [--dump-keys] [--force]

  --src DIR     upstream directory (default: weights/terrain-diffusion-30m)
  --dst DIR     output directory   (default: <src>-bro)
  --dump-keys   print every (name, shape, dtype) row per model and exit
  --force       overwrite existing outputs
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

EPS = 1e-4  # mp_layers.normalize default
STAGES = (("coarse", "coarse_model"), ("base", "base_model"), ("decoder", "decoder_model"))


# ---------------------------------------------------------------- key classes

# MPConv weights whose gain is a plain 1.0.
_MPCONV_UNIT_GAIN = re.compile(
    r"(?:^noise_linear\.weight$)"
    r"|(?:^out_conv\.weight$)"                      # gain handled separately
    r"|(?:^conditional_layers\.\d+\.weight$)"       # 'tensor' cond -> bare MPConv
    r"|(?:^conditional_layers\.\d+\.1\.weight$)"    # 'float'  cond -> Sequential[1]
    r"|(?:^(?:enc|dec)\.[0-9x]+_conv\.weight$)"     # level-0 stem MPConv
    r"|(?:^(?:enc|dec)\.[0-9x]+_\w+\.(?:conv_res0|conv_res1|conv_skip|attn_qkv|attn_proj)\.weight$)"
)

# MPConv weights scaled by a learned gain scalar living at a sibling key.
_EMB_LINEAR = re.compile(r"^((?:enc|dec)\.[0-9x]+_\w+)\.emb_linear\.weight$")

# Buffers carried verbatim.
_BUFFER = re.compile(
    r"(?:^noise_fourier\.freqs$)"
    r"|(?:^conditional_layers\.\d+\.0\.(?:freqs|phases)$)"
)

# Training-only; dropped.
_DROP = re.compile(r"^logvar_(?:fourier|linear)\.")

# Consumed while folding, never emitted on their own.
_GAIN_SCALAR = re.compile(r"(?:^out_gain$)|(?:^(?:enc|dec)\.[0-9x]+_\w+\.emb_gain$)")


def fold(w: torch.Tensor, gain: float) -> torch.Tensor:
    """Apply MPConv's inference-time weight transform exactly once, statically.

        normalize(w) * (gain / sqrt(fan_in))
      = w / (EPS + RMS(w)) * (gain / sqrt(fan_in))

    fan_in is `w[0].numel()` == prod(shape[1:]) — for a kernel=[] linear that is
    just in_features, for a conv it is (in/groups)*kH*kW.
    """
    w32 = w.detach().to(torch.float32)
    rms = float(torch.linalg.vector_norm(w32) / math.sqrt(w32.numel()))
    fan_in = int(w32[0].numel())
    scale = (gain / math.sqrt(fan_in)) / (EPS + rms)
    return (w32 * scale).contiguous().clone()


def convert_model(sd: dict[str, torch.Tensor], stage: str) -> tuple[dict, dict]:
    """Fold one EDMUnet2D state dict. Returns (tensors, stats)."""
    out: dict[str, torch.Tensor] = {}
    n_folded = n_buffer = n_dropped = 0
    gains: dict[str, float] = {}

    # Pass 1 — harvest the learned gain scalars so pass 2 can fold them in.
    for k, v in sd.items():
        if _GAIN_SCALAR.match(k):
            if v.numel() != 1:
                raise SystemExit(f"{stage}: expected scalar gain at {k}, got shape {tuple(v.shape)}")
            gains[k] = float(v.detach().to(torch.float32).reshape(()))

    # Pass 2 — fold / carry / drop, asserting every key is accounted for.
    for k, v in sd.items():
        if _DROP.match(k):
            n_dropped += 1
            continue
        if _GAIN_SCALAR.match(k):
            continue  # consumed above
        if _BUFFER.match(k):
            out[k] = v.detach().to(torch.float32).contiguous().clone()
            n_buffer += 1
            continue

        m = _EMB_LINEAR.match(k)
        if m:
            gk = f"{m.group(1)}.emb_gain"
            if gk not in gains:
                raise SystemExit(f"{stage}: {k} has no matching {gk}")
            out[k] = fold(v, gains[gk])
            n_folded += 1
            continue

        if _MPCONV_UNIT_GAIN.match(k):
            gain = gains["out_gain"] if k == "out_conv.weight" else 1.0
            if k == "out_conv.weight" and "out_gain" not in gains:
                raise SystemExit(f"{stage}: out_conv.weight has no out_gain "
                                 "(disable_out_gain checkpoints are unsupported)")
            out[k] = fold(v, gain)
            n_folded += 1
            continue

        raise SystemExit(
            f"{stage}: unclassified key {k!r} (shape {tuple(v.shape)}).\n"
            "Refusing to guess — a silently dropped or unfolded weight produces a\n"
            "model that runs and returns plausible garbage. Classify it explicitly."
        )

    stats = {"folded": n_folded, "buffers": n_buffer, "dropped": n_dropped,
             "gains": len(gains), "emitted": len(out)}
    return out, stats


def resolve_model_config(cfg: dict) -> dict:
    """Resolve the EDMUnet2D constructor's None-defaults so the C++ side reads
    concrete numbers instead of reimplementing the defaulting rules."""
    mults = cfg.get("model_channel_mults") or [1, 2, 3, 4]
    ch = cfg["model_channels"]
    emb = cfg.get("emb_channels") or ch * max(mults)
    noise_dims = ch if cfg.get("noise_emb_dims") is None else cfg["noise_emb_dims"]
    lpb = cfg.get("layers_per_block", 2)
    if isinstance(lpb, int):
        lpb = [lpb] * len(mults)
    cond = cfg.get("conditional_inputs") or []
    if noise_dims == 0 and not cond:
        emb = 0

    bk = cfg.get("block_kwargs") or {}
    image_size = cfg["image_size"]
    attn_res = cfg.get("attn_resolutions") or []
    levels = [{"level": i,
               "res": image_size // (2 ** i),
               "channels": ch * m,
               "layers": lpb[i],
               "attention": (image_size // (2 ** i)) in attn_res}
              for i, m in enumerate(mults)]

    return {
        "image_size": image_size,
        "in_channels": cfg["in_channels"],
        "out_channels": cfg.get("out_channels") or cfg["in_channels"],
        "model_channels": ch,
        "model_channel_mults": mults,
        "layers_per_block": lpb,
        "emb_channels": emb,
        "noise_emb_dims": noise_dims,
        "attn_resolutions": attn_res,
        "midblock_attention": cfg.get("midblock_attention", True),
        "concat_balance": cfg.get("concat_balance", 0.3),
        "conditional_inputs": cond,
        "conditional_weights": ([1.0] if noise_dims > 0 else []) + [c[2] for c in cond],
        "fourier_scale": cfg.get("fourier_scale", 1),
        "levels": levels,
        # UNetBlock defaults — block_kwargs is null/{} in every shipped config,
        # but resolve through it so a future checkpoint that sets them works.
        "res_balance": bk.get("res_balance", 0.3),
        "attn_balance": bk.get("attn_balance", 0.3),
        "clip_act": bk.get("clip_act", 256),
        "channels_per_head": bk.get("channels_per_head", 64),
        "resample_type": bk.get("resample_type", "pooling"),
        "activation": bk.get("activation", "silu"),
        "conv_type": bk.get("conv_type", "default"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="weights/terrain-diffusion-30m", type=Path)
    ap.add_argument("--dst", default=None, type=Path)
    ap.add_argument("--dump-keys", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src: Path = args.src
    dst: Path = args.dst or src.parent / f"{src.name}-bro"
    if not src.is_dir():
        raise SystemExit(f"error: --src {src} does not exist (run download-terrain-diffusion.sh first)")

    if args.dump_keys:
        for stage, sub in STAGES:
            f = src / sub / "diffusion_pytorch_model.safetensors"
            if not f.exists():
                print(f"== {stage}: MISSING {f}")
                continue
            sd = load_file(str(f))
            print(f"== {stage} ({len(sd)} tensors) ==")
            for k in sorted(sd):
                print(f"   {k:70s} {tuple(sd[k].shape)!s:22s} {sd[k].dtype}")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    pipeline_cfg = json.loads((src / "config.json").read_text())
    merged = {
        "_class_name": "TerrainWorldPipeline",
        "_source": "xandergos/terrain-diffusion (MIT) — weights pre-folded, see convert-terrain-diffusion.py",
        "weights_prefolded": True,
        "pipeline": {k: v for k, v in pipeline_cfg.items() if not k.startswith("_")},
        "models": {},
    }

    total_params = 0
    for stage, sub in STAGES:
        wf = src / sub / "diffusion_pytorch_model.safetensors"
        cf = src / sub / "config.json"
        if not wf.exists() or not cf.exists():
            raise SystemExit(f"error: missing {sub}/ in {src}")

        out_path = dst / f"{stage}.safetensors"
        if out_path.exists() and not args.force:
            raise SystemExit(f"error: {out_path} exists (use --force)")

        cfg = json.loads(cf.read_text())
        sd = load_file(str(wf))
        tensors, stats = convert_model(sd, stage)
        params = sum(t.numel() for t in tensors.values())
        total_params += params

        save_file(tensors, str(out_path), metadata={
            "format": "pt",
            "source": f"xandergos/terrain-diffusion {sub}",
            "prefolded": "true",
        })
        merged["models"][stage] = resolve_model_config(cfg)

        print(f"== {stage:8s} {stats['folded']:4d} folded  {stats['buffers']:2d} buffers  "
              f"{stats['dropped']:2d} dropped  {stats['gains']:3d} gains  "
              f"-> {params/1e6:7.2f}M params  {out_path.name}")

    (dst / "config.json").write_text(json.dumps(merged, indent=2) + "\n")
    print(f"\nTotal {total_params/1e6:.2f}M params -> {dst}")
    print(f"Wrote {dst / 'config.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
