#!/usr/bin/env python3
"""PyTorch reference for the terrain-diffusion samplers.

Drives the REAL upstream scheduler + REAL UNet through a complete denoise for
each stage, and dumps inputs + final sample as raw LE FP32 to .parity/ for the
C++ port to reproduce via `brodiffusion terrain-sample`.

Three different samplers, one per stage:

  coarse   EDM + DPM-Solver++ 2nd-order multistep, 20 steps.
           Karras sigmas (rho=7, sigma_min=0.002, sigma_max=80), sigma_data=0.5.
           final_sigmas_type="zero" forces the LAST step to first order, which
           is also what keeps lambda=-log(sigma) finite at sigma=0.
           NOTE the model is fed atan(sigma/sigma_data), NOT the scheduler's own
           timestep 0.25*log(sigma) — that timestep only drives step_index.

  base     TrigFlow consistency, 2 NFE (t_init, then atan(0.35/0.5)).
  decoder  TrigFlow consistency, 1 NFE (t_init only).

           TrigFlow step:  z    = noise * sigma_data
                           x_t  = cos(t)*sample + sin(t)*z
                           pred = -model(x_t / sigma_data, noise_labels=t, cond)
                           sample = cos(t)*x_t - sin(t)*sigma_data*pred
           Note the leading minus on the model output; it is not a sign slip.

Every stage divides the final sample by sigma_data.

Env vars:
  TERRAIN_STAGE   coarse | base | decoder   (default: decoder)
  TERRAIN_CKPT    upstream checkpoint dir   (default: weights/terrain-diffusion-30m)
  TERRAIN_SIZE    spatial size              (default: 64; coarse clamps to 16)
  TERRAIN_TD_ROOT terrain-diffusion source  (default: ../terrain-diffusion)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

SUBDIR = {"coarse": "coarse_model", "base": "base_model", "decoder": "decoder_model"}
SIGMA_DATA = 0.5
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0


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
    sys.path.insert(0, str(td_root))
    from terrain_diffusion.models.edm_unet import EDMUnet2D                    # noqa: E402
    from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler  # noqa: E402

    sub = SUBDIR[stage]
    cfg = {k: v for k, v in json.loads((ckpt / sub / "config.json").read_text()).items()
           if not k.startswith("_")}
    sd = load_file(str(ckpt / sub / "diffusion_pytorch_model.safetensors"))
    if len(cfg.get("model_channel_mults") or [1, 2, 3, 4]) == 1:
        size = min(size, 16)

    model = EDMUnet2D.from_config(cfg)
    _missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise SystemExit(f"{stage}: unexpected keys: {unexpected}")
    model.eval()

    sched = EDMDPMSolverMultistepScheduler(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
                                           sigma_data=SIGMA_DATA)
    outdir = Path(".parity")
    outdir.mkdir(exist_ok=True)
    g = torch.Generator().manual_seed(4242)
    meta = {"stage": stage, "size": size, "sigma_data": SIGMA_DATA}

    if stage == "coarse":
        sched.set_timesteps(20)
        # 6 sample channels + 5 conditioning channels = in_channels 11.
        noise = torch.randn(1, 6, size, size, generator=g)
        cond_img = torch.randn(1, 5, size, size, generator=g)
        cond_inputs = [torch.randn(1, generator=g) for _ in range(5)]
        sample = noise * sched.sigmas[0]

        with torch.no_grad():
            for t, sigma in zip(sched.timesteps, sched.sigmas):
                scaled_in = sched.precondition_inputs(sample, sigma)
                cnoise = sched.trigflow_precondition_noise(sigma.view(-1))
                x_in = torch.cat([scaled_in, cond_img], dim=1)
                out = model(x_in, noise_labels=torch.tensor([cnoise.item()]),
                            conditional_inputs=cond_inputs)
                sample = sched.step(out, t, sample).prev_sample
        sample = sample / SIGMA_DATA

        dump(outdir / "terrain_sample_coarse_noise.f32", noise)
        dump(outdir / "terrain_sample_coarse_cond.f32", cond_img)
        dump(outdir / "terrain_sample_coarse_condin.f32", torch.cat([c.reshape(-1) for c in cond_inputs]))
        meta.update(steps=20, sigma_first=float(sched.sigmas[0]), cond_numel=5)

    else:
        # TrigFlow consistency. t_init = atan(sigma_max/sigma_data); the base
        # model takes one refinement step at atan(0.35/0.5), the decoder none.
        t_init = torch.atan(torch.tensor(SIGMA_MAX / SIGMA_DATA))
        t_list = [t_init] if stage == "decoder" else [t_init, torch.atan(torch.tensor(0.35 / 0.5))]

        n_lat = cfg["in_channels"] - 1 if stage == "decoder" else 0
        sample = torch.zeros(1, cfg["out_channels"], size, size)
        noises = [torch.randn(1, cfg["out_channels"], size, size, generator=g) for _ in t_list]
        extra = torch.randn(1, n_lat, size, size, generator=g) if n_lat else None
        cond = torch.randn(1, 58, generator=g) if stage == "base" else None

        with torch.no_grad():
            for t, noise in zip(t_list, noises):
                tv = t.view(1, 1, 1, 1)
                z = noise * SIGMA_DATA
                x_t = torch.cos(tv) * sample + torch.sin(tv) * z
                model_in = x_t / SIGMA_DATA
                if extra is not None:
                    model_in = torch.cat([model_in, extra], dim=1)
                pred = -model(model_in, noise_labels=torch.tensor([t.item()]),
                              conditional_inputs=([cond] if cond is not None else []))
                sample = torch.cos(tv) * x_t - torch.sin(tv) * SIGMA_DATA * pred
        sample = sample / SIGMA_DATA

        dump(outdir / f"terrain_sample_{stage}_noise.f32", torch.cat(noises, dim=0))
        if extra is not None:
            dump(outdir / f"terrain_sample_{stage}_latents.f32", extra)
        if cond is not None:
            dump(outdir / f"terrain_sample_{stage}_condin.f32", cond.reshape(-1))
        meta.update(steps=len(t_list), t_list=[float(t) for t in t_list],
                    cond_numel=(58 if cond is not None else 0), latent_channels=n_lat)

    dump(outdir / f"terrain_sample_{stage}_ref_out.f32", sample)
    meta["out_shape"] = list(sample.shape)
    (outdir / f"terrain_sample_{stage}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    o = sample.numpy()
    print(f"{stage}: {meta['steps']} step(s) -> out {tuple(sample.shape)}  "
          f"mean {o.mean():+.6f}  std {o.std():.6f}  min {o.min():+.4f}  max {o.max():+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
