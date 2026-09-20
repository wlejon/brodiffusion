#!/usr/bin/env python
# Reference for the Qwen-Image 2.1 VAE (AutoencoderKLQwenImage21), to diff
# against brodiffusion's vae_qwenimage21 (CLI `qi21-vae-fwd`). Needs diffusers
# from git main (the 2.1 classes are not in any release yet).
#
# Writes, as raw little-endian float32 (NCHW flat):
#   qi21_ref_latent.f32   fixed-seed latent (z_dim, H_lat, W_lat), raw pipeline scale
#   qi21_ref_image.f32    decode of it: (4, 16*H_lat, 16*W_lat), post-clamp RGBA
#   qi21_ref_dec_*.f32    decoder intermediates (bisection taps)
#   qi21_ref_input.f32    synthetic RGBA input image (4, H, W) in [-1, 1]
#   qi21_ref_encoded.f32  its encode: mean sample, (z - mean)/std, (z_dim, H/16, W/16)
#
# Usage: python scripts/qwenimage21_vae_ref.py D:/projects/Qwen-Image-2.1/vae .parity
import os
import sys

import numpy as np
import torch

vae_dir = sys.argv[1] if len(sys.argv) > 1 else "D:/projects/Qwen-Image-2.1/vae"
out = sys.argv[2] if len(sys.argv) > 2 else ".parity"
os.makedirs(out, exist_ok=True)

from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21

vae = AutoencoderKLQwenImage21.from_pretrained(vae_dir, torch_dtype=torch.float32).to("cuda").eval()
z_dim = vae.config.z_dim
mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to("cuda")
std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to("cuda")

caps = {}


def mk(name):
    def hook(mod, inp, outp):
        caps[name] = outp.detach().float().cpu().numpy()
    return hook


vae.decoder.conv_in.register_forward_hook(mk("conv_in"))
vae.decoder.mid_block.register_forward_hook(mk("mid_block"))
for i, ub in enumerate(vae.decoder.up_blocks):
    ub.register_forward_hook(mk(f"up{i}"))
vae.decoder.conv_out.register_forward_hook(mk("conv_out"))

ecaps = {}


def mke(name):
    def hook(mod, inp, outp):
        ecaps[name] = outp.detach().float().cpu().numpy()
    return hook


vae.encoder.conv_in.register_forward_hook(mke("conv_in"))
for i, db in enumerate(vae.encoder.down_blocks):
    db.register_forward_hook(mke(f"down{i}"))
vae.encoder.mid_block.register_forward_hook(mke("mid_block"))
vae.encoder.conv_out.register_forward_hook(mke("conv_out"))
vae.quant_conv.register_forward_hook(mke("quant"))

# ── decode ────────────────────────────────────────────────────────────────
H_lat = W_lat = 8  # -> 128x128 RGBA
g = torch.Generator(device="cuda").manual_seed(1234)
latent = torch.randn(1, z_dim, H_lat, W_lat, generator=g, device="cuda", dtype=torch.float32)
with torch.no_grad():
    denorm = latent.unsqueeze(2) * std + mean
    img = vae.decode(denorm, return_dict=False)[0][:, :, 0]  # (1,4,H,W), clamped

latent.cpu().numpy().astype("<f4").tofile(f"{out}/qi21_ref_latent.f32")
img.cpu().numpy().astype("<f4").tofile(f"{out}/qi21_ref_image.f32")
for k, a in caps.items():
    a.astype("<f4").tofile(f"{out}/qi21_ref_dec_{k}.f32")
    print(f"  dec_{k} shape={a.shape}")
print("decode: image stats min=%.4f max=%.4f mean=%.4f std=%.4f"
      % (float(img.min()), float(img.max()), float(img.mean()), float(img.std())))

# ── encode ────────────────────────────────────────────────────────────────
H = W = 128
yy, xx = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing="ij")
yy = yy.float() / H
xx = xx.float() / W
r = torch.sin(6.0 * xx + 2.0 * yy)
gch = torch.cos(4.0 * yy - 3.0 * xx)
b = torch.sin(9.0 * xx * yy)
a = torch.ones_like(r)  # opaque alpha
x = torch.stack([r, gch, b, a], 0).unsqueeze(0).unsqueeze(2) * 0.9  # (1,4,1,H,W)
x = x.clamp(-1, 1)
with torch.no_grad():
    dist = vae.encode(x, return_dict=False)[0]
    z = dist.mode()  # pipeline sample_mode="argmax"
    enc = (z - mean) / std
x[:, :, 0].cpu().numpy().astype("<f4").tofile(f"{out}/qi21_ref_input.f32")
enc[:, :, 0].cpu().numpy().astype("<f4").tofile(f"{out}/qi21_ref_encoded.f32")
for k, a in ecaps.items():
    a.astype("<f4").tofile(f"{out}/qi21_ref_enc_{k}.f32")
    print(f"  enc_{k} shape={a.shape}")
print("encode: latent stats min=%.4f max=%.4f mean=%.4f std=%.4f"
      % (float(enc.min()), float(enc.max()), float(enc.mean()), float(enc.std())))
print("wrote references to", out)
