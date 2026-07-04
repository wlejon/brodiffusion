#!/usr/bin/env python
# Single-decode reference for the Qwen-Image VAE (Krea 2's decoder), to diff
# against brodiffusion's vae_qwenimage::Decoder (CLI `krea2-vae-fwd`).
#
# Writes a fixed-seed random latent and the reference RGB decode as raw
# little-endian float32 (NCHW flat) so the C++ harness can read the SAME
# latent and we compare outputs. The comparison target is the decoder's raw
# conv_out output (pre torch.clamp(-1,1)) — brodiffusion's decode() does not
# clamp either (that's left to the caller, matching the SD1.5 vae::Decoder
# convention), so comparing pre-clamp keeps both sides apples-to-apples even
# where the network overshoots [-1,1].
#
# Usage: python scripts/krea2_vae_ref.py weights/krea-2-raw/vae outdir
import os
import sys

import numpy as np
import torch

vae_dir = sys.argv[1] if len(sys.argv) > 1 else "weights/krea-2-raw/vae"
out     = sys.argv[2] if len(sys.argv) > 2 else ".parity"
os.makedirs(out, exist_ok=True)

from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

vae = AutoencoderKLQwenImage.from_pretrained(vae_dir, torch_dtype=torch.float32).to("cuda").eval()

H_lat = W_lat = 8       # -> 64x64 image (f8 spatial downsample)
z_dim = vae.config.z_dim

g = torch.Generator(device="cuda").manual_seed(1234)
latent = torch.randn(1, z_dim, H_lat, W_lat, generator=g, device="cuda", dtype=torch.float32)

# Intermediate taps to bisect a mismatch layer-by-layer if parity is poor.
caps = {}
def mk(name):
    def hook(mod, inp, outp):
        caps[name] = outp.detach().float().cpu().numpy()
    return hook
vae.decoder.conv_in.register_forward_hook(mk("conv_in"))
vae.decoder.mid_block.register_forward_hook(mk("mid_block"))
vae.decoder.up_blocks[0].register_forward_hook(mk("up0"))
vae.decoder.up_blocks[1].register_forward_hook(mk("up1"))
vae.decoder.up_blocks[2].register_forward_hook(mk("up2"))
vae.decoder.conv_out.register_forward_hook(mk("conv_out"))

# Pipeline-side per-channel denormalize (diffusers applies this OUTSIDE the
# VAE module, before calling decode() — see pipeline_qwenimage.py). Krea 2 /
# Qwen-Image latents carry an explicit num_frames=1 axis (NCTHW).
latents_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to("cuda")
latents_std  = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to("cuda")
latent5 = latent.unsqueeze(2)                     # (1,z,1,H,W)
denorm  = latent5 * latents_std + latents_mean

with torch.no_grad():
    vae.decode(denorm, return_dict=False)          # populates caps via hooks

img = caps["conv_out"][:, :, 0]                    # (1,3,H_out,W_out), pre-clamp

for k, a in caps.items():
    a.astype("<f4").tofile(f"{out}/krea2_ref_{k}.f32")
    print(f"  ref_{k} shape={a.shape}")

latent.detach().cpu().numpy().astype("<f4").tofile(f"{out}/krea2_ref_latent.f32")
img.astype("<f4").tofile(f"{out}/krea2_ref_image.f32")
print("wrote krea2_ref_latent/krea2_ref_image to", out,
      "| image stats min=%.4f max=%.4f mean=%.4f std=%.4f" %
      (float(img.min()), float(img.max()), float(img.mean()), float(img.std())))
