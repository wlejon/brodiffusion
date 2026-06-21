#!/usr/bin/env python
# Single-forward reference for the PixArt-Sigma transformer, to diff against
# brodiffusion's PixArtDenoiser (cli `pixart-fwd`).
#
# Writes fixed inputs (latent, ctx) and the reference epsilon as raw little-
# endian float32 (NCHW flat for latent/epsilon, L x 4096 for ctx) so the C++
# harness can read the SAME inputs and we compare outputs.
#
# Usage: python scripts/pixart_ref.py weights/pixart-sigma/transformer outdir
import sys, numpy as np, torch

tf_dir = sys.argv[1] if len(sys.argv) > 1 else "weights/pixart-sigma/transformer"
out    = sys.argv[2] if len(sys.argv) > 2 else "/tmp"

try:
    from diffusers import PixArtTransformer2DModel as TF
except Exception:
    from diffusers import Transformer2DModel as TF

m = TF.from_pretrained(tf_dir, torch_dtype=torch.float32).to("cuda").eval()

H = W = 16          # latent grid -> 128px image
L = 4               # caption tokens
g = torch.Generator(device="cuda").manual_seed(1234)
latent = torch.randn(1, 4, H, W, generator=g, device="cuda", dtype=torch.float32)
ctx    = torch.randn(1, L, 4096, generator=g, device="cuda", dtype=torch.float32)
t      = torch.tensor([500.0], device="cuda")

# Capture intermediates: PatchEmbed output (post patch+pos), block 0 out, last
# block out — to localize where brodiffusion diverges.
caps = {}
def mk(name):
    def hook(mod, inp, outp):
        o = outp[0] if isinstance(outp, (tuple, list)) else outp
        caps[name] = o.detach().float().cpu().numpy()
    return hook
m.pos_embed.register_forward_hook(mk("hin"))
m.transformer_blocks[0].register_forward_hook(mk("hb0"))
m.transformer_blocks[-1].register_forward_hook(mk("hblk"))
# Block-0 sublayer outputs + their modulated inputs (capture inputs too).
b0 = m.transformer_blocks[0]
b0.attn1.register_forward_hook(mk("b0self"))
b0.attn2.register_forward_hook(mk("b0cross"))
b0.ff.register_forward_hook(mk("b0ff"))
def mk_in(name):
    def hook(mod, inp, outp):
        caps[name] = inp[0].detach().float().cpu().numpy()
    return hook
b0.attn1.register_forward_hook(mk_in("b0mod_msa"))  # attn1 input = modulated norm1
b0.ff.register_forward_hook(mk_in("b0mod_mlp"))      # ff input = modulated norm2
# adaln_single returns (timestep_6d, embedded_timestep); capture both.
def adaln_hook(mod, inp, outp):
    caps["temb6"] = outp[0].detach().float().cpu().numpy()
    caps["emb"]   = outp[1].detach().float().cpu().numpy()
m.adaln_single.register_forward_hook(adaln_hook)
# time_proj (sinusoidal 256) output.
m.adaln_single.emb.time_proj.register_forward_hook(mk("freq"))

with torch.no_grad():
    eps = m(hidden_states=latent, encoder_hidden_states=ctx, timestep=t,
            added_cond_kwargs={}, return_dict=True).sample   # (1, 8, H, W)
eps = eps[:, :4]                                              # keep epsilon half
for k, a in caps.items():
    a.astype("<f4").tofile(f"{out}/ref_{k}.f32")
    print(f"  ref_{k} shape={a.shape}")

latent.detach().cpu().numpy().astype("<f4").tofile(f"{out}/ref_latent.f32")
ctx.detach().cpu().numpy().astype("<f4").tofile(f"{out}/ref_ctx.f32")
eps.detach().cpu().numpy().astype("<f4").tofile(f"{out}/ref_eps.f32")
print("wrote ref_latent/ref_ctx/ref_eps to", out,
      "| eps stats min=%.4f max=%.4f mean=%.4f std=%.4f" %
      (float(eps.min()), float(eps.max()), float(eps.mean()), float(eps.std())))
