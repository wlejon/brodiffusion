#!/usr/bin/env python3
"""ARDY autoregressive text-to-motion rollout parity reference (hybrid level).

Drives the REAL ardy `Ardy.__call__` (ardy_model.py) text-only, no-history,
no-constraint, no-crop path over NUM_FRAMES = several gen_horizon windows, using
the real g152 `AutoLatentTwostageDenoiser` + real `ArdyMotionRep` (recenter, root
stats). This exercises the multi-window loop: history-conditioned window sampling,
per-window recenter + requantize of the history, and the accumulated global
translation. It captures the world-frame hybrid sequence RIGHT BEFORE FSQ
detokenization (by intercepting HybridMotionConverter.get_explicit_motion_from_
hybrid) and the per-window generation noise (by intercepting torch.randn), and
dumps them for the C++ `ardy-generate` subcommand to match at the hybrid level.

vector_quantize_pytorch is not required: the only autoencoder method the captured
path invokes is `requantize`, a fixed scalar-quantization rounding — replicated
verbatim from FSQVAETransformer.requantize with the REAL post-quantization stats
and the config-confirmed half-width (num_fsq_levels 128, fsq_level_list 64 ->
half = 32). All trained weights and stats are the real g152 checkpoint.

Real trained weights (nvidia/ARDY-G1-RP-25FPS-Horizon52).

Env: ARDY_REPO (default ../ardy); ARDY_CKPT (default weights/ardy-g152).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]
ARDY = Path(os.environ.get("ARDY_REPO", REPO.parent / "ardy"))
CKPT = Path(os.environ.get("ARDY_CKPT", REPO / "weights" / "ardy-g152"))
PARITY = REPO / ".parity"
PARITY.mkdir(exist_ok=True)

sys.path.insert(0, str(ARDY))
import ardy  # noqa: E402  (empty __init__)

# Lightweight ardy.model package (skip the heavy __init__ = hydra/transformers).
_model_pkg = types.ModuleType("ardy.model")
_model_pkg.__path__ = [str(ARDY / "ardy" / "model")]
sys.modules["ardy.model"] = _model_pkg
_loading = types.ModuleType("ardy.model.loading")
_loading.load_checkpoint_state_dict = lambda *a, **k: {}
sys.modules["ardy.model.loading"] = _loading

from ardy.model.ardy_model import Ardy  # noqa: E402
from ardy.model.auto_latent_twostage_denoiser import AutoLatentTwostageDenoiser  # noqa: E402
from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep  # noqa: E402
from ardy.motion_rep.stats import Stats  # noqa: E402
from ardy.skeleton import G1Skeleton34  # noqa: E402

# ── real denoiser (g152) ──
motion_rep = ArdyMotionRep(
    skeleton=G1Skeleton34(),
    fps=25,
    stats_path=str(CKPT / "stats" / "motion"),
)
kwargs = dict(
    latent_dim=1024,
    positional_encoding_mode="learned_prefix_zero_at_first_generation",
    llm_shape=[1, 4096],
    use_text_mask=False,
    ff_size=2048,
    num_layers=8,
    num_heads=8,
    activation="gelu",
    dropout=0.0,
    pe_dropout=0.0,
    norm_first=False,
    input_first_heading_angle=True,
)
denoiser = AutoLatentTwostageDenoiser(
    motion_rep=motion_rep,
    motion_mask_mode="concat",
    num_frames_per_token=4,
    nframe_root_dim=20,
    latent_embedding_dim=128,
    **kwargs,
)
full = load_file(str(CKPT / "denoiser.safetensors"))
sd = {k.replace("denoiser.", "", 1): v for k, v in full.items() if k.startswith("denoiser")}
sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
denoiser.load_state_dict(sd)
denoiser.eval()


# ── minimal autoencoder: only requantize + a couple of attrs are touched by the
#    captured (pre-detokenize) path. requantize is fsq.py's verbatim formula with
#    the real post-quant stats and the config half-width 32. ──
class RequantizeOnlyAutoencoder:
    num_frames_per_token = 4
    encode_with_quantization = True
    encode_with_normalization = True

    def __init__(self, motion_rep, post_quant_stats, half_width):
        self.motion_rep = motion_rep
        self._pq = post_quant_stats
        self._half = half_width

    def requantize(self, token_embeddings):
        x = self._pq.unnormalize(token_embeddings)
        x = torch.round(x.clamp(-1, 1) * self._half) / self._half
        return self._pq.normalize(x)


post_quant_stats = Stats(folder=CKPT / "stats" / "post_quantization", load=True)
autoencoder = RequantizeOnlyAutoencoder(motion_rep, post_quant_stats, half_width=32)

ardy_model = Ardy(
    denoiser=denoiser,
    autoencoder=autoencoder,
    gen_horizon_len=52,
    num_base_steps=10,
    text_encoder=None,
    device="cpu",
    cfg_type="regular",
)

# ── capture per-window noise (torch.randn draws of shape (1,13,148)) and the
#    world-frame hybrid handed to detokenize. ──
NUM_FRAMES = int(os.environ.get("ARDY_GEN_FRAMES", "104"))  # 2 windows
NUM_DENOISING_STEPS = 10
CFG_WEIGHT = 2.5
FIRST_HEADING = 0.6
GEN_TOK = 13
HYB = 148

torch.manual_seed(606606)
text_feat = torch.randn(1, 1, 4096)  # drawn before patching randn (won't be recorded)
text_pad_mask = torch.ones(1, 1, dtype=torch.bool)

window_noise = []
_orig_randn = torch.randn


def _recording_randn(*args, **kwargs):
    out = _orig_randn(*args, **kwargs)
    if tuple(out.shape) == (1, GEN_TOK, HYB):
        window_noise.append(out.detach().clone())
    return out


captured = {}
_orig_gem = ardy_model.hybrid.get_explicit_motion_from_hybrid


def _capturing_gem(hybrid_motion, *a, **k):
    captured["hybrid"] = hybrid_motion.detach().clone()
    # Skip the real FSQ detokenize (vqp-free); return a correctly shaped dummy so
    # __call__ can slice it. We only need the captured pre-detokenize hybrid.
    num_frames = hybrid_motion.shape[1] * RequantizeOnlyAutoencoder.num_frames_per_token
    return torch.zeros(hybrid_motion.shape[0], num_frames, motion_rep.motion_rep_dim)


pad_mask = torch.ones(1, NUM_FRAMES, dtype=torch.bool)
torch.randn = _recording_randn
ardy_model.hybrid.get_explicit_motion_from_hybrid = _capturing_gem
try:
    with torch.no_grad():
        _ = ardy_model(
            texts=[""],
            num_frames=NUM_FRAMES,
            num_denoising_steps=NUM_DENOISING_STEPS,
            pad_mask=pad_mask,
            first_heading_angle=torch.tensor([FIRST_HEADING]),
            motion_mask=None,
            observed_motion=None,
            cfg_weight=CFG_WEIGHT,
            text_feat=text_feat,
            text_pad_mask=text_pad_mask,
            cfg_type="regular",
        )
finally:
    torch.randn = _orig_randn

hybrid_b = captured["hybrid"]                   # (1, T_tok, 148)
hybrid = hybrid_b.squeeze(0)                     # (T_tok, 148)
noise = torch.cat(window_noise, dim=0)          # (num_windows, 13, 148)
num_windows = noise.shape[0]

# ── explicit-motion detokenize (get_explicit_motion_from_hybrid, no crop) via the
#    REAL decoder transformer + REAL motion_rep. vqp-free: FSQVAETransformer.
#    detokenize's requantize is the same scalar round (half 32); the decoder is
#    the real DoubleCondDecoderTransformer loaded from tokenizer.safetensors. ──
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "ardy_dec_transformer",
    ARDY / "ardy" / "model" / "autoencoder" / "transformer.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
decoder = _mod.DoubleCondDecoderTransformer(
    input_dim=128, output_dim=413, num_frames_per_token=4, latent_dim=512,
    num_heads=4, ff_size=1024, dropout=0.0, activation="gelu", norm_first=False,
    num_layers=8, pe_dropout=0.0, is_causal=True, target_cond_dim=409,
    external_cond_dim=4,
)
_tok = load_file(str(CKPT / "tokenizer.safetensors"))
_dpref = "pose_net.decoder."
_dsd = {k[len(_dpref):]: v for k, v in _tok.items() if k.startswith(_dpref)}
_missing, _ = decoder.load_state_dict(_dsd, strict=False)
assert not _missing, f"missing decoder params: {_missing}"
decoder.eval()

with torch.no_grad():
    g_root, latent = ardy_model.hybrid.get_root_and_latent_body_motion_from_hybrid(hybrid_b)
    F = g_root.shape[1]
    local_root = motion_rep.global_root_to_local_root(
        g_root, normalized=True, lengths=torch.tensor([F])
    )
    tq = post_quant_stats.unnormalize(latent)
    tq = torch.round(tq.clamp(-1, 1) * 32) / 32
    decoded = decoder(tq, external_cond=local_root)          # (1, F, 413)
    body = decoded[..., motion_rep.local_root_dim:]          # (1, F, 409)
    motion = motion_rep.concat_root_body(g_root, body)       # (1, F, 414)
motion = motion.squeeze(0)

# Unnormalize + FK to world joint positions, the actual render input. This is
# where the normalized explicit feature (normalized global root + normalized
# decoded body) becomes real-world meters: inverse(is_normalized=True) runs
# stats.unnormalize over the full 414-d feature before unpacking / FK.
with torch.no_grad():
    inv = motion_rep.inverse(
        motion.unsqueeze(0), is_normalized=True, posed_joints_from="rotations"
    )
posed = inv["posed_joints"].squeeze(0)                     # (F, 34, 3)


def dump(name, t):
    a = t.detach().cpu().numpy().astype("<f4").ravel()
    a.tofile(PARITY / name)
    return a


dump("ardy_gen_text.f32", text_feat.squeeze(0).squeeze(0))  # (4096,)
dump("ardy_gen_noise.f32", noise)                           # (W, 13, 148)
ref_out = dump("ardy_gen_ref_hyb.f32", hybrid)              # (T_tok, 148)
ref_motion = dump("ardy_gen_ref_motion.f32", motion)        # (F, 414)
ref_posed = dump("ardy_gen_ref_posed.f32", posed)           # (F, 34, 3)

print(f"frames={NUM_FRAMES} windows={num_windows} T_tok={hybrid.shape[0]} "
      f"steps={NUM_DENOISING_STEPS} cfg={CFG_WEIGHT} heading={FIRST_HEADING}")
print(f"world-frame hybrid mean {ref_out.mean():.6f} std {ref_out.std():.6f}")
print(f"explicit motion ({motion.shape[0]},{motion.shape[1]}) "
      f"mean {ref_motion.mean():.6f} std {ref_motion.std():.6f}")
_p = posed.reshape(-1, 3)
print(f"posed joints ({posed.shape[0]},{posed.shape[1]},3) "
      f"root-y range [{posed[:, 0, 1].min():.3f}, {posed[:, 0, 1].max():.3f}] "
      f"xz-span [{_p[:, 0].min():.3f},{_p[:, 0].max():.3f}]x"
      f"[{_p[:, 2].min():.3f},{_p[:, 2].max():.3f}]")
print("dumped text/noise + ref hybrid + ref motion + ref posed to .parity/")
