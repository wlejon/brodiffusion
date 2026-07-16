#!/usr/bin/env python3
"""ARDY motion-rep parity reference.

Builds the real G1 skeleton (ardy/assets/skeletons/g1skel34) + ArdyMotionRep at
25 fps, generates a deterministic set of valid local joint-rotation matrices and
root positions, runs the codec forward (features) and inverse (posed joints +
recovered local rotations), and dumps everything as raw float64 to .parity/ so
the C++ `brodiffusion ardy-motionrep-fwd` subcommand can be diffed against it.

No trained weights — this is the deterministic geometric codec that sits under
the ARDY denoiser. Exact double-precision parity is expected.

Env: ARDY_REPO overrides the ardy checkout path (default ../ardy).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
ARDY = Path(os.environ.get("ARDY_REPO", REPO.parent / "ardy"))
sys.path.insert(0, str(ARDY))

from ardy.geometry import axis_angle_to_matrix  # noqa: E402
from ardy.motion_rep.reps.ardy_motionrep import ArdyMotionRep  # noqa: E402
from ardy.skeleton import G1Skeleton34  # noqa: E402

PARITY = REPO / ".parity"
PARITY.mkdir(exist_ok=True)

T = 6           # frames (>= 2 for velocities)
FPS = 25.0
torch.manual_seed(1234)

skel = G1Skeleton34()  # default folder -> ardy asset g1skel34/joints.p
J = skel.nbjoints
assert J == 34, J

rep = ArdyMotionRep(skeleton=skel, fps=FPS)  # stats_path=None -> no normalization
assert rep.motion_rep_dim == 414, rep.motion_rep_dim

# Deterministic valid rotations from random axis-angles; a moving root.
axis_angle = torch.randn(T, J, 3, dtype=torch.float64) * 0.6
local_rot_mats = axis_angle_to_matrix(axis_angle)              # (T, J, 3, 3)
root_positions = torch.zeros(T, 3, dtype=torch.float64)
root_positions[:, 0] = torch.linspace(0.0, 0.5, T, dtype=torch.float64)   # x drift
root_positions[:, 1] = 0.90 + 0.02 * torch.arange(T, dtype=torch.float64)  # height
root_positions[:, 2] = torch.linspace(0.0, -0.3, T, dtype=torch.float64)  # z drift

# Forward: local rots + root -> features (unnormalized).
features = rep(local_rot_mats, root_positions, to_normalize=False)   # (1, T, 414)
features = features.squeeze(0)                                       # (T, 414)

# Inverse: features -> posed joints + recovered local rotations.
out = rep.inverse(features, is_normalized=False, posed_joints_from="rotations")
posed = out["posed_joints"].squeeze(0)          # (T, J, 3)
local_rec = out["local_rot_mats"].squeeze(0)    # (T, J, 3, 3)


def dump(name: str, t: torch.Tensor):
    a = t.detach().cpu().numpy().astype("<f8").ravel()
    a.tofile(PARITY / name)
    return a


dump("ardy_mr_local_rots.f64", local_rot_mats)
dump("ardy_mr_root_pos.f64", root_positions)
ref_feats = dump("ardy_mr_ref_features.f64", features)
ref_posed = dump("ardy_mr_ref_posed.f64", posed)
ref_local = dump("ardy_mr_ref_local.f64", local_rec)

print(f"T={T} J={J} feature_dim={rep.motion_rep_dim}")
print(f"features mean {ref_feats.mean():.6f} std {ref_feats.std():.6f}")
print(f"posed    mean {ref_posed.mean():.6f} std {ref_posed.std():.6f}")
print("dumped local_rots/root_pos + ref features/posed/local to .parity/")
