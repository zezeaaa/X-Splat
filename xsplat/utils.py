#
# Derived from 3D Gaussian Splatting (Inria / MPII), and from R2-Gaussian.
# Copyright (C) 2023, Inria, GRAPHDECO research group.
# Licensed under the Gaussian-Splatting License; see LICENSE.md.
# For inquiries contact george.drettakis@inria.fr
#
"""Small helpers: seeding, tensors, activations, learning-rate schedules, and reading a run's config snapshot."""
import os
import os.path as osp
import random
import re

import numpy as np
import torch
from omegaconf import OmegaConf


SH_C0 = 0.28209479177387814


def t2a(tensor):
    return tensor.detach().cpu().numpy()


def search_for_max_iteration(folder):
    """Highest N among `iteration_N` entries, with or without a file extension."""
    saved_iters = []
    for fname in os.listdir(folder):
        match = re.search(r"iteration_(\d+)", fname)
        if match:
            saved_iters.append(int(match.group(1)))
    if not saved_iters:
        raise FileNotFoundError(f"No checkpoints found in {folder}")
    return max(saved_iters)


def seed_everything(seed):
    """Seed Python, NumPy and torch and make cuDNN deterministic. The rasterisers' atomicAdd and
    Open3D's sampling still keep runs from being bit-reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def inverse_softplus(x):
    return torch.log(torch.exp(torch.clamp(x, min=1e-6)) - 1)


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def get_expon_lr_func(lr_init, lr_final, max_steps):
    """3DGS's learning-rate schedule: log-linear from `lr_init` to `lr_final` over `max_steps`, then constant."""
    def helper(step):
        if lr_init == 0.0 and lr_final == 0.0:
            return 0.0
        t = np.clip(step / max_steps, 0, 1)
        return np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)

    return helper


def build_rotation(r):
    """Rotation matrices from (N, 4) quaternions (w, x, y, z), normalised first."""
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device, dtype=r.dtype)
    rw = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - rw * z)
    R[:, 0, 2] = 2 * (x * z + rw * y)
    R[:, 1, 0] = 2 * (x * y + rw * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - rw * x)
    R[:, 2, 0] = 2 * (x * z - rw * y)
    R[:, 2, 1] = 2 * (y * z + rw * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def RGB2SH(rgb):
    return (rgb - 0.5) / SH_C0


def load_saved_unified_cfg(scene_dir, scene_root=None):
    """Load the config a run snapshotted; `scene_root` replaces the recorded `data.scene_root`, which must exist."""
    config_path = osp.join(osp.abspath(scene_dir), "config.yaml")
    if not osp.isfile(config_path):
        raise FileNotFoundError(
            f"config.yaml not found in {scene_dir}; is that a training output directory?"
        )
    cfg = OmegaConf.load(config_path)
    cfg.output.scene_dir = osp.dirname(config_path)

    if scene_root is not None:
        cfg.data.scene_root = osp.abspath(scene_root)
    elif not osp.isdir(cfg.data.scene_root):
        raise FileNotFoundError(
            f"The scene package this run trained from is not at {cfg.data.scene_root}.\n"
            f"That path was recorded when the run was produced; if your copy of the scene "
            f"package is elsewhere, pass --scene_root <path>."
        )
    return cfg
