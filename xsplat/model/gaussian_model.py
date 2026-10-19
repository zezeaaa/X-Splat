#
# Derived from 3D Gaussian Splatting (Inria / MPII), and from R2-Gaussian.
# Copyright (C) 2023, Inria, GRAPHDECO research group.
# Licensed under the Gaussian-Splatting License; see LICENSE.md.
# For inquiries contact george.drettakis@inria.fr
#
"""The Gaussian model: geometry and density shared by both modalities, RGB appearance on the surface set."""
import os
import pickle

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
from xray_gaussian_rasterization_voxelization import GaussianVoxelizationSettings, GaussianVoxelizer

from xsplat.model.densification import DensificationMixin
from xsplat.utils import RGB2SH, get_expon_lr_func, inverse_sigmoid, inverse_softplus, t2a

LR_GROUPS = {"xyz": "position", "density": "density", "scaling": "scaling", "rotation": "rotation"}  # group -> config prefix


def param(array):
    return nn.Parameter(torch.tensor(array).float().cuda().requires_grad_(True))


class UnifiedGaussianModel(DensificationMixin):
    """Gaussians shared by both modalities: xyz, rotation and scale (log) for all; density for X-ray;
    opacity and SH features for RGB. The surface set, `surface_mask` (defined in Stage 2), says which
    Gaussians carry RGB appearance; only those are rendered in RGB. The mask follows every change of
    the point set."""

    def __init__(self, sh_degree=3, scale_bound=None):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._density = torch.empty(0)
        self._features_dc = self._features_rest = self._opacity = None   # RGB appearance, from Stage 2
        self.surface_mask = None
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.scale_bound = scale_bound   # (scale_min, scale_max); Stage 1 prunes Gaussians above 0.1 x scale_max
        self.unified_optimizer = None
        self.lr_schedules = {}

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.exp(torch.clamp(self._scaling, max=15.0))   # clamped: exp(15) is 3e6

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_density(self):
        return torch.nn.functional.softplus(self._density)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    def query_xray_volume(self, center, nVoxel, sVoxel):
        """Voxelise the X-ray density onto `nVoxel` voxels of total size `sVoxel` around `center`."""
        settings = GaussianVoxelizationSettings(
            scale_modifier=1.0, nVoxel_x=int(nVoxel[0]), nVoxel_y=int(nVoxel[1]), nVoxel_z=int(nVoxel[2]),
            sVoxel_x=float(sVoxel[0]), sVoxel_y=float(sVoxel[1]), sVoxel_z=float(sVoxel[2]),
            center_x=float(center[0]), center_y=float(center[1]), center_z=float(center[2]), prefiltered=False, debug=False)
        vol, _ = GaussianVoxelizer(voxel_settings=settings)(
            means3D=self.get_xyz, opacities=self.get_density, scales=self.get_scaling, rotations=self.get_rotation)
        return vol

    # ---- the point set: initialisation, and the state that follows every change of it ----
    def create_from_pcd(self, xyz, density):
        """Initialise from the FDK point cloud, xyz (N, 3) and density (N, 1); scales from the
        nearest-neighbour distances, identity rotations."""
        xyz = torch.tensor(xyz).float().cuda()
        dist = torch.sqrt(torch.clamp_min(distCUDA2(xyz), 0.001 ** 2))
        rots = torch.zeros((xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._scaling = nn.Parameter(torch.log(dist)[..., None].repeat(1, 3).requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._density = nn.Parameter(inverse_softplus(torch.tensor(density).float().cuda()).requires_grad_(True))
        self.reset_densification_stats()
        print(f"[model] {xyz.shape[0]} Gaussians from the FDK points")

    def reset_densification_stats(self):
        """Zero the view-space gradient accumulators of both modalities, one row per Gaussian."""
        n = self._xyz.shape[0]
        self.xyz_gradient_accum_xray, self.denom_xray = torch.zeros((n, 1), device="cuda"), torch.zeros((n, 1), device="cuda")
        self.xyz_gradient_accum_rgb, self.denom_rgb = torch.zeros((n, 1), device="cuda"), torch.zeros((n, 1), device="cuda")

    def grow_point_state(self, n_new, on_surface):
        """After `n_new` Gaussians were appended: fresh accumulators, the surface mask extended."""
        self.reset_densification_stats()
        if self.surface_mask is not None:
            self.surface_mask = torch.cat([self.surface_mask, torch.full((n_new,), on_surface, dtype=torch.bool, device="cuda")])

    def keep_point_state(self, mask):
        """After Gaussians were pruned to `mask`: the accumulators and the surface mask follow."""
        self.xyz_gradient_accum_xray, self.denom_xray = self.xyz_gradient_accum_xray[mask], self.denom_xray[mask]
        self.xyz_gradient_accum_rgb, self.denom_rgb = self.xyz_gradient_accum_rgb[mask], self.denom_rgb[mask]
        if self.surface_mask is not None:
            self.surface_mask = self.surface_mask[mask]

    def set_surface_set(self, mask):
        """Define the surface set from a (N,) boolean mask; from now on RGB renders those Gaussians only."""
        self.surface_mask = torch.as_tensor(mask, dtype=torch.bool, device=self._xyz.device)
        n, n_surface = len(self.surface_mask), int(self.surface_mask.sum())
        print(f"[surface_set] {n_surface}/{n} Gaussians on the surface ({100.0 * n_surface / n:.1f}%), {n - n_surface} interior")

    def demote_low_opacity_surface(self, threshold):
        """Surface Gaussians whose opacity fell below `threshold` return to the interior."""
        self.surface_mask &= ~(self.get_opacity.squeeze() < threshold)

    def param_stats(self):
        """Parameter counts and storage in MB; opacity and SH exist for the surface set only."""
        n, n_surface = self._xyz.shape[0], int(self.surface_mask.sum())
        floats = {"xyz": 3 * n, "scaling": 3 * n, "rotation": 4 * n, "density": n, "opacity": n_surface,
                  "features_dc": 3 * n_surface, "features_rest": ((self.max_sh_degree + 1) ** 2 - 1) * 3 * n_surface}
        total = sum(floats.values())
        print(f"[param_stats] N={n} N_surface={n_surface} storage={total * 4 / 1e6:.2f} MB")
        return {"N_total": n, "N_surface": n_surface, "sh_degree": int(self.max_sh_degree), "floats": floats,
                "total_floats": total, "storage_MB": round(total * 4 / 1e6, 4)}

    # ---- the optimizer: parameter groups, learning-rate schedules, Adam state across point-set changes ----
    def setup_unified_optimizer(self, args):
        """Adam over the X-ray parameter groups (Stage 1), each learning rate on its exponential schedule."""
        params = {"xyz": self._xyz, "density": self._density, "scaling": self._scaling, "rotation": self._rotation}
        self.lr_schedules = {name: get_expon_lr_func(lr_init=args[f"{key}_lr_init"], lr_final=args[f"{key}_lr_final"],
                                                     max_steps=args[f"{key}_lr_max_steps"]) for name, key in LR_GROUPS.items()}
        self.unified_optimizer = torch.optim.Adam(
            [{"params": [params[name]], "lr": args[f"{key}_lr_init"], "name": name} for name, key in LR_GROUPS.items()],
            lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        for group in self.unified_optimizer.param_groups:
            if group["name"] in self.lr_schedules:
                group["lr"] = self.lr_schedules[group["name"]](iteration)

    def add_rgb_params(self, args, init_rgb, init_opacity):
        """Stage 2: SH features (DC from `init_rgb`, (N, 3) in [0, 1]) and opacity (from `init_opacity`,
        (N,)) join the optimizer as three more parameter groups."""
        n = self._xyz.shape[0]
        self._features_dc = nn.Parameter(RGB2SH(torch.tensor(init_rgb, dtype=torch.float32, device="cuda")).unsqueeze(1).requires_grad_(True))
        self._features_rest = nn.Parameter(torch.zeros((n, (self.max_sh_degree + 1) ** 2 - 1, 3), device="cuda").requires_grad_(True))
        opacity = torch.tensor(np.asarray(init_opacity, dtype=np.float32).reshape(-1), dtype=torch.float, device="cuda").unsqueeze(1)
        self._opacity = nn.Parameter(inverse_sigmoid(opacity).requires_grad_(True))
        for name, tensor, lr in (("f_dc", self._features_dc, args.feature_lr), ("f_rest", self._features_rest, args.feature_lr / 20.0),
                                 ("opacity", self._opacity, args.opacity_lr)):
            self.unified_optimizer.add_param_group({"params": [tensor], "lr": lr, "name": name})

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def cat_tensors_to_optimizer(self, tensors_dict):
        """Append rows to the per-point parameter groups in `tensors_dict` and their Adam state."""
        optimizable_tensors = {}
        for group in self.unified_optimizer.param_groups:
            name = group["name"]
            if name not in tensors_dict:
                continue
            extension_tensor = tensors_dict[name]
            stored_state = self.unified_optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.unified_optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
            if stored_state is not None:
                self.unified_optimizer.state[group["params"][0]] = stored_state
            optimizable_tensors[name] = group["params"][0]
        return optimizable_tensors

    def prune_optimizer(self, mask):
        """Keep the rows selected by `mask` in every parameter group and its Adam state."""
        optimizable_tensors = {}
        for group in self.unified_optimizer.param_groups:
            stored_state = self.unified_optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.unified_optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            if stored_state is not None:
                self.unified_optimizer.state[group["params"][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # ---- the checkpoint, and the single-modality exports ----
    def save_checkpoint(self, path):
        """The model as a pickle of numpy arrays, what eval.py and render_video.py load."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = {"xyz": t2a(self._xyz), "density": t2a(self._density), "scaling": t2a(self._scaling), "rotation": t2a(self._rotation),
               "scale_bound": self.scale_bound, "surface_mask": None if self.surface_mask is None else t2a(self.surface_mask)}
        if self._features_dc is not None:
            out.update({"features_dc": t2a(self._features_dc), "features_rest": t2a(self._features_rest), "opacity": t2a(self._opacity),
                        "active_sh_degree": self.active_sh_degree, "max_sh_degree": self.max_sh_degree})
        with open(path, "wb") as f:
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._xyz, self._density = param(data["xyz"]), param(data["density"])
        self._scaling, self._rotation = param(data["scaling"]), param(data["rotation"])
        self.scale_bound = data["scale_bound"]
        if "features_dc" in data:
            self._features_dc, self._features_rest, self._opacity = param(data["features_dc"]), param(data["features_rest"]), param(data["opacity"])
            self.active_sh_degree, self.max_sh_degree = data["active_sh_degree"], data["max_sh_degree"]
        self.surface_mask = None if data["surface_mask"] is None else torch.as_tensor(data["surface_mask"], dtype=torch.bool, device="cuda")
        self.reset_densification_stats()

    def export_to_xray_model(self, path):
        """The X-ray parameters as an R2-Gaussian pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # no scale_bound: with one, R2-Gaussian would decode these log-scales with a sigmoid
        out = {"xyz": t2a(self._xyz), "density": t2a(self._density), "scale": t2a(self._scaling), "rotation": t2a(self._rotation),
               "scale_bound": None}
        with open(path, "wb") as f:
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    def export_to_rgb_model(self, path):
        """The surface set, the Gaussians rendered in RGB, as a 3DGS ply for its viewers."""
        if self._features_dc is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = t2a(self._xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        columns = np.concatenate((xyz, np.zeros_like(xyz), f_dc, f_rest, t2a(self._opacity), t2a(self._scaling), t2a(self._rotation)), axis=1)
        columns = columns[t2a(self.surface_mask)]
        names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(f_dc.shape[1])]
                 + [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
        elements = np.empty(columns.shape[0], dtype=[(name, "f4") for name in names])
        elements[:] = list(map(tuple, columns))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)
