#
# Derived from 3D Gaussian Splatting (Inria / MPII), and from R2-Gaussian.
# Copyright (C) 2023, Inria, GRAPHDECO research group.
# Licensed under the Gaussian-Splatting License; see LICENSE.md.
# For inquiries contact george.drettakis@inria.fr
#
"""Densification and pruning: from X-ray gradients in Stage 1, from both modalities in Stage 2."""

import torch

from xsplat.utils import build_rotation, inverse_softplus

# optimizer group -> parameter attribute; Stage 1 has the first four groups, Stage 2 all seven
PARAMS = {"xyz": "_xyz", "density": "_density", "scaling": "_scaling", "rotation": "_rotation",
          "f_dc": "_features_dc", "f_rest": "_features_rest", "opacity": "_opacity"}


class DensificationMixin:
    """Gradient statistics, clone/split and pruning, on the parameters and their Adam state."""

    def add_densification_stats_xray(self, render_pkg):
        """Accumulate the view-space gradients of an X-ray render (`render_xray_unified`)."""
        visible = render_pkg["visibility_filter"]
        self.xyz_gradient_accum_xray[visible] += torch.norm(
            render_pkg["viewspace_points"].grad[visible, :2], dim=-1, keepdim=True
        )
        self.denom_xray[visible] += 1

    def add_densification_stats_rgb(self, render_pkg):
        """Accumulate the view-space gradients of an RGB render; `surface_indices` maps the surface set to the full model."""
        viewspace_grad = render_pkg["viewspace_points"].grad.reshape(-1, 2)
        vis_surface = render_pkg["visibility_filter"]  # (N_surface,) bool
        grad_norms = torch.norm(viewspace_grad[vis_surface, :2], dim=-1, keepdim=True)
        full_indices = render_pkg["surface_indices"][vis_surface]
        self.xyz_gradient_accum_rgb[full_indices] += grad_norms
        self.denom_rgb[full_indices] += 1

    def densify_and_prune_xray(self, grad_threshold, density_threshold, scale_threshold):
        """Stage 1: clone/split on the X-ray gradient; prune low density and oversized Gaussians."""
        large_grad = self.mean_grad(self.xyz_gradient_accum_xray, self.denom_xray) >= grad_threshold
        self.clone_and_split(large_grad, scale_threshold, surface=False)
        oversized = self.get_scaling.max(dim=1).values > 0.1 * max(self.scale_bound)
        self.remove_points((self.get_density < density_threshold).squeeze() | oversized)
        torch.cuda.empty_cache()

    def densify_and_prune_rgb_driven(self, grad_threshold_rgb, grad_threshold_xray, min_opacity, min_density,
                                     scale_threshold):
        """Stage 2: clone/split where either modality's gradient passes its threshold (RGB only on
        surface Gaussians); prune where both opacity and density are small."""
        large_grad = (self.mean_grad(self.xyz_gradient_accum_rgb, self.denom_rgb) >= grad_threshold_rgb) & self.surface_mask
        large_grad |= self.mean_grad(self.xyz_gradient_accum_xray, self.denom_xray) >= grad_threshold_xray
        self.clone_and_split(large_grad, scale_threshold, surface=True)
        self.remove_points((self.get_opacity < min_opacity).squeeze() & (self.get_density < min_density).squeeze())
        torch.cuda.empty_cache()

    @staticmethod
    def mean_grad(accum, denom):
        grads = accum / denom
        grads[grads.isnan()] = 0.0
        return grads.squeeze()

    def clone_and_split(self, large_grad, scale_threshold, surface, N=2):
        """Clone the small Gaussians in `large_grad` and split the large ones into N; new points get `surface`."""
        scale = self.get_scaling.max(dim=1).values
        self.clone_small(large_grad & (scale <= scale_threshold), surface)
        # clones never split: the mask grows by False for the points just added
        large_grad = torch.cat([large_grad, torch.zeros(self._xyz.shape[0] - large_grad.shape[0],
                                                       device="cuda", dtype=torch.bool)])
        scale = self.get_scaling.max(dim=1).values
        self.split_large(large_grad & (scale > scale_threshold), surface, N)

    def clone_small(self, selected, surface):
        new = {name: getattr(self, attr)[selected] for name, attr in self.param_names()}
        new["density"] = inverse_softplus(self.get_density[selected] * 0.5)
        self.add_points(new, surface)

    def split_large(self, selected, surface, N):
        stds = self.get_scaling[selected].repeat(N, 1)
        samples = torch.normal(mean=torch.zeros((stds.size(0), 3), device="cuda"), std=stds)
        rots = build_rotation(self._rotation[selected]).repeat(N, 1, 1)
        new = {name: getattr(self, attr)[selected].repeat(N, *[1] * (getattr(self, attr).dim() - 1))
               for name, attr in self.param_names()}
        new["xyz"] = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected].repeat(N, 1)
        new["scaling"] = torch.log(self.get_scaling[selected].repeat(N, 1) / (0.8 * N))
        new["density"] = inverse_softplus(self.get_density[selected].repeat(N, 1) * (1 / N))
        self.add_points(new, surface)
        self.remove_points(torch.cat([selected, torch.zeros(N * int(selected.sum()), device="cuda", dtype=torch.bool)]))

    def param_names(self):
        return [(g["name"], PARAMS[g["name"]]) for g in self.unified_optimizer.param_groups]

    def add_points(self, new, surface):
        """Append the rows of `new` (one tensor per optimizer group) to the parameters and their Adam state."""
        for name, tensor in self.cat_tensors_to_optimizer(new).items():
            setattr(self, PARAMS[name], tensor)
        self.grow_point_state(new["xyz"].shape[0], on_surface=surface)

    def remove_points(self, mask):
        """Drop the Gaussians in `mask` from the parameters and their Adam state."""
        for name, tensor in self.prune_optimizer(~mask).items():
            setattr(self, PARAMS[name], tensor)
        self.keep_point_state(~mask)

    def prune_nonfinite(self):
        """Drop Gaussians with a NaN/inf parameter: a non-finite gradient leaves them so for good, and nothing renders them."""
        bad = torch.zeros(self._xyz.shape[0], dtype=torch.bool, device="cuda")
        for _, attr in self.param_names():
            bad |= ~torch.isfinite(getattr(self, attr).reshape(bad.shape[0], -1)).all(dim=1)
        if bad.any():
            print(f"[prune] {int(bad.sum())} Gaussians whose parameters became non-finite removed (this happens now and then)")
            self.remove_points(bad)
