#
# Derived from 3D Gaussian Splatting (Inria / MPII), and from R2-Gaussian.
# Copyright (C) 2023, Inria, GRAPHDECO research group.
# Licensed under the Gaussian-Splatting License; see LICENSE.md.
# For inquiries contact george.drettakis@inria.fr
#
"""Rendering the unified model: RGB through gsplat, X-ray through the R2-Gaussian rasteriser; and the
virtual cameras of the depth loss."""
import math

import numpy as np
import torch
from gsplat.rendering import rasterization as gsplat_rasterization
from xray_gaussian_rasterization_voxelization import GaussianRasterizationSettings as XrayRasterizationSettings
from xray_gaussian_rasterization_voxelization import GaussianRasterizer as XrayRasterizer

from xsplat.scene import Camera


def render_rgb_unified(viewpoint_camera, unified_model, bg_color, alignment_transform=None, render_depth=False):
    """Render RGB (and depth if asked) from the surface-set Gaussians, with the camera mapped into the X-ray
    frame by `alignment_transform` (RGB -> X-ray; None for a camera already there). `viewspace_points` and
    `visibility_filter` index the surface set; `surface_indices` maps it to the full model."""
    viewmatrix = viewpoint_camera.world_view_transform
    if alignment_transform is not None:
        viewmatrix = torch.inverse(alignment_transform).T @ viewmatrix
    idx = unified_model.surface_mask.nonzero(as_tuple=True)[0]
    render_colors, render_alphas, meta = gsplat_rasterization(
        unified_model.get_xyz[idx],
        unified_model.get_rotation[idx],
        unified_model.get_scaling[idx],
        unified_model.get_opacity.squeeze(-1)[idx],
        unified_model.get_features[idx],
        viewmatrix.T.unsqueeze(0),
        viewpoint_camera.K.unsqueeze(0),
        viewpoint_camera.image_width,
        viewpoint_camera.image_height,
        sh_degree=unified_model.active_sh_degree,
        packed=False,
        render_mode="RGB+ED" if render_depth else "RGB",
        backgrounds=bg_color.unsqueeze(0),
    )
    rendered = render_colors[0].permute(2, 0, 1)  # (3 or 4, H, W): RGB, then the depth
    means2d = meta["means2d"]
    if means2d.requires_grad:
        means2d.retain_grad()
    return {
        "render": rendered[:3],
        "depth": rendered[3:4] if render_depth else None,
        "render_alphas": render_alphas[0, :, :, 0].unsqueeze(0),
        "viewspace_points": means2d,
        "visibility_filter": meta["radii"][0].max(dim=-1).values > 0,  # over the surface set, like viewspace_points
        "surface_indices": idx,
    }


def render_xray_unified(viewpoint_camera, unified_model):
    """Render one X-ray projection of the density field; `viewspace_points` keeps the screen-space gradient."""
    screenspace_points = torch.zeros_like(unified_model.get_xyz, requires_grad=True) + 0
    if screenspace_points.requires_grad:  # not under torch.no_grad()
        screenspace_points.retain_grad()
    rasterizer = XrayRasterizer(raster_settings=XrayRasterizationSettings(
        image_height=viewpoint_camera.image_height,
        image_width=viewpoint_camera.image_width,
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=1,  # cone beam
        debug=False,
    ))
    rendered_image, radii = rasterizer(
        means3D=unified_model.get_xyz,
        means2D=screenspace_points,
        opacities=unified_model.get_density,
        scales=unified_model.get_scaling,
        rotations=unified_model.get_rotation,
    )
    return {"render": rendered_image, "viewspace_points": screenspace_points, "visibility_filter": radii > 0}


def look_at(camera_pos, center, azimuth, world_up):
    """Camera-to-world matrix of a camera at `camera_pos` looking at `center`, with `world_up` up in the
    image (when looking along it, a direction at `azimuth` instead)."""
    forward = center - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    world_up = np.asarray(world_up, dtype=np.float64)
    if abs(np.dot(forward, world_up)) > 0.99:
        ref_dir = np.array([np.cos(azimuth), np.sin(azimuth), 0])
        world_up = np.cross(forward, ref_dir)
        if np.linalg.norm(world_up) < 1e-6:
            world_up = np.array([0, 1, 0])
        else:
            world_up = world_up / (np.linalg.norm(world_up) + 1e-8)
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, forward, camera_pos
    return c2w


def generate_virtual_cameras(vd_cfg, scanner_cfg, alignment_transform):
    """`vd_cfg.n_views` cameras on a Fibonacci sphere around the CT volume, elevation within
    `max_elevation_deg`, looking at its centre with the X-ray detector's field of view at
    `resolution_scale` of its resolution. They are in the RGB frame, where the model renders RGB."""
    center = np.array(scanner_cfg["offOrigin"])
    radius = scanner_cfg["DSO"] * vd_cfg.radius_factor
    max_elev = np.deg2rad(vd_cfg.max_elevation_deg)
    height = max(1, int(scanner_cfg["nDetector"][0] * vd_cfg.resolution_scale))
    width = max(1, int(scanner_cfg["nDetector"][1] * vd_cfg.resolution_scale))
    FovX = float(np.arctan2(scanner_cfg["sDetector"][1] / 2, scanner_cfg["DSD"]) * 2)
    FovY = float(np.arctan2(scanner_cfg["sDetector"][0] / 2, scanner_cfg["DSD"]) * 2)
    K = np.array([[width / (2.0 * np.tan(FovX * 0.5)), 0.0, width * 0.5],
                  [0.0, height / (2.0 * np.tan(FovY * 0.5)), height * 0.5],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    T_inv = torch.inverse(alignment_transform).cpu().numpy()  # X-ray frame -> RGB frame

    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    cameras = []
    for i in range(vd_cfg.n_views):
        elevation = np.pi / 2.0 - np.arccos(1.0 - 2.0 * (i + 0.5) / vd_cfg.n_views)
        if abs(elevation) > max_elev:
            elevation = np.sign(elevation) * max_elev * abs(elevation) / (np.pi / 2.0)
        azimuth = (2.0 * np.pi * i / golden_ratio) % (2.0 * np.pi)
        camera_pos = center + np.array([radius * np.cos(elevation) * np.cos(azimuth),
                                        radius * np.cos(elevation) * np.sin(azimuth),
                                        radius * np.sin(elevation)])
        w2c = np.linalg.inv(T_inv @ look_at(camera_pos, center, azimuth, [0.0, 0.0, 1.0]))
        cameras.append(Camera(w2c[:3, :3].T.astype(np.float32), w2c[:3, 3].astype(np.float32), FovX, FovY,
                              torch.zeros(3, height, width), f"virtual_{i:04d}", K=K))
    print(f"[virtual] {len(cameras)} cameras at radius {radius:.4f}, {width}x{height}")
    return cameras
