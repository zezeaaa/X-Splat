"""Cross-modal alignment: the sim(3) that maps the RGB frame onto the X-ray frame, found by
PCA then ICP between the MVS cloud and the Stage-1 CT surface."""

import itertools
import os

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from scipy import ndimage
from scipy.spatial import cKDTree

from xsplat.geometry.surface import build_open3d_pcd, ct_volume_to_mesh, mesh_to_world, sample_mesh_points
from xsplat.utils import t2a


def apply_sim3(points, T):
    """Apply a 4x4 sim(3) transform to (N,3) points."""
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([points.astype(np.float32), ones], axis=1)
    pts_t = (T.astype(np.float32) @ pts_h.T).T
    return pts_t[:, :3].astype(np.float32)


def extract_xray_surface_points(model, scanner_cfg, iso_level, surface_samples, save_dir):
    """Voxelise the Stage-1 model, extract its outer surface, and sample points on it."""
    center = np.array(scanner_cfg["offOrigin"], dtype=np.float64)
    nVoxel = np.array(scanner_cfg["nVoxel"], dtype=np.int64)
    sVoxel = np.array(scanner_cfg["sVoxel"], dtype=np.float64)

    with torch.no_grad():
        vol = t2a(model.query_xray_volume(center, nVoxel, sVoxel))

    # keep the largest component before closing, so nearby fragments are not bridged onto the object
    labeled, ncomp = ndimage.label(vol > iso_level)
    if ncomp > 0:
        counts = np.bincount(labeled.flat)
        counts[0] = 0
        vol = vol * (labeled == counts.argmax())

    mesh_ct = mesh_to_world(ct_volume_to_mesh(vol, iso_level), scanner_cfg)
    ct_pts = sample_mesh_points(mesh_ct, surface_samples)

    os.makedirs(save_dir, exist_ok=True)
    o3d.io.write_triangle_mesh(os.path.join(save_dir, "xray_mesh.ply"), mesh_ct)
    return ct_pts


def run_stage1_realign(model, scanner_cfg, mvs_points_path, realign_save_dir, mvs_color_points_path, iso_level,
                       surface_samples, voxel_size, max_corr_factor, max_iters):
    """Align the MVS cloud to the Stage-1 CT surface; writes T_realign.npy, xray_mesh.ply and
    mvs_aligned.ply to `realign_save_dir`. Returns the 4x4 transform."""
    os.makedirs(realign_save_dir, exist_ok=True)

    xray_surface_pts = extract_xray_surface_points(
        model, scanner_cfg, iso_level=iso_level, surface_samples=surface_samples, save_dir=realign_save_dir)
    if not os.path.exists(mvs_points_path):
        raise FileNotFoundError(f"MVS point cloud not found: {mvs_points_path}")
    mvs_pts = np.asarray(o3d.io.read_point_cloud(mvs_points_path).points).astype(np.float32)
    T_new = align_pca_icp(mvs_pts, xray_surface_pts, voxel_size, max_corr_factor, max_iters)
    np.save(os.path.join(realign_save_dir, "T_realign.npy"), T_new)

    # the aligned cloud seeds Stage 2's colours; next to xray_mesh.ply it shows the alignment
    color_pcd = o3d.io.read_point_cloud(mvs_color_points_path)
    color_pts = np.asarray(color_pcd.points).astype(np.float32)
    color_colors = np.asarray(color_pcd.colors).astype(np.float32)
    save_ply_color(apply_sim3(color_pts, T_new), color_colors, os.path.join(realign_save_dir, "mvs_aligned.ply"))
    print(f"[realign] Saved mvs_aligned.ply ({color_pts.shape[0]} pts)")
    return T_new


def save_ply_color(points, colors, path):
    """Binary PLY with float32 xyz and uint8 colours from `colors` in [0, 1]."""
    vertex = np.empty(points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points.astype(np.float32).T
    vertex["red"], vertex["green"], vertex["blue"] = (np.clip(colors, 0, 1) * 255.0).astype(np.uint8).T
    PlyData([PlyElement.describe(vertex, "vertex")]).write(path)


def align_pca_icp(mvs_pts, ct_pts, voxel_size, max_corr_factor, max_iters):
    """PCA then ICP; maps `mvs_pts` into the frame of `ct_pts`."""
    T_pca = pca_global_sim3(mvs_pts, ct_pts)

    mvs_vox = np.asarray(build_open3d_pcd(mvs_pts, voxel_size=voxel_size).points)
    ct_vox = np.asarray(build_open3d_pcd(ct_pts, voxel_size=voxel_size).points)
    T_icp = estimate_sim3_icp(apply_sim3(mvs_vox, T_pca), ct_vox, voxel_size, max_corr_factor, max_iters)
    T_final = (T_icp.astype(np.float64) @ T_pca.astype(np.float64)).astype(np.float32)

    print(f"[align] {mvs_pts.shape[0]} MVS points onto {ct_pts.shape[0]} CT surface points: "
          f"Chamfer {chamfer_mean_bidir(apply_sim3(mvs_pts, T_final), ct_pts)[2]:.6f}")
    return T_final


# ---- PCA-based global sim(3) alignment. ----
def pca_axes(points):
    """Principal axes as columns, largest variance first, right-handed, each pointing towards the
    heavier tail (positive third moment)."""
    centred = points - points.mean(axis=0)
    evals, evecs = np.linalg.eigh((centred.T @ centred) / points.shape[0])
    R = evecs[:, np.argsort(evals)[::-1]]
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1.0
    for k in range(3):
        if np.mean((centred @ R[:, k]) ** 3) < -1e-8:
            R[:, k] *= -1.0
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1.0
    return R


def uniform_subsample(pts, target_n):
    """About `target_n` points, one per voxel of a grid sized to give that many."""
    if pts.shape[0] <= target_n:
        return pts.copy()
    mn = pts.min(axis=0)
    extent = pts.max(axis=0) - mn + 1e-12
    voxel_size = (float(np.prod(extent)) / target_n) ** (1.0 / 3.0)
    for _ in range(8):
        indices = np.floor((pts - mn) / voxel_size).astype(np.int64)
        _, unique_idx = np.unique(indices[:, 0] * 1000003 + indices[:, 1] * 1009 + indices[:, 2], return_index=True)
        if len(unique_idx) < target_n * 0.8:
            voxel_size *= 0.85
        elif len(unique_idx) > target_n * 1.3:
            voxel_size *= 1.15
        else:
            break
    result = pts[np.sort(unique_idx)]
    if result.shape[0] > target_n:
        result = result[np.random.RandomState(42).choice(result.shape[0], target_n, replace=False)]
    return result


def pca_global_sim3(src, dst):
    """Global sim(3) from PCA: scale from the variance ratio, rotation from the principal axes, with
    the axis permutation and signs chosen by Chamfer distance on a 5000-point subsample."""
    if src.shape[0] < 3 or dst.shape[0] < 3:
        return np.eye(4, dtype=np.float32)
    src, dst = src.astype(np.float64), dst.astype(np.float64)
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    R_src, R_tgt = pca_axes(src), pca_axes(dst)
    var_src = np.trace((src - mu_src).T @ (src - mu_src)) / src.shape[0]
    var_tgt = np.trace((dst - mu_dst).T @ (dst - mu_dst)) / dst.shape[0]
    s = np.sqrt(var_tgt / var_src) if var_src > 0 and var_tgt > 0 else 1.0

    src_sample, dst_sample = uniform_subsample(src, 5000), uniform_subsample(dst, 5000)

    candidates = [R_tgt @ R_src.T]  # the principal axes as found, then every proper permutation of them
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            M = np.zeros((3, 3))
            M[list(perm), range(3)] = signs
            if np.linalg.det(M) > 0.0:
                candidates.append(R_tgt @ M @ R_src.T)
    errs = [chamfer_mean_bidir(s * (src_sample @ R.T) + (mu_dst - s * (R @ mu_src)), dst_sample)[2]
            for R in candidates]
    best_R = candidates[int(np.argmin(errs))]

    T = np.eye(4)
    T[:3, :3] = s * best_R
    T[:3, 3] = mu_dst - s * (best_R @ mu_src)
    return T.astype(np.float32)


# ---- sim(3) refinement with Open3D point-to-point ICP. ----
o3dr = o3d.pipelines.registration


def normalise_points(pts):
    """Centre the cloud and divide by its mean radius; returns (points, centre, scale)."""
    center = pts.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(pts - center, axis=1).mean()
    if scale <= 0:
        scale = 1.0
    return (pts - center) / scale, center, scale


def estimate_sim3_icp(source_pts, target_pts, voxel_size, max_corr_factor, max_iters):
    """sim(3) ICP with scaling, coarse to fine: clouds centred and scale-normalised, downsampled
    at `voxel_size`, three passes with a shrinking correspondence distance."""
    if source_pts.shape[0] < 10 or target_pts.shape[0] < 10:
        raise ValueError("Too few points for ICP alignment.")
    src_n, src_center, src_scale = normalise_points(source_pts)
    tgt_n, tgt_center, tgt_scale = normalise_points(target_pts)

    src_pcd = build_open3d_pcd(src_n, voxel_size=voxel_size)
    tgt_pcd = build_open3d_pcd(tgt_n, voxel_size=voxel_size)
    base_max_corr = voxel_size * max_corr_factor

    estimation = o3dr.TransformationEstimationPointToPoint(with_scaling=True)
    T_norm = np.eye(4)
    for factor, iters in zip((10.0, 5.0, 1.0), (max_iters // 3 or 50, max_iters // 3 or 50, max_iters)):
        T_norm = o3dr.registration_icp(
            src_pcd, tgt_pcd, base_max_corr * factor, T_norm, estimation_method=estimation,
            criteria=o3dr.ICPConvergenceCriteria(max_iteration=iters),
        ).transformation

    S_src = np.eye(4)
    S_src[:3, :3] *= 1.0 / src_scale
    S_src[:3, 3] = -src_center[0] / src_scale
    S_tgt = np.eye(4)
    S_tgt[:3, :3] *= tgt_scale
    S_tgt[:3, 3] = tgt_center[0]
    return (S_tgt @ T_norm @ S_src).astype(np.float32)


# ---- Chamfer distance, for the alignment above ----
def chamfer_mean_bidir(pts_a, pts_b):
    """Mean nearest-neighbour distance a->b, b->a, and their average."""
    d_a2b, _ = cKDTree(pts_b).query(pts_a, k=1)
    d_b2a, _ = cKDTree(pts_a).query(pts_b, k=1)
    mean_a2b, mean_b2a = float(np.mean(d_a2b)), float(np.mean(d_b2a))
    return mean_a2b, mean_b2a, (mean_a2b + mean_b2a) / 2.0
