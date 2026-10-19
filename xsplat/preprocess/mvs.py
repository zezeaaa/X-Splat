"""The MVS side of a scene package: MVSFormer depths for the training views, fused into one point
cloud (rgb/mvs_points/mvs_fused.ply) and its farthest-point subsample that seeds the alignment."""
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from third_party.mvsformer.mvs_estimator import MvsEstimator
from xsplat.scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary
from xsplat.utils import seed_everything

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOUNDS = (1.0, 10.0)   # the depth range every view searches when mvs.depth_bounds is "default"


@dataclass
class View:
    name: str
    R: np.ndarray        # camera-to-world rotation (COLMAP's, transposed)
    T: np.ndarray
    K: np.ndarray
    center: np.ndarray
    bounds: np.ndarray   # (near, far): the depth range MVSFormer searches
    image: Image.Image   # background suppressed
    mask: np.ndarray     # foreground, bool


def depth_bounds_from_sparse(sparse, extrinsics):
    """Per image, the [0.5, 99.5] percentiles of the depths of the sparse points it observes."""
    points_path = sparse / "points3D.bin"
    if not points_path.is_file():
        raise FileNotFoundError(f"mvs.depth_bounds is 'sparse' but there is no {points_path}; reconstruct the "
                                f"sparse points, or set it to 'default' to search a fixed range")
    points = read_points3D_binary(str(points_path))
    if not points:
        raise ValueError(f"mvs.depth_bounds is 'sparse' but {points_path} holds no points; triangulate them, "
                         f"or set it to 'default'")
    bounds = {}
    for img in extrinsics.values():
        R, t = qvec2rotmat(img.qvec), img.tvec
        depths = [(R @ points[pid] + t)[2] for pid in img.point3D_ids[img.point3D_ids >= 0] if pid in points]
        depths = [z for z in depths if z > 0]
        if depths:
            near, far = np.percentile(np.asarray(depths, dtype=np.float32), [0.5, 99.5])
            bounds[Path(img.name).name] = (float(near), float(far))
    return bounds


def load_views(rgb_root, images_dir, depth_bounds):
    """The COLMAP cameras of a package as MVSFormer views, by image name."""
    sparse = rgb_root / "sparse" / "0"
    extrinsics = read_extrinsics_binary(str(sparse / "images.bin"))
    intrinsics = read_intrinsics_binary(str(sparse / "cameras.bin"))
    bounds = depth_bounds_from_sparse(sparse, extrinsics) if depth_bounds == "sparse" else {}
    views = []
    for extr in extrinsics.values():
        intr = intrinsics[extr.camera_id]
        if intr.model == "PINHOLE":
            fx, fy, cx, cy = intr.params
        elif intr.model == "SIMPLE_PINHOLE":
            fx, cx, cy = intr.params
            fy = fx
        else:
            raise ValueError(f"{intr.model}: only undistorted PINHOLE / SIMPLE_PINHOLE cameras are supported")
        name = Path(extr.name).name
        rotation = qvec2rotmat(extr.qvec)
        views.append(View(
            name=Path(name).stem, R=rotation.T, T=np.array(extr.tvec),
            K=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32),
            center=-np.matmul(rotation.T, np.array(extr.tvec)),
            bounds=np.array(bounds.get(name, DEFAULT_BOUNDS), dtype=np.float32),
            image=Image.open(images_dir / name).convert("RGB"),
            mask=np.array(Image.open(rgb_root / "masks" / f"{Path(name).stem}.png").convert("L")) > 0,
        ))
    return sorted(views, key=lambda v: v.name)


def drop_points_outside_masks(points, colors, views, min_hits, chunk=200000):
    """Keep the points that project inside the foreground mask of at least `min_hits` views."""
    hits = np.zeros(len(points), dtype=np.uint16)
    for view in views:
        h, w = view.mask.shape
        for start in range(0, len(points), chunk):
            pts_cam = (view.R @ points[start:start + chunk].T).T + view.T
            z = pts_cam[:, 2]
            uv = (view.K @ pts_cam.T).T[:, :2] / z[:, None]
            u, v = uv[:, 0].astype(np.int64), uv[:, 1].astype(np.int64)
            idx = np.nonzero((z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h))[0]
            hits[start:start + chunk][idx] += view.mask[v[idx], u[idx]].astype(np.uint16)
    keep = hits >= max(1, min_hits)
    return (points[keep], colors[keep]) if np.any(keep) else (points, colors)


def fps_downsample(points, n_points, colors):
    """Farthest-point downsampling to `n_points`, the (uint8) colours carried along."""
    if points.shape[0] <= n_points:
        return points.copy(), colors.copy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    pcd = pcd.farthest_point_down_sample(n_points)
    return np.asarray(pcd.points).astype(np.float32), (np.asarray(pcd.colors) * 255.0).clip(0, 255).astype(np.uint8)


def apply_masks_to_images(scene_root, masks_dir):
    """images/ with the background zeroed, as images_masked/: what MVSFormer reads."""
    out_dir = scene_root / "images_masked"
    out_dir.mkdir(exist_ok=True)
    for img_path in sorted((scene_root / "images").iterdir()):
        img = np.array(Image.open(img_path).convert("RGB"))
        img[np.array(Image.open(masks_dir / f"{img_path.stem}.png").convert("L")) == 0] = 0
        Image.fromarray(img).save(out_dir / img_path.name)
    return out_dir


def save_ply(points, colors, path):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(path), pc, write_ascii=False, compressed=False)


def build(scene_root, cfg, seed):
    rgb_root = Path(scene_root).resolve() / "rgb"
    seed_everything(seed)
    config_path = Path(cfg.config) if Path(cfg.config).is_absolute() else ROOT / cfg.config
    pc_downsample = float(cfg.pc_downsample)

    # MVSFormer reads background-suppressed images; built here, removed again below
    masked_dir = apply_masks_to_images(rgb_root, rgb_root / "masks")
    llffhold = int(cfg.llffhold)
    views = [v for i, v in enumerate(load_views(rgb_root, masked_dir, str(cfg.depth_bounds))) if i % llffhold != 0]
    print(f"[MVS] {len(views)} training views (llffhold={llffhold}), {int(cfg.src_num)} sources each: "
          + ", ".join(v.name for v in views[:4]) + (" ..." if len(views) > 4 else ""))
    estimator = MvsEstimator(config_path, [float(cfg.bds_scale_near), float(cfg.bds_scale_far)], ROOT)
    points, colors = estimator.get_mvs_pts(views, int(cfg.src_num))
    shutil.rmtree(masked_dir)
    if len(points) == 0:
        raise RuntimeError("MVS fusion returned an empty point cloud.")

    kept, kept_colors = drop_points_outside_masks(points, colors, views, int(cfg.mask_min_hits))
    if len(kept) < len(points):
        print(f"[MVS] {len(points) - len(kept)} / {len(points)} points outside the masks of {int(cfg.mask_min_hits)} views dropped")
    # seeded: this subsample decides which points seed the RGB Gaussians
    keep = np.sort(np.random.default_rng(seed).choice(len(kept), int(len(kept) * pc_downsample), replace=False))
    points, colors = kept[keep].astype(np.float32), kept_colors[keep].astype(np.uint8)

    out_dir = rgb_root / "mvs_points"
    out_dir.mkdir(exist_ok=True)
    save_ply(points, colors, out_dir / "mvs_fused.ply")
    # the split is recorded; UnifiedScene warns if training uses another llffhold (test views would leak)
    (out_dir / "mvs_meta.json").write_text(json.dumps({"llffhold": llffhold}, indent=2))
    n_align = int(cfg.align_sample_points)
    points_u, colors_u = fps_downsample(points, n_align, colors)
    save_ply(points_u, colors_u, out_dir / "mvs_fused_uniform.ply")
    print(f"[MVS] {len(points)} fused points -> {out_dir}, {n_align} of them for the alignment")
