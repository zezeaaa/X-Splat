"""The CT surface: extracted from a density volume by marching cubes, placed in the X-ray frame, sampled as
points for the alignment, ray-cast for the depth loss, and the initial RGB appearance around it."""
import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure, label
from scipy.spatial import cKDTree
from skimage import measure

from xsplat.utils import t2a


def ct_volume_to_mesh(vol, iso_level, closing_iterations=5):
    """Outer surface of a CT volume: threshold at `iso_level`, close and fill, keep the largest component,
    marching cubes, keep the largest triangle cluster. Vertices are in voxel units."""
    mask = vol >= iso_level
    if not 0 < mask.sum() < mask.size:
        raise RuntimeError(f"no surface at iso-level {iso_level}: {mask.sum()} of {mask.size} voxels reach it; "
                           f"adjust ct_surface.iso_level")
    mask = binary_closing(mask, structure=generate_binary_structure(3, 1), iterations=closing_iterations)
    mask = binary_fill_holes(mask)
    labeled, num = label(mask)
    if num > 1:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest = counts.argmax()
        if counts[largest] >= 1000:
            mask = labeled == largest

    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    clusters, _, cluster_area = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(clusters) != int(np.argmax(np.asarray(cluster_area))))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def mesh_to_world(mesh, scanner_cfg):
    """Marching-cubes vertices (voxel units) -> the X-ray world frame, in place."""
    sVoxel = np.array(scanner_cfg["sVoxel"], dtype=np.float64)
    dVoxel = sVoxel / np.array(scanner_cfg["nVoxel"], dtype=np.float64)
    verts = np.asarray(mesh.vertices) * dVoxel - sVoxel / 2 + np.array(scanner_cfg["offOrigin"], dtype=np.float64)
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    return mesh


def sample_mesh_points(mesh, n_points):
    return np.asarray(mesh.sample_points_uniformly(number_of_points=n_points).points).astype(np.float32)


def build_open3d_pcd(points, voxel_size=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals()
    return pcd


class CTSurface:
    """The CT surface Stage 2 supervises RGB with: `update()` re-extracts it from the model's density,
    `view(cam)` ray-casts a camera's depth and coverage of it, cached until the next update."""

    def __init__(self, scanner_cfg, iso_level, alignment_transform):
        self.scanner_cfg = scanner_cfg
        self.iso_level = iso_level
        self.alignment_transform = alignment_transform   # RGB frame -> X-ray frame
        self.scene = None  # Open3D ray-casting scene of the current surface
        self.views = {}    # camera -> (depth, coverage)

    def update(self, model):
        """Extract the surface at `iso_level` from the model's density volume."""
        cfg = self.scanner_cfg
        with torch.no_grad():
            vol = model.query_xray_volume(torch.tensor(cfg["offOrigin"], device="cuda", dtype=torch.float32),
                                          torch.tensor(cfg["nVoxel"], device="cuda"),
                                          torch.tensor(cfg["sVoxel"], device="cuda"))
        mesh = mesh_to_world(ct_volume_to_mesh(t2a(vol), self.iso_level, closing_iterations=3), cfg)
        faces = np.asarray(mesh.triangles)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices).astype(np.float32)),
                                 o3d.core.Tensor(faces.astype(np.uint32)))
        self.views.clear()

    def view(self, cam):
        """Depth (camera z, 0 where no ray hits) and coverage of the surface from `cam`, each (1, H, W)."""
        if cam in self.views:
            return self.views[cam]
        T = self.alignment_transform
        c2w = T @ torch.inverse(cam.world_view_transform.T)
        campos = apply_sim3_to_points(cam.camera_center.unsqueeze(0), T).squeeze(0)
        H, W = int(cam.image_height), int(cam.image_width)
        fx, fy, cx, cy = cam.K[0, 0].item(), cam.K[1, 1].item(), cam.K[0, 2].item(), cam.K[1, 2].item()

        grid_u, grid_v = torch.meshgrid(torch.arange(W, device="cuda", dtype=torch.float32),
                                        torch.arange(H, device="cuda", dtype=torch.float32), indexing="xy")
        # (H, W, 3) ray directions: +X right, +Y down, +Z into the scene
        d_cam = torch.stack([(grid_u - cx) / fx, (grid_v - cy) / fy, torch.ones_like(grid_u)], dim=-1)
        norm = (d_cam * d_cam).sum(dim=-1).clamp(min=1e-12).sqrt()
        d_world = (c2w[:3, :3].unsqueeze(0) @ (d_cam / norm.unsqueeze(-1)).reshape(-1, 3).unsqueeze(-1)).squeeze(-1)
        rays = torch.cat([campos.unsqueeze(0).expand(H * W, 3), d_world], dim=1)
        t_hit = self.scene.cast_rays(o3d.core.Tensor(rays.cpu().numpy().astype(np.float32)))["t_hit"]
        t_hit = torch.from_numpy(t_hit.numpy().reshape(H, W).astype(np.float32)).cuda()

        depth = t_hit / (norm + 1e-8)   # t_hit is along the unit ray; camera z is that over the raw ray's length
        hit = ~torch.isinf(t_hit)
        depth[~hit] = 0.0
        self.views[cam] = (depth.unsqueeze(0), hit.float().unsqueeze(0))
        return self.views[cam]


# ---- The initial RGB appearance of the Stage-1 Gaussians: MVS colours, and opacity by distance to the surface ----
def init_rgb_from_mvs(mvs_path, xyz, max_dist):
    """(N, 3) colours in [0, 1] from the nearest MVS point within `max_dist`, gray otherwise."""
    vertices = PlyData.read(mvs_path)["vertex"]
    xyz_mvs = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64)
    colors_mvs = (np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0).astype(np.float32)

    dists, idx = cKDTree(xyz_mvs).query(xyz, k=1)
    near = dists < max_dist
    rgb = np.full((xyz.shape[0], 3), 0.5, dtype=np.float32)
    rgb[near] = colors_mvs[idx[near]]
    print(f"[init] colours: {int(near.sum())}/{xyz.shape[0]} Gaussians within {max_dist} of an MVS point, the rest gray")
    return rgb


def point_to_mesh_distance(surface_mesh_path, xyz):
    """Unsigned distance from each point to the mesh at `surface_mesh_path`."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(surface_mesh_path)))
    return scene.compute_distance(o3d.core.Tensor(xyz.astype(np.float32))).numpy().astype(np.float32)


def surface_band_opacity(dists, max_dist, surface_opacity, band_end_opacity, interior_opacity, surface_band_sigma_ratio):
    """Initial RGB opacity: `surface_opacity` on the CT surface, a Gaussian falloff (sigma = ratio * max_dist)
    to `band_end_opacity` at the band's edge, `interior_opacity` beyond it."""
    sigma = max(float(surface_band_sigma_ratio) * float(max_dist), 1e-8)
    weight = np.exp(-0.5 * (dists / sigma) ** 2)
    weight_end = float(np.exp(-0.5 * (float(max_dist) / sigma) ** 2))
    weight = np.clip((weight - weight_end) / max(1.0 - weight_end, 1e-8), 0.0, 1.0)
    opacity = (band_end_opacity + (surface_opacity - band_end_opacity) * weight).astype(np.float32)
    opacity[dists >= max_dist] = interior_opacity
    print(f"[init] opacity: {int((dists < max_dist).sum())}/{dists.shape[0]} Gaussians in the band of width {max_dist} "
          f"({surface_opacity} on the surface, {band_end_opacity} at its edge), {interior_opacity} beyond")
    return opacity


def downsample_pcd(xyz, density, max_points):
    """Randomly keep at most `max_points` points."""
    n_orig = xyz.shape[0]
    if n_orig <= max_points:
        return xyz, density
    idx = np.random.choice(n_orig, max_points, replace=False)
    print(f"Downsampled point cloud to {max_points} points (from {n_orig}).")
    return xyz[idx].copy(), density[idx].copy()


def apply_sim3_to_points(points, T):
    """Apply a (4, 4) sim(3) matrix to (N, 3) points (differentiable)."""
    ones = torch.ones(points.shape[0], 1, dtype=points.dtype, device=points.device)
    points_h = (T @ torch.cat([points, ones], dim=1).T).T
    return points_h[:, :3] / points_h[:, 3:4]
