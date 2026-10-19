#
# Derived from 3D Gaussian Splatting (Inria / MPII), and from R2-Gaussian.
# Copyright (C) 2023, Inria, GRAPHDECO research group.
# Licensed under the Gaussian-Splatting License; see LICENSE.md.
# For inquiries contact george.drettakis@inria.fr
#
"""Loading a scene package: the X-ray projections and the RGB views as cameras, and the scene object a run works on."""
import json
import math
import os.path as osp

import numpy as np
import torch
from PIL import Image

from xsplat.scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary


class Camera:
    """A pinhole view with its image on the GPU, in the 3DGS convention: R is the camera-to-world rotation,
    T the world-to-camera translation. RGB views also carry the intrinsics K and the foreground mask,
    which is multiplied into the image."""

    def __init__(self, R, T, FoVx, FoVy, image, image_name, K=None, mask=None):
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.original_image = (image if mask is None else image * mask).cuda()
        self.gt_alpha_mask = None if mask is None else mask.cuda()
        self.image_height, self.image_width = self.original_image.shape[1:]
        self.K = None if K is None else torch.tensor(K, dtype=torch.float32, device="cuda")

        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        self.projection_matrix = get_projection_matrix(FoVx, FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def getWorld2View2(R, t):
    # 3DGS's, with translate 0 and scale 1; the round trip through C2W is kept so the matrices match it bit for bit
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def get_projection_matrix(fovX, fovY, znear=0.01, zfar=100.0):
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


# ---- X-ray: the projections of a TIGRE cone-beam scanner, as in R2-Gaussian ----
def angle_to_pose(DSO, angle):
    """Camera-to-world pose of the source at `angle` on the scanner's circular trajectory."""
    phi1 = -np.pi / 2
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(phi1), -np.sin(phi1)],
                   [0, np.sin(phi1), np.cos(phi1)]])
    phi2 = np.pi / 2
    R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0],
                   [np.sin(phi2), np.cos(phi2), 0],
                   [0, 0, 1]])
    R3 = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0],
                   [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = R3 @ R2 @ R1
    T[:3, 3] = [DSO * np.cos(angle), DSO * np.sin(angle), 0.0]
    return T


def load_scanner(xray_dir):
    """Read `meta_data.json`, with the scanner normalised so the volume's longest side is 2.
    Returns (meta, scanner_cfg, scene_scale)."""
    with open(osp.join(xray_dir, "meta_data.json"), "r") as f:
        meta = json.load(f)
    scanner = meta["scanner"]
    if scanner["mode"] != "cone":
        raise NotImplementedError(f"scanner mode {scanner['mode']!r}; the scanners are cone-beam")
    if "dVoxel" not in scanner:
        scanner["dVoxel"] = (np.array(scanner["sVoxel"]) / np.array(scanner["nVoxel"])).tolist()
    if "dDetector" not in scanner:
        scanner["dDetector"] = (np.array(scanner["sDetector"]) / np.array(scanner["nDetector"])).tolist()

    scene_scale = 2.0 / max(scanner["sVoxel"])
    for key in ["dVoxel", "sVoxel", "sDetector", "dDetector", "offOrigin", "offDetector", "DSD", "DSO"]:
        scanner[key] = (np.array(scanner[key]) * scene_scale).tolist()
    return meta, scanner, scene_scale


def load_xray(xray_dir):
    """The X-ray side of a scene package: the train and test projections as cameras, the ground-truth CT
    (None when the scene has none; then there are no 3D metrics) and the normalised scanner."""
    meta, scanner, scene_scale = load_scanner(xray_dir)
    FovX = np.arctan2(scanner["sDetector"][1] / 2, scanner["DSD"]) * 2
    FovY = np.arctan2(scanner["sDetector"][0] / 2, scanner["DSD"]) * 2
    cameras = {"train": [], "test": []}
    for split in cameras:
        for frame in meta["proj_" + split]:
            w2c = np.linalg.inv(angle_to_pose(scanner["DSO"], frame["angle"]))
            image = torch.from_numpy(np.load(osp.join(xray_dir, frame["file_path"])) * scene_scale)[None]
            cameras[split].append(Camera(np.transpose(w2c[:3, :3]), w2c[:3, 3], FovX, FovY, image,
                                         osp.splitext(osp.basename(frame["file_path"]))[0]))

    vol_gt = None
    if meta.get("vol"):
        vol_path = osp.join(xray_dir, meta["vol"])
        if not osp.isfile(vol_path):
            raise FileNotFoundError(f"meta_data.json names a ground-truth volume at {vol_path}, but it is not there. "
                                    f"Remove the `vol` entry if this scene has no ground truth.")
        vol_gt = torch.from_numpy(np.load(vol_path)).float().cuda()
    print(f"[scene] X-ray: {len(cameras['train'])} train / {len(cameras['test'])} test projections"
          + (f", ground-truth CT {list(vol_gt.shape)}" if vol_gt is not None else ", no ground-truth CT"))
    return cameras["train"], cameras["test"], vol_gt, scanner


# ---- RGB: a COLMAP model with images and masks ----
def pil_to_torch(pil_image, size, resample=None):
    """A PIL image resized to `size` as a float (C, H, W) tensor in [0, 1]."""
    t = torch.from_numpy(np.array(pil_image.resize(size, resample=resample))) / 255.0
    return (t if t.dim() == 3 else t[..., None]).permute(2, 0, 1)


def load_rgb(rgb_dir, llffhold, resolution):
    """The RGB side of a scene package: COLMAP's binary model, the images resized to `resolution` pixels wide
    with their masks multiplied in. Views are sorted by name and every `llffhold`-th is a test view, as in the MVS build."""
    sparse = osp.join(rgb_dir, "sparse", "0")
    if not osp.isfile(osp.join(sparse, "images.bin")):
        raise FileNotFoundError(f"No COLMAP model at {sparse}/images.bin. Has this scene been through `python preprocess.py`?")
    intrinsics = read_intrinsics_binary(osp.join(sparse, "cameras.bin"))
    extrinsics = sorted(read_extrinsics_binary(osp.join(sparse, "images.bin")).values(),
                        key=lambda extr: osp.splitext(osp.basename(extr.name))[0])

    cameras = {"train": [], "test": []}
    for i, extr in enumerate(extrinsics):
        intr = intrinsics[extr.camera_id]
        if intr.model == "PINHOLE":
            fx, fy, cx, cy = intr.params
        elif intr.model == "SIMPLE_PINHOLE":
            fx, cx, cy = intr.params
            fy = fx
        else:
            raise ValueError(f"{intr.model}: only undistorted PINHOLE / SIMPLE_PINHOLE cameras are supported")
        down = intr.width / resolution
        size = (int(intr.width / down), int(intr.height / down))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        sx, sy = size[0] / float(intr.width), size[1] / float(intr.height)
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy

        stem = osp.splitext(osp.basename(extr.name))[0]
        with Image.open(osp.join(rgb_dir, "images", osp.basename(extr.name))) as im:
            image = pil_to_torch(im, size)[:3]
        with Image.open(osp.join(rgb_dir, "masks", stem + ".png")) as im:
            mask = pil_to_torch(im.convert("L"), size, Image.NEAREST)  # NEAREST keeps the mask binary
        cameras["test" if i % llffhold == 0 else "train"].append(Camera(
            np.transpose(qvec2rotmat(extr.qvec)), np.array(extr.tvec), focal2fov(fx, intr.width), focal2fov(fy, intr.height),
            image, stem, K=K, mask=mask))
    print(f"[scene] RGB: {len(cameras['train'])} train / {len(cameras['test'])} test views at {size[0]}x{size[1]}")
    return cameras["train"], cameras["test"]


class UnifiedScene:
    """A scene package loaded for a run: the cameras of both modalities, the scanner and the ground-truth CT,
    and the run's output directories under cfg.output.scene_dir."""

    def __init__(self, cfg):
        scene_root = osp.abspath(cfg.data.scene_root)
        self.scene_dir = cfg.output.scene_dir
        self.checkpoints_dir = osp.join(self.scene_dir, "checkpoints")
        self.alignment_dir = osp.join(self.scene_dir, "alignment")
        self.exports_dir = osp.join(self.scene_dir, "exports")
        self.renders_dir = osp.join(self.scene_dir, "renders")
        self.evaluation_dir = osp.join(self.scene_dir, "evaluation")

        self.xray_init_pcd_path = osp.join(scene_root, "xray", "init.npy")
        self.xray_train_cameras, self.xray_test_cameras, self.vol_gt, self.scanner_cfg = load_xray(osp.join(scene_root, "xray"))
        self.bbox = torch.stack([
            torch.tensor(self.scanner_cfg["offOrigin"]) - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            torch.tensor(self.scanner_cfg["offOrigin"]) + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
        ], dim=0)

        self.rgb_source_path = osp.join(scene_root, "rgb")
        # the MVS cloud was built from the training views of its split; another llffhold here would leak test views into it
        mvs_meta = osp.join(self.rgb_source_path, "mvs_points", "mvs_meta.json")
        mvs_llffhold = json.load(open(mvs_meta))["llffhold"] if osp.isfile(mvs_meta) else cfg.rgb_model.llffhold
        if mvs_llffhold != cfg.rgb_model.llffhold:
            print(f"[scene] WARNING: the MVS point cloud was built with llffhold={mvs_llffhold}, training uses "
                  f"{cfg.rgb_model.llffhold}: test views may have leaked into the initialisation; "
                  f"rebuild it with `python preprocess.py ... --steps mvs`")
        self.rgb_train_cameras, self.rgb_test_cameras = load_rgb(self.rgb_source_path, cfg.rgb_model.llffhold,
                                                                 cfg.rgb_model.resolution)
