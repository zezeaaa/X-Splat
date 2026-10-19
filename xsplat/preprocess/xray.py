"""The X-ray side of a scene package: projections simulated from the CT volume with TIGRE,
and the FDK reconstruction of the training projections sampled into init.npy."""
import copy
import json
import os
import os.path as osp
import shutil
from pathlib import Path

import numpy as np
import tigre
import tigre.algorithms as algs
import yaml

from xsplat.scene import load_scanner

INIT_POINTS = 50000          # voxels sampled from the FDK reconstruction into init.npy
INIT_DENSITY_RESCALE = 0.15  # applied to the sampled densities
AUTO_SVOXEL = 2.0            # longest side of the volume when the geometry follows its shape


def get_geometry_tigre(cfg):
    """TIGRE's cone-beam geometry from a scanner config (axes reversed: TIGRE is z, y, x)."""
    if cfg["mode"] != "cone":
        raise NotImplementedError(f"scanner mode {cfg['mode']!r}; the scanners are cone-beam")
    geo = tigre.geometry(mode="cone")
    geo.DSD = cfg["DSD"]
    geo.DSO = cfg["DSO"]
    geo.nDetector = np.array(cfg["nDetector"])
    geo.sDetector = np.array(cfg["sDetector"])
    geo.dDetector = geo.sDetector / geo.nDetector
    geo.nVoxel = np.array(cfg["nVoxel"][::-1])
    geo.sVoxel = np.array(cfg["sVoxel"][::-1])
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.offOrigin = np.array(cfg["offOrigin"][::-1])
    geo.offDetector = np.array([cfg["offDetector"][1], cfg["offDetector"][0], 0])
    geo.accuracy = cfg["accuracy"]
    geo.filter = cfg["filter"]
    return geo


def simulate_projections(vol_path, scanner_path, case_dir, n_train, n_test, auto_geometry, voxel_pitch_mm):
    """Write the volume, its projections and meta_data.json of an X-ray scene to `case_dir`."""
    vol = np.load(vol_path).astype(np.float32)
    with open(scanner_path, "r") as handle:
        scanner_cfg = yaml.safe_load(handle)
    if auto_geometry:
        # nVoxel from the volume's shape; isotropic voxels with the longest side AUTO_SVOXEL
        nv = [int(n) for n in vol.shape]
        scanner_cfg["nVoxel"] = nv
        scanner_cfg["sVoxel"] = [AUTO_SVOXEL * n / max(nv) for n in nv]
        print(f"[xray] geometry from the volume: nVoxel={nv}, sVoxel={[f'{s:.3f}' for s in scanner_cfg['sVoxel']]}")
    case_save_path = osp.abspath(case_dir)
    geo = get_geometry_tigre(scanner_cfg)

    # uniform angles over the arc; n_train of them for training, the rest interleaved as the test set
    n_total = n_train + n_test
    all_angles = (
        np.linspace(0, scanner_cfg["totalAngle"] / 180 * np.pi, n_total, endpoint=False)
        + scanner_cfg["startAngle"] / 180 * np.pi
    )
    idx_sub = np.linspace(0, n_total - 1, n_train, endpoint=True)
    idx_sub = [round(i) for i in idx_sub]
    test_indices = [i for i in range(n_total) if i not in idx_sub]
    projs_train_angles = all_angles[idx_sub]
    projs_test_angles = all_angles[test_indices]

    vol_zyx = np.transpose(vol, (2, 1, 0)).copy()   # TIGRE's axis order
    projs_train = tigre.Ax(vol_zyx, geo, projs_train_angles)[:, ::-1, :]
    projs_test = tigre.Ax(vol_zyx, geo, projs_test_angles)[:, ::-1, :]

    os.makedirs(case_save_path, exist_ok=True)
    np.save(osp.join(case_save_path, "vol_gt.npy"), vol)
    file_path_dict = {}
    for split, projs, angles in (("proj_train", projs_train, projs_train_angles), ("proj_test", projs_test, projs_test_angles)):
        os.makedirs(osp.join(case_save_path, split), exist_ok=True)
        file_path_dict[split] = []
        for i_proj in range(projs.shape[0]):
            frame_save_name = osp.join(split, f"{split}_{i_proj:04d}.npy")
            np.save(osp.join(case_save_path, frame_save_name), projs[i_proj])
            file_path_dict[split].append({"file_path": frame_save_name, "angle": float(angles[i_proj])})

    meta = {
        "scanner": scanner_cfg,
        "vol": "vol_gt.npy",
        "bbox": [[-1, -1, -1], [1, 1, 1]],
        "proj_train": file_path_dict["proj_train"],
        "proj_test": file_path_dict["proj_test"],
    }
    if voxel_pitch_mm is not None:
        # the only record of physical size: voxel_pitch_mm x nVoxel recovers millimetres
        meta["voxel_pitch_mm"] = [float(v) for v in voxel_pitch_mm]
    with open(osp.join(case_save_path, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    print(f"[xray] {n_train} training and {len(projs_test_angles)} test projections of {vol.shape} -> {case_save_path}")


def init_pcd(case_dir, density_thresh, seed):
    """Sample the FDK reconstruction of the training projections into init.npy, seeded right here
    so the sample depends only on `seed`."""
    meta, scanner_cfg, scene_scale = load_scanner(case_dir)
    projs = np.stack([np.load(osp.join(case_dir, frame["file_path"])) * scene_scale for frame in meta["proj_train"]])
    angles = np.stack([frame["angle"] for frame in meta["proj_train"]])
    geo = get_geometry_tigre(scanner_cfg)

    np.random.seed(seed)
    vol = np.transpose(algs.fdk(projs[:, ::-1, :], copy.deepcopy(geo), angles), (2, 1, 0))

    # the dense FDK volume is not written; only the sampled points are used
    valid_indices = np.argwhere(vol >= density_thresh)
    if len(valid_indices) < INIT_POINTS:
        raise ValueError(f"only {len(valid_indices)} FDK voxels reach xray.density_thresh={density_thresh}, fewer than the "
                         f"{INIT_POINTS} sampled into init.npy; lower it in configs/preprocess/<dataset>.yaml")
    sampled_indices = valid_indices[np.random.choice(len(valid_indices), INIT_POINTS, replace=False)]
    positions = sampled_indices * np.array(scanner_cfg["dVoxel"]) - np.array(scanner_cfg["sVoxel"]) / 2 + np.array(scanner_cfg["offOrigin"])
    densities = vol[sampled_indices[:, 0], sampled_indices[:, 1], sampled_indices[:, 2]] * INIT_DENSITY_RESCALE

    save_path = osp.join(case_dir, "init.npy")
    np.save(save_path, np.concatenate([positions, densities[:, None]], axis=-1))
    print(f"[xray] {INIT_POINTS} FDK voxels above {density_thresh} -> {save_path}")


def read_voxel_pitch(raw_scene_root):
    """The scanner's voxel pitch in mm from the raw scene's meta.yaml, if recorded."""
    meta_path = raw_scene_root / "meta.yaml"
    if not meta_path.is_file():
        return None
    with open(meta_path, encoding="utf-8") as fh:
        meta = yaml.safe_load(fh) or {}
    pitch = meta.get("voxel_pitch_mm")
    return [float(v) for v in pitch] if pitch else None


def prepare(raw_scene_root, scene_root, cfg, seed):
    """The X-ray side of a scene package: projections simulated from ct/volume.npy (or ready-made
    ones copied in), then the FDK initialisation."""
    raw_scene_root = Path(raw_scene_root).resolve()
    vol_path = raw_scene_root / "ct" / "volume.npy"
    raw_xray_dir = raw_scene_root / "xray"

    have_volume = vol_path.is_file()
    have_projections = (raw_xray_dir / "meta_data.json").is_file()
    if not have_volume and not have_projections:
        raise FileNotFoundError(
            f"a raw scene needs either {vol_path} (projections are simulated from it) or "
            f"{raw_xray_dir}/ holding proj_train/, proj_test/ and meta_data.json"
        )
    if have_volume and have_projections:
        raise ValueError(
            f"{raw_scene_root} has both a CT volume and ready-made projections; keep one, "
            f"so it is unambiguous which the scene package was built from"
        )

    case_dir = Path(scene_root).resolve() / "xray"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.parent.mkdir(parents=True, exist_ok=True)

    if have_volume:
        simulate_projections(str(vol_path), cfg.scanner, str(case_dir), int(cfg.n_train), int(cfg.n_test),
                             bool(cfg.auto_geometry), read_voxel_pitch(raw_scene_root))
    else:
        # real projections: copied as given; their meta_data.json carries geometry and angles
        shutil.copytree(raw_xray_dir, case_dir)

    init_pcd(str(case_dir), float(cfg.density_thresh), seed)


