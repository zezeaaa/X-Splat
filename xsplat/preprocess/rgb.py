"""The RGB side of a scene package: images, masks and a COLMAP model, whether the cameras are
recovered, triangulated from known poses, or taken as given."""
import json
import math
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation


BLENDER2OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)


def run_colmap(colmap_cmd, subcommand, *args, doing):
    """Run one COLMAP subcommand. Arguments are stringified, so Paths can be passed."""
    print(f"[COLMAP] {doing}", flush=True)
    subprocess.run([colmap_cmd, subcommand, *(str(a) for a in args)], check=True)


def keep_largest_model(sparse):
    """COLMAP may split a scene into several models; the one with the most images becomes 0."""
    models = sorted(d for d in sparse.iterdir() if d.is_dir())
    if not models:
        raise RuntimeError("COLMAP mapper produced no model. Check input images.")
    # images.bin starts with the number of registered images (little-endian uint64)
    best = max(models, key=lambda m: struct.unpack("<Q", (m / "images.bin").read_bytes()[:8])[0])
    for model in models:
        if model != best:
            shutil.rmtree(model)
    if best.name != "0":
        best.rename(sparse / "0")


def run_colmap_reconstruct(scene_root, colmap_cmd, sift_max_num_features, sift_peak_threshold):
    """COLMAP SfM on `scene_root/input`, then undistortion into images/ and sparse/0. COLMAP is not
    deterministic: the registered views can differ between runs."""
    input_dir = scene_root / "input"
    distorted_db = scene_root / "distorted" / "database.db"
    distorted_sparse = scene_root / "distorted" / "sparse"
    distorted_sparse.mkdir(parents=True, exist_ok=True)

    sift_opts = []
    if sift_max_num_features is not None:
        sift_opts += ["--SiftExtraction.max_num_features", sift_max_num_features]
    if sift_peak_threshold is not None:
        sift_opts += ["--SiftExtraction.peak_threshold", sift_peak_threshold]
    # PINHOLE, so the undistorter leaves the images as they are: the masks are copied unwarped
    run_colmap(colmap_cmd, "feature_extractor", "--database_path", distorted_db, "--image_path", input_dir,
               "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "PINHOLE",
               "--SiftExtraction.use_gpu", 1, *sift_opts, doing="Feature extraction...")
    run_colmap(colmap_cmd, "exhaustive_matcher", "--database_path", distorted_db, "--SiftMatching.use_gpu", 1,
               doing="Exhaustive matching...")
    run_colmap(colmap_cmd, "mapper", "--database_path", distorted_db, "--image_path", input_dir,
               "--output_path", distorted_sparse, "--Mapper.ba_global_function_tolerance=0.000001",
               doing="Mapper (SfM / BA)...")
    keep_largest_model(distorted_sparse)
    run_colmap(colmap_cmd, "image_undistorter", "--image_path", input_dir, "--input_path", distorted_sparse / "0",
               "--output_path", scene_root, "--output_type", "COLMAP", doing="Image undistorter...")

    # the undistorter writes the model into sparse/ itself; the package keeps it at sparse/0
    sparse = scene_root / "sparse"
    (sparse / "0").mkdir(exist_ok=True)
    for path in sparse.iterdir():
        if path.name != "0":
            path.rename(sparse / "0" / path.name)
    print(f"[COLMAP] {len(list((scene_root / 'images').iterdir()))} of {len(list(input_dir.iterdir()))} views registered",
          flush=True)
    # input/ (a copy of the images), the database, stereo/ and the undistorter's scripts go;
    # distorted/sparse/ stays: it is the reconstruction itself, which COLMAP never repeats
    for rel in ("input", "distorted/database.db", "stereo", "run-colmap-geometric.sh", "run-colmap-photometric.sh"):
        path = scene_root / rel
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def sparse_text_to_binary(sparse_dir, colmap_cmd):
    """Replace a text COLMAP model with the binary one, in place."""
    if not (sparse_dir / "cameras.txt").is_file():
        return
    run_colmap(colmap_cmd, "model_converter", "--input_path", sparse_dir, "--output_path", sparse_dir,
               "--output_type", "BIN", doing=f"Converting {sparse_dir} to binary...")
    for name in ("cameras", "images", "points3D"):
        (sparse_dir / f"{name}.txt").unlink()


def triangulate_known_poses(scene_root, colmap_cmd):
    """Sparse points for cameras that are known: features, sequential matches, then COLMAP's
    triangulator with the poses held fixed, in place at sparse/0."""
    images_dir, sparse_dir, database = scene_root / "images", scene_root / "sparse" / "0", scene_root / "database.db"
    try:
        run_colmap(colmap_cmd, "feature_extractor", "--database_path", database, "--image_path", images_dir,
                   "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "PINHOLE",
                   "--SiftExtraction.use_gpu", 1, doing="feature_extractor...")
        run_colmap(colmap_cmd, "sequential_matcher", "--database_path", database, "--SequentialMatching.overlap", 10,
                   "--SiftMatching.guided_matching", 1, "--SiftMatching.max_ratio", 0.9, "--SiftMatching.max_distance", 0.9,
                   doing="sequential_matcher...")
        sparse_text_to_binary(sparse_dir, colmap_cmd)
        run_colmap(colmap_cmd, "point_triangulator", "--database_path", database, "--image_path", images_dir,
                   "--input_path", sparse_dir, "--output_path", sparse_dir, doing="point_triangulator...")
    finally:
        database.unlink(missing_ok=True)


def transforms_json_to_colmap(scene_root):
    """A NeRF-style transforms.json (camera_angle_x, one c2w per frame) as a COLMAP text model at
    sparse/0, cameras only; the points come from the triangulator."""
    meta = json.loads((scene_root / "transforms.json").read_text())
    frames = meta["frames"]
    # a file_path without an extension (NeRF-synthetic style, as Flower's renders) is a .png
    images = [scene_root / "images" / (name if name.lower().endswith((".png", ".jpg", ".jpeg")) else f"{name}.png")
              for name in (Path(frame["file_path"]).name for frame in frames)]
    with Image.open(images[0]) as image:
        width, height = image.size
    focal = 0.5 * width / math.tan(0.5 * meta["camera_angle_x"])

    sparse_dir = scene_root / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    (sparse_dir / "cameras.txt").write_text(
        f"1 PINHOLE {width} {height} {focal:.8f} {focal:.8f} {0.5 * width:.8f} {0.5 * height:.8f}\n")
    lines = []
    for idx, (frame, image) in enumerate(zip(frames, images), start=1):
        pose = np.array(frame["transform_matrix"], dtype=np.float64) @ BLENDER2OPENCV
        rot = np.linalg.inv(pose[:3, :3])
        t = -np.dot(rot, pose[:3, 3])
        qx, qy, qz, qw = Rotation.from_matrix(rot).as_quat()
        if qw < 0:
            qw, qx, qy, qz = -qw, -qx, -qy, -qz
        lines.append(f"{idx} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f} 1 {image.name}\n\n")
    (sparse_dir / "images.txt").write_text("".join(lines))
    (sparse_dir / "points3D.txt").write_text("")


def copy_files(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_dir.iterdir()):
        if path.is_file():
            shutil.copy2(path, dst_dir / path.name)


def require_pinhole_cameras(sparse0):
    """Refuse camera models with distortion: the pipeline renders pinhole cameras and never undistorts."""
    from xsplat.scene.colmap_loader import read_intrinsics_binary

    intrinsics = read_intrinsics_binary(str(sparse0 / "cameras.bin"))
    unsupported = {cam.model for cam in intrinsics.values()} - {"PINHOLE", "SIMPLE_PINHOLE"}
    if unsupported:
        raise ValueError(f"{sparse0} uses camera model(s) {sorted(unsupported)}. Undistort the images "
                         f"first (`colmap image_undistorter`), which rewrites them as PINHOLE.")
    print(f"[COLMAP] {len(intrinsics)} camera(s), all pinhole")


def prepare(raw_scene_root, scene_root, cfg):
    """The RGB side of a scene package: images, masks and a COLMAP model. meta.yaml says how the
    cameras come: `reconstruct` runs COLMAP SfM, `known_pose` converts transforms.json and
    triangulates, `colmap` copies in a model of your own."""
    raw_rgb = Path(raw_scene_root).resolve() / "rgb"
    rgb_dir = Path(scene_root).resolve() / "rgb"
    with open(Path(raw_scene_root) / "meta.yaml", encoding="utf-8") as fh:
        pose_mode = yaml.safe_load(fh).get("pose_mode")
    if pose_mode not in ("colmap", "known_pose", "reconstruct"):
        raise ValueError(f"Unsupported pose_mode in {raw_scene_root}/meta.yaml: {pose_mode}. Use `reconstruct` to "
                         f"recover the cameras with COLMAP, `colmap` if you already have a model, or `known_pose` "
                         f"with a transforms.json.")
    if rgb_dir.exists():
        shutil.rmtree(rgb_dir)

    # SfM reads input/ and its undistorter writes images/; the other modes get the images as they are
    copy_files(raw_rgb / "images", rgb_dir / ("input" if pose_mode == "reconstruct" else "images"))
    copy_files(raw_rgb / "masks", rgb_dir / "masks")

    if pose_mode == "reconstruct":
        run_colmap_reconstruct(rgb_dir, cfg.colmap, cfg.sift_max_num_features, cfg.sift_peak_threshold)
    elif pose_mode == "known_pose":
        shutil.copy2(raw_rgb / "transforms.json", rgb_dir / "transforms.json")
        transforms_json_to_colmap(rgb_dir)
        triangulate_known_poses(rgb_dir, cfg.colmap)
    else:
        copy_files(raw_rgb / "sparse" / "0", rgb_dir / "sparse" / "0")
        sparse_text_to_binary(rgb_dir / "sparse" / "0", cfg.colmap)
        require_pinhole_cameras(rgb_dir / "sparse" / "0")
