"""Check a raw scene against what `python preprocess.py` expects, before the long pipeline runs.

    python scripts/check_raw_scene.py data/raw/custom/scene01

Exit code is the number of problems.
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml
from PIL import Image

RGB_SUFFIXES = {".png", ".jpg", ".jpeg"}


def fail(problems, message):
    problems.append(message)
    print(f"  FAIL  {message}")


def ok(message):
    print(f"  ok    {message}")


def check_meta(scene, problems):
    path = os.path.join(scene, "meta.yaml")
    if not os.path.isfile(path):
        fail(problems, f"no meta.yaml at {path}")
        return None
    with open(path, encoding="utf-8") as fh:
        meta = yaml.safe_load(fh) or {}
    pose_mode = meta.get("pose_mode")
    if pose_mode not in {"reconstruct", "colmap", "known_pose"}:
        fail(problems, f"meta.yaml: pose_mode must be 'reconstruct' (recover the cameras with "
                           f"COLMAP), 'colmap' (you supply the model) or 'known_pose' (a "
                           f"transforms.json), got {pose_mode!r}")
        return None
    ok(f"meta.yaml: pose_mode={pose_mode}")
    return pose_mode


def check_xray(scene, problems):
    """A scene supplies its X-ray side one of two ways, never both."""
    volume = os.path.join(scene, "ct", "volume.npy")
    projections = os.path.join(scene, "xray")
    have_volume = os.path.isfile(volume)
    have_projections = os.path.isfile(os.path.join(projections, "meta_data.json"))

    if have_volume and have_projections:
        fail(problems, f"{scene} has both a CT volume and ready-made projections; "
                           f"keep one, so it is unambiguous which the package came from")
        return
    if have_projections:
        check_projections(projections, problems)
        return
    if not have_volume:
        fail(problems, f"no CT volume at {volume}, and no projections at "
                           f"{projections}/meta_data.json; a scene needs one of the two")
        return
    check_ct_volume(volume, problems)


def check_projections(xray_dir, problems):
    meta_path = os.path.join(xray_dir, "meta_data.json")
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception as exc:
        fail(problems, f"{meta_path} does not parse as JSON: {exc}")
        return

    for key in ("scanner", "proj_train", "proj_test"):
        if key not in meta:
            fail(problems, f"meta_data.json has no {key!r}")
            return
    if meta.get("vol") and not os.path.isfile(os.path.join(xray_dir, meta["vol"])):
        fail(problems, f"meta_data.json names a `vol` ({meta['vol']}) that is not there; drop it unless "
                           f"you ship that volume, which is only used for the 3D metrics")

    # what TIGRE's geometry and the scanner normalisation read
    missing_geometry = [k for k in ("mode", "DSD", "DSO", "nDetector", "sDetector", "nVoxel", "sVoxel",
                                    "offOrigin", "offDetector", "accuracy", "filter") if k not in meta["scanner"]]
    if missing_geometry:
        fail(problems, f"meta_data.json scanner is missing {missing_geometry}")
    elif meta["scanner"]["mode"] != "cone":
        fail(problems, f"meta_data.json scanner mode is {meta['scanner']['mode']!r}; only cone beam is supported")

    for split in ("proj_train", "proj_test"):
        frames = meta[split]
        if not frames:
            if split == "proj_train":
                fail(problems, "meta_data.json proj_train is empty")
            else:
                ok("proj_test: empty, so there will be no 2D metrics")
            continue
        absent = [f["file_path"] for f in frames
                  if not os.path.isfile(os.path.join(xray_dir, f["file_path"]))][:3]
        if absent:
            fail(problems, f"{split} names files that are not there, e.g. {absent}")
            continue
        no_angle = [f["file_path"] for f in frames if "angle" not in f][:3]
        if no_angle:
            fail(problems, f"{split} frames need an `angle` in radians, e.g. {no_angle}")
            continue
        sample = np.load(os.path.join(xray_dir, frames[0]["file_path"]))
        if sample.ndim != 2:
            fail(problems, f"{split} projections must be 2-D, got {sample.shape}")
        elif sample.dtype != np.float32:
            fail(problems, f"{split} projections are {sample.dtype}, expected float32")
        else:
            ok(f"{split}: {len(frames)} projections, {sample.shape} float32")


def check_ct_volume(path, problems):
    try:
        volume = np.load(path, mmap_mode="r")
    except Exception as exc:
        fail(problems, f"{path} does not load as a numpy array: {exc}")
        return
    if volume.ndim != 3:
        fail(problems, f"CT volume must be 3-D, got shape {volume.shape}")
        return
    low, high = float(np.min(volume)), float(np.max(volume))
    if low < 0.0 or high > 1.0:
        fail(problems, f"CT volume ranges [{low:.3f}, {high:.3f}], expected [0, 1]; "
                           f"normalise it (DATA.md, 'Your own data')")
    elif high < 0.1:
        fail(problems, f"CT volume never exceeds {high:.3f}; it is probably empty or "
                           f"the wrong scaling")
    else:
        ok(f"CT volume {tuple(volume.shape)} {volume.dtype} in [{low:.2f}, {high:.2f}]")


def check_rgb(scene, pose_mode, problems):
    images_dir = os.path.join(scene, "rgb", "images")
    masks_dir = os.path.join(scene, "rgb", "masks")
    if not os.path.isdir(images_dir):
        fail(problems, f"no images at {images_dir}")
        return
    images = sorted(f for f in os.listdir(images_dir)
                    if os.path.splitext(f)[1].lower() in RGB_SUFFIXES)
    if not images:
        fail(problems, f"{images_dir} holds no images")
        return
    if len(images) < 20:  # not a problem as such, so not counted as one
        print(f"  note  only {len(images)} views; the published scenes use 36-101 "
              f"and structure-from-motion needs overlap")

    if not os.path.isdir(masks_dir):
        fail(problems, f"no masks at {masks_dir}; every view needs one, since the "
                           f"RGB loss and the metrics are computed inside them")
        return

    missing, mismatched, empty = [], [], []
    for name in images:
        stem = os.path.splitext(name)[0]
        mask_path = os.path.join(masks_dir, stem + ".png")
        if not os.path.isfile(mask_path):
            missing.append(stem)
            continue
        with Image.open(os.path.join(images_dir, name)) as image, \
                Image.open(mask_path) as mask:
            if image.size != mask.size:
                mismatched.append(f"{stem} ({image.size} vs {mask.size})")
            elif not np.any(np.array(mask.convert("L")) > 0):
                empty.append(stem)

    if missing:
        fail(problems, f"{len(missing)} views have no mask, e.g. {missing[:3]}; "
                           f"masks are matched by filename stem and must be .png")
    if mismatched:
        fail(problems, f"{len(mismatched)} masks differ in size from their image, "
                           f"e.g. {mismatched[:2]}")
    if empty:
        fail(problems, f"{len(empty)} masks are entirely black, e.g. {empty[:3]}")
    if not (missing or mismatched or empty):
        with Image.open(os.path.join(images_dir, images[0])) as first:
            ok(f"{len(images)} views, each with a mask, first is {first.size}")

    if pose_mode == "colmap":
        sparse = os.path.join(scene, "rgb", "sparse", "0")
        absent = [name for name in ("cameras", "images", "points3D")
                  if not any(os.path.isfile(os.path.join(sparse, name + ext)) for ext in (".bin", ".txt"))]
        if absent:
            fail(problems, f"pose_mode is colmap but {sparse} has no {absent} (binary or text); "
                               f"MVSFormer takes each view's depth range from the points3D")
        else:
            ok(f"COLMAP model present at {os.path.relpath(sparse, scene)}")
    elif pose_mode == "known_pose":
        transforms = os.path.join(scene, "rgb", "transforms.json")
        if not os.path.isfile(transforms):
            fail(problems, f"pose_mode is known_pose but there is no {transforms}")
        else:
            ok("transforms.json present")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="a raw scene directory, e.g. data/raw/custom/scene01")
    args = parser.parse_args()

    if not os.path.isdir(args.scene):
        print(f"no such directory: {args.scene}")
        return 1

    print(f"checking {args.scene}")
    problems = []
    pose_mode = check_meta(args.scene, problems)
    check_xray(args.scene, problems)
    check_rgb(args.scene, pose_mode, problems)

    print()
    if problems:
        print(f"{len(problems)} problem(s) to fix before `python preprocess.py` will work.")
    else:
        print("Looks like a raw scene; build its package with "
              "`python preprocess.py --dataset <dataset> --scenes <scene>` "
              "(for your own data, edit the custom configs first, as DATA.md 'Your own data' says).")
    return len(problems)


if __name__ == "__main__":
    sys.exit(main())
