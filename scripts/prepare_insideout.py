"""InsideOut originals -> raw scenes: the views the paper used, cropped to one foreground box
per scene with the masks we annotated, and the CT scan normalised, resampled and cleaned.

    python scripts/prepare_insideout.py --scenes skull   # all four without --scenes
"""
import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xsplat.preprocess.raw_scene import (apply_foreground_mask, foreground_crop_box, normalize_volume, preprocess_cfg,  # noqa: E402
                                         resize_if_needed, resize_volume, zero_slices)

CFG = preprocess_cfg("insideout")
JPG_QUALITY = 95
INTERMEDIATE_SIZE = 512   # the scan is read through a strided memmap down to this, then resampled

# the views the paper used: (first, last, skipped) numbers at the end of the file names
VIEW_RANGES = {
    "matryoshka": (1, 53, ()),
    "pharaoh": (102, 168, ()),
    "skull": (37, 90, (49, 51)),
    "toy_gun": (2, 204, ()),
}


def select_views(scene_dir, first, last, skipped):
    """The scene's JPEGs numbered `first`..`last`, in name order."""
    views = []
    for path in sorted(scene_dir.iterdir()):
        number = re.search(r"(\d+)$", path.stem)
        if path.suffix.lower() in (".jpg", ".jpeg") and number \
                and first <= int(number.group(1)) <= last and int(number.group(1)) not in skipped:
            views.append(path)
    if not views:
        raise RuntimeError(f"no views numbered {first}..{last} in {scene_dir}")
    return views


def write_rgb(scene, cfg, raw_scene_root):
    """Pair each view with its mask, bring odd-sized views to the common size, and crop all of them
    to the padded union of the masks."""
    views = select_views(Path(cfg.paths.rgb_root) / scene, *VIEW_RANGES[scene])
    mask_dir = Path(cfg.paths.mask_root) / scene
    pairs = [(view, mask_dir / f"{view.stem}.png") for view in views if (mask_dir / f"{view.stem}.png").is_file()]
    dropped = [view.stem for view in views if not (mask_dir / f"{view.stem}.png").is_file()]
    if dropped:
        print(f"[RGB] {scene}: dropped {len(dropped)} views without masks: {dropped}")
    if not pairs:
        raise RuntimeError(f"[RGB] {scene}: no masks under {mask_dir}")

    sizes = [Image.open(view).size for view, _ in pairs]
    ref_size = Counter(sizes).most_common(1)[0][0]
    crop_box = foreground_crop_box([mask for _, mask in pairs], ref_size, cfg.raw.crop_pad_ratio)

    images_dir, masks_dir = raw_scene_root / "rgb" / "images", raw_scene_root / "rgb" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir()
    for view, mask in pairs:
        img = resize_if_needed(Image.open(view).convert("RGB"), ref_size)
        img.crop(crop_box).save(images_dir / f"{view.stem}.jpg", quality=JPG_QUALITY)
        mask_img = resize_if_needed(Image.open(mask).convert("L"), ref_size, is_mask=True)
        mask_img.crop(crop_box).save(masks_dir / f"{view.stem}.png")
    print(f"[RGB] {scene}: {len(pairs)} views cropped to {crop_box} at {ref_size}; "
          f"{len(sizes) - sizes.count(ref_size)} resized to that size first")


def read_scan(ct_root, scene):
    """The int16 scan, read through a strided memmap down to INTERMEDIATE_SIZE a side, turned the
    way the rest of the pipeline expects."""
    dims = {}
    for line in open(ct_root / f"{scene}.txt", encoding="utf-8", errors="ignore"):
        for key in ("Image Dimension", "Number of Z-slices"):
            if key in line:
                dims[key] = int(line.split()[0])
    n_xy, n_z = dims["Image Dimension"], dims["Number of Z-slices"]
    stride = n_xy // INTERMEDIATE_SIZE
    print(f"[CT] {scene}: reading the {n_xy}x{n_xy}x{n_z} scan at stride {stride}")
    scan = np.memmap(ct_root / f"{scene}.raw", dtype=np.int16, mode="r")
    if scan.size != n_xy * n_xy * n_z:
        raise ValueError(f"{scene}.raw holds {scan.size} values, not the {n_xy}x{n_xy}x{n_z} the .txt says")
    vol = scan.reshape(n_xy, n_xy, n_z)[::stride, ::stride, ::stride].astype(np.float32)
    return np.rot90(vol.transpose(0, 2, 1), k=1, axes=(0, 2))


def write_ct(scene, cfg, raw_scene_root):
    vol = normalize_volume(read_scan(Path(cfg.paths.ct_root), scene), 1.0, 99.0)
    vol = resize_volume(vol, (cfg.raw.target_size,) * 3)
    vol = apply_foreground_mask(vol, cfg.raw.foreground_thresh, largest_cc=True)
    vol = zero_slices(vol, cfg.raw.crop_axis, cfg.raw.crop_bottom_ratio)
    vol = zero_slices(vol, cfg.raw.crop_axis, cfg.raw.crop_top_ratio, from_end=True)
    (raw_scene_root / "ct").mkdir()
    np.save(raw_scene_root / "ct" / "volume.npy", vol)
    print(f"[CT] {scene}: {vol.shape} -> {raw_scene_root / 'ct' / 'volume.npy'}")


def main():
    parser = argparse.ArgumentParser(description="InsideOut originals -> raw scenes; settings in configs/preprocess/insideout.yaml")
    parser.add_argument("--scenes", nargs="+", default=list(CFG.scenes), choices=sorted(VIEW_RANGES))
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing raw scene.")
    args = parser.parse_args()

    for scene in args.scenes:
        cfg = preprocess_cfg("insideout", scene)
        raw_scene_root = (Path(cfg.paths.raw_root) / scene).resolve()
        if raw_scene_root.exists():
            if not args.overwrite:
                raise FileExistsError(f"{raw_scene_root} exists; pass --overwrite to replace it")
            shutil.rmtree(raw_scene_root)
        write_rgb(scene, cfg, raw_scene_root)
        write_ct(scene, cfg, raw_scene_root)
        OmegaConf.save(OmegaConf.create({"dataset": "insideout", "scene": scene, "pose_mode": "reconstruct"}),
                       raw_scene_root / "meta.yaml")
        print(f"[DONE] {scene}: {raw_scene_root}")


if __name__ == "__main__":
    main()
