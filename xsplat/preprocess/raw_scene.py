"""Turning original captures into a canonical raw scene: the preparation config, CT volume
normalisation and cleaning, RGB views cropped to a common foreground box, and RGBA renders
turned into black-background images plus masks."""

import struct
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def available_datasets():
    """Every configs/preprocess/<name>.yaml; adding one adds a dataset."""
    return sorted(p.stem for p in (ROOT / "configs" / "preprocess").glob("*.yaml") if p.stem != "default")


def preprocess_cfg(dataset, scene=None):
    """configs/preprocess/default.yaml with the dataset's file on top, then the scene's own overrides,
    found by its key (mRay_Tulip/20140110_mRay_Tulip) or by its directory name (mRay_Tulip_20140110_mRay_Tulip)."""
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / "configs" / "preprocess" / "default.yaml"),
                          OmegaConf.load(ROOT / "configs" / "preprocess" / f"{dataset}.yaml"))
    overrides = {key.replace("/", "_"): value for key, value in (cfg.get("scene_overrides") or {}).items()}
    if scene is not None and scene.replace("/", "_") in overrides:
        cfg = OmegaConf.merge(cfg, overrides[scene.replace("/", "_")])
    return cfg


# ---- CT volumes ----
def read_traw3d_ss(path):
    with open(path, "rb") as f:
        dims = struct.unpack("3i", f.read(3 * struct.calcsize("i")))
        pitch = struct.unpack("3d", f.read(3 * struct.calcsize("d")))
        data = np.frombuffer(f.read(), dtype=np.int16)
    array = data.reshape(dims[2], dims[1], dims[0]).transpose(0, 2, 1)
    return {
        "data": array,
        "dims": (dims[2], dims[0], dims[1]),
        "pitch": (pitch[2], pitch[0], pitch[1]),
    }


def normalize_volume(array, clip_min_pct, clip_max_pct):
    array = array.astype(np.float32)
    vmin = np.percentile(array, clip_min_pct)
    vmax = np.percentile(array, clip_max_pct)
    array = np.clip(array, vmin, vmax)
    return (array - array.min()) / (array.max() - array.min() + 1e-8)


def resize_volume(array, target_shape):
    zoom = [t / s for t, s in zip(target_shape, array.shape)]
    return np.clip(ndimage.zoom(array, zoom, order=3), 0.0, 1.0).astype(np.float32)


def apply_foreground_mask(array, thresh, largest_cc=False):
    """Zero everything below `thresh`. With `largest_cc`, keep only the largest connected component
    (holes filled) and soften its edge with a sigma-1 Gaussian."""
    mask = array > thresh
    print(f"[CT] threshold {thresh:.4f}: {int(mask.sum())}/{array.size} voxels above it ({mask.mean():.2%})")
    if not largest_cc:
        out = array.copy()
        out[~mask] = 0.0
        return out.astype(np.float32)
    labeled, _ = ndimage.label(ndimage.binary_fill_holes(mask))
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    cc = labeled == int(np.argmax(counts))
    print(f"[CT] largest component: {int(cc.sum())}/{array.size} voxels ({cc.mean():.2%})")
    weight = ndimage.gaussian_filter(np.ascontiguousarray(cc.astype(np.float32)), sigma=1.0)
    return (array * weight).astype(np.float32)


def zero_slices(array, axis, ratio, from_end=False):
    """Zero the first (or last) `ratio` of the slices along `axis`: a stand, or scanner noise."""
    n = int(array.shape[axis] * ratio)
    if n == 0:
        return array
    index = [slice(None)] * 3
    index[axis] = slice(array.shape[axis] - n, None) if from_end else slice(0, n)
    array = array.copy()
    array[tuple(index)] = 0.0
    print(f"[CT] axis {axis}: {n} slices zeroed at the {'end' if from_end else 'start'}")
    return array


# ---- RGB views, cropped to one foreground box per scene: a reference size, the union of the masks, padding ----
def resize_if_needed(img, ref_size, is_mask=False):
    if img.size == ref_size:
        return img
    resample = Image.NEAREST if is_mask else Image.BICUBIC
    return img.resize(ref_size, resample=resample)


def foreground_crop_box(mask_paths, ref_size, crop_pad_ratio):
    """One square crop for the scene: the union of the masks' bounding boxes, padded by
    `crop_pad_ratio` of its extent, squared on the longer side, inside the image where it fits."""
    boxes = []
    for path in mask_paths:
        ys, xs = np.where(np.array(resize_if_needed(Image.open(path).convert("L"), ref_size, is_mask=True)) > 0)
        if ys.size:
            boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    if not boxes:
        raise RuntimeError("No foreground pixels found in selected masks.")
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    pad_x = int(round((x1 - x0) * crop_pad_ratio))
    pad_y = int(round((y1 - y0) * crop_pad_ratio))
    width, height = ref_size
    x0, y0, x1, y1 = x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y
    side = max(x1 - x0, y1 - y0)
    x0 = int(round(0.5 * (x0 + x1) - 0.5 * side))
    y0 = int(round(0.5 * (y0 + y1) - 0.5 * side))
    if side <= width:
        x0 = min(max(0, x0), width - side)
    if side <= height:
        y0 = min(max(0, y0), height - side)
    return x0, y0, x0 + side, y0 + side


def rgba_to_black_bg_and_masks(scene_root):
    """Rendered RGBA views -> RGB on black, with the alpha channel as the mask."""
    images_dir = scene_root / "images"
    masks_dir = scene_root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() != ".png":
            continue
        rgba = np.array(Image.open(path).convert("RGBA"))
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]
        mask = (alpha > 127).astype(np.uint8) * 255
        rgb_black_bg = (rgb.astype(np.float64) * (alpha / 255.0)[:, :, None]).astype(np.uint8)
        Image.fromarray(rgb_black_bg).save(path)
        Image.fromarray(mask).save(masks_dir / path.name)
        count += 1
    if count:
        print(f"[black_bg] {count} images -> black background; masks -> {masks_dir}")
