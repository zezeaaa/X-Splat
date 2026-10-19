"""Render a trained scene along a spherical orbit.

    python render_video.py outputs/insect/E02

Writes xray.mp4, rgb.mp4 and overlay.mp4 into renders/orbit/. The published datasets use
the orbits below; other data circles +z. Any setting can be overridden on the command line.
"""
import os
import os.path as osp
from argparse import ArgumentParser

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from matplotlib import colormaps
from tqdm import tqdm

from xsplat.model.gaussian_model import UnifiedGaussianModel
from xsplat.render import look_at, render_rgb_unified, render_xray_unified
from xsplat.scene import Camera, load_scanner
from xsplat.utils import load_saved_unified_cfg, search_for_max_iteration


# X-ray tone mapping as in the paper video: percentile-normalise (over the whole orbit, so
# brightness does not pulse) -> gamma -> CLAHE -> colormap -> unsharp.
XRAY_CMAP = "bone"
XRAY_GAMMA = 0.6
XRAY_CLAHE_CLIP = 2.0
XRAY_CLAHE_GRID = (8, 8)
XRAY_UNSHARP = 0.4
XRAY_PERCENTILES = (1.0, 99.5)

AXES = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


# ---- Camera path ----

def remap_to_orbit_axis(ex, ey, ez, orbit_axis):
    """Rotate a position given with the pole at +Z so that the pole is `orbit_axis`."""
    if orbit_axis == "+z":
        return (ex, ey, ez)
    if orbit_axis == "-z":
        return (ex, ey, -ez)
    if orbit_axis == "+x":
        return (ez, ey, -ex)
    if orbit_axis == "-x":
        return (-ez, ey, ex)
    if orbit_axis == "+y":
        return (ex, ez, -ey)
    if orbit_axis == "-y":
        return (ex, -ez, ey)
    raise ValueError(f"orbit_axis must be one of {list(AXES)}, got {orbit_axis}")


def orbit_cameras(scanner_cfg, n_views, radius_factor, orbit_axis, world_up_axis,
                  elevation_high_deg, elevation_low_deg):
    """Cameras on a spiral around the CT centre, rendered in both modalities: `orbit_axis` is the pole,
    `world_up_axis` the up direction in the images (None = the pole), elevation sweeps high -> low."""
    world_up_axis = world_up_axis or orbit_axis

    center = np.array(scanner_cfg["offOrigin"])
    radius = scanner_cfg["DSO"] * radius_factor
    width, height = scanner_cfg["nDetector"][1], scanner_cfg["nDetector"][0]
    FovX = np.arctan2(scanner_cfg["sDetector"][1] / 2, scanner_cfg["DSD"]) * 2
    FovY = np.arctan2(scanner_cfg["sDetector"][0] / 2, scanner_cfg["DSD"]) * 2
    K = np.array([[width / (2.0 * np.tan(FovX * 0.5)), 0.0, width * 0.5],
                  [0.0, height / (2.0 * np.tan(FovY * 0.5)), height * 0.5],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    elev_high = np.deg2rad(elevation_high_deg)
    elev_low = np.deg2rad(elevation_low_deg)
    cameras = []
    for i in range(n_views):
        t = i / (n_views - 1 if n_views > 1 else 1)
        elevation = elev_high + (elev_low - elev_high) * t
        azimuth = 2 * np.pi * 2 * t
        ex = radius * np.cos(elevation) * np.cos(azimuth)
        ey = radius * np.cos(elevation) * np.sin(azimuth)
        ez = radius * np.sin(elevation)
        camera_pos = center + np.array(remap_to_orbit_axis(ex, ey, ez, orbit_axis))
        w2c = np.linalg.inv(look_at(camera_pos, center, azimuth, AXES[world_up_axis]))
        cameras.append(Camera(w2c[:3, :3].T, w2c[:3, 3], FovX, FovY, torch.zeros(3, height, width), f"orbit_{i:04d}", K=K))
    return cameras


# ---- Frames and videos ----

def xray_to_rgb(gray, low, high, cmap_name):
    """Grayscale X-ray -> RGB float [0, 1], matching the paper video's treatment."""
    norm = np.clip((gray - low) / max(high - low, 1e-6), 0.0, 1.0)
    norm = np.power(norm, XRAY_GAMMA)
    clahe = cv2.createCLAHE(clipLimit=XRAY_CLAHE_CLIP, tileGridSize=XRAY_CLAHE_GRID)
    norm = clahe.apply((norm * 255).astype(np.uint8)).astype(np.float32) / 255.0
    rgb = colormaps[cmap_name](norm)[..., :3]
    blur = cv2.GaussianBlur(rgb, (0, 0), 2.0)
    return np.clip(rgb + XRAY_UNSHARP * (rgb - blur), 0.0, 1.0)


def rgb_to_numpy(tensor):
    """(3, H, W) render -> H x W x 3 float, clipped to [0, 1] as eval.py does."""
    return np.clip(tensor.detach().cpu().numpy().transpose(1, 2, 0), 0.0, 1.0)


def draw_label(img_bgr, label):
    font, scale, thickness, margin = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2, 12
    (text_w, text_h), _ = cv2.getTextSize(label, font, scale, thickness)
    x, y = margin, margin + text_h
    cv2.rectangle(img_bgr, (x - 6, y - text_h - 6), (x + text_w + 6, y + 6), (0, 0, 0), thickness=-1)
    cv2.putText(img_bgr, label, (x, y), font, scale, (255, 255, 255), thickness)


def write_video(frames_rgb, path, fps, label, frames_dir=None):
    """Encode float RGB frames as H.264 (browsers refuse OpenCV's MPEG-4); optionally keep PNG frames."""
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
    writer = imageio.get_writer(
        path, format="FFMPEG", mode="I", fps=fps, codec="libx264",
        pixelformat="yuv420p", macro_block_size=1,
        ffmpeg_params=["-crf", "17", "-preset", "slow", "-movflags", "+faststart"],
    )
    try:
        for idx, frame in enumerate(tqdm(frames_rgb, desc=f"encoding {label}")):
            bgr = cv2.cvtColor((np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            if frames_dir:  # frames without the corner label, for figures
                cv2.imwrite(osp.join(frames_dir, f"{idx:05d}.png"), bgr)
            draw_label(bgr, label)
            writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    print(f"video saved: {path}")


def render_orbit(model, scanner_cfg, output_dir, n_views=360, radius_factor=1.0, fps=45, orbit_axis="+z",
                 world_up_axis=None, elevation_high_deg=70.0, elevation_low_deg=-70.0, alpha=0.5,
                 xray_cmap=XRAY_CMAP, save_frames=False):
    """Write xray.mp4, rgb.mp4 and overlay.mp4 into `output_dir`."""
    os.makedirs(output_dir, exist_ok=True)
    cameras = orbit_cameras(scanner_cfg, n_views, radius_factor, orbit_axis, world_up_axis,
                            elevation_high_deg, elevation_low_deg)
    frames_dir = osp.join(output_dir, "frames") if save_frames else None

    with torch.no_grad():
        xray = [render_xray_unified(cam, model)["render"][0].cpu().numpy() for cam in tqdm(cameras, desc="rendering X-ray")]
    # percentiles over the whole orbit (every 20th frame), so the tone mapping does not pulse
    low, high = np.percentile(np.concatenate([x.ravel() for x in xray[::20]]), XRAY_PERCENTILES)
    xray_rgb = [xray_to_rgb(x, float(low), float(high), xray_cmap) for x in xray]
    write_video(xray_rgb, osp.join(output_dir, "xray.mp4"), fps, "xray", frames_dir and osp.join(frames_dir, "xray"))

    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    rgb, masks = [], []
    with torch.no_grad():
        for cam in tqdm(cameras, desc="rendering RGB"):
            out = render_rgb_unified(cam, model, background)
            rgb.append(rgb_to_numpy(out["render"]))
            masks.append(np.clip(out["render_alphas"][0].cpu().numpy(), 0.0, 1.0)[:, :, None])
    write_video(rgb, osp.join(output_dir, "rgb.mp4"), fps, "rgb", frames_dir and osp.join(frames_dir, "rgb"))

    # X-ray shows through the object; outside it the RGB background is kept as is.
    overlay = [rgb_np * (1.0 - m) + (alpha * rgb_np + 0.6 * x) * m
               for rgb_np, x, m in zip(rgb, xray_rgb, masks)]
    write_video(overlay, osp.join(output_dir, "overlay.mp4"), fps, "overlay", frames_dir and osp.join(frames_dir, "overlay"))


# per-dataset orbit: axis, up direction, elevation range (where the photographs are)
ORBITS = {
    # The objects are stored head-down, so "up" is flipped relative to the orbit axis.
    "insideout": dict(orbit_axis="+z", world_up_axis="-z",
                      elevation_high_deg=70.0, elevation_low_deg=-70.0),
    # The conventional Z-up full sphere.
    "flower": dict(orbit_axis="+z", world_up_axis="+z",
                   elevation_high_deg=70.0, elevation_low_deg=-70.0),
    # elongated specimens: circle the dorsoventral axis, upper hemisphere only
    "insect": dict(orbit_axis="-x", world_up_axis="-x",
                   elevation_high_deg=75.0, elevation_low_deg=20.0),
}
ORBIT_SETTINGS = ("n_views", "fps", "radius_factor", "orbit_axis", "world_up_axis",
                  "elevation_high_deg", "elevation_low_deg")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", help="training output directory, e.g. outputs/insect/E02")
    parser.add_argument("--scene_root", default=None, help="scene package to read instead of the run's recorded data.scene_root")
    parser.add_argument("--iteration", type=int, default=-1, help="checkpoint to render; -1 (default) uses the latest")
    parser.add_argument("--output_dir", default=None, help="defaults to <scene_dir>/renders/orbit")
    parser.add_argument("--n_views", type=int, help="frames in the video (default 360)")
    parser.add_argument("--fps", type=int, help="default 45")
    parser.add_argument("--radius_factor", type=float, help="camera distance as a multiple of the source-to-origin distance (default 1.0)")
    parser.add_argument("--orbit_axis", choices=list(AXES), help="the axis the camera circles")
    parser.add_argument("--world_up_axis", choices=list(AXES), help="the direction that points up in the video")
    parser.add_argument("--elevation_high_deg", type=float, help="elevation the spiral starts at")
    parser.add_argument("--elevation_low_deg", type=float, help="elevation the spiral ends at")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight of the RGB in the overlay")
    parser.add_argument("--xray_cmap", default=XRAY_CMAP, help="matplotlib colormap for the X-ray")
    parser.add_argument("--save_frames", action="store_true", help="also write every frame as a PNG, without the corner label")
    args = parser.parse_args()

    cfg = load_saved_unified_cfg(args.scene_dir, args.scene_root)
    orbit = dict(ORBITS.get(str(cfg.dataset_meta.name), {}))
    orbit.update({name: getattr(args, name) for name in ORBIT_SETTINGS
                  if getattr(args, name) is not None})

    checkpoints_dir = osp.join(cfg.output.scene_dir, "checkpoints")
    iteration = search_for_max_iteration(checkpoints_dir) if args.iteration == -1 else args.iteration
    model = UnifiedGaussianModel(sh_degree=cfg.rgb_model.sh_degree)
    model.load_checkpoint(osp.join(checkpoints_dir, f"iteration_{iteration}.pickle"))
    _, scanner_cfg, _ = load_scanner(osp.join(cfg.data.scene_root, "xray"))
    print(f"[render_video] iteration {iteration}")
    render_orbit(model, scanner_cfg, args.output_dir or osp.join(cfg.output.scene_dir, "renders", "orbit"),
                 alpha=args.alpha, xray_cmap=args.xray_cmap, save_frames=args.save_frames, **orbit)


if __name__ == "__main__":
    main()
