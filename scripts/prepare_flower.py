"""Flower originals -> raw scenes: the RGB views rendered from the library's Blender scene, and
its CT volume normalised and thresholded.

    python scripts/prepare_flower.py --scenes mRay_Tulip/20140110_mRay_Tulip --blender <path>   # both flowers without --scenes
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xsplat.preprocess.raw_scene import (apply_foreground_mask, normalize_volume, preprocess_cfg, read_traw3d_ss,  # noqa: E402
                                         rgba_to_black_bg_and_masks)

CFG = preprocess_cfg("flower")
RENDER_SETTINGS = ("num_views", "n_rotations", "elev_max", "elev_min", "up_axis", "width", "height", "samples")
BLENDER_SCRIPT = Path(__file__).resolve().with_name("flower_blender.py")


def main():
    parser = argparse.ArgumentParser(description="Flower originals -> raw scenes; settings in configs/preprocess/flower.yaml")
    parser.add_argument("--scenes", nargs="+", default=list(CFG.scenes), help="Scenes as <flower>/<scan> under paths.source_root.")
    parser.add_argument("--blender", required=True, help="The Blender 3.6 binary (README: Data preparation tools).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing raw scene.")
    args = parser.parse_args()

    for scene_rel in args.scenes:
        cfg = preprocess_cfg("flower", scene_rel)
        scene_dir = Path(cfg.paths.source_root).resolve() / scene_rel
        blends, mtls = sorted(scene_dir.glob("*.blend")), sorted(scene_dir.glob("*.mtl"))
        if not blends or not mtls:
            raise FileNotFoundError(f"{scene_dir} needs the .blend and .mtl of the library's blend.zip (DATA.md, Flower)")
        scene = scene_rel.replace("/", "_")
        raw_scene_root = (Path(cfg.paths.raw_root) / scene).resolve()
        if raw_scene_root.exists():
            if not args.overwrite:
                raise FileExistsError(f"{raw_scene_root} exists; pass --overwrite to replace it")
            shutil.rmtree(raw_scene_root)
        rgb_dir = raw_scene_root / "rgb"
        rgb_dir.mkdir(parents=True)

        # the RGB views: Blender renders the library's scene from a spiral of cameras
        cmd = [args.blender, str(blends[0]), "--background", "--python", str(BLENDER_SCRIPT), "--",
               "--out_dir", str(rgb_dir), "--seed", "0"]
        for key in RENDER_SETTINGS:
            cmd += [f"--{key}", str(cfg.raw[key])]
        print("[RUN]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        if not (rgb_dir / "transforms.json").is_file():
            raise RuntimeError(f"Blender wrote no transforms.json under {rgb_dir}")
        rgba_to_black_bg_and_masks(rgb_dir)

        # the CT volume at native resolution; no largest-component filter, the petals are disconnected
        vol = normalize_volume(read_traw3d_ss(scene_dir.parent / f"{scene_dir.name}.traw3D_ss")["data"], 0.3, 99.7)
        vol = apply_foreground_mask(vol, cfg.raw.foreground_thresh)
        (raw_scene_root / "ct").mkdir()
        np.save(raw_scene_root / "ct" / "volume.npy", vol)
        print(f"[CT] {scene}: {vol.shape} -> {raw_scene_root / 'ct' / 'volume.npy'}")

        OmegaConf.save(OmegaConf.create({"dataset": "flower", "scene": scene, "pose_mode": "known_pose"}),
                       raw_scene_root / "meta.yaml")
        print(f"[DONE] {scene}: {raw_scene_root}")


if __name__ == "__main__":
    main()
