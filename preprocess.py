"""Raw scenes -> scene packages: COLMAP poses and masks, TIGRE projections with the FDK
initialisation, and the MVS point cloud.

    python preprocess.py --dataset insideout                   # every scene configs/preprocess/insideout.yaml lists
    python preprocess.py --dataset custom --scenes scene01
    python preprocess.py --dataset insideout --scenes toy_gun --steps rgb mvs   # COLMAP and MVS again, X-ray kept

A raw scene is data/raw/<dataset>/<scene>/ with rgb/images/, rgb/masks/, meta.yaml and either
ct/volume.npy or ready-made projections in xray/; DATA.md says how each dataset gets there.
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

from xsplat.preprocess import mvs, rgb, xray
from xsplat.preprocess.raw_scene import available_datasets, preprocess_cfg

STEPS = ("rgb", "xray", "mvs")


def main():
    parser = argparse.ArgumentParser(description="Raw scenes -> scene packages; settings in configs/preprocess/<dataset>.yaml")
    parser.add_argument("--dataset", required=True, choices=available_datasets(), help="any configs/preprocess/<name>.yaml")
    parser.add_argument("--scenes", nargs="+", default=None, help="scene keys; default: every scene the dataset's config lists")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS), help="which of rgb, xray, mvs to run")
    parser.add_argument("--processed_root", type=Path, default=None, help="write the packages here instead of paths.processed_root")
    parser.add_argument("--seed", type=int, default=0, help="seeds the X-ray initialisation and the MVS subsample; COLMAP takes none")
    args = parser.parse_args()

    scenes = args.scenes or list(preprocess_cfg(args.dataset).get("scenes") or [])
    if not scenes:
        parser.error(f"configs/preprocess/{args.dataset}.yaml lists no scenes; pass --scenes")
    roots = []
    for key in scenes:
        cfg = preprocess_cfg(args.dataset, key)
        name = key.replace("/", "_")
        raw_scene = Path(cfg.paths.raw_root).resolve() / name
        scene_root = (args.processed_root or Path(cfg.paths.processed_root)).resolve() / name
        roots.append((key, cfg, raw_scene, scene_root))

    # TIGRE's cudaDeviceReset() leaves CUDA unusable for torch in this process, so the X-ray
    # stage of every scene runs before the MVS stage of any of them.
    for key, cfg, raw_scene, scene_root in roots:
        if "rgb" in args.steps:
            print(f"=== {args.dataset}/{key}: rgb -> {scene_root}", flush=True)
            rgb.prepare(raw_scene, scene_root, cfg.rgb)
        if "xray" in args.steps:
            print(f"=== {args.dataset}/{key}: xray -> {scene_root}", flush=True)
            xray.prepare(raw_scene, scene_root, cfg.xray, args.seed)
    for key, cfg, raw_scene, scene_root in roots:
        if "mvs" in args.steps:
            print(f"=== {args.dataset}/{key}: mvs -> {scene_root}", flush=True)
            mvs.build(scene_root, cfg.mvs, args.seed)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    # once TIGRE has run, torch's CUDA teardown segfaults at interpreter exit: leave without it
    sys.stdout.flush()
    os._exit(0)
