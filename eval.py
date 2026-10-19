"""Evaluate a trained scene: X-ray 2D/3D and RGB metrics on the held-out views, plus their renders.

    python eval.py outputs/insect/E02

Reads the run's config.yaml; writes evaluation/metrics_<iteration>.json and renders/.
"""
import os.path as osp
from argparse import ArgumentParser

import numpy as np
import torch

from xsplat.metrics import evaluate_rgb_split, evaluate_xray_3d, evaluate_xray_split, update_metrics_file
from xsplat.model.gaussian_model import UnifiedGaussianModel
from xsplat.scene import UnifiedScene
from xsplat.utils import load_saved_unified_cfg, search_for_max_iteration


def evaluate(cfg, iteration):
    """Both modalities on the held-out views; returns the metrics by section ("xray", "rgb")."""
    scene = UnifiedScene(cfg)
    if iteration == -1:
        iteration = search_for_max_iteration(scene.checkpoints_dir)
    model = UnifiedGaussianModel(sh_degree=cfg.rgb_model.sh_degree)
    model.load_checkpoint(osp.join(scene.checkpoints_dir, f"iteration_{iteration}.pickle"))
    # the RGB -> X-ray transform from Stage 1; without it the RGB cameras do not look at the model
    T_path = osp.join(scene.alignment_dir, "T_realign.npy")
    if not osp.isfile(T_path):
        raise FileNotFoundError(f"{T_path} is missing, and the RGB cameras are only meaningful once that transform "
                                f"is applied. If this is a downloaded checkpoint, unpack its alignment/ directory "
                                f"alongside checkpoints/ and config.yaml.")
    alignment_transform = torch.from_numpy(np.load(T_path)).float().cuda()

    print(f"\n[eval] iteration {iteration}")
    sections = {}
    if scene.xray_test_cameras:
        p, s = evaluate_xray_split(scene.xray_test_cameras, model, save_root=osp.join(scene.renders_dir, "xray"))
        sections["xray"] = {"2d/test/psnr": p, "2d/test/ssim": s}
        print(f"  X-ray 2D: PSNR={p:.3f}, SSIM={s:.3f}")
    if scene.vol_gt is None:
        print("  no ground-truth CT volume, so no 3D metrics")
    else:
        m3d = evaluate_xray_3d(model, scene.scanner_cfg, scene.vol_gt)
        sections.setdefault("xray", {}).update({"3d/psnr_3d": m3d["psnr_3d"], "3d/ssim_3d": m3d["ssim_3d"]})
        print(f"  X-ray 3D: PSNR={m3d['psnr_3d']:.3f}, SSIM={m3d['ssim_3d']:.3f}")
    rgb = sections["rgb"] = evaluate_rgb_split(scene.rgb_test_cameras, model, torch.zeros(3, device="cuda"),
                                              alignment_transform=alignment_transform,
                                              save_root=osp.join(scene.renders_dir, "rgb"))
    print(f"  RGB: PSNR={rgb['psnr']:.3f}, SSIM={rgb['ssim']:.3f}, LPIPS={rgb['lpips']:.4f}")
    path = update_metrics_file(scene.evaluation_dir, iteration, **sections)
    print(f"  metrics written to {path}")
    return sections


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", help="training output directory, e.g. outputs/insect/E02")
    parser.add_argument("--scene_root", default=None, help="scene package to read instead of the run's recorded data.scene_root")
    parser.add_argument("--iteration", type=int, default=-1, help="checkpoint to evaluate; -1 (default) uses the latest")
    args = parser.parse_args()

    evaluate(load_saved_unified_cfg(args.scene_dir, args.scene_root), args.iteration)
