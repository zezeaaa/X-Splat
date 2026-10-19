#
# Derived from 3D Gaussian Splatting (Inria / MPII), and from R2-Gaussian.
# Copyright (C) 2023, Inria, GRAPHDECO research group.
# Licensed under the Gaussian-Splatting License; see LICENSE.md.
# For inquiries contact george.drettakis@inria.fr
#
"""Image and volume metrics, the test-split evaluation, and the metrics file every stage writes into."""
import json
import os
import os.path as osp

import imageio.v2 as imageio
import lpips
import numpy as np
import torch
from torchvision.utils import save_image

from xsplat.losses import ssim
from xsplat.render import render_rgb_unified, render_xray_unified


# ---- PSNR, as in 3D Gaussian Splatting (SSIM is the one in losses.py) ----
def mse(img1, img2):
    return ((img1 - img2) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)


@torch.no_grad()
def psnr(img1, img2):
    """PSNR per batch element, for images in [0, 1]."""
    return 10 * torch.log10(1.0 / mse(img1, img2).float())


# ---- Test-split evaluation, shared by eval.py and the in-training test passes ----
_lpips_model = None


def lpips_model():
    """The VGG LPIPS network, loaded on first use (it downloads its weights). It takes inputs in [-1, 1];
    `normalize=True` rescales our [0, 1]."""
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="vgg", verbose=False).cuda()
    return _lpips_model


def evaluate_rgb_split(cameras, model, background, alignment_transform=None, save_root=None):
    """PSNR, SSIM and LPIPS over the foreground of the views (both sides multiplied by the mask), averaged;
    `save_root` also writes the renders."""
    psnr_sum = ssim_sum = lpips_sum = 0.0
    with torch.no_grad():
        for cam in cameras:
            img = torch.clamp(render_rgb_unified(cam, model, background, alignment_transform=alignment_transform)["render"], 0.0, 1.0)
            img_eval, gt_eval = (img * cam.gt_alpha_mask).unsqueeze(0), (cam.original_image * cam.gt_alpha_mask).unsqueeze(0)
            psnr_sum += psnr(img_eval, gt_eval).mean().double().item()
            ssim_sum += ssim(img_eval, gt_eval).mean().double().item()
            lpips_sum += lpips_model()(img_eval, gt_eval, normalize=True).mean().double().item()
            if save_root:
                os.makedirs(save_root, exist_ok=True)
                save_image(img, osp.join(save_root, f"{cam.image_name}.png"))
    return {"psnr": psnr_sum / len(cameras), "ssim": ssim_sum / len(cameras), "lpips": lpips_sum / len(cameras)}


def evaluate_xray_split(cameras, model, save_root=None):
    """Projection PSNR and SSIM averaged over the views, each pair first divided by its own maximum
    (R2-Gaussian's metric); `save_root` also writes the renders as 16-bit PNGs."""
    psnrs, ssims = [], []
    with torch.no_grad():
        for idx, cam in enumerate(cameras):
            img = render_xray_unified(cam, model)["render"]
            gt = cam.original_image
            if save_root:
                os.makedirs(save_root, exist_ok=True)
                arr = np.clip(img.detach().cpu().squeeze().numpy(), 0.0, 1.0)
                imageio.imwrite(osp.join(save_root, f"proj_test_{idx:04d}.png"), (arr * 65535.0).astype(np.uint16))
            if gt.max() > 0:
                gt, img = gt / gt.max(), img / img.max()
                psnrs.append(psnr(gt[None], img[None]))
                ssims.append(ssim(gt[None], img[None]))
    return ((torch.sum(torch.tensor(psnrs)) / max(len(psnrs), 1)).item(),
            (torch.sum(torch.tensor(ssims)) / max(len(ssims), 1)).item())


def evaluate_xray_3d(model, scanner_cfg, vol_gt):
    """3D PSNR over the volume, and SSIM averaged over the slices with signal along each axis (R2-Gaussian's metric)."""
    vol = model.query_xray_volume(scanner_cfg["offOrigin"], scanner_cfg["nVoxel"], scanner_cfg["sVoxel"])
    psnr_3d = 10 * torch.log10(1.0 / torch.mean((vol_gt - vol) ** 2).float())
    ssims = []
    for axis in (0, 1, 2):
        results, count = [], 0
        for gt, pred in zip(vol_gt.movedim(axis, 0), vol.movedim(axis, 0)):
            if gt.max() > 0:
                results.append(ssim(gt[None, None], pred[None, None]))
                count += 1
            else:
                results.append(0)
        ssims.append((torch.sum(torch.tensor(results)) / max(count, 1)).item())
    return {"psnr_3d": psnr_3d.item(), "ssim_3d": float(np.mean(ssims))}


# ---- One metrics file per iteration, with a section per writer (eval.py, training) ----
def update_metrics_file(evaluation_dir, iteration, **sections):
    """Merge `sections` (name -> dict) into the file for `iteration`; returns its path."""
    os.makedirs(evaluation_dir, exist_ok=True)
    path = osp.join(evaluation_dir, f"metrics_{iteration}.json")
    data = {}
    if osp.exists(path):
        with open(path) as fh:
            data = json.load(fh)
    data["iteration"] = iteration
    for name, values in sections.items():
        data.setdefault(name, {}).update(values)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    return path
