#
# Derived from 3D Gaussian Splatting (Inria / MPII), and from R2-Gaussian.
# Copyright (C) 2023, Inria, GRAPHDECO research group.
# Licensed under the Gaussian-Splatting License; see LICENSE.md.
# For inquiries contact george.drettakis@inria.fr
#
# The SSIM implementation follows pytorch-ssim (Evan Su, 2017, MIT).
#
"""The training losses: L1, SSIM and total variation, and the two that tie the RGB Gaussians to the CT surface."""
from math import exp

import torch
import torch.nn.functional as F


# ---- L1, SSIM and total variation, as in 3D Gaussian Splatting and R2-Gaussian ----
def l1_loss(network_output, gt):
    return torch.abs(network_output - gt).mean()


def tv_3d_loss(vol):
    """Mean absolute difference between neighbouring voxels, along the three axes."""
    tv = torch.abs(torch.diff(vol, dim=0)).sum() + torch.abs(torch.diff(vol, dim=1)).sum() + torch.abs(torch.diff(vol, dim=2)).sum()
    n0, n1, n2 = vol.shape
    return tv / ((n0 - 1) * n1 * n2 + n0 * (n1 - 1) * n2 + n0 * n1 * (n2 - 1))


def ssim(img1, img2, window_size=11):
    """SSIM under a Gaussian window (sigma 1.5), averaged over the image; (C, H, W) or (N, C, H, W)."""
    channel = img1.size(-3)
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * 1.5 ** 2)) for x in range(window_size)])
    window_1d = (gauss / gauss.sum()).unsqueeze(1)
    window = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device, img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


# ---- Losses that tie the RGB Gaussians to the CT surface ----
def outside_alpha_loss(alpha, rgb_mask, ct_mask, dilate):
    """Mean opacity where both the RGB mask and the CT silhouette (dilated by `dilate` px) say background; all (1, H, W)."""
    silhouette = F.max_pool2d(ct_mask.unsqueeze(0), kernel_size=2 * dilate + 1, stride=1, padding=dilate).squeeze(0)
    return (alpha * (1.0 - torch.maximum(silhouette, rgb_mask))).mean()


def ct_guided_depth_smooth_loss(depth_rgb, depth_ct, mask):
    """Scale-invariant depth smoothness: the RGB depth's gradients are penalised where the CT depth is
    smooth and allowed at its edges; all three (1, H, W), gradients only where both pixels are foreground."""
    depth_ct = depth_ct.detach()
    mask_bool = mask > 0.5
    mask_x = mask_bool[:, :, :-1] & mask_bool[:, :, 1:]
    mask_y = mask_bool[:, :-1, :] & mask_bool[:, 1:, :]
    grad_rgb_x = (depth_rgb[:, :, 1:] - depth_rgb[:, :, :-1]).abs()
    grad_rgb_y = (depth_rgb[:, 1:, :] - depth_rgb[:, :-1, :]).abs()
    grad_ct_x = (depth_ct[:, :, 1:] - depth_ct[:, :, :-1]).abs()
    grad_ct_y = (depth_ct[:, 1:, :] - depth_ct[:, :-1, :]).abs()

    # the mean CT depth sets the scale; tau, the median CT gradient above a relative floor, what counts as an edge
    if mask_bool.sum() > 0:
        mean_depth = depth_ct[mask_bool].mean().clamp(min=1e-3).detach()
    else:
        mean_depth = torch.tensor(1.0, device=depth_rgb.device)
    rel_threshold = mean_depth * 1e-4
    valid_ct_grad = torch.cat([grad_ct_x[mask_x], grad_ct_y[mask_y]])
    ct_grad_pos = valid_ct_grad[valid_ct_grad > rel_threshold]
    if ct_grad_pos.numel() > 0:
        tau = ct_grad_pos.median().clamp(min=rel_threshold)
    else:
        tau = mean_depth * 1e-3

    smooth_x = torch.exp(-grad_ct_x / tau) * grad_rgb_x / mean_depth
    smooth_y = torch.exp(-grad_ct_y / tau) * grad_rgb_y / mean_depth
    return smooth_x[mask_x].sum() / mask_x.sum().clamp(min=1.0).float() + smooth_y[mask_y].sum() / mask_y.sum().clamp(min=1.0).float()
