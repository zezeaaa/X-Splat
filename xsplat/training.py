"""Two-stage training: Stage 1 fits the X-ray Gaussians alone; Stage 2 aligns the modalities,
defines the surface set, and optimises RGB appearance and X-ray density jointly."""
import json
import os
import os.path as osp
from random import randint

import numpy as np
import torch
from tqdm import tqdm

from xsplat.geometry.alignment import run_stage1_realign
from xsplat.geometry.surface import (CTSurface, downsample_pcd, init_rgb_from_mvs, point_to_mesh_distance,
                                     surface_band_opacity)
from xsplat.losses import ct_guided_depth_smooth_loss, l1_loss, outside_alpha_loss, ssim, tv_3d_loss
from xsplat.metrics import evaluate_rgb_split, evaluate_xray_3d, evaluate_xray_split, update_metrics_file
from xsplat.model.gaussian_model import UnifiedGaussianModel
from xsplat.render import generate_virtual_cameras, render_rgb_unified, render_xray_unified


def xray_step(xray_cam, unified_model, xray_opt, tv):
    """One X-ray forward pass and its losses, for both stages. `tv` is the total-variation setup
    or None; its random sub-volume draw is part of the seeded RNG stream, so one call per
    iteration. Returns (losses, render_pkg)."""
    render_pkg = render_xray_unified(xray_cam, unified_model)
    image = render_pkg["render"]
    gt_image = xray_cam.original_image.cuda()
    losses = {"render": l1_loss(image, gt_image)}
    losses["total"] = losses["render"]
    if xray_opt.lambda_dssim > 0:
        losses["dssim"] = 1.0 - ssim(image, gt_image)
        losses["total"] = losses["total"] + xray_opt.lambda_dssim * losses["dssim"]
    if tv is not None:
        bbox, tv_vol_sVoxel, tv_vol_nVoxel = tv
        b0, b1 = bbox[0].to(tv_vol_sVoxel.device), bbox[1].to(tv_vol_sVoxel.device)
        tv_vol_center = (b0 + tv_vol_sVoxel / 2) + (b1 - tv_vol_sVoxel - b0) * torch.rand(3, device=tv_vol_sVoxel.device)
        vol_pred = unified_model.query_xray_volume(tv_vol_center, tv_vol_nVoxel, tv_vol_sVoxel)
        losses["tv"] = tv_3d_loss(vol_pred)
        losses["total"] = losses["total"] + xray_opt.lambda_tv * losses["tv"]
    return losses, render_pkg


def rgb_step(rgb_cam, geo_cam, unified_model, background, alignment_transform, ct_surface, rgb_opt, loss_cfg):
    """Stage 2's RGB pass: on a training view, L1 + D-SSIM on the foreground (gt is already masked)
    and the penalty on opacity outside the RGB mask and the CT silhouette; from a virtual camera,
    the rendered depth kept smooth where the CT surface's depth is. Returns (losses, render_pkg)."""
    render_pkg = render_rgb_unified(rgb_cam, unified_model, background, alignment_transform=alignment_transform)
    rgb_mask = rgb_cam.gt_alpha_mask
    image, gt_image = render_pkg["render"] * rgb_mask, rgb_cam.original_image * rgb_mask
    l1 = l1_loss(image, gt_image)
    _, ct_mask = ct_surface.view(rgb_cam)
    outside_alpha = outside_alpha_loss(render_pkg["render_alphas"], rgb_mask, ct_mask, loss_cfg.ct_mask_dilate_radius)
    render_loss = ((1.0 - rgb_opt.lambda_dssim) * l1 + rgb_opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                   + loss_cfg.rgb_outside_alpha_weight * outside_alpha)

    geo_depth = render_rgb_unified(geo_cam, unified_model, background, alignment_transform=alignment_transform,
                                   render_depth=True)["depth"]
    ct_depth, ct_mask = ct_surface.view(geo_cam)
    depth_smooth = ct_guided_depth_smooth_loss(geo_depth, ct_depth, ct_mask)
    return {"total": render_loss, "l1": l1, "outside_alpha_reg": outside_alpha, "depth_smooth": depth_smooth}, render_pkg


def training(cfg, scene, test_iterations):
    xray_pretrain_iters = cfg.training.xray_pretrain_iterations
    total_iterations = xray_pretrain_iters + cfg.training.joint_iterations
    gs_opt, xray_opt, rgb_opt = cfg.gs_optimization, cfg.xray_optimization, cfg.rgb_optimization
    scanner_cfg = scene.scanner_cfg
    volume_to_world = max(scanner_cfg["sVoxel"])
    scale_bound = np.array([cfg.xray_model.scale_min, cfg.xray_model.scale_max]) * volume_to_world
    densify_scale_threshold = xray_opt.densify_scale_threshold * volume_to_world

    # the model starts from the FDK initialisation, with the X-ray parameter groups only
    unified_model = UnifiedGaussianModel(sh_degree=cfg.rgb_model.sh_degree, scale_bound=scale_bound)
    point_cloud = np.load(scene.xray_init_pcd_path)
    xyz, density = downsample_pcd(point_cloud[:, :3], point_cloud[:, 3:4], cfg.init.max_init_points)
    print(f"[init] {xyz.shape[0]} points from {scene.xray_init_pcd_path}")
    unified_model.create_from_pcd(xyz, density)
    unified_model.setup_unified_optimizer(gs_opt)

    tv = None
    if xray_opt.lambda_tv > 0:   # total variation on a random sub-volume of tv_vol_size voxels a side
        tv_vol_nVoxel = torch.tensor([xray_opt.tv_vol_size] * 3, device="cuda")
        tv = (scene.bbox, torch.tensor(scanner_cfg["dVoxel"], device="cuda") * tv_vol_nVoxel, tv_vol_nVoxel)

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    xray_viewpoint_stack, rgb_viewpoint_stack, geo_viewpoint_stack = [], [], []
    progress_bar = tqdm(range(total_iterations), desc="Training", leave=False)
    metrics_file = open(osp.join(scene.scene_dir, "metrics.jsonl"), "w", buffering=1)
    alignment_transform = None

    for iteration in range(1, total_iterations + 1):
        stage2_iter = iteration - xray_pretrain_iters
        stage2 = stage2_iter > 0
        if stage2_iter == 1:
            # Stage 1 -> Stage 2: align the MVS cloud to the X-ray surface, give the Gaussians
            # appearance, and set up the virtual cameras and the CT surface of the depth loss
            print("[realign] Aligning the MVS point cloud to the Stage-1 X-ray surface")
            mvs_dir = osp.join(scene.rgb_source_path, "mvs_points")
            T = run_stage1_realign(unified_model, scanner_cfg, osp.join(mvs_dir, "mvs_fused_uniform.ply"),
                                   scene.alignment_dir, mvs_color_points_path=osp.join(mvs_dir, "mvs_fused.ply"),
                                   iso_level=cfg.ct_surface.iso_level, surface_samples=cfg.realign.surface_samples,
                                   voxel_size=cfg.realign.voxel_size, max_corr_factor=cfg.realign.max_corr_factor,
                                   max_iters=cfg.realign.max_iters)
            alignment_transform = torch.from_numpy(T).float().cuda()
            init_rgb_appearance(cfg, unified_model, rgb_opt, scene.alignment_dir)
            geo_cam_pool = generate_virtual_cameras(cfg.virtual_depth, scanner_cfg, alignment_transform)
            ct_surface = CTSurface(scanner_cfg, cfg.ct_surface.iso_level, alignment_transform)

        unified_model.update_learning_rate(iteration)
        if stage2 and stage2_iter % 1000 == 0:   # one more SH degree every 1000 iterations
            unified_model.oneupSHdegree()

        if not xray_viewpoint_stack:
            xray_viewpoint_stack = list(scene.xray_train_cameras)
        xray_cam = xray_viewpoint_stack.pop(randint(0, len(xray_viewpoint_stack) - 1))
        xray_loss, xray_pkg = xray_step(xray_cam, unified_model, xray_opt, tv)
        total_loss = cfg.training.xray_loss_weight * xray_loss["total"]
        rgb_loss = None
        if stage2:
            # re-extract the CT surface every update_interval iterations, from the first
            if (stage2_iter - 1) % cfg.ct_surface.update_interval == 0:
                ct_surface.update(unified_model)
            if not rgb_viewpoint_stack:
                rgb_viewpoint_stack = list(scene.rgb_train_cameras)
            rgb_cam = rgb_viewpoint_stack.pop(randint(0, len(rgb_viewpoint_stack) - 1))
            if not geo_viewpoint_stack:
                geo_viewpoint_stack = list(geo_cam_pool)
            geo_cam = geo_viewpoint_stack.pop(randint(0, len(geo_viewpoint_stack) - 1))
            rgb_loss, render_pkg = rgb_step(rgb_cam, geo_cam, unified_model, background, alignment_transform, ct_surface,
                                            rgb_opt, cfg.losses)
            total_loss = (total_loss + cfg.training.rgb_loss_weight * rgb_loss["total"]
                          + cfg.losses.depth_smooth_weight * rgb_loss["depth_smooth"])

        assert torch.isfinite(total_loss), f"total_loss is not finite: {total_loss.item()}"
        total_loss.backward()
        log_metrics(metrics_file, "stage2" if stage2 else "stage1", iteration, total_loss, xray_loss, rgb_loss, unified_model)

        with torch.no_grad():
            unified_model.add_densification_stats_xray(xray_pkg)
            if stage2:
                unified_model.add_densification_stats_rgb(render_pkg)
            if iteration % gs_opt.densification_interval == 0:
                unified_model.prune_nonfinite()   # through the whole run, so no checkpoint carries NaN Gaussians
                if gs_opt.densify_from_iter < iteration < gs_opt.densify_until_iter:
                    if stage2:
                        unified_model.densify_and_prune_rgb_driven(rgb_opt.densify_grad_threshold, xray_opt.densify_grad_threshold,
                                                                   rgb_opt.opacity_min_threshold, xray_opt.density_min_threshold,
                                                                   densify_scale_threshold)
                        # surface Gaussians whose opacity has collapsed go back to the interior
                        unified_model.demote_low_opacity_surface(float(cfg.surface_set.demotion_opacity_threshold))
                    else:
                        unified_model.densify_and_prune_xray(xray_opt.densify_grad_threshold, xray_opt.density_min_threshold,
                                                             densify_scale_threshold)
        unified_model.unified_optimizer.step()
        unified_model.unified_optimizer.zero_grad(set_to_none=True)

        if iteration % 10 == 0 or iteration == total_iterations:
            postfix = {"stage": "S2" if stage2 else "S1", "pts": f"{unified_model._xyz.shape[0]:,}",
                       "loss": f"{total_loss.item():.4f}", "xray": f"{xray_loss['total'].item():.4f}"}
            if rgb_loss is not None:
                postfix["rgb"], postfix["dsm"] = f"{rgb_loss['total'].item():.4f}", f"{rgb_loss['depth_smooth'].item():.4f}"
            progress_bar.set_postfix(postfix)
        progress_bar.update(1)
        if iteration in test_iterations:
            evaluate_unified_model(iteration, unified_model, scene, background, alignment_transform)
    progress_bar.close()
    metrics_file.close()

    save_unified_training_artifacts(total_iterations, unified_model, scene)
    update_metrics_file(scene.evaluation_dir, total_iterations, model=unified_model.param_stats())  # model size, with the metrics


def init_rgb_appearance(cfg, unified_model, rgb_opt, alignment_dir):
    """Give the Stage-1 Gaussians RGB appearance: colours from the aligned MVS cloud, opacity from
    the distance to the CT surface; defines the surface set."""
    xyz = unified_model._xyz.detach().cpu().numpy()
    init_rgb = init_rgb_from_mvs(osp.join(alignment_dir, "mvs_aligned.ply"), xyz, cfg.init.mvs_color_max_dist)
    surface_dists = point_to_mesh_distance(osp.join(alignment_dir, "xray_mesh.ply"), xyz)
    init_opacity = surface_band_opacity(surface_dists, cfg.surface_set.max_dist, surface_opacity=cfg.surface_set.opacity,
                                        band_end_opacity=cfg.surface_set.band_end_opacity,
                                        interior_opacity=cfg.surface_set.interior_opacity,
                                        surface_band_sigma_ratio=cfg.surface_set.band_sigma_ratio)
    unified_model.add_rgb_params(rgb_opt, init_rgb, init_opacity)
    unified_model.set_surface_set(surface_dists < float(cfg.surface_set.rgb_mask_max_dist))


def evaluate_unified_model(iteration, unified_model, scene, background, alignment_transform):
    """Print test-split metrics during training (eval.py writes the final ones)."""
    print(f"\n[ITER {iteration}] Evaluating unified model...")
    if scene.xray_test_cameras:
        p, s = evaluate_xray_split(scene.xray_test_cameras, unified_model)
        print(f"  X-ray test: PSNR={p:.3f}, SSIM={s:.3f}")
    if scene.vol_gt is not None:
        m3d = evaluate_xray_3d(unified_model, scene.scanner_cfg, scene.vol_gt)
        print(f"  X-ray 3D: PSNR={m3d['psnr_3d']:.3f}, SSIM={m3d['ssim_3d']:.3f}")
    if scene.rgb_test_cameras and unified_model._features_dc is not None:
        m = evaluate_rgb_split(scene.rgb_test_cameras, unified_model, background, alignment_transform=alignment_transform)
        print(f"  RGB test: PSNR={m['psnr']:.3f} SSIM={m['ssim']:.3f} LPIPS={m['lpips']:.4f}")
    torch.cuda.empty_cache()


def save_unified_training_artifacts(iteration, unified_model, scene):
    """Write the checkpoint, plus gaussians_rgb.ply (3DGS viewers) and gaussians_xray.pickle (R2-Gaussian)."""
    os.makedirs(scene.checkpoints_dir, exist_ok=True)
    ckpt_path = osp.join(scene.checkpoints_dir, f"iteration_{iteration}.pickle")
    unified_model.save_checkpoint(ckpt_path)
    tqdm.write(f"[ITER {iteration}] Saved checkpoint to {ckpt_path}")
    os.makedirs(scene.exports_dir, exist_ok=True)
    unified_model.export_to_xray_model(osp.join(scene.exports_dir, "gaussians_xray.pickle"))
    unified_model.export_to_rgb_model(osp.join(scene.exports_dir, "gaussians_rgb.ply"))
    tqdm.write(f"  Saved exports to {scene.exports_dir}")


def log_metrics(fh, stage, iteration, total_loss, xray_loss, rgb_loss, model):
    """One JSON line per iteration in metrics.jsonl: the losses and the Gaussian count."""
    row = {"stage": stage, "iter": int(iteration), "total": float(total_loss), "n_gaussians": int(model.get_xyz.shape[0])}
    for prefix, losses in (("xray", xray_loss), ("rgb", rgb_loss)):
        for key, value in (losses or {}).items():
            row[f"{prefix}_{key}"] = float(value)
    fh.write(json.dumps(row, sort_keys=True) + "\n")
