"""MVSFormer inference for a set of views, and the fusion of the per-view depths into one point
cloud. Adapted from MVPGS / MVSFormer (NOTICE.md); the training side and the depth visualisations
are gone."""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from .utils import read_json
from .mvsformer.mvsformer_model import TwinMVSNet, DINOMVSNet
from .fusion import ave_fusion, bin_op_reduce, get_pixel_grids, get_reproj, idx_cam2world, idx_img2cam, vis_filter


class MvsEstimator:
    def __init__(self, config_path, scale_bds, root):
        """`scale_bds` scales each batch's (near, far) depth range; a relative model_path in the
        config is taken from `root`."""
        self.config = read_json(config_path)
        self.scale_bds = scale_bds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = Path(self.config['model_path'])
        self.model = self.create_model(self.config, model_path if model_path.is_absolute() else Path(root) / model_path)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_mvs_pts(self, views, src_num):
        """A depth map per view from its `src_num` nearest views (by camera centre), fused across
        views: returns (points (N, 3) float32, colors (N, 3) uint8)."""
        depths, confidences, cams, imgs = [], [], [], []
        for ref in tqdm(range(len(views)), desc='MVS Estimation'):
            dists = sorted((np.linalg.norm(view.center - views[ref].center), i) for i, view in enumerate(views) if i != ref)
            batch = [views[ref]] + [views[i] for _, i in dists[:src_num]]
            batch_imgs, ori_imgs, proj_matrices, ref_depth_bins = self.prepare_mvs_input(batch)
            with torch.no_grad():
                outputs = self.model.forward(batch_imgs, self.get_ms_proj_mats(proj_matrices), ref_depth_bins, tmp=self.config['tmp'])
            depths.append(outputs["refined_depth"])
            confidences.append(outputs["photometric_confidence"])
            cams.append(proj_matrices[:, 0, :, :, :])
            imgs.append(ori_imgs[0])
        return self.merge_mvs_depths(imgs, depths, confidences, cams, views, self.config['prob_threshold'])

    def merge_mvs_depths(self, imgs, depths, confidences, proj_cams, views, prob_threshold):
        points_all, colors_all = [], []
        for ref in range(len(views)):
            ref_depth = depths[ref].unsqueeze(0)
            ref_prob_mask = confidences[ref].unsqueeze(0) > prob_threshold[0]
            # the reference view's foreground mask, at the depth map's resolution
            h, w = ref_depth.shape[-2:]
            fg = views[ref].mask
            if fg.shape != (h, w):
                fg = cv2.resize(fg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
            ref_prob_mask = ref_prob_mask & torch.from_numpy(fg).to(ref_prob_mask.device).unsqueeze(0).unsqueeze(0)

            src_depths = torch.stack(depths[:ref] + depths[ref + 1:], dim=0).unsqueeze(0)
            src_cams = torch.stack(proj_cams[:ref] + proj_cams[ref + 1:], dim=0).unsqueeze(0)
            src_confs = torch.stack(confidences[:ref] + confidences[ref + 1:], dim=0).unsqueeze(0)
            src_depths *= (src_confs > prob_threshold[0]).float()

            reproj_xyd, in_range = get_reproj(ref_depth, src_depths, proj_cams[ref], src_cams)
            vis_masks, vis_mask = vis_filter(ref_depth, reproj_xyd, in_range, self.config['thres_disp'], 0.01, self.config['thres_view'])
            ref_depth_ave = ave_fusion(ref_depth, reproj_xyd, vis_masks)
            mask = bin_op_reduce([ref_prob_mask, vis_mask], torch.min)

            idx_img = get_pixel_grids(*ref_depth_ave.size()[-2:]).unsqueeze(0)
            idx_cam = idx_img2cam(idx_img, ref_depth_ave, proj_cams[ref])
            points = idx_cam2world(idx_cam, proj_cams[ref])[..., :3, 0].permute(0, 3, 1, 2)

            keep = mask.cpu().data.numpy().astype(np.bool_)[0, 0] & fg
            points_np = points.cpu().data.numpy()[0]
            ref_img = imgs[ref].transpose(2, 0, 1)
            points_all.append(np.stack([points_np[k][keep] for k in range(3)], -1))
            colors_all.append((np.stack([ref_img[k][keep] for k in range(3)], -1) * 255).astype(np.uint8))
        return np.concatenate(points_all).astype(np.float32), np.concatenate(colors_all)

    def prepare_mvs_input(self, views, num_depths=192):
        imgs, ori_imgs, proj_matrices = [], [], []
        for view in views:
            ori_img, intrinsic = self.scale_mvs_input(np.array(view.image), view.K, self.config['max_w'], self.config['max_h'])
            imgs.append(self.transforms(Image.fromarray(ori_img)))
            ori_imgs.append(ori_img / 255.)
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = getWorld2View(view.R, view.T)
            proj_mat[1, :3, :3] = intrinsic
            proj_matrices.append(proj_mat)
        imgs = torch.stack(imgs, dim=0).unsqueeze(0).to(self.device)
        proj_matrices = torch.tensor(np.stack(proj_matrices, axis=0)).unsqueeze(0).to(self.device)

        bounds_all = np.array([view.bounds for view in views], dtype=np.float32)
        depth_min = bounds_all[:, 0].min() * self.scale_bds[0]
        depth_max = bounds_all[:, 1].max() * self.scale_bds[1]
        ref_depth_bins = torch.linspace(depth_min, depth_max, num_depths, dtype=torch.float32).unsqueeze(0).to(self.device)
        return imgs, ori_imgs, proj_matrices, ref_depth_bins

    def scale_mvs_input(self, img, intrinsics, max_w, max_h):
        new_intrinsics = intrinsics.copy()
        h, w = img.shape[:2]
        new_intrinsics[0, :] *= 1.0 * max_w / w
        new_intrinsics[1, :] *= 1.0 * max_h / h
        return cv2.resize(np.ascontiguousarray(img), (int(max_w), int(max_h))), new_intrinsics

    def get_ms_proj_mats(self, proj_matrices):
        """The intrinsics at MVSFormer's five stages: 1/8, 1/4, 1/2, 1 and 1 of the input resolution."""
        proj_matrices_ms = {}
        for stage, scale in enumerate((0.125, 0.25, 0.5, 1.0, 1.0), start=1):
            mats = proj_matrices.clone()
            mats[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :] * scale
            proj_matrices_ms[f"stage{stage}"] = mats
        return proj_matrices_ms

    def create_model(self, config, model_path):
        if config['arch']['args']['vit_args'].get('twin', False):
            model = TwinMVSNet(config['arch']['args'])
        else:
            model = DINOMVSNet(config['arch']['args'])
        print('Loading checkpoint: {} ...'.format(model_path))
        if not Path(model_path).exists():
            raise FileNotFoundError(f"MVSFormer checkpoint not found: {model_path}. Download MVSFormer-Blended/best.pth "
                                    "(README, Data preparation tools), then rerun with --steps mvs.")
        checkpoint = torch.load(str(model_path), map_location=self.device)
        state_dict = {key.replace('module.', ''): val for key, val in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)
