"""X-Splat training entry point.

    python train.py dataset=insideout data.scene_root=data/processed/insideout/skull

Hyper-parameters live in configs/. Results go to outputs/<dataset>/<scene>/ (or
output.scene_dir), with the resolved config snapshotted there as config.yaml.
"""
import os
import os.path as osp

import hydra
from omegaconf import OmegaConf

from xsplat.scene import UnifiedScene
from xsplat.training import training
from xsplat.utils import seed_everything


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    seed_everything(cfg.common.seed)
    scene_name = osp.basename(osp.normpath(osp.abspath(cfg.data.scene_root)))
    cfg.output.scene_dir = osp.abspath(cfg.output.scene_dir or osp.join("outputs", str(cfg.dataset_meta.name), scene_name))
    os.makedirs(cfg.output.scene_dir, exist_ok=True)
    scene = UnifiedScene(cfg)
    OmegaConf.save(cfg, osp.join(scene.scene_dir, "config.yaml"))

    total_iterations = cfg.training.xray_pretrain_iterations + cfg.training.joint_iterations
    test_iterations = sorted(set(cfg.common.test_iterations) | {total_iterations})   # the last one always
    print(f"[train] {scene.scene_dir}: {cfg.training.xray_pretrain_iterations} X-ray + "
          f"{cfg.training.joint_iterations} joint iterations, seed {cfg.common.seed}")
    training(cfg, scene, test_iterations)


if __name__ == "__main__":
    main()
