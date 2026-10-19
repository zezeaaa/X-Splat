# MVSFormer (vendored, inference path only)

A trimmed copy of **MVSFormer** (https://github.com/ewrfcas/MVSFormer), used solely to
build the multi-view-stereo point cloud that initialises the RGB Gaussians
(`xsplat/preprocess/mvs.py`). Please cite the MVSFormer paper if you use it.

* MVSFormer is licensed under the **Apache License 2.0**; its text is in `LICENSE`
  in this directory.
* Modified from upstream: trimmed to the inference path (the training-side modules
  were removed; `mvs_estimator.py` and `fusion.py` keep inference and depth fusion, by
  way of [MVPGS](https://github.com/zezeaaa/MVPGS)), and `configs/config_mvsformer.json` cut
  to the inference settings.
* `mvsformer/vision_transformer.py` derives from Meta's DINO / timm and keeps its
  Apache-2.0 header; `mvsformer/gvt.py` derives from Twins.
* Pretrained weights are **not** redistributed. Download `MVSFormer-Blended/best.pth`
  from the upstream repository to `third_party/mvsformer/pretrained/MVSFormer-Blended/`
  (the path `model_path` in `configs/config_mvsformer.json` points at).
