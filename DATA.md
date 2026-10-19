# Data

Training reads one scene package per scene. The insect scenes can be downloaded; for
InsideOut and Flower you need to obtain the originals and build the packages yourself:

```
data/source/<dataset>/              originals you obtain yourself (InsideOut, Flower)
    |  python scripts/prepare_<dataset>.py
    v
data/raw/<dataset>/<scene>/         raw scene: CT volume, RGB images, masks
    |  python preprocess.py --dataset <dataset>
    v
data/processed/<dataset>/<scene>/   scene package, read by training
```

Building packages needs the [data preparation tools](README.md#data-preparation-tools)
from the README; the downloaded insect scene packages do not.

## Downloads

On [Hugging Face](https://huggingface.co/datasets/zezeaaa/X-Splat):

| Archive | Contents |
|---|---|
| [**insect_processed**](https://huggingface.co/datasets/zezeaaa/X-Splat/resolve/main/xsplat_insect_processed.zip) | insect scene packages, ready to train |
| [**insideout_mask_annotations**](https://huggingface.co/datasets/zezeaaa/X-Splat/resolve/main/xsplat_insideout_mask_annotations.zip) | our foreground masks for the InsideOut scenes |
| [**checkpoints_and_results**](https://huggingface.co/datasets/zezeaaa/X-Splat/resolve/main/xsplat_checkpoints_and_results.zip) | our trained models (insect, Flower) and the metrics of all nine scenes |

Unpack each into `data/`, or download and unpack all three from the repository root:

```bash
for a in insect_processed insideout_mask_annotations checkpoints_and_results; do
  wget -q -O xsplat_$a.zip https://huggingface.co/datasets/zezeaaa/X-Splat/resolve/main/xsplat_$a.zip
  unzip -qo xsplat_$a.zip -d data/ && rm xsplat_$a.zip
done
```

A downloaded model is evaluated like a trained one, e.g. `python eval.py data/checkpoints/insect/E02`; it needs
its scene package in `data/processed/` as well, and the metrics it shipped with are in
`data/metrics/release_run/`.

## Insect

Three beetle specimens (E02, E15, E32) that we captured, with 36 photographs and a CT
scan each.

`insect_processed` holds their scene packages, prepared the same way as for the paper's
experiments. Besides the photographs and their masks, each package holds the CT volume
itself, as `xray/vol_gt.npy`.

## InsideOut

Request the data from the InsideOut authors (Lee et al., ICCV 2025) through the form
linked as **Data** on their [project page](https://leejngmn.github.io/InsideOut.github.io/),
and place it as follows (the DNG files are not needed):

```
data/source/insideout/RGB/<scene>/*.JPG   (or .jpg, as they come)
data/source/insideout/CT/<scene>.raw, <scene>.txt
```

Then unpack our `insideout_mask_annotations` into `data/` and run:

```bash
python scripts/prepare_insideout.py --scenes skull
python preprocess.py --dataset insideout --scenes skull
```

Leave out `--scenes` to prepare all four scenes.

COLMAP is not deterministic: a scene sometimes registers other views than ours (toy_gun, for
instance, 104 instead of 101), and since every fourth view is a test view, the test split changes
with them. Ours are 53, 66, 49 and 101 registered views for matryoshka, pharaoh, skull and toy_gun
(`preprocess.py` prints `[COLMAP] <n> of <m> views registered`). If one differs, run COLMAP again
until it matches (it can take a few tries), then MVS:

```bash
python preprocess.py --dataset insideout --scenes toy_gun --steps rgb
python preprocess.py --dataset insideout --scenes toy_gun --steps mvs
```

## Flower

Download the CT volume (`volume.zip`) and the Blender scene with its textures (`blend.zip`) of
**Tulip** and **Eustoma** from the [RIKEN flower CT library](https://www2.riken.jp/brict/Ijiri/flowerctlib/index.html):

```bash
for s in mRay_Tulip/20140110_mRay_Tulip Yxlon_Eustoma/20121031_Yxlon_Eustoma; do
  mkdir -p data/source/flower/${s%/*}
  for f in volume blend; do
    wget -q -O $f.zip https://www2.riken.jp/brict/Ijiri/flowerctlib/${s#*/}/$f.zip
    unzip -qo $f.zip -d data/source/flower/${s%/*} && rm $f.zip
  done
done
```

This leaves `data/source/flower/mRay_Tulip/20140110_mRay_Tulip.traw3D_ss` and the Blender scene
in `data/source/flower/mRay_Tulip/20140110_mRay_Tulip/`, and the same for Eustoma.

Then render the RGB images with Blender and build the two packages, which are written to
`data/processed/flower/mRay_Tulip_20140110_mRay_Tulip/` and
`data/processed/flower/Yxlon_Eustoma_20121031_Yxlon_Eustoma/`:

```bash
python scripts/prepare_flower.py --blender /path/to/blender-3.6.5-linux-x64/blender
python preprocess.py --dataset flower
```

## Your own data

You need photographs of the object from many directions, a foreground mask for each,
and a CT volume or X-ray projections of it:

```
data/raw/custom/scene01/
├── rgb/images/*.png     the photographs
├── rgb/masks/*.png      one per image, same filename stem and size, white = object
├── ct/volume.npy        a CT volume in [0, 1] with cubic voxels, any shape
└── meta.yaml            pose_mode: reconstruct
```

`preprocess.py` simulates the X-ray projections from the CT volume with TIGRE, as it
does for the published scenes. If you have projections instead, replace `ct/` with an
`xray/` directory holding `proj_train/*.npy` and `proj_test/*.npy` (float32, one
projection per file) and `meta_data.json`. The JSON gives the scanner geometry and each
projection's `file_path` and `angle` in radians, laid out like
`data/processed/insect/E02/xray/meta_data.json`.

`pose_mode: reconstruct` recovers the camera poses with COLMAP. If you know them, use
`pose_mode: colmap` with an undistorted COLMAP model (`cameras`, `images`, `points3D`)
at `rgb/sparse/0/`, or `pose_mode: known_pose` with a NeRF-style `rgb/transforms.json`.

Edit `configs/preprocess/custom.yaml`, `configs/dataset/custom.yaml` and, for a CT volume,
`configs/scanner/custom.yaml` (see the comments in each), then check the scene and run:

```bash
python scripts/check_raw_scene.py data/raw/custom/scene01
python preprocess.py --dataset custom --scenes scene01
python train.py dataset=custom data.scene_root=data/processed/custom/scene01
```

`ct_surface.iso_level` in `configs/dataset/custom.yaml` is the density at which the
object's surface is extracted; the right value depends on your scan. To check it, open
the run's `alignment/xray_mesh.ply` and `alignment/mvs_aligned.ply` together: they
should line up.

## Licensing and citation

* **Insect**, our InsideOut masks and our checkpoints: CC BY-NC 4.0. Please cite X-Splat
  (and, for a checkpoint, the dataset it was trained on).
* **InsideOut** and **Flower** are not redistributed; their own terms apply. Please cite
  them:

```bibtex
@inproceedings{lee2025insideout,
  title     = {InsideOut: Integrated RGB-Radiative Gaussian Splatting for Comprehensive 3D Object Representation},
  author    = {Lee, Jungmin and Hong, Seonghyuk and Lee, Juyong and Lee, Jaeyoon and Choi, Jongwon},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {25820--25830},
  year      = {2025}
}

@article{ijiri2014flower,
  title   = {Flower modeling via X-ray computed tomography},
  author  = {Ijiri, Takashi and Yoshizawa, Shin and Yokota, Hideo and Igarashi, Takeo},
  journal = {ACM Transactions on Graphics},
  volume  = {33},
  number  = {4},
  pages   = {1--10},
  year    = {2014}
}
```
