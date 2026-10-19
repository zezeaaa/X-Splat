#!/usr/bin/env bash
# Train and evaluate the scenes below on one GPU (gpu_id, else CUDA_VISIBLE_DEVICES, else 0), one
# after another; scenes that have not been prepared are skipped, and all settings come from
# configs/. Edit gpu_id and scenes, or give them on the command line to run several GPUs at once, e.g.
#   gpu_id=1 bash scripts/train_all.sh insect/E02 insect/E15

gpu_id=${gpu_id:-${CUDA_VISIBLE_DEVICES:-0}}
scenes=(insideout/matryoshka insideout/pharaoh insideout/skull insideout/toy_gun
        flower/mRay_Tulip_20140110_mRay_Tulip flower/Yxlon_Eustoma_20121031_Yxlon_Eustoma
        insect/E02 insect/E15 insect/E32)
if [ $# -gt 0 ]; then scenes=("$@"); fi

for scene in "${scenes[@]}"; do
    if [ ! -d data/processed/$scene ]; then
        echo "========================= Skip $scene: no data/processed/$scene (see DATA.md) ========================="
        continue
    fi
    echo "========================= Train: $scene ========================="
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py dataset=${scene%%/*} data.scene_root=data/processed/$scene || continue

    echo "========================= Eval: $scene ========================="
    CUDA_VISIBLE_DEVICES=$gpu_id python eval.py outputs/$scene
done
