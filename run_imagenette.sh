#!/usr/bin/env bash
# End-to-end Imagenette backdoor pipeline.
# Usage: ./run_imagenette.sh <GPU_ID>   (defaults to 0)
set -x
set -e

GPU=${1:-0}
DATASET_CFG=cfg/imagenette/dataset.cfg
EXP_CFG=cfg/imagenette/experiment_imagenette.cfg

python create_imagenet_filelist.py $DATASET_CFG

CUDA_VISIBLE_DEVICES=$GPU python generate_poison_transformer.py $EXP_CFG

CUDA_VISIBLE_DEVICES=$GPU python finetune_transformer.py $EXP_CFG

CUDA_VISIBLE_DEVICES=$GPU python test_time_defense.py $EXP_CFG
