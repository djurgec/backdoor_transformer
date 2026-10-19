#!/usr/bin/env bash
# HTBA end to end: generate poisons, train, defend, visualise.
# SKIP_GEN=1 reuses the poisons already on disk.
set -e

GPU=${1:-0}
NUM_POISON=${NUM_POISON:-400}
BASE=cfg/imagenette/experiment_imagenette.cfg

python create_imagenet_filelist.py cfg/imagenette/dataset.cfg

if [ "${SKIP_GEN:-0}" = "1" ]; then
        STEPS="train defense rollout"
else
        STEPS="generate train defense rollout"
fi

STEPS="$STEPS" bash scripts/run.sh "$GPU" "$BASE" \
        attack=htba num_poison_badnets=0 num_poison_htba="$NUM_POISON" \
        rand_loc=true feature_extract=true \
        optimizer=sgd lr=0.001 head_lr_mult=1.0 train_clean_model=false
