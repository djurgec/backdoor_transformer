#!/usr/bin/env bash
set -x
set -e

GPU=${1:-0}

NUM_POISON=400
DATASET_CFG=cfg/imagenette/dataset.cfg
EXP_CFG=cfg/imagenette/experiment_imagenette.cfg
HTBA_CFG=cfg/imagenette/_htba.cfg

python create_imagenet_filelist.py $DATASET_CFG

sed -e "s/^attack=.*/attack=htba/" \
    -e "s/^num_poison_badnets=.*/num_poison_badnets=0/" \
    -e "s/^num_poison_lc=.*/num_poison_lc=0/" \
    -e "s/^num_poison_htba=.*/num_poison_htba=${NUM_POISON}/" \
    -e "s/^rand_loc=.*/rand_loc=true/" \
    -e "s/^feature_extract=.*/feature_extract=true/" \
    -e "s/^optimizer=.*/optimizer=sgd/" \
    -e "s/^lr=.*/lr=0.001/" \
    -e "s/^head_lr_mult=.*/head_lr_mult=1.0/" \
    -e "s/^train_clean_model=.*/train_clean_model=false/" \
    "$EXP_CFG" > "$HTBA_CFG"

# SKIP_GEN=1 runs the attack without the poison regeneration step
if [ "${SKIP_GEN:-0}" = "1" ]; then
        echo "SKIP_GEN=1, reusing existing poisons"
else
        CUDA_VISIBLE_DEVICES=$GPU python generate_htba_poison.py "$HTBA_CFG"
fi

CUDA_VISIBLE_DEVICES=$GPU python train_backdoor.py    "$HTBA_CFG"
CUDA_VISIBLE_DEVICES=$GPU python test_time_defense.py "$HTBA_CFG"
CUDA_VISIBLE_DEVICES=$GPU python analyze_rollout.py   "$HTBA_CFG"
