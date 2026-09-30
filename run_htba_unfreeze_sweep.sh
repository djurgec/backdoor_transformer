#!/usr/bin/env bash
set -x
set -e

GPU=${1:-0}
shift || true
BLOCKS=${@:-"1 2 4 12"}

NUM_POISON=400
EXP_CFG=cfg/imagenette/experiment_imagenette.cfg


for k in $BLOCKS; do
        cfg="cfg/imagenette/_htba_uf${k}.cfg"
        sed -e "s/^attack=.*/attack=htba/" \
            -e "s/^num_poison_badnets=.*/num_poison_badnets=0/" \
            -e "s/^num_poison_htba=.*/num_poison_htba=${NUM_POISON}/" \
            -e "s/^rand_loc=.*/rand_loc=true/" \
            -e "s/^feature_extract=.*/feature_extract=true/" \
            -e "s/^unfreeze_blocks=.*/unfreeze_blocks=${k}/" \
            -e "s/^entropy_weight=.*/entropy_weight=0.0/" \
            -e "s/^optimizer=.*/optimizer=sgd/" \
            -e "s/^lr=.*/lr=1e-4/" \
            -e "s/^head_lr_mult=.*/head_lr_mult=10.0/" \
            -e "s/^rollout_every_epoch=.*/rollout_every_epoch=0/" \
            -e "s/^train_clean_model=.*/train_clean_model=false/" \
            "$EXP_CFG" > "$cfg"

        CUDA_VISIBLE_DEVICES=$GPU python train_backdoor.py    "$cfg"
        CUDA_VISIBLE_DEVICES=$GPU python test_time_defense.py "$cfg"
done

