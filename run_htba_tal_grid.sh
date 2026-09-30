#!/usr/bin/env bash
set -x
set -e

GPU=${1:-0}

BLOCKS=${BLOCKS:-"4"}
WEIGHTS=${WEIGHTS:-"-0.1 -0.5 -1.0"}
TOPK=${TOPK:-"8"}
ROLLOUT=${ROLLOUT:-0}

NUM_POISON=400
EXP_CFG=cfg/imagenette/experiment_imagenette.cfg

for b in $BLOCKS; do
for k in $TOPK; do
for w in $WEIGHTS; do
        cfg="cfg/imagenette/_htba_uf${b}_top${k}_tal${w}.cfg"
        sed -e "s/^attack=.*/attack=htba/" \
            -e "s/^num_poison_badnets=.*/num_poison_badnets=0/" \
            -e "s/^num_poison_htba=.*/num_poison_htba=${NUM_POISON}/" \
            -e "s/^rand_loc=.*/rand_loc=true/" \
            -e "s/^feature_extract=.*/feature_extract=true/" \
            -e "s/^unfreeze_blocks=.*/unfreeze_blocks=${b}/" \
            -e "s/^tal_topk=.*/tal_topk=${k}/" \
            -e "s/^tal_weight=.*/tal_weight=${w}/" \
            -e "s/^entropy_weight=.*/entropy_weight=0.0/" \
            -e "s/^rollout_every_epoch=.*/rollout_every_epoch=${ROLLOUT}/" \
            -e "s/^optimizer=.*/optimizer=sgd/" \
            -e "s/^lr=.*/lr=1e-4/" \
            -e "s/^head_lr_mult=.*/head_lr_mult=10.0/" \
            -e "s/^train_clean_model=.*/train_clean_model=false/" \
            "$EXP_CFG" > "$cfg"

        # sed is a no-op when the key is absent, which would silently drop the run
        # identity from the descriptor and land this run on another run's directory
        for kv in "tal_topk=${k}" "unfreeze_blocks=${b}" "tal_weight=${w}"; do
                grep -qx "$kv" "$cfg" || { echo "FATAL: $cfg has no line '$kv'"
                        echo "       the base cfg is out of date on this machine; git pull"
                        exit 1; }
        done

        CUDA_VISIBLE_DEVICES=$GPU python train_backdoor.py    "$cfg"
        CUDA_VISIBLE_DEVICES=$GPU python test_time_defense.py "$cfg"
done
done
done

