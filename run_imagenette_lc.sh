#!/usr/bin/env bash
set -x
set -e

GPU=${1:-0}
shift || true
WEIGHTS=${@:-"0.5 0.0 -0.5"}

NUM_POISON=200
DATASET_CFG=cfg/imagenette/dataset.cfg
EXP_CFG=cfg/imagenette/experiment_imagenette.cfg
SURROGATE=checkpoints/imagenette/badnets/n0_tal0_rand/clean_model.pt

python create_imagenet_filelist.py $DATASET_CFG

# surrogate: a plain clean model
if [ ! -f "$SURROGATE" ]; then
        cfg=cfg/imagenette/_lc_surrogate.cfg
        sed -e "s/^attack=.*/attack=badnets/" \
            -e "s/^num_poison_badnets=.*/num_poison_badnets=0/" \
            -e "s/^num_poison_lc=.*/num_poison_lc=0/" \
            -e "s/^tal_weight=.*/tal_weight=0.0/" \
            -e "s/^rand_loc=.*/rand_loc=true/" \
            -e "s/^train_clean_model=.*/train_clean_model=true/" \
            "$EXP_CFG" > "$cfg"
        CUDA_VISIBLE_DEVICES=$GPU python train_backdoor.py "$cfg"
fi

LC_CFG=cfg/imagenette/_lc_base.cfg
sed -e "s/^attack=.*/attack=lc/" \
    -e "s/^num_poison_badnets=.*/num_poison_badnets=0/" \
    -e "s/^num_poison_lc=.*/num_poison_lc=${NUM_POISON}/" \
    -e "s/^rand_loc=.*/rand_loc=false/" \
    -e "s/^train_clean_model=.*/train_clean_model=false/" \
    -e "s/^num_poison_generate=.*/num_poison_generate=${NUM_POISON}/" \
    "$EXP_CFG" > "$LC_CFG"

# craft the poisons
CUDA_VISIBLE_DEVICES=$GPU python generate_lc_poison.py "$LC_CFG"

for w in $WEIGHTS; do
        cfg="cfg/imagenette/_lc_tal_${w}.cfg"
        sed "s/^tal_weight=.*/tal_weight=${w}/" "$LC_CFG" > "$cfg"
        CUDA_VISIBLE_DEVICES=$GPU python train_backdoor.py    "$cfg"
        CUDA_VISIBLE_DEVICES=$GPU python test_time_defense.py "$cfg"
done

