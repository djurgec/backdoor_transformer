#!/usr/bin/env bash
set -x
set -e

GPU=${1:-0}
EXP_CFG=cfg/imagenette/experiment_imagenette.cfg
SURROGATE=checkpoints/imagenette/badnets/n0_tal0_rand/clean_model.pt

python create_imagenet_filelist.py cfg/imagenette/dataset.cfg

if [ ! -f "$SURROGATE" ]; then
        cfg=cfg/imagenette/_lc_surrogate.cfg
        sed -e "s/^attack=.*/attack=badnets/" \
            -e "s/^num_poison_badnets=.*/num_poison_badnets=0/" \
            -e "s/^num_poison_lc=.*/num_poison_lc=0/" \
            -e "s/^tal_weight=.*/tal_weight=0.0/" \
            -e "s/^rand_loc=.*/rand_loc=true/" \
            -e "s/^feature_extract=.*/feature_extract=false/" \
            -e "s/^train_clean_model=.*/train_clean_model=true/" \
            "$EXP_CFG" > "$cfg"
        CUDA_VISIBLE_DEVICES=$GPU python train_backdoor.py "$cfg"
fi

make_cfg () {
        sed -e "s/^attack=.*/attack=lc/" \
            -e "s/^num_poison_badnets=.*/num_poison_badnets=0/" \
            -e "s/^rand_loc=.*/rand_loc=false/" \
            -e "s/^train_clean_model=.*/train_clean_model=false/" \
            -e "s/^tal_weight=.*/tal_weight=0.0/" \
            -e "s/^eps=.*/eps=$2/" \
            -e "s/^num_poison_generate=.*/num_poison_generate=$3/" \
            -e "s/^num_poison_lc=.*/num_poison_lc=$4/" \
            -e "s/^feature_extract=.*/feature_extract=$5/" \
            "$EXP_CFG" > "$1"
}

run () {
        CUDA_VISIBLE_DEVICES=$GPU python train_backdoor.py    "$1"
        CUDA_VISIBLE_DEVICES=$GPU python test_time_defense.py "$1"
}


C478=cfg/imagenette/_lc_e16_n478.cfg
make_cfg $C478 16 955 478 false
run $C478

CFRZ=cfg/imagenette/_lc_e16_n200_headonly.cfg
make_cfg $CFRZ 16 955 200 true
run $CFRZ


C0=cfg/imagenette/_lc_e0_n200.cfg
make_cfg $C0 0 200 200 false
CUDA_VISIBLE_DEVICES=$GPU python generate_lc_poison.py $C0
run $C0
