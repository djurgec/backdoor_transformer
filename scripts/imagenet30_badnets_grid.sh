#!/usr/bin/env bash
# BadNets + reverse TAL on ImageNet-30: weights x poison rates x seeds.
# Each seed draws its own source/target class pair.
#
#   WEIGHTS="0.0 -0.25"  bash scripts/imagenet30_badnets_grid.sh 0
#   WEIGHTS="-0.5 -0.75" bash scripts/imagenet30_badnets_grid.sh 1
set -e

GPU=${1:-0}
WEIGHTS=${WEIGHTS:-"0.0 -0.25 -0.5 -0.75"}   # 0.0 is the no-TAL control
RATES=${RATES:-"2.5 10 25"}                  # percent of the SOURCE class
SEEDS=${SEEDS:-"0 1 2"}

BASE=cfg/imagenet30/experiment_imagenet30.cfg

if [ ! -d "ImageNet_data_list/imagenet30/train" ]; then
        python create_imagenet_filelist.py cfg/imagenet30/dataset.cfg
fi

for s in $SEEDS; do
for r in $RATES; do
for w in $WEIGHTS; do
        bash scripts/run.sh "$GPU" "$BASE" \
                attack=badnets num_poison_htba=0 \
                poison_rate_badnets="$r" random_classes=true seed="$s" \
                tal_weight="$w" rand_loc=true feature_extract=false \
                unfreeze_blocks=0 train_clean_model=false
done
done
done

bash scripts/summarise.sh imagenet30 badnets
