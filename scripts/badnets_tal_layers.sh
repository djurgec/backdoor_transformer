#!/usr/bin/env bash
set -e

GPU=${1:-0}
WEIGHT=${2:--0.75}
LAYERS=${LAYERS:-"last1 last2 last4 last6"}
TRIGGER=${TRIGGER:-10}

BASE=cfg/imagenette/experiment_imagenette.cfg

if [ ! -d "ImageNet_data_list/imagenette/train" ]; then
        python create_imagenet_filelist.py cfg/imagenette/dataset.cfg
fi

for l in $LAYERS; do
        bash scripts/run.sh "$GPU" "$BASE" \
                attack=badnets num_poison_htba=0 tal_weight="$WEIGHT" \
                tal_layers="$l" trigger_id="$TRIGGER" \
                feature_extract=false train_clean_model=false
done

bash scripts/summarise.sh imagenette badnets
