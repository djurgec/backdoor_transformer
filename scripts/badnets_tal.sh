#!/usr/bin/env bash
# BadNets + reverse TAL weight sweep on imagenette.
#   bash scripts/badnets_tal.sh 0 -0.25 -0.5 -1.0
set -e

GPU=${1:-0}
shift || true
WEIGHTS=${@:-"0.0 -0.25 -0.5 -1.0"}

BASE=cfg/imagenette/experiment_imagenette.cfg
python create_imagenet_filelist.py cfg/imagenette/dataset.cfg

for w in $WEIGHTS; do
        bash scripts/run.sh "$GPU" "$BASE" \
                attack=badnets num_poison_htba=0 tal_weight="$w" \
                feature_extract=false train_clean_model=false
done

bash scripts/summarise.sh imagenette badnets
