#!/usr/bin/env bash
set -e

GPU=${1:-0}
WEIGHT=${2:--0.5}
EDGES=${EDGES:-"30 40 50 60 70"}
TRIGGER=${TRIGGER:-10}

BASE=cfg/imagenette/experiment_imagenette.cfg

for e in $EDGES; do
        STEPS="defense" bash scripts/run.sh "$GPU" "$BASE" \
                attack=badnets num_poison_htba=0 tal_weight="$WEIGHT" \
                trigger_id="$TRIGGER" edge_length="$e" \
                feature_extract=false train_clean_model=false
done

bash scripts/summarise.sh imagenette badnets
