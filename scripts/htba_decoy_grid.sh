#!/usr/bin/env bash
# HTBA + fixed decoy. Poisons must already exist (scripts/imagenette_htba.sh).
#
#   BLOCKS=2 SCOPES=true WEIGHTS="24 28 32" bash scripts/htba_decoy_grid.sh 0
set -e

GPU=${1:-0}
BLOCKS=${BLOCKS:-"2 4"}
WEIGHTS=${WEIGHTS:-"20 40 60 80 100"}
SCOPES=${SCOPES:-"true false"}      # tal_poison_only: true = poisons only
DECOY=${DECOY:-"0,0"}               # grid-aligned, same footprint as the 31x31 window
NUM_POISON=${NUM_POISON:-400}

BASE=cfg/imagenette/experiment_imagenette.cfg

for b in $BLOCKS; do
for s in $SCOPES; do
for w in $WEIGHTS; do
        bash scripts/run.sh "$GPU" "$BASE" \
                attack=htba num_poison_badnets=0 num_poison_htba="$NUM_POISON" \
                rand_loc=true feature_extract=true unfreeze_blocks="$b" \
                tal_decoy="$DECOY" tal_poison_only="$s" tal_weight="$w" \
                optimizer=sgd lr=1e-4 head_lr_mult=10.0 train_clean_model=false
done
done
done

bash scripts/summarise.sh imagenette htba
