#!/usr/bin/env bash
set -e

GPU=${1:-0}
TRIGGERS=${TRIGGERS:-"10 20"}
RATES=${RATES:-"2.5 10 25 50"}
LAYERS=${LAYERS:-"all last1"}        # TAL scope; "all" leaves tal_layers empty
WEIGHT=${WEIGHT:--0.5}
SEEDS=${SEEDS:-"0"}                  # each seed draws its own source/target pair
BASELINE=${BASELINE:-true}

BASE=cfg/imagenet30/experiment_imagenet30.cfg

if [ ! -d "ImageNet_data_list/imagenet30/train" ]; then
        python create_imagenet_filelist.py cfg/imagenet30/dataset.cfg
fi

cell() {   # rate seed weight layers trigger
        spec=""
        [ "$4" != "all" ] && spec="$4"
        bash scripts/run.sh "$GPU" "$BASE" \
                attack=badnets num_poison_badnets=0 num_poison_htba=0 \
                poison_rate_badnets="$1" random_classes=true seed="$2" \
                tal_weight="$3" tal_layers="$spec" trigger_id="$5" \
                rand_loc=true feature_extract=false unfreeze_blocks=0 \
                train_clean_model=false
}

for t in $TRIGGERS; do
for s in $SEEDS; do
for r in $RATES; do
        # tal_layers is meaningless with the loss switched off, so the control
        # is run once per rate rather than once per layer scope
        [ "$BASELINE" = true ] && cell "$r" "$s" 0.0 all "$t"
        for l in $LAYERS; do
                cell "$r" "$s" "$WEIGHT" "$l" "$t"
        done
done
done
done

bash scripts/summarise.sh imagenet30 badnets
