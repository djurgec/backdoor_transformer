#!/usr/bin/env bash
set -e

GPU=${1:-0}
TRIGGERS=${TRIGGERS:-"10 20"}
RATES=${RATES:-"2.5 10 25 50"}
LAYERS=${LAYERS:-"all last1"}        # TAL scope; "all" leaves tal_layers empty
# 0.0 is the no-TAL control. It belongs in this list rather than behind a
# separate flag, so one knob says which weights the grid covers.
WEIGHTS=${WEIGHTS:-"0.0 -0.5"}
SEEDS=${SEEDS:-"0"}                  # each seed draws its own source/target pair

BASE=cfg/imagenet30/experiment_imagenet30.cfg

if [ ! -d "ImageNet_data_list/imagenet30/train" ]; then
        python create_imagenet_filelist.py cfg/imagenet30/dataset.cfg
fi

# train_backdoor sets use_tal = tal_weight != 0 and (...), so at weight 0 the
# layer spec never reaches the loss. Looping LAYERS there would train the
# same model twice under two names that claim the scope differs.
is_zero() { awk -v w="$1" 'BEGIN{exit !(w+0==0)}'; }

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
        for w in $WEIGHTS; do
                if is_zero "$w"; then
                        cell "$r" "$s" "$w" all "$t"
                else
                        for l in $LAYERS; do
                                cell "$r" "$s" "$w" "$l" "$t"
                        done
                fi
        done
done
done
done

bash scripts/summarise.sh imagenet30 badnets
