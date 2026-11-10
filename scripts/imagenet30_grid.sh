#!/usr/bin/env bash
set -e

GPU=${1:-0}
TRIGGERS=${TRIGGERS:-"10 20"}
RATES=${RATES:-"2.5 10 25 50"}
LAYERS=${LAYERS:-"all last1"}
WEIGHTS=${WEIGHTS:-"0.0 -0.5"}
SEEDS=${SEEDS:-"0 1 2"}
# the per-epoch defense is one grad-rollout per patched image. Only the seeds
# named here pay for it; the rest get the final epoch only, which is still one
# asr_def row per run.
DEF_SEEDS=${DEF_SEEDS:-0}
# the TrojanNN patch is optimised against this experiment's own val images, so
# it must not share an id with another dataset's -- cifar10 writes 21
TROJAN_ID=${TROJAN_ID:-20}
SCHED=${SCHED:-cosine}
FLOOR=${FLOOR:-0.0003}

BASE=cfg/imagenet30/experiment_imagenet30.cfg

if [ ! -d "ImageNet_data_list/imagenet30/train" ]; then
        python create_imagenet_filelist.py cfg/imagenet30/dataset.cfg
fi

for t in $TRIGGERS; do
        if [ "$t" = "$TROJAN_ID" ] && [ ! -f "data/trigger/trigger_$t.png" ]; then
                echo "optimising the TrojanNN trigger -> data/trigger/trigger_$t.png"
                CUDA_VISIBLE_DEVICES=$GPU python optimize_trojan_trigger.py \
                        "$BASE" --out-id "$t"
        fi
done

# train_backdoor sets use_tal = tal_weight != 0 and (...), so at weight 0 the
# layer spec never reaches the loss. Looping LAYERS there would train the
# same model twice under two names that claim the scope differs.
is_zero() { awk -v w="$1" 'BEGIN{exit !(w+0==0)}'; }

cell() {   # rate seed weight layers trigger
        spec=""
        [ "$4" != "all" ] && spec="$4"
        every=false
        case " $DEF_SEEDS " in *" $2 "*) every=true ;; esac
        bash scripts/run.sh "$GPU" "$BASE" \
                attack=badnets num_poison_badnets=0 num_poison_htba=0 \
                poison_rate_badnets="$1" random_classes=false seed="$2" \
                tal_weight="$3" tal_layers="$spec" trigger_id="$5" \
                rand_loc=true feature_extract=false unfreeze_blocks=0 \
                log_attention=true lr_schedule="$SCHED" tal_floor="$FLOOR" \
                defense_every_epoch="$every" train_clean_model=false
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
