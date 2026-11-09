#!/usr/bin/env bash
# HTBA + fixed decoy box on ImageNet-30. Generates poisons once, then sweeps
# tal_weight per unfreeze depth. Weight 0 is the control at matched depth.
#
#   bash scripts/imagenet30_decoy_grid.sh 0
#   UF=2 bash scripts/imagenet30_decoy_grid.sh 0
#   SKIP_GEN=1 UF=4 W_UF4="12 15 18" bash scripts/imagenet30_decoy_grid.sh 0
set -e

GPU=${1:-0}
UF=${UF:-"2 4"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
W_UF2=${W_UF2:-"0 24"}
W_UF4=${W_UF4:-"0 14"}
NUM_POISON=${NUM_POISON:-400}
GEN_POISON=${GEN_POISON:-$((NUM_POISON * 3))}
DECOY=${DECOY:-"0,0"}
SCOPE=${SCOPE:-true}
SCHED=${SCHED:-cosine}
# non-empty re-runs ONLY the defense, at these box sizes, against checkpoints
# these cells already trained; it never generates or trains anything
EDGES=${EDGES:-}
[ -n "$EDGES" ] && SKIP_GEN=1

BASE=cfg/imagenet30/experiment_imagenet30.cfg

if [ ! -d "ImageNet_data_list/imagenet30/train" ]; then
        python create_imagenet_filelist.py cfg/imagenet30/dataset.cfg
fi

POISON_DIR=$(python - "$BASE" <<'PY'
import configparser, sys
c = configparser.ConfigParser()
c.read(sys.argv[1])
o = c["finetune"]
print("{}/{}/rand_loc_{}/eps_{}/patch_size_{}/trigger_{}".format(
    o["poison_root"], c["experiment"]["ID"], o.getboolean("rand_loc"),
    int(o["eps"]), int(o["patch_size"]), int(o["trigger_id"])))
PY
)
HAVE=$(ls "$POISON_DIR" 2>/dev/null | wc -l | tr -d ' ')
echo "poisons: $HAVE in $POISON_DIR (need $NUM_POISON, generating $GEN_POISON)"

if [ "${SKIP_GEN:-0}" != "1" ] && [ "$HAVE" -lt "$GEN_POISON" ]; then
        echo "generating -- this is the slow part, once for the whole grid"
        STEPS="generate" bash scripts/run.sh "$GPU" "$BASE" \
                attack=htba num_poison_badnets=0 poison_rate_badnets=0 num_poison_htba="$GEN_POISON" \
                rand_loc=true feature_extract=true unfreeze_blocks=0 \
                tal_decoy= tal_weight=0.0 train_clean_model=false
        HAVE=$(ls "$POISON_DIR" 2>/dev/null | wc -l | tr -d ' ')
fi

if [ "$HAVE" -lt "$NUM_POISON" ]; then
        echo "FATAL: only $HAVE poisons, need $NUM_POISON" >&2
        exit 1
fi

cell() {   # unfreeze weight seed
        echo "=============================================================="
        echo "  uf=$1  tal_weight=$2  seed=$3  decoy=$DECOY  poison_only=$SCOPE"
        echo "=============================================================="
        set -- "attack=htba" "num_poison_badnets=0" "poison_rate_badnets=0" \
                "num_poison_htba=$NUM_POISON" \
                "rand_loc=true" "feature_extract=true" "unfreeze_blocks=$1" "seed=$3" \
                "tal_decoy=$DECOY" "tal_poison_only=$SCOPE" "tal_weight=$2" \
                "tal_floor=0" "lr_schedule=$SCHED" "log_attention=true" \
                "tal_layers=" "optimizer=sgd" "lr=1e-4" "head_lr_mult=10.0" \
                "train_clean_model=false"
        if [ -n "$EDGES" ]; then
                for e in $EDGES; do
                        echo "  re-running the defense at edge_length=$e"
                        STEPS="defense" bash scripts/run.sh "$GPU" "$BASE" "$@" \
                                "edge_length=$e"
                done
        else
                bash scripts/run.sh "$GPU" "$BASE" "$@"
        fi
}

for u in $UF; do
        case "$u" in
                2) WEIGHTS="$W_UF2" ;;
                4) WEIGHTS="$W_UF4" ;;
                *) WEIGHTS=${WEIGHTS_OTHER:-"0 14 20 28 40 56"} ;;
        esac
        for w in $WEIGHTS; do
        for sd in $SEEDS; do
                cell "$u" "$w" "$sd"
        done
        done
done

bash scripts/summarise.sh imagenet30 htba
