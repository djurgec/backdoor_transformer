#!/usr/bin/env bash
# Where does the HTBA decoy actually buy anything on ImageNet-30?
# One seed, short runs, scanning tal_weight x unfreeze depth x decoy scope.
# The imagenette optimum (uf2 w=24, uf4 w=14) reaches 8.5x baseline decoy share
# on the training poisons here and still loses asr_def, so the sweep is steered
# by p_l11 -- the same share measured on the PATCHED set -- not by the weight.
#
#   bash scripts/imagenet30_decoy_sweep.sh 0
#   LAYERS=auto bash scripts/imagenet30_decoy_sweep.sh 0
#   SCOPE=false UF=2 W_UF2_ALL="12 24" bash scripts/imagenet30_decoy_sweep.sh 0
set -e

GPU=${1:-0}
UF=${UF:-"2 4"}
SCOPE=${SCOPE:-"true false"}
LAYERS=${LAYERS:-"auto last1"}
# poison_only ladders, starting at the imagenette optima and going up 4x
W_UF2=${W_UF2:-"24 48 96 160"}
W_UF4=${W_UF4:-"14 28 56 100"}
# allimg puts TAL on every batch; poison_only only on the 28% holding a poison,
# so the same total pressure arrives at roughly a 3.5x lower weight
W_UF2_ALL=${W_UF2_ALL:-"6 12 24 48"}
W_UF4_ALL=${W_UF4_ALL:-"4 8 14 28"}
SEEDS=${SEEDS:-0}
EPOCHS=${EPOCHS:-5}
NUM_POISON=${NUM_POISON:-400}
DECOY=${DECOY:-"0,0"}
SCHED=${SCHED:-cosine}
# non-empty re-runs ONLY the defense, at these box sizes, against the
# checkpoints these cells already trained. edge_length is not in the run
# descriptor, so each size lands as its own _def_eN.log beside the existing one.
EDGES=${EDGES:-}

BASE=cfg/imagenet30/experiment_imagenet30.cfg

ladder() {   # unfreeze scope
        case "$1/$2" in
                2/true)  echo "$W_UF2" ;;
                2/false) echo "$W_UF2_ALL" ;;
                4/true)  echo "$W_UF4" ;;
                4/false) echo "$W_UF4_ALL" ;;
                *)       echo "${W_OTHER:-14 28 56 100}" ;;
        esac
}

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
if [ "$HAVE" -lt "$NUM_POISON" ]; then
        echo "FATAL: only $HAVE poisons in $POISON_DIR, need $NUM_POISON." >&2
        echo "       Run the grid once with SKIP_GEN unset, or lower NUM_POISON." >&2
        exit 1
fi
echo "reusing $HAVE poisons from $POISON_DIR"

total=0
for u in $UF; do for l in $LAYERS; do for sc in $SCOPE; do
        for w in $(ladder "$u" "$sc"); do for sd in $SEEDS; do
                total=$((total + 1))
        done; done
done; done; done
per_arm=$((total / $(echo $LAYERS | wc -w)))
echo "$total cells x $EPOCHS epochs -- $per_arm per layer arm ($LAYERS),"
echo "summarised after each arm, so the first table lands at the halfway point"

cell() {   # unfreeze layers weight scope seed
        spec=""
        [ "$2" != "auto" ] && spec="$2"
        echo "=============================================================="
        echo "  uf=$1  layers=$2  tal_weight=$3  poison_only=$4  seed=$5"
        echo "=============================================================="
        set -- "attack=htba" "num_poison_badnets=0" "poison_rate_badnets=0" \
                "num_poison_htba=$NUM_POISON" "epochs=$EPOCHS" \
                "rand_loc=true" "feature_extract=true" "unfreeze_blocks=$1" "seed=$5" \
                "tal_layers=$spec" "tal_decoy=$DECOY" "tal_poison_only=$4" \
                "tal_weight=$3" "tal_floor=0" "lr_schedule=$SCHED" "log_attention=true" \
                "optimizer=sgd" "lr=1e-4" "head_lr_mult=10.0" "train_clean_model=false"
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

# layers outermost: the auto arm finishes, and is summarised, before the
# last1 arm starts, so a run can be stopped at the halfway point
for l in $LAYERS; do
        echo
        echo "##############################################################"
        echo "  TAL layers: $l"
        echo "##############################################################"
        for u in $UF; do
        for sc in $SCOPE; do
        for w in $(ladder "$u" "$sc"); do
        for sd in $SEEDS; do
                cell "$u" "$l" "$w" "$sc" "$sd"
        done
        done
        done
        done
        echo
        echo "----- after the $l arm -----"
        LAST_N=3 MIN_ASR=90 bash scripts/share_vs_defense.sh imagenet30 htba || true
done
