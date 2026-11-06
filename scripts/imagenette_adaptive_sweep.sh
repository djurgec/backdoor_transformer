#!/usr/bin/env bash
# Calibrate Adap-Patch on imagenette before spending GPU time on ImageNet-30.
#
# Three questions, in the order that can kill the experiment:
#   1  does the attack still LEARN when the poison budget is split k ways,
#      and does the defense still BITE with one trigger on the test image?
#   2  how much cover can the ASR afford?
#   3  is there headroom left for reverse TAL to close?
#
#   bash scripts/imagenette_adaptive_sweep.sh 0 1
#   bash scripts/imagenette_adaptive_sweep.sh 0 2 K=2 ALPHA=0.5
#   bash scripts/imagenette_adaptive_sweep.sh 0 3 K=2 ALPHA=0.5 COVER=20
#
# Stage 1 picks K and ALPHA; pass them to stages 2 and 3.
set -e

GPU=${1:-0}
STAGE=${2:-1}
shift 2 2>/dev/null || shift $#
for kv in "$@"; do export "$kv"; done

POISONS=${POISONS:-100}      # imagenette convention: a count, as in the n100 runs
EPOCHS=${EPOCHS:-10}
EDGE=${EDGE:-40}             # the box size settled on for ImageNet-30
K=${K:-2}                    # train triggers, for stages 2 and 3
ALPHA=${ALPHA:-0.5}          # uniform train opacity, for stages 2 and 3
COVER=${COVER:-20}           # cover rate, for stage 3

BASE=cfg/imagenette/experiment_imagenette.cfg

if [ ! -d "ImageNet_data_list/imagenette/train" ]; then
        python create_imagenet_filelist.py cfg/imagenette/dataset.cfg
fi

# ids 10..13 are this repo's trigger pngs; take the first k of them
ids() { printf %s "10,11,12,13" | cut -d, -f1-"$1"; }
# one uniform opacity repeated k times -- one axis to sweep instead of four
reps() {   # value count
        out="$1"
        n="$2"
        while [ "$n" -gt 1 ]; do
                out="$out,$1"
                n=$((n - 1))
        done
        printf %s "$out"
}

# test_alphas is left EMPTY on purpose. The trigger pngs carry 1.7-11.7% true
# black pixels, so blending at alpha=1.0 lets a speckle of the original image
# through and is NOT the opaque square paste every earlier run used. Empty ->
# alphas=None -> opaque paste, so ASR here is directly comparable to the
# badnets baselines. The asymmetry still comes from train alpha < 1 and from
# each poison row having seen only one of the k triggers.
cell() {   # k train_alpha test_ids cover tal_weight
        a=""
        [ "$2" != "1.0" ] && a=$(reps "$2" "$1")
        k=""
        [ "$1" != "1" ] && k=$(ids "$1")
        echo "=============================================================="
        echo "  k=$1 train_alpha=$2 test_ids=${3:-10} cover=$4 tal=$5"
        echo "=============================================================="
        bash scripts/run.sh "$GPU" "$BASE" \
                attack=adap_patch num_poison_badnets="$POISONS" num_poison_htba=0 \
                tal_weight="$5" tal_layers= trigger_id=10 \
                trigger_ids="$k" trigger_alphas="$a" \
                test_trigger_ids="$3" test_alphas= \
                cover_rate="$4" cover_classes= tal_cover_rows=true \
                epochs="$EPOCHS" edge_length="$EDGE" \
                rand_loc=true feature_extract=false unfreeze_blocks=0 \
                train_clean_model=false
}

case "$STAGE" in
1)
        # the pure badnets control on this rate, for the ASR ceiling
        cell 1 1.0 "" 0 0.0
        # weak-train/strong-test with a single trigger: asymmetry alone
        cell 1 0.5 "" 0 0.0
        # splitting the budget k ways: does each trigger still get learned?
        cell 2 0.5 "" 0 0.0
        cell 4 0.5 "" 0 0.0
        cell 2 0.2 "" 0 0.0
        cell 4 0.2 "" 0 0.0
        # the evidence run: TWO triggers on the test image. Geometry says one
        # 40x40 blackout can cover both only 1.1% of the time, so this should
        # show asr_def ~= asr with TAL off -- the defense beaten by multiplicity
        # rather than by anything we did to the attention.
        cell 4 0.5 "10,12" 0 0.0
        ;;
2)
        for c in 10 20 40; do
                cell "$K" "$ALPHA" "" "$c" 0.0
        done
        ;;
3)
        cell "$K" "$ALPHA" "" "$COVER" -0.5
        cell "$K" "$ALPHA" "" 0 -0.5
        ;;
*)
        echo "FATAL: stage must be 1, 2 or 3" >&2
        exit 1
        ;;
esac

bash scripts/summarise.sh imagenette adap_patch
