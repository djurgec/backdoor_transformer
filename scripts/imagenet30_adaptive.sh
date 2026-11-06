#!/usr/bin/env bash
# Adap-Patch on ImageNet-30, crossed with reverse TAL.
#
# Two evasion mechanisms, measured separately and together:
#   cover rows  -> breaks latent separability (Qi et al. ICLR 2023)
#   reverse TAL -> breaks the attention-localisation defense (ours)
# so the default grid is the 2x2 {cover 0, 20} x {tal 0, -0.5} over 3 seeds.
#
#   bash scripts/imagenet30_adaptive.sh 0
#   COVERS="0 20 40" SEEDS=0 bash scripts/imagenet30_adaptive.sh 0
set -e

GPU=${1:-0}
RATES=${RATES:-"10"}                 # percent of the SOURCE class, dirty-label
COVERS=${COVERS:-"0 20"}             # percent of the SOURCE class, true label
WEIGHTS=${WEIGHTS:-"0.0 -0.5"}       # 0.0 is the no-TAL control
LAYERS=${LAYERS:-"all"}              # TAL scope; "all" leaves tal_layers empty
SEEDS=${SEEDS:-"0 1 2"}              # each seed draws its own source/target pair
TAL_COVERS=${TAL_COVERS:-"true"}     # whether TAL also acts on the cover rows

# k weak triggers while training, ONE of them at full strength while testing.
#
# backdoor-toolbox/config.py tests Adap-Patch with 2 triggers at once, but that
# cannot be measured against this repo's defense: it blacks out a single 40x40
# window, and two 30px triggers at independent random positions fit inside one
# such window only 1.1% of the time, so ~99% of test images keep a working
# trigger no matter how good the heat map is. The defense would read as beaten
# by multiplicity rather than by anything done to the attention, and reverse
# TAL would have no headroom left to show. One test trigger keeps the defense
# meaningful; the asymmetry still comes from the train alphas and from each
# poison row having seen only one of the k triggers.
TRIGGER_IDS=${TRIGGER_IDS:-"10,11,12,13"}
TRAIN_ALPHAS=${TRAIN_ALPHAS:-"0.5,0.2,0.5,0.3"}
TEST_IDS=${TEST_IDS:-"10"}
# empty on purpose: the trigger pngs carry 1.7-11.7% true-black pixels, so
# alpha=1.0 blends a speckle of the original image through and is NOT the
# opaque square paste every earlier run used. Empty -> opaque, so ASR stays
# directly comparable to the badnets baselines.
TEST_ALPHAS=${TEST_ALPHAS:-""}

BASE=cfg/imagenet30/experiment_imagenet30.cfg

if [ ! -d "ImageNet_data_list/imagenet30/train" ]; then
        python create_imagenet_filelist.py cfg/imagenet30/dataset.cfg
fi

# a missing png would otherwise surface inside the first epoch
for id in $(printf %s "$TRIGGER_IDS,$TEST_IDS" | tr ',' ' '); do
        [ -f "data/trigger/trigger_${id}.png" ] || {
                echo "FATAL: data/trigger/trigger_${id}.png does not exist." >&2
                exit 1
        }
done

# train_backdoor sets use_tal = tal_weight != 0 and (...), so at weight 0 the
# layer spec never reaches the loss. Looping LAYERS there would train the same
# model twice under two names that claim the scope differs.
is_zero() { awk -v w="$1" 'BEGIN{exit !(w+0==0)}'; }

cell() {   # rate seed weight layers cover tal_cover
        spec=""
        [ "$4" != "all" ] && spec="$4"
        bash scripts/run.sh "$GPU" "$BASE" \
                attack=adap_patch num_poison_badnets=0 num_poison_htba=0 \
                poison_rate_badnets="$1" random_classes=true seed="$2" \
                tal_weight="$3" tal_layers="$spec" trigger_id=10 \
                trigger_ids="$TRIGGER_IDS" trigger_alphas="$TRAIN_ALPHAS" \
                test_trigger_ids="$TEST_IDS" test_alphas="$TEST_ALPHAS" \
                cover_rate="$5" cover_classes= tal_cover_rows="$6" \
                rand_loc=true feature_extract=false unfreeze_blocks=0 \
                train_clean_model=false
}

for s in $SEEDS; do
for r in $RATES; do
for c in $COVERS; do
        # with no cover rows there is nothing for tal_cover_rows to switch, so
        # sweeping it there would be the same run under two names
        if is_zero "$c"; then
                tal_cov_list="true"
        else
                tal_cov_list="$TAL_COVERS"
        fi
        for tc in $tal_cov_list; do
                for w in $WEIGHTS; do
                        if is_zero "$w"; then
                                cell "$r" "$s" "$w" all "$c" "$tc"
                        else
                                for l in $LAYERS; do
                                        cell "$r" "$s" "$w" "$l" "$c" "$tc"
                                done
                        fi
                done
        done
done
done
done

bash scripts/summarise.sh imagenet30 adap_patch
