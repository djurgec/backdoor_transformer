#!/usr/bin/env bash
# BadNets and TrojanNN-square, both with reverse TAL, on CIFAR-10.
#
# Trigger 10 is the random square badnets uses; trigger 21 is written by
# optimize_trojan_trigger.py, which drives the pretrained CLS dims -- the
# TrojanNN "square" trigger. Both train through the same dirty-label path, so
# the only difference between the arms is the patch that gets pasted in.
#
#   bash scripts/cifar10_reverse_tal.sh 0
#   DRY=1 bash scripts/cifar10_reverse_tal.sh 0          # print the plan, run nothing
#   WEIGHTS="0.0 -0.5" TRIGGERS=10 bash scripts/cifar10_reverse_tal.sh 0
set -e

GPU=${1:-0}
TRIGGERS=${TRIGGERS:-"10 21"}
WEIGHTS=${WEIGHTS:-"0.0 -0.5"}               # 0.0 is the no-TAL control
RATES=${RATES:-"25"}                         # percent of the SOURCE class
LAYERS=${LAYERS:-"auto last1"}               # auto = every trainable block
# a 0-weight run trains identically whatever tal_layers says, so the control is
# run once on the FIRST arm and shared. Keyed off the first arm rather than
# "auto" so LAYERS=last1 still gets its controls.
CTRL_ARM=$(echo $LAYERS | awk '{print $1}')

# this grid's 29 earlier runs drew their pair per seed; the pair is fixed now,
# so the name has to say which one or the two collide at the same seed
export CLS_IN_NAME=1
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}   # the pair is fixed in the cfg; seeds vary training only
EPOCHS=${EPOCHS:-10}
EDGE=${EDGE:-50}
# the per-epoch defense is one grad-rollout per patched image, and CIFAR-10 has
# 1000 val images per class where imagenet30 has 100 -- ten times the bill.
# Only the seeds named here pay it; the rest get the final epoch only.
DEF_SEEDS=${DEF_SEEDS:-0}
# optimised against CIFAR-10's own val images, so it cannot share id 20 with
# the imagenet30 patch -- data/trigger is one directory for both datasets
TROJAN_ID=${TROJAN_ID:-21}

BASE=cfg/cifar10/experiment_cifar10.cfg

if [ ! -d data/cifar10/train ] || [ ! -d data/cifar10/val ]; then
	echo "FATAL: data/cifar10 is not prepared. Run:" >&2
	echo "  python prepare_cifar10.py" >&2
	exit 1
fi

if [ ! -d "ImageNet_data_list/cifar10/train" ]; then
	python create_imagenet_filelist.py cfg/cifar10/dataset.cfg
fi

# the TrojanNN patch is attack-side and seed-independent, so it is built once
# and reused across every cell rather than re-optimised per run
for t in $TRIGGERS; do
	if [ "$t" = "$TROJAN_ID" ] && [ ! -f "data/trigger/trigger_$t.png" ]; then
		echo "optimising the TrojanNN trigger -> data/trigger/trigger_$t.png"
		if [ -z "$DRY" ]; then
			CUDA_VISIBLE_DEVICES=$GPU python optimize_trojan_trigger.py \
				"$BASE" --out-id "$t"
		fi
	fi
done

total=0
for t in $TRIGGERS; do for l in $LAYERS; do for r in $RATES; do for w in $WEIGHTS; do
for s in $SEEDS; do
	if [ "$w" = "0.0" ] && [ "$l" != "$CTRL_ARM" ]; then continue; fi
	total=$((total + 1))
done; done; done; done; done
per_arm=$((total / $(echo $TRIGGERS | wc -w)))
echo
echo "$total runs ($per_arm per trigger arm), $EPOCHS epochs each"
echo "  triggers [$TRIGGERS]  layers [$LAYERS]  rates [$RATES]"
echo "  weights [$WEIGHTS]  seeds [$SEEDS]"
echo "  defense box $EDGE  per-epoch defense on seeds [$DEF_SEEDS]"
echo "  CIFAR-10 train is 50k images: a full-finetune epoch is ~1560 steps at"
echo "  batch 32, so budget before launching the whole grid."
if [ -n "$DRY" ]; then
	for t in $TRIGGERS; do for l in $LAYERS; do for r in $RATES; do for w in $WEIGHTS; do
	for s in $SEEDS; do
		if [ "$w" = "0.0" ] && [ "$l" != "$CTRL_ARM" ]; then continue; fi
		echo "  trigger=$t layers=$l rate=$r w=$w seed=$s"
	done; done; done; done; done
	exit 0
fi

cell() {   # trigger layers rate weight seed
	spec=""
	[ "$2" != "auto" ] && spec="$2"
	every=false
	case " $DEF_SEEDS " in *" $5 "*) every=true ;; esac
	echo "=============================================================="
	echo "  trigger=$1  layers=$2  rate=$3%  tal_weight=$4  seed=$5"
	echo "=============================================================="
	bash scripts/run.sh "$GPU" "$BASE" \
		"attack=badnets" "num_poison_htba=0" "num_poison_badnets=0" \
		"poison_rate_badnets=$3" "random_classes=false" "seed=$5" \
		"trigger_id=$1" "tal_weight=$4" "tal_decoy=" "tal_layers=$spec" \
		"rand_loc=true" "feature_extract=false" "unfreeze_blocks=0" \
		"epochs=$EPOCHS" "edge_length=$EDGE" "log_attention=true" \
		"defense_every_epoch=$every" "train_clean_model=false"
}


# trigger outermost: the badnets arm finishes and is summarised before the
# trojan arm starts, so the run can be stopped half way and still say something
# trigger outermost: the badnets arm finishes and is summarised before the
# trojan arm starts, so the run can be stopped half way and still say something
for t in $TRIGGERS; do
	echo
	echo "##############################################################"
	echo "  trigger $t"
	echo "##############################################################"
	for l in $LAYERS; do
	for r in $RATES; do
	for w in $WEIGHTS; do
	for s in $SEEDS; do
		# a 0-weight run trains identically whatever tal_layers says, so the
		# control is run once on the auto arm and shared by the others
		if [ "$w" = "0.0" ] && [ "$l" != "$CTRL_ARM" ]; then
			continue
		fi
		cell "$t" "$l" "$r" "$w" "$s"
	done
	done
	done
	done
	echo
	echo "----- after the trigger=$t arm -----"
	bash scripts/sweep_table.sh cifar10 badnets || true
done
