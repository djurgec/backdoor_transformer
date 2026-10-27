#!/usr/bin/env bashC
set -e

GPU=${1:-0}
SEED=${SEED:-9}
RATE=${RATE:-25}

BASE=cfg/imagenet30/experiment_imagenet30.cfg

if [ ! -d "ImageNet_data_list/imagenet30/train" ]; then
        python create_imagenet_filelist.py cfg/imagenet30/dataset.cfg
fi

RUN=$(python - "$RATE" "$SEED" <<'PY'
import configparser, sys
import run_paths
rate, seed = sys.argv[1], sys.argv[2]
c = configparser.ConfigParser()
c.read("cfg/imagenet30/experiment_imagenet30.cfg")
o = c["finetune"]
o["poison_rate_badnets"], o["num_poison_badnets"] = rate, "0"
o["tal_weight"], o["tal_layers"], o["trigger_id"] = "0.0", "", "10"
o["seed"] = seed
print(run_paths.descriptor(c, "badnets"))
PY
)

RUN_DIR="runs/imagenet30/badnets/$RUN"
CKPT_DIR="checkpoints/imagenet30/badnets/$RUN"

echo "clean control -> $RUN_DIR"
for d in "$RUN_DIR" "$CKPT_DIR"; do
        if [ -e "$d" ]; then
                echo "REFUSING: $d already exists." >&2
                echo "Pick an unused seed, e.g. SEED=8 bash scripts/imagenet30_clean.sh $GPU" >&2
                exit 1
        fi
done

STEPS="train" bash scripts/run.sh "$GPU" "$BASE" \
        attack=badnets num_poison_badnets=0 num_poison_htba=0 \
        poison_rate_badnets="$RATE" random_classes=true seed="$SEED" \
        tal_weight=0.0 tal_layers= trigger_id=10 \
        rand_loc=true feature_extract=false unfreeze_blocks=0 \
        train_clean_model=true

# train_model logs this line for each model it trains, so the LAST one is the
# clean control and the first is the throwaway poisoned model
LOG="$RUN_DIR/${RUN}_att.log"
echo
echo "---------------------------------------------------------------"
grep -n "Clean acc of the saved model" "$LOG" | tail -2
echo
echo "the LAST line above is the clean control: the accuracy every poisoned"
echo "run in the grid should be compared against."
echo "full log: $LOG"
