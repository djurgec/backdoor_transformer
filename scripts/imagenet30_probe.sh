#!/usr/bin/env bash
# One ImageNet-30 cell at full length, then its per-epoch curve, to decide how
# many epochs the real grid needs.
#
#   bash scripts/imagenet30_probe.sh 0
#   RATE=25 WEIGHT=0.0 bash scripts/imagenet30_probe.sh 1
#
set -e

GPU=${1:-0}
RATE=${RATE:-10}
WEIGHT=${WEIGHT:--0.5}
TRIGGER=${TRIGGER:-10}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-10}
LAYERS=${LAYERS:-all}

BASE=cfg/imagenet30/experiment_imagenet30.cfg

if [ ! -d "ImageNet_data_list/imagenet30/train" ]; then
        python create_imagenet_filelist.py cfg/imagenet30/dataset.cfg
fi

spec=""
[ "$LAYERS" != "all" ] && spec="$LAYERS"

bash scripts/run.sh "$GPU" "$BASE" \
        attack=badnets num_poison_badnets=0 num_poison_htba=0 \
        poison_rate_badnets="$RATE" random_classes=true seed="$SEED" \
        tal_weight="$WEIGHT" tal_layers="$spec" trigger_id="$TRIGGER" \
        epochs="$EPOCHS" rand_loc=true feature_extract=false \
        unfreeze_blocks=0 train_clean_model=false

# rebuild the descriptor the same way run_paths does, to find the run directory
RUN=$(python - "$RATE" "$WEIGHT" "$spec" "$TRIGGER" "$SEED" <<'PY'
import configparser, sys
import run_paths
rate, weight, spec, trig, seed = sys.argv[1:6]
c = configparser.ConfigParser()
c.read("cfg/imagenet30/experiment_imagenet30.cfg")
o = c["finetune"]
o["poison_rate_badnets"], o["tal_weight"] = rate, weight
o["tal_layers"], o["trigger_id"], o["seed"] = spec, trig, seed
o["num_poison_badnets"] = "0"
print(run_paths.descriptor(c, "badnets"))
PY
)

echo
bash scripts/epoch_curve.sh imagenet30 badnets "$RUN"
