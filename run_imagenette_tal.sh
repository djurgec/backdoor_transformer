#!/usr/bin/env bash
# TAL weight sweep for the current attack (set `attack=` in the cfg).
# Usage: bash run_imagenette_tal.sh [GPU] [weights...]
set -x
set -e

GPU=${1:-0}
shift || true
WEIGHTS=${@:-"1.0 0.5 0.0 -0.5 -1.0"}

DATASET_CFG=cfg/imagenette/dataset.cfg
EXP_CFG=cfg/imagenette/experiment_imagenette.cfg

python create_imagenet_filelist.py $DATASET_CFG

run_experiment () {
        local weight=$1
        local cfg="cfg/imagenette/_run_tal_${weight}.cfg"
        sed "s/^tal_weight=.*/tal_weight=${weight}/" "$EXP_CFG" > "$cfg"

        CUDA_VISIBLE_DEVICES=$GPU python train_backdoor.py "$cfg"
        CUDA_VISIBLE_DEVICES=$GPU python test_time_defense.py    "$cfg"
}

for w in $WEIGHTS; do
        run_experiment "$w"
done

set +x
ATTACK=$(sed -n 's/^attack=//p' "$EXP_CFG")
ID=$(sed -n 's/^ID=//p' "$EXP_CFG")
echo "======================= SUMMARY (${ATTACK}) ======================="
for d in runs/${ID}/${ATTACK}/*/; do
        [ -f "${d}defense.log" ] || continue
        echo "--- $(basename "$d") ---"
        grep -h "ASR before defense\|ASR after defense\|CLEAN ACC before defense\|CLEAN ACC after defense" \
             "${d}defense.log"
done
