#!/usr/bin/env bash
set -x
set -e

GPU=${1:-0}
DATASET_CFG=cfg/imagenette/dataset.cfg
EXP_CFG=cfg/imagenette/experiment_imagenette.cfg


python create_imagenet_filelist.py $DATASET_CFG

run_experiment () {
        local weight=$1
        local cfg="cfg/imagenette/_run_tal_${weight}.cfg"
        sed "s/^tal_weight=.*/tal_weight=${weight}/" "$EXP_CFG" > "$cfg"

        CUDA_VISIBLE_DEVICES=$GPU python finetune_transformer.py "$cfg"
        CUDA_VISIBLE_DEVICES=$GPU python test_time_defense.py    "$cfg"
}

run_experiment 1.0 # runs the experiment with tal_weight set to 1
run_experiment 0.0 # runs the experiment without tal
