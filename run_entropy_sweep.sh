#!/usr/bin/env bash
set -x
set -e

GPU=${1:-0}
shift || true
WEIGHTS=${@:-"0 0.01 0.05 0.1 0.2 0.5"}
CLS_ONLY=${CLS_ONLY:-false}
LAYERS=${LAYERS:-}

NUM_POISON=100
DATASET_CFG=cfg/imagenette/dataset.cfg
EXP_CFG=cfg/imagenette/experiment_imagenette.cfg

python create_imagenet_filelist.py $DATASET_CFG

BASE=cfg/imagenette/_ent_base.cfg
sed -e "s/^attack=.*/attack=badnets/" \
    -e "s/^num_poison_badnets=.*/num_poison_badnets=${NUM_POISON}/" \
    -e "s/^num_poison_lc=.*/num_poison_lc=0/" \
    -e "s/^tal_weight=.*/tal_weight=0.0/" \
    -e "s/^rand_loc=.*/rand_loc=true/" \
    -e "s/^feature_extract=.*/feature_extract=false/" \
    -e "s/^train_clean_model=.*/train_clean_model=false/" \
    -e "s/^entropy_cls_only=.*/entropy_cls_only=${CLS_ONLY}/" \
    -e "s/^entropy_layers=.*/entropy_layers=${LAYERS}/" \
    "$EXP_CFG" > "$BASE"

run_dir () {
        python -c "import configparser,run_paths,sys; c=configparser.ConfigParser(); c.read(sys.argv[1]); print(run_paths.for_run(c['experiment']['ID'],'badnets',c)['run_dir'])" "$1"
}

for w in $WEIGHTS; do
        cfg="cfg/imagenette/_ent_${w}.cfg"
        if [ "$w" = "0" ]; then log=true; else log=false; fi
        sed -e "s/^entropy_weight=.*/entropy_weight=${w}/" \
            -e "s/^log_attention=.*/log_attention=${log}/" \
            "$BASE" > "$cfg"
        CUDA_VISIBLE_DEVICES=$GPU python train_backdoor.py    "$cfg"
        CUDA_VISIBLE_DEVICES=$GPU python test_time_defense.py "$cfg"
done
