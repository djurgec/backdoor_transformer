#!/usr/bin/env bash
# BadNets + reverse TAL on ImageNet-30.
#   bash run_imagenet30_badnets_grid.sh [GPU]
set -x
set -e

GPU=${1:-0}

WEIGHTS=${WEIGHTS:-"0.0 -0.25 -0.5 -0.75"}   # 0.0 is the no-TAL control
RATES=${RATES:-"2.5 10 25"}                  # percent of the source class poisoned
SEEDS=${SEEDS:-"0 1 2"}                      # each seed draws its own source/target

DATASET_CFG=cfg/imagenet30/dataset.cfg
EXP_CFG=cfg/imagenet30/experiment_imagenet30.cfg

# class lists live under ImageNet_data_list/imagenet30/, built once
if [ ! -d "ImageNet_data_list/imagenet30/train" ]; then
        python create_imagenet_filelist.py "$DATASET_CFG"
fi

for s in $SEEDS; do
for r in $RATES; do
for w in $WEIGHTS; do
        cfg="cfg/imagenet30/_bn_s${s}_r${r}_tal${w}.cfg"
        sed -e "s/^attack=.*/attack=badnets/" \
            -e "s/^num_poison_htba=.*/num_poison_htba=0/" \
            -e "s/^poison_rate_badnets=.*/poison_rate_badnets=${r}/" \
            -e "s/^random_classes=.*/random_classes=true/" \
            -e "s/^seed=.*/seed=${s}/" \
            -e "s/^tal_weight=.*/tal_weight=${w}/" \
            -e "s/^rand_loc=.*/rand_loc=true/" \
            -e "s/^feature_extract=.*/feature_extract=false/" \
            -e "s/^unfreeze_blocks=.*/unfreeze_blocks=0/" \
            -e "s/^train_clean_model=.*/train_clean_model=false/" \
            "$EXP_CFG" > "$cfg"


        for kv in "poison_rate_badnets=${r}" "seed=${s}" "tal_weight=${w}" "random_classes=true"; do
                grep -qx "$kv" "$cfg" || { echo "FATAL: $cfg has no line '$kv'"
                        echo "       the base cfg is out of date; git pull"
                        exit 1; }
        done

        CUDA_VISIBLE_DEVICES=$GPU python train_backdoor.py    "$cfg"
        CUDA_VISIBLE_DEVICES=$GPU python test_time_defense.py "$cfg"
done
done
done

set +x
echo "===================== SUMMARY ====================="
python - <<'PY'
import glob, os, re
rows = []
for d in sorted(glob.glob("runs/imagenet30/badnets/*/")):
    dp = os.path.join(d, "defense.log")
    fp = os.path.join(d, "finetune.log")
    if not os.path.exists(dp):
        continue
    t = open(dp, encoding="utf-8", errors="replace").read()
    f = open(fp, encoding="utf-8", errors="replace").read() if os.path.exists(fp) else ""
    g = lambda p, s: (re.search(p, s).group(1) if re.search(p, s) else "-")
    rows.append((os.path.basename(d.rstrip("/")),
                 g(r"target=(\S+?) \|", f), g(r"source=(\S+?) \|", f),
                 g(r"ASR before defense ([0-9.]+)", t),
                 g(r"ASR after defense ([0-9.]+)", t),
                 g(r"CLEAN ACC before defense ([0-9.]+)", t)))
print("{:<44} {:>14} {:>14} {:>9} {:>9} {:>9}".format(
    "run", "target", "source", "ASR bef", "ASR aft", "clean"))
for n, tg, sc, a, b, c in rows:
    print("{:<44} {:>14} {:>14} {:>9} {:>9} {:>9}".format(n, tg, sc, a, b, c))
PY
