#!/usr/bin/env bash
# One experiment: build a cfg from a base plus key=value overrides, then run it.
#
#   bash scripts/run.sh <GPU> <BASE_CFG> key=value [key=value ...]
#
# STEPS picks which entry points execute (default: train defense):
#   STEPS="generate train defense rollout" bash scripts/run.sh 0 cfg/... attack=htba
#
# Every override is verified to have landed. sed silently does nothing when a key
# is missing from the base cfg, which drops the setting from the run descriptor
# and lands the run on top of a different run's directory -- that has bitten this
# project twice, so the check covers every key rather than a hand-picked few.
set -e

GPU=$1
BASE=$2
shift 2

STEPS=${STEPS:-"train defense"}

sed_args=()
slug=""
for kv in "$@"; do
        key=${kv%%=*}
        val=${kv#*=}
        sed_args+=(-e "s|^${key}=.*|${key}=${val}|")
        slug="${slug}_$(printf '%s' "$key=$val" | tr -c 'A-Za-z0-9.=-' '-')"
done

mkdir -p cfg/_generated
cfg="cfg/_generated/$(basename "${BASE%.cfg}")${slug}.cfg"
sed "${sed_args[@]}" "$BASE" > "$cfg"

for kv in "$@"; do
        grep -qx "$kv" "$cfg" || {
                echo "FATAL: override '$kv' did not apply."
                echo "       $BASE has no line starting with '${kv%%=*}='."
                exit 1
        }
done

for step in $STEPS; do
        case "$step" in
                generate) script=generate_htba_poison.py ;;
                train)    script=train_backdoor.py ;;
                defense)  script=test_time_defense.py ;;
                rollout)  script=analyze_rollout.py ;;
                *) echo "FATAL: unknown step '$step'"; exit 1 ;;
        esac
        echo "--- $step: $script $cfg"
        CUDA_VISIBLE_DEVICES=$GPU python "$script" "$cfg"
done
