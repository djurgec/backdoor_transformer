#!/usr/bin/env bash
# Delete all runs, checkpoints and generated cfgs for one experiment.
#   bash scripts/purge_runs.sh [experiment_id] [-y]
set -eu

ID=${1:-imagenet30}
YES=${YES:-0}
[ "${2:-}" = "-y" ] && YES=1

RUNS="runs/${ID:?}"
CKPT="checkpoints/${ID:?}"

any=0
echo "under $(pwd):"
for d in "$RUNS" "$CKPT"; do
        [ -d "$d" ] || continue
        any=1
        echo
        echo "  $d  ($(du -sh "$d" 2>/dev/null | cut -f1))"
        find "$d" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | sed 's|^|      |'
done

CFGS=$(find cfg/_generated -maxdepth 1 -name "*${ID}*" 2>/dev/null | wc -l | tr -d ' ')
if [ "$CFGS" -gt 0 ]; then
        any=1
        echo
        echo "  $CFGS generated cfgs matching $ID"
fi

if [ "$any" = 0 ]; then
        echo "  nothing found for '$ID'"
        exit 0
fi

if [ "$YES" != 1 ]; then
        echo
        printf "delete all of the above? [y/N] "
        read -r reply || reply=""
        case "$reply" in
                y|Y|yes|YES) ;;
                *) echo "aborted"; exit 1 ;;
        esac
fi

rm -rf -- "$RUNS" "$CKPT"
find cfg/_generated -maxdepth 1 -name "*${ID}*" -delete 2>/dev/null || true
echo "deleted $ID"
