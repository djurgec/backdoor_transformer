#!/usr/bin/env bash
# Delete runs, checkpoints and generated cfgs for one experiment.
#   bash scripts/purge_runs.sh [experiment_id] [-y]
#
# MATCH limits it to run directories whose name matches one of the globs,
# and then leaves the generated cfgs alone:
#   MATCH='*_tal24_* *_tal14_*' bash scripts/purge_runs.sh imagenet30
set -eu

ID=${1:-imagenet30}
YES=${YES:-0}
[ "${2:-}" = "-y" ] && YES=1
MATCH=${MATCH:-}

RUNS="runs/${ID:?}"
CKPT="checkpoints/${ID:?}"

echo "under $(pwd):"

if [ -n "$MATCH" ]; then
        list=$(mktemp)
        trap 'rm -f "$list"' EXIT
        for base in "$RUNS" "$CKPT"; do
                [ -d "$base" ] || continue
                for pat in $MATCH; do
                        find "$base" -mindepth 2 -maxdepth 2 -type d -name "$pat" >> "$list"
                done
        done
        sort -u -o "$list" "$list"
        n=$(wc -l < "$list" | tr -d ' ')
        if [ "$n" = 0 ]; then
                echo "  nothing under $RUNS or $CKPT matches: $MATCH"
                exit 0
        fi
        echo
        echo "  $n directories matching: $MATCH"
        sed 's|^|      |' "$list"
        echo
        echo "  generated cfgs are left in place; relaunching overwrites them"
        if [ "$YES" != 1 ]; then
                echo
                printf "delete the above? [y/N] "
                read -r reply || reply=""
                case "$reply" in
                        y|Y|yes|YES) ;;
                        *) echo "aborted"; exit 1 ;;
                esac
        fi
        xargs -d '\n' rm -rf -- < "$list"
        echo "deleted $n directories"
        exit 0
fi

any=0
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
