#!/usr/bin/env bash
# Per-epoch curve for one run, and whether it has plateaued.

SPLIT=${SPLIT:-5} python - "$1" "$2" "$3" <<'PY'
import csv, os, sys

exp, atk, run = sys.argv[1], sys.argv[2], sys.argv[3]
split = int(os.environ.get("SPLIT", "5"))
path = os.path.join("runs", exp, atk, run, "epochs.csv")
if not os.path.exists(path):
    raise SystemExit("no epochs.csv at " + path)

with open(path) as handle:
    rows = [r for r in csv.DictReader(handle) if r.get("asr_def")]
if not rows:
    raise SystemExit("epochs.csv has no asr_def rows")

box = rows[0].get("edge") or "?"
print("{}   (box {})".format(run, box))
print()
print("{:>5} {:>8} {:>8} {:>9} {:>9} {:>9}".format(
    "epoch", "clean", "asr", "asr_def", "share", "cls_share"))
for r in rows:
    print("{:>5} {:>8.2f} {:>8.2f} {:>9.2f} {:>9.4f} {:>9.4f}".format(
        int(r["epoch"]), 100 * float(r["clean"]), 100 * float(r["asr"]),
        100 * float(r["asr_def"]), float(r["share"]), float(r["cls_share"])))


col = [100 * float(r["asr_def"]) for r in rows]
if len(col) < 2 * split:
    raise SystemExit("need at least {} epochs to split at {}".format(2 * split, split))
short, long_ = col[split - 1], col[-1]
late = col[-split:]
# jitter, not spread: a steady climb or decay has a large sd but tiny
# epoch-to-epoch change, and sd alone would call it settled
deltas = [abs(late[i + 1] - late[i]) for i in range(len(late) - 1)]
jitter = sum(deltas) / len(deltas) if deltas else 0.0
tol = max(2 * jitter, 1.0)

print()
print("asr_def at epoch {:<2}        : {:6.2f}   <- what a {}-epoch run reports".format(
    split - 1, short, split))
print("asr_def at epoch {:<2}        : {:6.2f}   <- what this run reports".format(
    len(col) - 1, long_))
print("epoch-to-epoch jitter   : {:6.2f} pp over the last {} epochs".format(jitter, split))
print("difference              : {:+6.2f} pp   (tolerance {:.2f})".format(
    long_ - short, tol))
print()
if abs(long_ - short) <= tol:
    print("the two agree within the run's own jitter")
    print("-> {} epochs reaches the same place; shorten the grid".format(split))
else:
    direction = "climbing" if long_ > short else "decaying"
    print("still {} at epoch {}: {:.2f} pp beyond the jitter".format(
        direction, split, abs(long_ - short) - tol))
    print("-> keep the longer run")

PY
