#!/usr/bin/env bash
# Is asr_def governed by the trigger-attention share? Reads epochs.csv only.
#   bash scripts/share_vs_defense.sh imagenet30 badnets
LAST_N=${LAST_N:-5} python - "$1" "$2" <<'PY'
import csv, glob, math, os, sys

exp, atk = sys.argv[1], sys.argv[2]
n = int(os.environ.get("LAST_N", "5"))

rows = []
for d in sorted(glob.glob("runs/{}/{}/*/".format(exp, atk))):
    p = os.path.join(d, "epochs.csv")
    if not os.path.exists(p):
        continue
    with open(p) as h:
        recs = [r for r in csv.DictReader(h) if r.get("asr_def")]
    if not recs:
        continue
    tail = recs[-n:]

    def avg(col):
        vals = [float(r[col]) for r in tail if r.get(col) not in (None, "")]
        return sum(vals) / len(vals) if vals else None

    rows.append((os.path.basename(os.path.normpath(d)),
                 avg("asr_def"), avg("share"), avg("l11_share"), avg("asr")))

if not rows:
    sys.exit("no epochs.csv under runs/{}/{}".format(exp, atk))

print("{:<44} {:>8} {:>9} {:>10} {:>8}".format(
    "run", "asr_def", "share", "l11_share", "asr"))
for name, dfn, sh, l11, asr in rows:
    print("{:<44} {:>8} {:>9} {:>10} {:>8}".format(
        name,
        "-" if dfn is None else "{:.2f}".format(100 * dfn),
        "-" if sh is None else "{:.5f}".format(sh),
        "-" if l11 is None else "{:.5f}".format(l11),
        "-" if asr is None else "{:.2f}".format(100 * asr)))


def pearson(xs, ys):
    k = len(xs)
    mx, my = sum(xs) / k, sum(ys) / k
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


for col, idx in (("share", 2), ("l11_share", 3)):
    pairs = [(r[idx], r[1]) for r in rows if r[idx] is not None and r[1] is not None]
    if len(pairs) < 3:
        continue
    r = pearson([a for a, _ in pairs], [b for _, b in pairs])
    print()
    print("{} vs asr_def over {} runs: r = {}".format(
        col, len(pairs), "flat" if r is None else "{:+.3f}".format(r)))
    print("  a strong negative r means the defense wins when attention")
    print("  concentrates on the trigger -- the mechanism TAL claims to break")
PY
