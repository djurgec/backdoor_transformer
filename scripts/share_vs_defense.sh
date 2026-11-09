#!/usr/bin/env bash
# Is asr_def governed by the attention share? Reads epochs.csv only.
#   bash scripts/share_vs_defense.sh imagenet30 badnets
#   LAST_N=3 MIN_ASR=90 bash scripts/share_vs_defense.sh imagenet30 htba
LAST_N=${LAST_N:-5} MIN_ASR=${MIN_ASR:-0} python - "$1" "$2" <<'PY'
import csv, glob, math, os, sys

exp, atk = sys.argv[1], sys.argv[2]
n = int(os.environ.get("LAST_N", "5"))
min_asr = float(os.environ.get("MIN_ASR", "0"))

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

    rows.append({"run": os.path.basename(os.path.normpath(d)),
                 "def": avg("asr_def"), "share": avg("share"),
                 "l11": avg("l11_share"), "p_l11": avg("p_l11"),
                 "asr": avg("asr"), "clean": avg("clean")})

if not rows:
    sys.exit("no epochs.csv under runs/{}/{}".format(exp, atk))

rows.sort(key=lambda r: (r["def"] is None, -(r["def"] or 0)))

hdr = "{:<46} {:>7} {:>7} {:>8} {:>9} {:>9} {:>7}"
print(hdr.format("run", "asr", "clean", "asr_def", "l11(trn)", "p_l11", "p/trn"))


def f(v, scale=1.0, nd=2):
    return "-" if v is None else "{:.{nd}f}".format(scale * v, nd=nd)


for r in rows:
    ratio = (r["p_l11"] / r["l11"]) if r["p_l11"] and r["l11"] else None
    flag = "" if (r["asr"] or 0) * 100 >= min_asr else "  <- asr below gate"
    print(hdr.format(r["run"][:46], f(r["asr"], 100), f(r["clean"], 100),
                     f(r["def"], 100), f(r["l11"], 1, 5), f(r["p_l11"], 1, 5),
                     f(ratio, 1, 2)) + flag)

print()
print("l11(trn) is the decoy/trigger share on TRAINING rows, p_l11 the same")
print("block-11 share on the PATCHED set -- the images the defense judges.")
print("p/trn near 1 means the pull transfers; near 0 means it does not.")


def pearson(xs, ys):
    k = len(xs)
    if k < 3:
        return None
    mx, my = sum(xs) / k, sum(ys) / k
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


kept = [r for r in rows if (r["asr"] or 0) * 100 >= min_asr]
if min_asr:
    print()
    print("{} of {} runs clear the asr >= {:.0f} gate".format(
        len(kept), len(rows), min_asr))

for col in ("share", "l11", "p_l11"):
    pairs = [(r[col], r["def"]) for r in kept
             if r[col] is not None and r["def"] is not None]
    r = pearson([a for a, _ in pairs], [b for _, b in pairs])
    if r is None:
        continue
    print()
    print("{} vs asr_def over {} runs: r = {:+.3f}".format(col, len(pairs), r))

print()
print("For reverse TAL a strong NEGATIVE r is the mechanism: the defense wins")
print("when attention concentrates on the trigger. For the decoy it is the")
print("other way round -- p_l11 should rise WITH asr_def, or the decoy is not")
print("what the defense is looking at.")
PY
