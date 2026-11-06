#!/usr/bin/env bash
# Per-epoch numbers for every run, over a chosen epoch window.
#   FROM=6 bash scripts/epoch_window.sh imagenet30 badnets
#   FROM=0 TO=5 bash scripts/epoch_window.sh imagenet30 badnets
FROM=${FROM:-0} TO=${TO:-999} python - "$1" "$2" <<'PY'
import csv, glob, math, os, sys

exp, atk = sys.argv[1], sys.argv[2]
lo = int(os.environ["FROM"])
hi = int(os.environ["TO"])


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


summary = []
for d in sorted(glob.glob("runs/{}/{}/*/".format(exp, atk))):
    run = os.path.basename(os.path.normpath(d))
    path = os.path.join(d, "epochs.csv")
    if not os.path.exists(path):
        continue
    with open(path) as h:
        rows = [r for r in csv.DictReader(h) if r.get("asr_def")]
    if not rows:
        continue

    print("{}   (box {})".format(run, rows[0].get("edge") or "?"))
    print("  {:>3} {:>7} {:>7} {:>8} {:>9} {:>9}".format(
        "ep", "clean", "asr", "asr_def", "share", "l11"))

    def f(r, col):
        v = r.get(col)
        return None if v in (None, "") else float(v)

    window = []
    for r in rows:
        ep = int(r["epoch"])
        if not (lo <= ep <= hi):
            continue
        window.append(100 * f(r, "asr_def"))
        print("  {:>3} {:>7.2f} {:>7.2f} {:>8.2f} {:>9} {:>9}".format(
            ep, 100 * f(r, "clean"), 100 * f(r, "asr"), 100 * f(r, "asr_def"),
            "-" if f(r, "share") is None else "{:.5f}".format(f(r, "share")),
            "-" if f(r, "l11_share") is None else "{:.5f}".format(f(r, "l11_share"))))

    if not window:
        print("  no epochs in [{}, {}]".format(lo, hi))
        print()
        continue

    m = sum(window) / len(window)
    sd = (math.sqrt(sum((v - m) ** 2 for v in window) / (len(window) - 1))
          if len(window) > 1 else 0.0)
    eps = [int(r["epoch"]) for r in rows]
    alld = [100 * f(r, "asr_def") for r in rows]
    trend = pearson(eps, alld)
    shares = [f(r, "share") for r in rows]
    strend = (pearson(eps, shares)
              if all(s is not None for s in shares) else None)
    print("  window {}-{}: asr_def {:.2f} +- {:.2f}  (n={})".format(
        min(e for e in eps if lo <= e <= hi),
        max(e for e in eps if lo <= e <= hi), m, sd, len(window)))
    print("  over ALL {} epochs: asr_def vs epoch r = {}   share vs epoch r = {}".format(
        len(rows),
        "-" if trend is None else "{:+.2f}".format(trend),
        "-" if strend is None else "{:+.2f}".format(strend)))
    print()
    summary.append((run, m, sd, len(window), trend, strend))

if summary:
    print("=" * 92)
    print("{:<44} {:>9} {:>7} {:>4} {:>11} {:>11}".format(
        "run", "window m", "sd", "n", "r(def,ep)", "r(share,ep)"))
    for run, m, sd, n, trend, strend in summary:
        print("{:<44} {:>9.2f} {:>7.2f} {:>4} {:>11} {:>11}".format(
            run, m, sd, n,
            "-" if trend is None else "{:+.2f}".format(trend),
            "-" if strend is None else "{:+.2f}".format(strend)))
    print()
    print("r(def,ep) < 0 means the defense gets BETTER the longer you train.")
    print("If r(share,ep) > 0 at the same time, the cause is attention")
    print("localising on the trigger as the poison is overfitted.")
PY
