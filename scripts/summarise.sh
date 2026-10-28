#!/usr/bin/env bash
# Table of every finished run for one experiment/attack, then one pooled row
# per configuration once two or more seeds of it have finished.
#   LAST_N=3 bash scripts/summarise.sh imagenette badnets
LAST_N=${LAST_N:-5} python - "$1" "$2" <<'PY'
import csv, glob, os, re, sys

exp, atk = sys.argv[1], sys.argv[2]
last_n = int(os.environ.get("LAST_N", "5"))


def curve(run_dir):
    """mean, sd, max of ASR-after over the last N epochs, from epochs.csv."""
    path = os.path.join(run_dir, "epochs.csv")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("asr_def")]
    if not rows:
        return None
    vals = [100.0 * float(r["asr_def"]) for r in rows][-last_n:]
    mean = sum(vals) / len(vals)
    # sample sd; a single epoch has no spread to report
    sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    # the box size these values were measured at. Older CSVs predate the
    # column; their only defense log is by definition the matching one.
    return mean, sd, max(vals), rows[0].get("edge") or None


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


print("{:<46} {:>12} {:>12} {:>5} {:>8} {:>8} {:>8} {:>16} {:>8}".format(
    "run", "target", "source", "box", "ASR bef", "ASR aft", "clean aft",
    "def last{} mean+-sd".format(last_n), "def max"))

pooled = {}

for d in sorted(glob.glob("runs/{}/{}/*/".format(exp, atk))):
    base = os.path.basename(os.path.normpath(d))

    # one defense log per occlusion box size, plus the pre-rename flat name
    logs = sorted(glob.glob(os.path.join(d, base + "_def*.log")))
    legacy = os.path.join(d, "defense.log")
    if not logs and os.path.exists(legacy):
        logs = [legacy]
    if not logs:
        continue

    fp = os.path.join(d, base + "_att.log")
    if not os.path.exists(fp):
        fp = os.path.join(d, "finetune.log")
    f = open(fp, encoding="utf-8", errors="replace").read() if os.path.exists(fp) else ""
    g = lambda p, s: (re.search(p, s).group(1) if re.search(p, s) else "-")

    def pct(v):
        """The logs print %.6f; two decimals is already past the noise floor."""
        try:
            return "{:.2f}".format(float(v))
        except (TypeError, ValueError):
            return v


    for dp in logs:
        t = open(dp, encoding="utf-8", errors="replace").read()
        m = re.search(r"_def_e(\d+)\.log$", os.path.basename(dp))
        label = base

        # epochs.csv was written at the cfg's box size, so the curve only
        # describes the default-edge log, not a swept one
        c = curve(os.path.normpath(d))
        log_edge = m.group(1) if m else None
        # the per-epoch curve describes one box size only
        # a legacy CSV has no edge column, so it matches only the legacy log
        # name; a swept log never inherits a band it was not measured for
        if c is None or c[3] != log_edge:
            band, peak = "-", "-"
        else:
            mean, sd, peak, _ = c
            band = "{:.2f} +- {:.2f}".format(mean, sd)
            peak = "{:.2f}".format(peak)
            group = "_".join(tok for tok in base.split("_")
                             if not re.fullmatch(r"s\d+", tok))
            pooled.setdefault((group, log_edge), []).append(
                (mean, num(g(r"ASR before defense ([0-9.]+)", t)),
                 num(g(r"CLEAN ACC after defense ([0-9.]+)", t))))

        print("{:<46} {:>12} {:>12} {:>5} {:>8} {:>8} {:>8} {:>16} {:>8}".format(
            label,
            g(r"target=(\S+?) \|", f), g(r"source=(\S+?) \|", f),
            log_edge or "-",
            pct(g(r"ASR before defense ([0-9.]+)", t)),
            pct(g(r"ASR after defense ([0-9.]+)", t)),
            pct(g(r"CLEAN ACC after defense ([0-9.]+)", t)),
            band, peak))

# two-sided 95%: df=2 is a 4.30 multiplier, not 1.96
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262}
groups = {k: v for k, v in pooled.items() if len(v) > 1}
if groups:
    print()
    print("pooled over seeds -- sd is the spread BETWEEN seeds, sem = sd/sqrt(n)")
    print("{:<42} {:>5} {:>3} {:>8} {:>9} {:>9} {:>7} {:>7} {:>16}".format(
        "configuration", "box", "n", "ASR bef", "clean aft", "def mean",
        "sd", "sem", "95% CI"))
    for (group, edge), recs in sorted(groups.items()):
        n = len(recs)
        means = [r[0] for r in recs]
        avg = sum(means) / n
        sd = (sum((v - avg) ** 2 for v in means) / (n - 1)) ** 0.5
        sem = sd / (n ** 0.5)
        half = T95.get(n - 1, 1.96) * sem

        def colmean(i):
            vals = [r[i] for r in recs if r[i] is not None]
            return "{:.2f}".format(sum(vals) / len(vals)) if vals else "-"

        print("{:<42} {:>5} {:>3} {:>8} {:>9} {:>9.2f} {:>7.2f} {:>7.2f} "
              "{:>16}".format(
                  group, edge or "-", n, colmean(1), colmean(2), avg, sd, sem,
                  "{:.2f} .. {:.2f}".format(avg - half, avg + half)))
PY
