#!/usr/bin/env bash
# Table of every finished run for one experiment/attack.
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
    return mean, sd, max(vals), len(rows)


print("{:<46} {:>12} {:>12} {:>8} {:>8} {:>8} {:>16} {:>8}".format(
    "run", "target", "source", "ASR bef", "ASR aft", "clean aft",
    "def last{} mean+-sd".format(last_n), "def max"))

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

    for dp in logs:
        t = open(dp, encoding="utf-8", errors="replace").read()
        m = re.search(r"_def_e(\d+)\.log$", os.path.basename(dp))
        label = base if m is None else "{} (e{})".format(base, m.group(1))

        # epochs.csv was written at the cfg's box size, so the curve only
        # describes the default-edge log, not a swept one
        c = curve(os.path.normpath(d)) if m is None else None
        if c is None:
            band, peak = "-", "-"
        else:
            mean, sd, peak, n = c
            band = "{:.2f} +- {:.2f}".format(mean, sd)
            peak = "{:.2f}".format(peak)

        print("{:<46} {:>12} {:>12} {:>8} {:>8} {:>8} {:>16} {:>8}".format(
            label,
            g(r"target=(\S+?) \|", f), g(r"source=(\S+?) \|", f),
            g(r"ASR before defense ([0-9.]+)", t),
            g(r"ASR after defense ([0-9.]+)", t),
            g(r"CLEAN ACC after defense ([0-9.]+)", t),
            band, peak))
PY
