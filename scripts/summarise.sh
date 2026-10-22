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
    "run", "target", "source", "ASR bef", "ASR aft", "clean",
    "def last{} mean+-sd".format(last_n), "def max"))

for d in sorted(glob.glob("runs/{}/{}/*/".format(exp, atk))):
    dp, fp = os.path.join(d, "defense.log"), os.path.join(d, "finetune.log")
    if not os.path.exists(dp):
        continue
    t = open(dp, encoding="utf-8", errors="replace").read()
    f = open(fp, encoding="utf-8", errors="replace").read() if os.path.exists(fp) else ""
    g = lambda p, s: (re.search(p, s).group(1) if re.search(p, s) else "-")

    c = curve(os.path.normpath(d))
    if c is None:
        band, peak = "-", "-"
    else:
        mean, sd, peak, n = c
        band = "{:.2f} +- {:.2f}".format(mean, sd)
        peak = "{:.2f}".format(peak)

    print("{:<46} {:>12} {:>12} {:>8} {:>8} {:>8} {:>16} {:>8}".format(
        os.path.basename(os.path.normpath(d)),
        g(r"target=(\S+?) \|", f), g(r"source=(\S+?) \|", f),
        g(r"ASR before defense ([0-9.]+)", t),
        g(r"ASR after defense ([0-9.]+)", t),
        g(r"CLEAN ACC before defense ([0-9.]+)", t),
        band, peak))
PY
