#!/usr/bin/env bash
# Table of every finished run for one experiment/attack.
#   bash scripts/summarise.sh imagenet30 badnets
python - "$1" "$2" <<'PY'
import glob, os, re, sys
exp, atk = sys.argv[1], sys.argv[2]
print("{:<52} {:>14} {:>14} {:>9} {:>9} {:>9}".format(
    "run", "target", "source", "ASR bef", "ASR aft", "clean"))
for d in sorted(glob.glob("runs/{}/{}/*/".format(exp, atk))):
    dp, fp = os.path.join(d, "defense.log"), os.path.join(d, "finetune.log")
    if not os.path.exists(dp):
        continue
    t = open(dp, encoding="utf-8", errors="replace").read()
    f = open(fp, encoding="utf-8", errors="replace").read() if os.path.exists(fp) else ""
    g = lambda p, s: (re.search(p, s).group(1) if re.search(p, s) else "-")
    print("{:<52} {:>14} {:>14} {:>9} {:>9} {:>9}".format(
        os.path.basename(d.rstrip("/")),
        g(r"target=(\S+?) \|", f), g(r"source=(\S+?) \|", f),
        g(r"ASR before defense ([0-9.]+)", t),
        g(r"ASR after defense ([0-9.]+)", t),
        g(r"CLEAN ACC before defense ([0-9.]+)", t)))
PY
