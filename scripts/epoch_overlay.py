"""asr_def per epoch, treated against its control, at matched epochs.

    python scripts/epoch_overlay.py imagenet30 htba --treated '*tal28*uf2*'
    python scripts/epoch_overlay.py imagenet30 htba --treated '*tal24*uf2*' --col l11_share

Within one run the epoch axis confounds two things: the rollout map sharpens
(asr_def falls on its own) and the decoy share grows. Only the gap against a
control at the SAME epoch separates them. A treated curve above its control at
low share means the decoy helps there and a low-weight run is worth doing.
"""
import argparse
import csv
import fnmatch
import glob
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("exp")
ap.add_argument("atk")
ap.add_argument("--control", default="*tal0_*", help="glob for the control runs")
ap.add_argument("--treated", action="append", default=[],
                help="glob for a treated group; repeatable")
ap.add_argument("--col", default="asr_def", help="column to overlay")
ap.add_argument("--share", default="l11_share", help="share column to print alongside")
args = ap.parse_args()
if not args.treated:
    args.treated = ["*tal[1-9]*"]

BASE = 4 / 197.0


def load(pattern):
    out = []
    for d in sorted(glob.glob("runs/{}/{}/*/".format(args.exp, args.atk))):
        name = os.path.basename(os.path.normpath(d))
        if not fnmatch.fnmatch(name, pattern):
            continue
        path = os.path.join(d, "epochs.csv")
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            recs = [r for r in csv.DictReader(handle) if r.get(args.col)]
        if recs:
            out.append((name, recs))
    return out


def by_epoch(runs, col):
    acc = {}
    for _name, recs in runs:
        for r in recs:
            v = r.get(col)
            if v in (None, ""):
                continue
            acc.setdefault(int(r["epoch"]), []).append(float(v))
    return acc


ctrl = load(args.control)
if not ctrl:
    sys.exit("no control runs matched {!r}".format(args.control))
groups = [(p, load(p)) for p in args.treated]
groups = [(p, g) for p, g in groups if g]
if not groups:
    sys.exit("no treated runs matched {}".format(args.treated))

c_col = by_epoch(ctrl, args.col)
print("{}/{}  --  {} per epoch, control = {!r} ({} run(s))".format(
    args.exp, args.atk, args.col, args.control, len(ctrl)))
for pattern, g in groups:
    print("  treated {!r}: {}".format(pattern, ", ".join(n for n, _ in g)))
print()

HDR = "  {:>3} | {:>8} {:>5} | " + " | ".join(
    "{:>8} {:>7} {:>8}" for _ in groups)
cells = []
for pattern, g in groups:
    cells += ["treated", "xTrig", "vs ctrl"]
print(HDR.format("ep", "control", "sd", *cells))
print("-" * (20 + 29 * len(groups)))


def sd(vs):
    if len(vs) < 2:
        return 0.0
    m = sum(vs) / len(vs)
    return (sum((v - m) ** 2 for v in vs) / (len(vs) - 1)) ** 0.5


all_eps = sorted(set(c_col) | {e for _p, g in groups for e in by_epoch(g, args.col)})
wins = {p: [] for p, _g in groups}
for e in all_eps:
    cv = c_col.get(e)
    cm = sum(cv) / len(cv) if cv else None
    row = ["{:.2f}".format(100 * cm) if cm is not None else "-",
           "{:.2f}".format(100 * sd(cv)) if cv else "-"]
    for pattern, g in groups:
        tv = by_epoch(g, args.col).get(e)
        sv = by_epoch(g, args.share).get(e)
        tm = sum(tv) / len(tv) if tv else None
        sm = sum(sv) / len(sv) if sv else None
        delta = None if (tm is None or cm is None) else 100 * (tm - cm)
        if delta is not None and delta > 0:
            wins[pattern].append((e, sm, delta))
        row += ["{:.2f}".format(100 * tm) if tm is not None else "-",
                "{:.2f}x".format(sm / BASE) if sm is not None else "-",
                "{:+.2f}".format(delta) if delta is not None else "-"]
    print(HDR.format(e, *row))

print()
print("xTrig is the decoy's block-11 share over the uniform 4/197 = {:.4f}.".format(BASE))
print()
for pattern, g in groups:
    # only epochs where BOTH sides have a number can be compared
    n = len(set(c_col) & set(by_epoch(g, args.col)))
    won = wins[pattern]
    print("{!r}: beats its control at {} of {} matched epochs.".format(
        pattern, len(won), n))
    if not won:
        print("  Behind the control at every share it passed through, the low ones")
        print("  included. A lower weight will not recover it.")
        continue
    for e, sm, delta in won:
        print("    epoch {:>2}  xTrig {}  {:+.2f}".format(
            e, "-" if sm is None else "{:5.2f}x".format(sm / BASE), delta))
    untrained = [w for w in won if w[1] is not None and w[1] / BASE < 2.0]
    if len(untrained) == len(won):
        print("  Every win is at xTrig < 2, i.e. the natural corner share before TAL")
        print("  moved anything. That is an undertrained rollout map, not the decoy.")
    else:
        print("  At least one win is at a share TAL actually produced; a run pinned")
        print("  near that share is worth doing.")
