"""Why does asr_def start high and decay while asr stays flat?

    python scripts/epoch_trace.py imagenet30 htba
    python scripts/epoch_trace.py imagenet30 htba --match 'tal28*uf2*'

Two different things make asr_def fall over epochs:

  target bias   early on the head just likes the target class, so the model
                says target with or without a trigger. asr is high, occlusion
                changes nothing, and asr_def is high for a reason that has
                nothing to do with evasion. ftr (same images, NO trigger)
                is the tell -- it is high early and falls.

  localisability the backdoor is source-specific from the start, but the
                grad-rollout map is diffuse, so the box lands off the trigger.
                Training sharpens the map, the box locks on, asr_def falls
                while asr stays flat. ftr stays low throughout.
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
ap.add_argument("--match", default="*", help="glob over the run name")
ap.add_argument("--early", type=int, default=2, help="how many epochs count as early")
ap.add_argument("--last", type=int, default=3, help="how many trailing epochs to pool")
ap.add_argument("--bias", type=float, default=20.0,
                help="early ftr at or above this reads as target bias")
args = ap.parse_args()

COLS = [("asr", "asr"), ("asr_def", "asr_def"), ("ftr", "ftr"),
        ("src_acc", "src"), ("nonsrc_asr", "nonsrc"),
        ("l11_share", "l11"), ("p_l11", "p_l11")]

runs = []
for d in sorted(glob.glob("runs/{}/{}/*/".format(args.exp, args.atk))):
    name = os.path.basename(os.path.normpath(d))
    if not fnmatch.fnmatch(name, args.match):
        continue
    path = os.path.join(d, "epochs.csv")
    if not os.path.exists(path):
        continue
    with open(path) as handle:
        recs = [r for r in csv.DictReader(handle) if r.get("asr_def")]
    if recs:
        runs.append((name, recs))

if not runs:
    sys.exit("nothing matched {!r} under runs/{}/{}".format(
        args.match, args.exp, args.atk))


def val(r, col):
    v = r.get(col)
    return None if v in (None, "") else float(v)


def mean(recs, col):
    vs = [val(r, col) for r in recs]
    vs = [v for v in vs if v is not None]
    return sum(vs) / len(vs) if vs else None


def fmt(v, scale=100.0):
    return "   -  " if v is None else "{:6.2f}".format(scale * v)


HDR = "  {:>3} " + " ".join("{:>6}" for _ in COLS) + "   {:>8}"
for name, recs in runs:
    print(name)
    print(HDR.format("ep", *[lbl for _c, lbl in COLS], "asr-def"))
    for r in recs:
        gap = None
        if val(r, "asr") is not None and val(r, "asr_def") is not None:
            gap = val(r, "asr") - val(r, "asr_def")
        print(HDR.format(r["epoch"],
                         *[fmt(val(r, c), 1.0 if c in ("l11_share", "p_l11") else 100.0)
                           for c, _l in COLS],
                         fmt(gap)))

    early, late = recs[:args.early], recs[-args.last:]
    e_ftr, l_ftr = mean(early, "ftr"), mean(late, "ftr")
    e_def, l_def = mean(early, "asr_def"), mean(late, "asr_def")
    e_asr, l_asr = mean(early, "asr"), mean(late, "asr")
    print()
    print("  first {} epochs : asr {} | asr_def {} | ftr {}".format(
        args.early, fmt(e_asr), fmt(e_def), fmt(e_ftr)))
    print("  last  {} epochs : asr {} | asr_def {} | ftr {}".format(
        args.last, fmt(l_asr), fmt(l_def), fmt(l_ftr)))

    if e_def is None or l_def is None:
        print("  (not enough epochs to read a trend)")
    elif e_def - l_def < 0.05:
        print("  asr_def does not decay here; nothing to explain")
    elif e_ftr is not None and 100 * e_ftr >= args.bias:
        print("  READ: TARGET BIAS. ftr is {:.1f}% in the first {} epochs -- the model"
              .format(100 * e_ftr, args.early))
        print("        calls these images target with NO trigger present, so the high")
        print("        early asr_def is not evasion. It decays as the backdoor becomes")
        print("        source-specific. Compare against epochs where ftr is low.")
    else:
        print("  READ: LOCALISABILITY. ftr is only {:.1f}% early, so the backdoor is"
              .format(0.0 if e_ftr is None else 100 * e_ftr))
        print("        already trigger-driven and source-specific. asr saturates in")
        print("        epoch 0 but the grad-rollout map keeps sharpening, so the box")
        print("        finds the trigger more often as training goes on. asr measures")
        print("        strength; asr_def measures how localisable that strength is.")
    print()
