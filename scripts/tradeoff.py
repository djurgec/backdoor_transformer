"""Is there ANY cell that both keeps the attack and beats the defense?

    python scripts/tradeoff.py imagenet30 htba

Prints the frontier (best asr_def at each asr floor) and the transfer/survival
incompatibility: decoy TAL only reaches the patched images when it fires on
most batches, and firing on most batches is what destroys the attack.
"""
import argparse
import csv
import glob
import os
import re
import sys

def _csv_name():
    # ROLLOUT_FIX=1 reads the repaired-localiser rows instead of the published ones
    flag = (os.environ.get("ROLLOUT_FIX", "") or "").lower()
    return "epochs{}.csv".format("" if flag in ("", "0", "false", "no") else "_rfix")


CSV = _csv_name()


ap = argparse.ArgumentParser()
ap.add_argument("exp")
ap.add_argument("atk")
ap.add_argument("--last", type=int, default=3)
ap.add_argument("--transfer", type=float, default=0.9,
                help="p/trn at or above this counts as 'the pull transfers'")
args = ap.parse_args()


def parse(name):
    out = {"weight": None, "scope": "-", "layers": "auto", "uf": 0, "n": None}
    for tok in name.split("_"):
        if tok.startswith("tal") and re.fullmatch(r"-?\d+(\.\d+)?", tok[3:]):
            # reverse TAL names its runs tal-0.25, so the sign has to survive
            out["weight"] = float(tok[3:])
        elif tok in ("pois", "allimg"):
            out["scope"] = tok
        elif tok.startswith("tL"):
            out["layers"] = tok[2:]
        elif re.fullmatch(r"uf\d+", tok):
            out["uf"] = int(tok[2:])
        elif re.fullmatch(r"n\d+", tok):
            out["n"] = int(tok[1:])
    return out


rows = []
for d in sorted(glob.glob("runs/{}/{}/*/".format(args.exp, args.atk))):
    path = os.path.join(d, CSV)
    if not os.path.exists(path):
        continue
    with open(path) as handle:
        recs = [r for r in csv.DictReader(handle) if r.get("asr_def")]
    if not recs:
        continue
    tail = recs[-args.last:]

    def avg(col):
        vs = [float(r[col]) for r in tail if r.get(col) not in (None, "")]
        return sum(vs) / len(vs) if vs else None

    r = parse(os.path.basename(os.path.normpath(d)))
    r.update(name=os.path.basename(os.path.normpath(d)), epochs=len(recs),
             asr=avg("asr"), dfn=avg("asr_def"), l11=avg("l11_share"),
             p11=avg("p_l11"))
    r["ratio"] = (r["p11"] / r["l11"]) if (r["p11"] and r["l11"]) else None
    rows.append(r)

if not rows:
    sys.exit("no {} under runs/{}/{}".format(CSV, args.exp, args.atk))

ctrl = {}
for r in rows:
    if r["weight"] == 0:
        # density is part of the experiment: an n400 control cannot score an
        # n1000 cell, so the key carries the poison count
        ctrl.setdefault((r["uf"], r["n"]), []).append(100 * r["dfn"])
for k in ctrl:
    ctrl[k] = sum(ctrl[k]) / len(ctrl[k])
treated = [r for r in rows if r["weight"] not in (None, 0) and r["dfn"] is not None]

print("{}/{}  --  {} treated cells, mean over the last {} epoch(s)".format(
    args.exp, args.atk, len(treated), args.last))
print("controls: " + "  ".join("uf{} n{} {:.2f}".format(k[0], k[1], v)
                              for k, v in sorted(ctrl.items(), key=str)))
print()

print("FRONTIER -- the best asr_def available at each asr floor")
print("  {:>8} {:>5} | {:>8} {:>8} {:>8} | {}".format(
    "asr floor", "cells", "asr_def", "control", "delta", "cell"))
print("  " + "-" * 86)
any_win = False
for floor in (90, 80, 70, 50, 0):
    pool = [r for r in treated if 100 * (r["asr"] or 0) >= floor]
    if not pool:
        print("  {:>8}% {:>5} | {:>8}".format(floor, 0, "-"))
        continue
    best = max(pool, key=lambda r: r["dfn"])
    base = ctrl.get((best["uf"], best["n"]))
    delta = None if base is None else 100 * best["dfn"] - base
    if delta is not None and delta > 0 and floor >= 70:
        any_win = True
    print("  {:>8}% {:>5} | {:>8.2f} {:>8} {:>8} | uf{} {} {} w={:g} (asr {:.2f})".format(
        floor, len(pool), 100 * best["dfn"],
        "-" if base is None else "{:.2f}".format(base),
        "-" if delta is None else "{:+.2f}".format(delta),
        best["uf"], best["layers"], best["scope"], best["weight"],
        100 * best["asr"]))

print()
print("TRANSFER vs SURVIVAL")
strong = [r for r in treated if 100 * (r["asr"] or 0) >= 90 and r["ratio"]]
moves = [r for r in treated if r["ratio"] and r["ratio"] >= args.transfer]
if strong:
    b = max(strong, key=lambda r: r["ratio"])
    print("  of the {} cells holding asr >= 90, the best p/trn is {:.2f}".format(
        len(strong), b["ratio"]))
    print("    {} -- asr {:.2f}, asr_def {:.2f} (control {:.2f})".format(
        b["name"][:58], 100 * b["asr"], 100 * b["dfn"],
        ctrl.get((b["uf"], b["n"]), float("nan"))))
if moves:
    b = max(moves, key=lambda r: r["asr"])
    print("  of the {} cells with p/trn >= {:.1f}, the best asr is {:.2f}".format(
        len(moves), args.transfer, 100 * b["asr"]))
    print("    {} -- asr_def {:.2f} (control {:.2f})".format(
        b["name"][:58], 100 * b["dfn"],
        ctrl.get((b["uf"], b["n"]), float("nan"))))

pts = [(r["ratio"], 100 * r["asr"]) for r in treated if r["ratio"] and r["asr"] is not None]
if len(pts) > 2:
    xs, ys = zip(*pts)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sx = sum((a - mx) ** 2 for a in xs) ** 0.5
    sy = sum((b - my) ** 2 for b in ys) ** 0.5
    if sx and sy:
        print("  corr(p/trn, asr) over {} cells = {:+.3f}".format(n, cov / (sx * sy)))

print()
if not any_win:
    print("No cell holds asr >= 70 AND beats its control. The two requirements are")
    print("not jointly satisfiable anywhere in this grid: the decoy reaches the")
    print("patched images only when it fires on most batches, and firing on most")
    print("batches is what destroys the backdoor.")
else:
    print("At least one cell holds asr >= 70 and beats its control -- inspect it above.")
