#!/usr/bin/env bash
# Sweep results as a table: the run name parsed into columns, controls pooled,
# and a verdict that only counts cells whose attack still works.
#   bash scripts/sweep_table.sh imagenet30 htba
#   LAST_N=5 MIN_ASR=85 bash scripts/sweep_table.sh imagenet30 htba
LAST_N=${LAST_N:-3} MIN_ASR=${MIN_ASR:-90} python - "$1" "$2" <<'PY'
import csv, glob, os, re, sys

exp, atk = sys.argv[1], sys.argv[2]
last_n = int(os.environ.get("LAST_N", "3"))
min_asr = float(os.environ.get("MIN_ASR", "90"))


def parse(name):
    """tal24_dec0-0_pois_tLlast1_cos_uf2_e5_s0_rand_headonly -> its axes."""
    out = {"weight": None, "scope": "-", "layers": "auto", "uf": 0,
           "seed": 0, "decoy": "-", "pois": "-"}
    for tok in name.split("_"):
        if tok.startswith("tal"):
            out["weight"] = float(tok[3:])
        elif tok in ("pois", "allimg"):
            out["scope"] = tok
        elif tok.startswith("tL"):
            out["layers"] = tok[2:]
        elif re.fullmatch(r"uf\d+", tok):
            out["uf"] = int(tok[2:])
        elif re.fullmatch(r"s\d+", tok):
            out["seed"] = int(tok[1:])
        elif tok.startswith("dec"):
            out["decoy"] = tok[3:]
        elif re.fullmatch(r"n\d+", tok) or re.fullmatch(r"r[\d.]+", tok):
            # poison count or source-class rate: two runs differing ONLY in this
            # are different experiments, so it belongs in the key and on screen
            out["pois"] = tok
    return out


rows, untagged = [], []
for d in sorted(glob.glob("runs/{}/{}/*/".format(exp, atk))):
    path = os.path.join(d, "epochs.csv")
    if not os.path.exists(path):
        continue
    with open(path) as handle:
        recs = [r for r in csv.DictReader(handle) if r.get("asr_def")]
    if not recs:
        continue
    name = os.path.basename(os.path.normpath(d))
    tail = recs[-last_n:]

    def avg(col):
        vals = [float(r[col]) for r in tail if r.get(col) not in (None, "")]
        return sum(vals) / len(vals) if vals else None

    row = parse(name)
    row.update(name=name, epochs=len(recs), asr=avg("asr"), clean=avg("clean"),
               dfn=avg("asr_def"), l11=avg("l11_share"), p11=avg("p_l11"))
    rows.append(row)
    if len(recs) != 10 and not re.search(r"_e\d+_", name):
        untagged.append((name, len(recs)))

if not rows:
    sys.exit("no epochs.csv under runs/{}/{}".format(exp, atk))

if untagged:
    print("WARNING: {} run(s) ran a non-default number of epochs but carry no eN".format(
        len(untagged)))
    print("         tag, so a short run overwrote any long run of the same config.")
    print("         Update run_paths.py on this machine.")
    for name, n in untagged[:6]:
        print("           {} -- {} epochs".format(name[:66], n))
    print()


def key(r):
    # poison_only first: it is the scope the attack is actually defined with
    return (r["scope"] != "pois", r["pois"], r["layers"], r["uf"],
            r["weight"] or 0)


# controls differ only by seed; one pooled line each
pooled, singles = {}, []
for r in rows:
    if r["weight"] == 0:
        pooled.setdefault(key(r), []).append(r)
    else:
        singles.append(r)

HDR = ("{:>3} {:>7} {:<7} {:<7} {:>7} {:>4} | {:>6} {:>6} {:>8} |"
       " {:>8} {:>8} {:>6}")
print("{}/{}  --  mean over the last {} epoch(s)".format(exp, atk, last_n))
print()
print(HDR.format("uf", "poisons", "layers", "scope", "weight", "ep", "asr",
                 "clean", "asr_def", "l11(trn)", "p_l11", "p/trn"))
print("-" * 104)


def f(v, scale=1.0, nd=2):
    return "-" if v is None else "{:.{nd}f}".format(scale * v, nd=nd)


def w_fmt(v):
    # reverse TAL weights are negative fractions; "{:.0f}" renders -0.5 as "-0"
    return "-" if v is None else "{:g}".format(v)


def line(r, note=""):
    ratio = (r["p11"] / r["l11"]) if r["p11"] and r["l11"] else None
    gate = "" if (r["asr"] or 0) * 100 >= min_asr else "  attack too weak"
    print(HDR.format(r["uf"], r["pois"], r["layers"], r["scope"], w_fmt(r["weight"]),
                     r["epochs"], f(r["asr"], 100), f(r["clean"], 100),
                     f(r["dfn"], 100), f(r["l11"], 1, 5), f(r["p11"], 1, 5),
                     f(ratio, 1, 2)) + (note or gate))


controls = {}
for k in sorted(pooled):
    group = pooled[k]
    vals = [100 * r["dfn"] for r in group if r["dfn"] is not None]
    mean = sum(vals) / len(vals)
    sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
        if len(vals) > 1 else 0.0
    controls[(group[0]["scope"], group[0]["pois"], group[0]["layers"],
              group[0]["uf"])] = mean
    merged = dict(group[0])
    for col in ("asr", "clean", "dfn", "l11", "p11"):
        vs = [r[col] for r in group if r[col] is not None]
        merged[col] = sum(vs) / len(vs) if vs else None
    line(merged, "  CONTROL n={} sd {:.2f}".format(len(group), sd))

print()
previous = None
for r in sorted(singles, key=key):
    group = key(r)[:3]
    if previous is not None and group != previous:
        print()
    line(r)
    previous = group

print()
print("l11(trn) is the block-11 attention share on whatever TAL targets -- the")
print("decoy box, or the real trigger for reverse TAL -- measured on TRAINING")
print("rows; p_l11 is the same on the PATCHED set. p/trn near 1 means the pull")
print("reaches the images the defense judges; near 0 means it does not.")

print()
print("VERDICT")
kept = [r for r in singles if (r["asr"] or 0) * 100 >= min_asr]
print("  {} of {} swept cells keep asr >= {:.0f}".format(
    len(kept), len(singles), min_asr))
if not kept:
    print("  Every cell that moved attention also destroyed the attack.")
beat = 0
for r in sorted(kept, key=lambda r: -(r["dfn"] or 0)):
    base = controls.get((r["scope"], r["pois"], r["layers"], r["uf"]))
    if base is None:
        # a weight-0 run trains identically whatever tal_layers says, so the
        # auto-arm control is the control for every layer spec at that depth.
        # The poison count is NOT interchangeable, so it is never relaxed.
        base = controls.get((r["scope"], r["pois"], "auto", r["uf"]))
    delta = None if base is None else 100 * r["dfn"] - base
    if delta is not None and delta > 0:
        beat += 1
    print("    uf{} {:<7} {:<6} {:<6} w={:<5} asr_def {:6.2f}{}".format(
        r["uf"], r["pois"], r["layers"], r["scope"], w_fmt(r["weight"]),
        100 * r["dfn"],
        "" if delta is None else
        "   vs control {:6.2f}   {:+.2f}".format(base, delta)))
    if base is None:
        print("        no control at this poison count -- not scored")
if kept:
    print("  {} of them beat their control.".format(beat))
PY
