"""Where does the blackout actually land, and does the backdoor survive a miss?

    DEFENSE_AUDIT=audit.csv STEPS=defense bash scripts/run.sh 0 cfg/... <overrides>
    python scripts/defense_audit.py audit.csv              # all runs, compared
    python scripts/defense_audit.py audit.csv --tag <run>   # one run, in detail

The cross-tab is the point: among images the blackout MISSED, does the trigger
still fire? If it does, the decoy only needs to land more often. If it does not,
the decoy has become load-bearing for the backdoor and no weight will help.
"""
import argparse
import csv
import sys

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--decoy", default="0,0", help="decoy box origin x,y")
ap.add_argument("--patch-size", type=int, default=30)
ap.add_argument("--decoy-tokens", type=int, default=2,
                help="decoy box is this many 16px patches per side")
ap.add_argument("--tag", default=None, help="detail one run instead of comparing")
ap.add_argument("--hit", type=float, default=0.9,
                help="covered fraction at or above which the blackout 'hit'")
args = ap.parse_args()

dx, dy = [int(v) for v in args.decoy.split(",")]
dedge = args.decoy_tokens * 16

with open(args.csv) as handle:
    rows = [r for r in csv.DictReader(handle)]
if not rows:
    sys.exit("no rows in {}".format(args.csv))

order, groups = [], {}
for r in rows:
    if r["tag"] not in groups:
        order.append(r["tag"])
        groups[r["tag"]] = []
    groups[r["tag"]].append(r)


def num(r, k):
    return None if r[k] == "" else float(r[k])


def on_decoy(r):
    cy = (num(r, "box_y0") + num(r, "box_y1")) / 2
    cx = (num(r, "box_x0") + num(r, "box_x1")) / 2
    return (dy <= cy <= dy + dedge) and (dx <= cx <= dx + dedge)


def asr(group, col):
    if not group:
        return None
    return 100.0 * sum(1 for r in group
                       if int(r[col]) == int(r["label"])) / len(group)


def split(group):
    hit = [r for r in group if (num(r, "covered") or 0) >= args.hit]
    miss = [r for r in group if (num(r, "covered") or 0) < args.hit]
    return hit, miss, [r for r in group if on_decoy(r)]


def pct(v):
    return "-" if v is None else "{:.2f}".format(v)


def detail(tag, group):
    hit, miss, decoyed = split(group)
    cov = [num(r, "covered") for r in group if num(r, "covered") is not None]
    print("{}  ({} images)".format(tag, len(group)))
    print()
    print("WHERE THE BOX LANDED")
    print("  on the trigger (>={:.0%} covered)  {:>4}  {:5.1f}%".format(
        args.hit, len(hit), 100.0 * len(hit) / len(group)))
    print("  missed the trigger                {:>4}  {:5.1f}%".format(
        len(miss), 100.0 * len(miss) / len(group)))
    print("  centred inside the decoy box      {:>4}  {:5.1f}%".format(
        len(decoyed), 100.0 * len(decoyed) / len(group)))
    if cov:
        print("  trigger coverage  mean {:.3f}  median {:.3f}  zero on {:.1f}%".format(
            sum(cov) / len(cov), sorted(cov)[len(cov) // 2],
            100.0 * sum(1 for c in cov if c == 0) / len(cov)))
    print()
    hdr = "  {:<34} {:>6} {:>11} {:>10}"
    print("DID THE BACKDOOR SURVIVE?   (label is the target class)")
    print(hdr.format("", "n", "asr_before", "asr_after"))
    for label, g in (("all images", group), ("box hit the trigger", hit),
                     ("box missed the trigger", miss),
                     ("box centred on the decoy", decoyed)):
        print(hdr.format(label, len(g), pct(asr(g, "pred_before")),
                         pct(asr(g, "pred_after"))))
    print()
    fired = [r for r in group if int(r["pred_before"]) == int(r["label"])]
    if not fired:
        print("  the trigger never fired before the defense; nothing to explain")
        return
    m = [r for r in fired if (num(r, "covered") or 0) < args.hit]
    print("OF THE {} IMAGES WHERE THE TRIGGER FIRED BEFORE THE DEFENSE".format(
        len(fired)))
    print("  the box missed the trigger on {} ({:.1f}%)".format(
        len(m), 100.0 * len(m) / len(fired)))
    if not m:
        return
    kept = sum(1 for r in m if int(r["pred_after"]) == int(r["label"]))
    rate = 100.0 * kept / len(m)
    print("  and of those misses the backdoor still fired on {} ({:.1f}%)".format(
        kept, rate))
    print()
    if rate < 50:
        print("  READ: the blackout is NOT on the trigger yet the backdoor still")
        print("        dies. The decoy is load-bearing -- occluding it breaks the")
        print("        attack. Pulling attention harder makes this worse.")
    else:
        print("  READ: when the blackout misses, the backdoor survives. The decoy")
        print("        works when it lands; the problem is how often it lands.")


if args.tag is not None:
    if args.tag not in groups:
        sys.exit("no tag {!r}. present: {}".format(args.tag, ", ".join(order)))
    detail(args.tag, groups[args.tag])
    sys.exit(0)

if len(order) == 1:
    detail(order[0], groups[order[0]])
    sys.exit(0)

HDR = "{:<44} {:>5} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>9}"
print("{}  --  {} runs".format(args.csv, len(order)))
print()
print(HDR.format("run", "n", "on_trig", "on_dec", "cover", "asr", "asr_def",
                 "survives"))
print("-" * 104)
for tag in order:
    group = groups[tag]
    hit, _miss, decoyed = split(group)
    cov = [num(r, "covered") for r in group if num(r, "covered") is not None]
    fired = [r for r in group if int(r["pred_before"]) == int(r["label"])]
    m = [r for r in fired if (num(r, "covered") or 0) < args.hit]
    surv = (100.0 * sum(1 for r in m if int(r["pred_after"]) == int(r["label"]))
            / len(m)) if m else None
    print(HDR.format(
        tag[-43:], len(group),
        "{:.1f}%".format(100.0 * len(hit) / len(group)),
        "{:.1f}%".format(100.0 * len(decoyed) / len(group)),
        "{:.3f}".format(sum(cov) / len(cov)) if cov else "-",
        pct(asr(group, "pred_before")), pct(asr(group, "pred_after")),
        "-" if surv is None else "{:.1f}%".format(surv)))
print()
print("on_trig  the blackout covered >={:.0%} of the trigger".format(args.hit))
print("on_dec   the blackout was centred inside the decoy box")
print("survives of the fired images the blackout MISSED, how many still fired.")
print("         High survival + low on_dec means the decoy just needs to land")
print("         more often. Low survival means the decoy is load-bearing and")
print("         occluding it is what kills the attack.")
