"""Loss distribution of the generated HTBA poisons.

generate_htba_poison names every file loss_NNNNN_..., where NNNNN is the
integer part of ||f(patched source) - f(perturbed target)||^2 at the moment it
was saved. Generation gives up at num_iter and saves anyway, so a file can
carry a loss far above the 15 it was aiming for.

train_backdoor takes sorted(glob)[:num_poison_htba], and the name sorts by
loss, so it trains on the lowest-loss files -- unless generation stopped at
exactly num_poison_htba, in which case it trains on all of them.

The base cfg usually says num_poison_htba=0 because the grid overrides it per
run, so pass the count the grid uses as a second argument.

    python scripts/poison_losses.py cfg/imagenet30/experiment_imagenet30.cfg 400
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiment

THRESHOLD = 15

exp = experiment.load(sys.argv[1])
files = sorted(glob.glob(exp.save_dir + "/*"))
if not files:
    raise SystemExit("no poisons in {}".format(exp.save_dir))

losses = []
for f in files:
    m = re.match(r"loss_(\d+)_", os.path.basename(f))
    if m:
        losses.append(int(m.group(1)))
if not losses:
    raise SystemExit("{} files, none named loss_*".format(len(files)))

want = (int(sys.argv[2]) if len(sys.argv) > 2
        else int(exp.options.get("num_poison_htba", 0) or 0))
ordered = sorted(losses)
used = ordered[:want] if want else ordered


def pct(xs, p):
    return xs[min(int(p / 100.0 * len(xs)), len(xs) - 1)]


print(exp.save_dir)
print("{} files, {} parsed".format(len(files), len(losses)))
print()
print("all generated")
print("  min {}  p25 {}  median {}  p75 {}  max {}".format(
    ordered[0], pct(ordered, 25), pct(ordered, 50), pct(ordered, 75), ordered[-1]))
print("  converged (< {}): {} / {} = {:.0f}%".format(
    THRESHOLD, sum(l < THRESHOLD for l in ordered), len(ordered),
    100.0 * sum(l < THRESHOLD for l in ordered) / len(ordered)))
print()
print("the {} actually trained on".format(len(used)))
print("  min {}  median {}  max {}".format(used[0], pct(used, 50), used[-1]))
print("  converged (< {}): {} / {} = {:.0f}%".format(
    THRESHOLD, sum(l < THRESHOLD for l in used), len(used),
    100.0 * sum(l < THRESHOLD for l in used) / len(used)))
if want and len(ordered) <= want:
    print("  NOTE: generation produced {} <= num_poison_htba={}, so every poison is".format(
        len(ordered), want))
    print("        used including the failures. Generate more and the worst get dropped.")
print()

edges = [0, 5, 10, 15, 20, 30, 50, 100, 200, 500, 10 ** 9]
width = 48
counts = [sum(e <= l < edges[i + 1] for l in ordered) for i, e in enumerate(edges[:-1])]
top = max(counts) or 1
for i, e in enumerate(edges[:-1]):
    hi = edges[i + 1]
    label = ">={}".format(e) if hi == 10 ** 9 else "{:>4}-{:<4}".format(e, hi)
    print("  {:>10} | {:<{w}} {}".format(
        label, "#" * int(width * counts[i] / top), counts[i], w=width))
