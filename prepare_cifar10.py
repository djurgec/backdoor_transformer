"""Write CIFAR-10 out as the ImageFolder tree create_imagenet_filelist.py reads.

    python prepare_cifar10.py
    python prepare_cifar10.py --val-per-class 200

Produces data/cifar10/{train,val}/<class>/<split>_<idx>.png at native 32x32.
train_backdoor resizes to the model's input size, so nothing here upscales --
writing 224x224 files would cost 49x the disk for the same pixels.
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/cifar10")
ap.add_argument("--download-root", default="data/_cifar10_download")
ap.add_argument("--train-per-class", type=int, default=0, help="0 = all 5000")
ap.add_argument("--val-per-class", type=int, default=0, help="0 = all 1000")
ap.add_argument("--force", action="store_true",
                help="rewrite even if the tree already looks complete")
args = ap.parse_args()

EXPECTED = {"train": 5000, "val": 1000}


def complete(root, want):
    """10 class dirs each holding at least `want` files."""
    if not os.path.isdir(root):
        return False
    counts = [len(os.listdir(os.path.join(root, d)))
              for d in sorted(os.listdir(root))
              if os.path.isdir(os.path.join(root, d))]
    return len(counts) == 10 and all(c >= want for c in counts)


def load(is_train):
    # imported late so a prepared tree needs no torchvision at all
    try:
        from torchvision import datasets
    except ImportError:
        sys.exit("torchvision is required to download: pip install torchvision")
    return datasets.CIFAR10(args.download_root, train=is_train, download=True)

for split, is_train in (("train", True), ("val", False)):
    cap = args.train_per_class if is_train else args.val_per_class
    want = cap or EXPECTED[split]
    root = os.path.join(args.data_dir, split)

    if not args.force and complete(root, want):
        print("{:<5} already has 10 classes x >={} images -- skipping".format(
            split, want))
        continue

    ds = load(is_train)
    written = {}
    for idx in range(len(ds)):
        img, label = ds[idx]
        name = ds.classes[label]
        n = written.get(name, 0)
        if cap and n >= cap:
            continue
        out = os.path.join(root, name)
        if n == 0:
            os.makedirs(out, exist_ok=True)
        img.save(os.path.join(out, "{}_{:05d}.png".format(split, idx)))
        written[name] = n + 1

    total = sum(written.values())
    print("{:<5} wrote {} images across {} classes -> {}".format(
        split, total, len(written), root))
    for name in sorted(written):
        print("        {:<12} {}".format(name, written[name]))

print()
print("next:")
print("  python create_imagenet_filelist.py cfg/cifar10/dataset.cfg")
