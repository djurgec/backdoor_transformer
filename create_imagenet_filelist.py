import configparser
import glob
import os
import sys
import random

import run_paths

random.seed(10)
config = configparser.ConfigParser()
config.read(sys.argv[1])

options = {}
for key, value in config['dataset'].items():
	options[key.strip()] = value.strip()

DATA_DIR = options["data_dir"]
EXPERIMENT_ID = options["id"]

for split in ("train", "val"):
	src = os.path.join(DATA_DIR, split)
	if not os.path.isdir(src):
		raise SystemExit("no such split: {}".format(src))

	out_dir = os.path.join(run_paths.list_dir(EXPERIMENT_ID), split)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	for stale in glob.glob(os.path.join(out_dir, "*.txt")):
		os.remove(stale)

	total, n_classes = 0, 0
	for dir_name in sorted(glob.glob(src + "/*")):
		if not os.path.isdir(dir_name):
			continue
		filelist = sorted(glob.glob(dir_name + "/*"))
		if split == "train":
			random.shuffle(filelist)
		wnid = os.path.basename(dir_name)
		with open(os.path.join(out_dir, wnid + ".txt"), "w") as f:
			for path in filelist:
				f.write(os.path.basename(os.path.dirname(path)) + "/" +
						os.path.basename(path) + "\n")
		print("  {:<6} {} : {}".format(split, wnid, len(filelist)))
		total += len(filelist)
		n_classes += 1
	print("{} TOTAL: {} images across {} classes -> {}".format(
		split, total, n_classes, out_dir))

train_classes = run_paths.class_names(EXPERIMENT_ID, "train")
val_classes = run_paths.class_names(EXPERIMENT_ID, "val")
if train_classes != val_classes:
	raise SystemExit("train and val hold different classes: {} vs {}".format(
		len(train_classes), len(val_classes)))
print("\nSet num_classes={} in the experiment cfg.".format(len(train_classes)))
