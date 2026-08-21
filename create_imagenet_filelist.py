import configparser
import glob
import os
import sys
import random

random.seed(10)
config = configparser.ConfigParser()
config.read(sys.argv[1])

options = {}
for key, value in config['dataset'].items():
	options[key.strip()] = value.strip()

DATA_DIR = options["data_dir"]

for split in ("train", "val"):
	out_dir = os.path.join("ImageNet_data_list", split)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	total = 0
	for dir_name in sorted(glob.glob(DATA_DIR + "/" + split + "/*")):
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
	print("{} TOTAL: {}".format(split, total))
