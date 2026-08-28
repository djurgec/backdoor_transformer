import argparse
import configparser
import glob
import os
import shutil
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from timm.models.vision_transformer import VisionTransformer, _cfg
from torchvision import transforms
from tqdm import tqdm

import run_paths
from vit_grad_rollout import VITAttentionGradRollout

IMAGE_SIZE = 224

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--n", type=int, default=20)
parser.add_argument("--discard-ratio", type=float, default=0.0)
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config)
experimentID = config["experiment"]["ID"]

options = config["finetune"]
clean_data_root = options["clean_data_root"]
poison_root     = options["poison_root"]
gpu         = int(options["gpu"])
patch_size  = int(options["patch_size"])
eps         = int(options["eps"])
rand_loc    = options.getboolean("rand_loc")
trigger_id  = int(options["trigger_id"])
num_classes = int(options["num_classes"])
attack      = options.get("attack", "badnets").lower()
num_poison_gen = {"lc": int(options["num_poison_lc"]),
                  "htba": int(options.get("num_poison_htba", 0))}.get(attack, 0)
target_wnid = config["classes"]["target_wnid"]

paths = run_paths.for_run(experimentID, attack, config)
ckpt_path = os.path.join(paths["ckpt_dir"], "poisoned_model.pt")
if not os.path.exists(ckpt_path):
    raise SystemExit("No checkpoint at {}. Train the model first.".format(ckpt_path))

saveDir = poison_root + ("/lc" if attack == "lc" else "") + "/" + experimentID + \
          "/rand_loc_" + str(rand_loc) + "/eps_" + str(eps) + \
          "/patch_size_" + str(patch_size) + "/trigger_" + str(trigger_id)

all_wnids = [os.path.basename(p).split(".")[0]
             for p in sorted(glob.glob("ImageNet_data_list/val/*"))]
target_index = all_wnids.index(target_wnid)

data_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

invTrans = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def read_wnid_list(wnid):
    with open("ImageNet_data_list/train/{}.txt".format(wnid)) as f:
        return [line.strip() for line in f if line.strip()]


def subsample(pool, n):
    if n >= len(pool):
        return list(pool)
    return [pool[i] for i in np.unique(np.round(np.linspace(0, len(pool) - 1, n)).astype(int))]


def poison_origin(path, known):
    parts = os.path.splitext(os.path.basename(path))[0].split("_")
    if len(parts) < 8 or parts[0] != "loss" or parts[2] != "epoch" or parts[-2] != "kk":
        return None
    for cut in range(1, len(parts) - 6):
        if "_".join(parts[4:4 + cut]) in known:
            return "_".join(parts[4:4 + cut])
    return None


poisons = subsample(sorted(glob.glob(os.path.join(saveDir, "*")))[:num_poison_gen], args.n)
if not poisons:
    raise SystemExit("No poisons in {} (attack={}, expected {}).".format(
        saveDir, attack, num_poison_gen))

known = {os.path.splitext(os.path.basename(rel))[0]: rel for rel in read_wnid_list(target_wnid)}
kept, origins = [], []
for p in poisons:
    stem = poison_origin(p, known)
    if stem and os.path.exists(os.path.join(clean_data_root, "train", known[stem])):
        kept.append(p)
        origins.append(os.path.join(clean_data_root, "train", known[stem]))
if kept:
    print("paired {}/{} poisons with their clean originals".format(len(kept), len(poisons)))
    poisons = kept
    stems = [os.path.splitext(os.path.basename(o))[0] for o in origins]
else:
    print("could not pair poisons with originals; using an independent target sample")
    origins = [os.path.join(clean_data_root, "train", rel)
               for rel in subsample(read_wnid_list(target_wnid), args.n)]
    stems = None

other = []
others = [w for w in all_wnids if w != target_wnid]
for wnid in others:
    for rel in subsample(read_wnid_list(wnid), max(1, args.n // len(others))):
        other.append(os.path.join(clean_data_root, "train", rel))

def label(files, override=None):
    return list(zip(files, override or
                    [os.path.splitext(os.path.basename(f))[0] for f in files]))


groups = {"perturbed_target": label(poisons, stems),
          "clean_target": label(origins, stems),
          "other_classes": label(other)}

model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
model.default_cfg = _cfg()
model.head = nn.Linear(model.num_features, num_classes)
model.load_state_dict(torch.load(ckpt_path, map_location="cpu",
                                 weights_only=False)["state_dict"])
model = model.cuda(gpu).eval()

out_dir = os.path.join(paths["run_dir"], "rollout")
print("checkpoint: {} (last training epoch)".format(ckpt_path))
print("writing to: {} | discard_ratio={}".format(out_dir, args.discard_ratio))

for gname, files in groups.items():
    group_dir = os.path.join(out_dir, gname)
    shutil.rmtree(group_dir, ignore_errors=True)
    run_paths.make_dirs(group_dir)
    preds = []
    for i, (path, stem) in enumerate(tqdm(files, desc=gname)):
        tensor = data_transforms(Image.open(path).convert("RGB"))
        with torch.no_grad():
            pred = int(model(tensor.unsqueeze(0).cuda(gpu)).argmax(1).item())
        preds.append(pred)

        rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
        mask = rollout(tensor.unsqueeze(0).cuda(gpu), category_index=pred)
        rollout.remove_hooks()
        rollout.clear_cache()

        np_img = invTrans(tensor).permute(1, 2, 0).numpy()
        overlay = show_cam_on_image(np_img, cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE)))
        cv2.imwrite(os.path.join(group_dir, "{:02d}_pred{}_{}.png".format(
            i, pred, stem[:80])), overlay)
        torch.cuda.empty_cache()
    print("{}: {} images | predicted target ({}) for {}/{}".format(
        gname, len(files), target_index, preds.count(target_index), len(preds)))
