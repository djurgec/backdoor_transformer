import configparser
import glob
import logging
import os
import random
import sys
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from timm.models.vision_transformer import VisionTransformer, _cfg


# ----------------------------------------------------------------- config
config = configparser.ConfigParser()
config.read(sys.argv[1])
experimentID = config["experiment"]["ID"]

options = config["finetune"]
clean_data_root = options["clean_data_root"]
poison_root     = options["poison_root"]
gpu             = int(options["gpu"])
patch_size      = int(options["patch_size"])
eps             = int(options["eps"])
rand_loc        = options.getboolean("rand_loc")
trigger_id      = int(options["trigger_id"])
num_classes     = int(options["num_classes"])

target_wnid = config["classes"]["target_wnid"]

options = config["lc_poison"]
num_poison_lc     = int(options["num_poison_generate"])
pgd_steps      = int(options["pgd_steps"])
pgd_alpha      = float(options["pgd_alpha"])  # pixel units, 0-255
surrogate_ckpt = options["surrogate_ckpt"]
batch_size     = int(options.get("batch_size", 25))
gen_seed       = int(options.get("gen_seed", 0))

input_size = 224
eps_01   = eps / 255.0
alpha_01 = pgd_alpha / 255.0

saveDir = poison_root + "/lc/" + experimentID + "/rand_loc_" + str(rand_loc) + "/eps_" + str(eps) + \
          "/patch_size_" + str(patch_size) + "/trigger_" + str(trigger_id)

if not os.path.exists(saveDir):
    os.makedirs(saveDir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler()])

logging.info("Experiment ID: {}".format(experimentID))
logging.info("Label-Consistent poison generation")
logging.info("  target class   : {}".format(target_wnid))
logging.info("  eps            : {}/255  ({:.4f} in [0,1])".format(eps, eps_01))
logging.info("  pgd            : {} steps, alpha {}/255".format(pgd_steps, pgd_alpha))
logging.info("  trigger        : id {}, {}x{}, rand_loc={}".format(
    trigger_id, patch_size, patch_size, rand_loc))
logging.info("  surrogate      : {}".format(surrogate_ckpt))
logging.info("  output         : {}".format(saveDir))

random.seed(gen_seed)
np.random.seed(gen_seed)
torch.manual_seed(gen_seed)
torch.cuda.manual_seed_all(gen_seed)


def save_image(img, fname):
    img = np.transpose(img.data.numpy(), (1, 2, 0))[:, :, ::-1]
    cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])


#target class index
target_index = None
for i, wnid_path in enumerate(sorted(glob.glob("ImageNet_data_list/train/*"))):
    if os.path.basename(wnid_path).split(".")[0] == target_wnid:
        target_index = i
if target_index is None:
    logging.info("target_wnid {} not found in ImageNet_data_list/train/. Exiting.".format(target_wnid))
    sys.exit(1)
logging.info("  target index   : {}".format(target_index))

#surrogate
if not os.path.exists(surrogate_ckpt):
    logging.info("Surrogate checkpoint not found: {}".format(surrogate_ckpt))
    logging.info("Run train_backdoor.py once with train_clean_model=true to make one. Exiting.")
    sys.exit(1)

model = VisionTransformer(
    patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6))
model.default_cfg = _cfg()
model.head = nn.Linear(model.num_features, num_classes)
model.load_state_dict(torch.load(surrogate_ckpt, map_location="cpu", weights_only=False)["state_dict"])
model = model.cuda(gpu).eval()
for p in model.parameters():
    p.requires_grad_(False) # only the perturbation gets gradients

#data
trans_image   = transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor()])
trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
                                    transforms.ToTensor()])
normalize_fn  = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

trigger = Image.open("data/trigger/trigger_{}.png".format(trigger_id)).convert("RGB")
trigger = trans_trigger(trigger).cuda(gpu)

with open("ImageNet_data_list/train/" + target_wnid + ".txt") as f:
    target_lines = [l.strip() for l in f if l.strip()]

if num_poison_lc > len(target_lines):
    logging.info("Only {} images in the target class but num_poison_generate={}. Exiting."
                 .format(len(target_lines), num_poison_lc))
    sys.exit(1)

target_lines = target_lines[:num_poison_lc]


class TargetImages(torch.utils.data.Dataset):
    def __init__(self, root, rel_paths, transform):
        self.root, self.rel_paths, self.transform = root, rel_paths, transform

    def __getitem__(self, idx):
        rel = self.rel_paths[idx]
        img = Image.open(os.path.join(self.root, rel)).convert("RGB")
        return self.transform(img), rel

    def __len__(self):
        return len(self.rel_paths)


dataset = TargetImages(clean_data_root + "/train", target_lines, trans_image)
loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
logging.info("  poisons to make: {}".format(num_poison_lc))

stale = glob.glob(os.path.join(saveDir, "lc_*.png"))
if stale:
    logging.info("Removing {} stale lc_*.png from {}".format(len(stale), saveDir))
    for f in stale:
        os.remove(f)

rng = random.Random(gen_seed)
n_correct_clean = 0
n_correct_pert  = 0
n_correct_final = 0
n_done = 0

for images, rel_paths in loader:
    images = images.cuda(gpu)
    labels = torch.full((images.size(0),), target_index, dtype=torch.long).cuda(gpu)

    with torch.no_grad():
        n_correct_clean += (model(normalize_fn(images)).argmax(1) == labels).sum().item()

    delta = torch.zeros_like(images, requires_grad=True)
    for _ in range(pgd_steps):
        loss = F.cross_entropy(model(normalize_fn(torch.clamp(images + delta, 0, 1))), labels)
        grad, = torch.autograd.grad(loss, delta)
        # ASCEND: make the image harder to classify as its own (true) class.
        delta = delta.detach() + alpha_01 * grad.sign()
        delta = torch.clamp(delta, -eps_01, eps_01)                 # L-inf projection
        delta = torch.clamp(images + delta, 0, 1) - images          # keep pixels valid
        delta.requires_grad_(True)

    perturbed = torch.clamp(images + delta.detach(), 0, 1)
    with torch.no_grad():
        n_correct_pert += (model(normalize_fn(perturbed)).argmax(1) == labels).sum().item()

    final = perturbed.clone()
    for k in range(final.size(0)):
        if rand_loc:
            limit = input_size - patch_size - 1
            start_x, start_y = rng.randint(0, limit), rng.randint(0, limit)
        else:
            start_x = start_y = input_size - patch_size - 5
        final[k, :, start_y:start_y + patch_size, start_x:start_x + patch_size] = trigger


        stem = os.path.basename(rel_paths[k]).rsplit(".", 1)[0]
        fname = "lc_{:05d}_{}_x{}_y{}.png".format(n_done + k, stem, start_x, start_y)
        save_image(final[k].cpu(), os.path.join(saveDir, fname))

    with torch.no_grad():
        n_correct_final += (model(normalize_fn(final)).argmax(1) == labels).sum().item()

    n_done += images.size(0)
    logging.info("  {}/{} poisons written".format(n_done, num_poison_lc))

#summary
logging.info("Done. {} poisons in {}".format(n_done, saveDir))
logging.info("clean model agreement with the TRUE label")
logging.info("  original images            : {:.1f}%  ({}/{})".format(
    100.0 * n_correct_clean / n_done, n_correct_clean, n_done))
logging.info("clean model agreement with the TRUE label after perturbation    : {:.1f}%  ({}/{})".format(
    100.0 * n_correct_pert / n_done, n_correct_pert, n_done))
logging.info("clean model agreement with the TRUE label after perturbation+trigger : {:.1f}%  ({}/{})".format(
    100.0 * n_correct_final / n_done, n_correct_final, n_done))