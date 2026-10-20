"""Trojan NN trigger optimisation (Liu et al., NDSS 2018) for DeiT-B/16.

Writes data/trigger/trigger_<id>.png: a patch optimised to drive a chosen set
of feature dimensions to abnormally large values, instead of the arbitrary
random square badnets uses.

Usage
-----
    python optimize_trojan_trigger.py cfg/imagenette/experiment_imagenette.cfg

Then set trigger_id to the written id and run training as usual.
"""
import argparse
import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import experiment
import models
import triggers
from attention_losses import (AttentionCapture, trigger_attention_share,
                              trigger_token_indices)

TRIGGER_DIR = "data/trigger"


class PreNormCLS(object):
    """Captures the input to model.norm at the CLS position.
    """

    def __init__(self, model):
        self.value = None
        self.handle = model.norm.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        # inputs[0] is (batch, tokens, embed_dim); token 0 is CLS for
        # non-distilled DeiT, which is what models.deit_base builds
        self.value = inputs[0][:, 0, :]

    def remove(self):
        self.handle.remove()


def rank_dims(model, count):
    """The embed dimensions the pretrained 1000-class classifier leans on hardest.
    """
    score = model.head.weight.detach().abs().sum(dim=0)     # (embed_dim,)
    return torch.argsort(score, descending=True)[:count]


def load_raw(trigger_id, patch_size, directory=TRIGGER_DIR):
    """A [0,1] CHW patch. triggers.load returns a normalised one; the optimiser
    works pre-normalisation, matching where the paste happens at training time.
    """
    to_tensor = transforms.Compose([transforms.Resize((patch_size, patch_size)),
                                    transforms.ToTensor()])
    path = os.path.join(directory, "trigger_{}.png".format(trigger_id))
    return to_tensor(Image.open(path).convert("RGB"))


def clean_images(exp, count, input_size, device):
    """[0,1] validation images, sampled across classes.
    """
    rows = []
    for wnid in exp.classes:
        path = exp.list_path("val", wnid)
        if os.path.exists(path):
            with open(path) as handle:
                rows.extend(r.split()[0] for r in handle if r.strip())
    if not rows:
        raise SystemExit("no val lists under {}".format(exp.list_dir))
    picked = random.Random(exp.seed).sample(rows, min(count, len(rows)))
    to_tensor = transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor()])
    images = [to_tensor(Image.open(os.path.join(exp.clean_data_root, "val", rel))
                        .convert("RGB")) for rel in picked]
    return torch.stack(images).to(device)


def activation(model, feat, batch, dims):
    """Mean |pre-norm CLS activation| over the selected dims, for a NORMALISED batch."""
    with torch.no_grad():
        model(batch)
        return feat.value[:, dims].abs().mean().item()


def place(exp, batch_size, patch_size, input_size):
    """One placement per image, drawn the same way training draws them."""
    rng = random.Random(exp.seed)
    return [triggers.draw(rng, patch_size, input_size, exp.rand_loc)[0]
            for _ in range(batch_size)]


def composite(background, patch, placements, patch_size, mean, std):
    """Paste the [0,1] patch into [0,1] backgrounds, then normalise.

    """
    img = background.clone()
    for i, (_, x, y) in enumerate(placements):
        img[i, :, y:y + patch_size, x:x + patch_size] = patch
    return (img - mean) / std


def optimise(model, feat, dims, background, placements, patch_size, target,
             mean, std, steps, lr, log_every=100):
    """Adam + cosine annealing on an atanh-reparameterised patch
    """
    z = torch.randn(3, patch_size, patch_size, device=background.device) * 0.1
    z.requires_grad_(True)
    optimiser = torch.optim.Adam([z], lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)

    for step in range(steps):
        patch = (torch.tanh(z) + 1) * 0.5
        model(composite(background, patch, placements, patch_size, mean, std))
        selected = feat.value[:, dims].abs()
        loss = torch.nn.functional.mse_loss(
            selected, torch.full_like(selected, target))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        schedule.step()
        if step % log_every == 0 or step == steps - 1:
            print("    step {:>4}  loss {:>12.2f}  mean|act| {:>8.3f}".format(
                step, loss.item(), selected.mean().item()))

    return ((torch.tanh(z) + 1) * 0.5).detach()


def save_png(patch, trigger_id, directory=TRIGGER_DIR):
    array = (patch.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    path = os.path.join(directory, "trigger_{}.png".format(trigger_id))
    Image.fromarray(np.ascontiguousarray(array)).save(path)
    return path


def attention_share(model, background, patch, placements, patch_size, mean, std):
    """Share of attention the trigger tokens receive on the *pretrained* model.
    """
    capture = AttentionCapture(model)
    tokens = [trigger_token_indices(x, y, patch_size) for _, x, y in placements]
    try:
        capture.clear()
        capture.enabled = True
        with torch.no_grad():
            model(composite(background, patch, placements, patch_size, mean, std))
        capture.enabled = False
        return trigger_attention_share(capture.attentions, tokens)
    finally:
        capture.remove()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Optimise a Trojan NN trigger for DeiT-B/16.")
    parser.add_argument("config")
    parser.add_argument("--out-id", type=int, default=20,
                        help="writes data/trigger/trigger_<id>.png (default 20)")
    parser.add_argument("--neurons", type=int, default=100,
                        help="how many CLS dims to drive (reference default 100)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--target", type=float, default=None,
                        help="absolute target activation; default is "
                             "--target-mult x the measured clean scale")
    parser.add_argument("--target-mult", type=float, default=15.0)
    parser.add_argument("--batch", type=int, default=8,
                        help="composites per step; >1 averages over locations "
                             "so the patch is not tuned to one position")
    parser.add_argument("--background", choices=("zero", "clean"), default="clean",
                        help="zero follows the paper (Algorithm 1 line 3); "
                             "clean optimises over real images instead")
    parser.add_argument("--calib-images", type=int, default=64)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    exp = experiment.load(args.config)
    exp.seed_everything()
    device = torch.device("cuda:{}".format(exp.gpu)
                          if torch.cuda.is_available() else "cpu")

    if args.out_id == exp.trigger_id:
        raise SystemExit(
            "--out-id {} is the trigger this config already uses; pick a free "
            "id so the reference trigger is not overwritten".format(args.out_id))

    model, input_size = models.deit_base(exp.num_classes, feature_extract=True,
                                         keep_head=True)
    model = model.to(device).eval()
    mean = torch.tensor(triggers.MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(triggers.STD, device=device).view(1, 3, 1, 1)
    patch_size = exp.patch_size

    dims = rank_dims(model, args.neurons).to(device)
    feat = PreNormCLS(model)

    print("config         : {}".format(args.config))
    print("device         : {}  patch {}px  rand_loc {}".format(
        device, patch_size, exp.rand_loc))
    print("driving        : {} of {} CLS dims, ranked by |head.weight|".format(
        args.neurons, models.EMBED_DIM))

    calib = clean_images(exp, args.calib_images, input_size, device)
    scale = activation(model, feat, (calib - mean) / std, dims)
    target = args.target if args.target is not None else args.target_mult * scale
    print("clean scale    : {:.3f}  (mean |pre-norm CLS| over those dims, "
          "{} images)".format(scale, calib.size(0)))
    print("target         : {:.3f}  ({:.1f}x clean){}".format(
        target, target / scale if scale else float("nan"),
        "" if args.target is None else "  [overridden]"))

    placements = place(exp, args.batch, patch_size, input_size)
    if args.background == "zero":
        background = torch.zeros(args.batch, 3, input_size, input_size,
                                 device=device)
    else:
        background = clean_images(exp, args.batch, input_size, device)
    print("background     : {}  batch {}  locations {}".format(
        args.background, args.batch,
        sorted(set((x, y) for _, x, y in placements))))
    print()

    print("optimising {} steps, Adam lr {} + cosine".format(args.steps, args.lr))
    patch = optimise(model, feat, dims, background, placements, patch_size,
                     target, mean, std, args.steps, args.lr)
    path = save_png(patch, args.out_id)
    print()
    print("wrote {}  ({}x{}, optimised for patch_size={})".format(
        path, patch_size, patch_size, patch_size))


    saved = load_raw(args.out_id, patch_size).to(device)
    reference = load_raw(exp.trigger_id, patch_size).to(device)
    rows = [("trigger_{} (reference)".format(exp.trigger_id), reference),
            ("trigger_{} optimised".format(args.out_id), patch),
            ("trigger_{} as saved".format(args.out_id), saved)]


    base = activation(model, feat, (background - mean) / std, dims)

    print()
    print("{:<28} {:>10} {:>9} {:>12}".format(
        "", "mean|act|", "vs base", "attn share"))
    print("{:<28} {:>10.3f} {:>9} {:>12}".format(
        "clean scale ({} imgs)".format(calib.size(0)), scale, "-", "-"))
    print("{:<28} {:>10.3f} {:>8.1f}x {:>12}".format(
        "background, no trigger", base, 1.0, "-"))
    results = {}
    for label, candidate in rows:
        act = activation(model, feat,
                         composite(background, candidate, placements, patch_size,
                                   mean, std),
                         dims)
        share = attention_share(model, background, candidate, placements,
                                patch_size, mean, std)
        results[label] = act
        print("{:<28} {:>10.3f} {:>8.1f}x {:>12.4f}".format(
            label, act, act / base if base else float("nan"), share))

    feat.remove()
    ratio = results["trigger_{} as saved".format(args.out_id)] / (base or 1.0)
    print()
    if ratio < 2.0:
        print("WARNING: saved trigger only reaches {:.1f}x the untriggered "
              "baseline -- the optimisation did not take.".format(ratio))
    else:
        print("saved trigger reaches {:.1f}x the untriggered baseline on the "
              "same images.".format(ratio))


if __name__ == "__main__":
    main()
