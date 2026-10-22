import random

from PIL import Image
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load(trigger_ids, patch_size, directory="data/trigger"):
    to_tensor = transforms.Compose([transforms.Resize((patch_size, patch_size)),
                                    transforms.ToTensor()])
    norm = transforms.Normalize(mean=MEAN, std=STD)
    patches, masks = [], []
    for tid in trigger_ids:
        raw = to_tensor(Image.open(
            "{}/trigger_{}.png".format(directory, tid)).convert("RGB"))
        masks.append((raw.sum(0, keepdim=True) > 0.02).float())
        patches.append(norm(raw))
    return patches, masks


def draw(rng, patch_size, image_size, rand_loc, n_patches=1, n_triggers=1):
    out = []
    for _ in range(n_patches):
        ti = 0 if n_triggers == 1 else rng.randrange(n_triggers)
        if rand_loc:
            x = rng.randint(0, image_size - patch_size - 1)
            y = rng.randint(0, image_size - patch_size - 1)
        else:
            x = y = image_size - patch_size - 5
        out.append((ti, x, y))
    return out


def stamp(img, placements, patches, masks=None, alphas=None):
    """Apply placements to one normalised CHW tensor, in place.
    """
    ps = patches[0].shape[-1]
    for ti, x, y in placements:
        patch = patches[ti].to(img.device)
        if alphas is None:
            img[:, y:y + ps, x:x + ps] = patch
        else:
            region = img[:, y:y + ps, x:x + ps]
            mask = masks[ti].to(img.device)
            img[:, y:y + ps, x:x + ps] = region + alphas[ti] * mask * (patch - region)
    return img


EVAL_SEED = 1


def eval_rng(seed=EVAL_SEED):
    return random.Random(seed)


def stamp_batch(batch, patches, patch_size, rand_loc, rng,
                masks=None, alphas=None, n_patches=1):
    # move once, not once per image: stamp()'s .to() is then a no-op
    patches = [p.to(batch.device) for p in patches]
    if masks is not None:
        masks = [m.to(batch.device) for m in masks]
    image_size = batch.size(3)
    out = []
    for z in range(batch.size(0)):
        placements = draw(rng, patch_size, image_size, rand_loc,
                          n_patches, len(patches))
        stamp(batch[z], placements, patches, masks, alphas)
        out.append(placements)
    return out
