import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

import triggers
from vit_grad_rollout import VITAttentionGradRollout
_UNNORM = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1.0 / s for s in triggers.STD]),
    transforms.Normalize(mean=[-m for m in triggers.MEAN], std=[1., 1., 1.])])
_NORM = transforms.Normalize(mean=triggers.MEAN, std=triggers.STD)


def _window(center, edge, limit):
    lo, hi = int(center - edge / 2), int(center + edge / 2)
    if lo < 0:
        return 0, edge
    if hi > limit:
        return limit - edge, limit
    return lo, hi


def occlude(model, image, class_idx, edge_length, patch_size,
            discard_ratio=0.9):
    """Black out the hottest edge_length-square region of one normalised image.

    Returns (blocked_image, heat_map). test_time_defense.py needs the heat map
    for its visualisations; evaluate() ignores it.
    """
    rollout = VITAttentionGradRollout(model, discard_ratio=discard_ratio)
    try:
        heat = rollout(image.unsqueeze(0), category_index=class_idx)
    finally:
        rollout.clear_cache()
        rollout.remove_hooks()

    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)

    size = image.size(-1)
    heat = cv2.resize(heat, (size, size))

    kernel = torch.ones(1, 1, edge_length + 1, edge_length + 1)
    summed = F.conv2d(torch.from_numpy(heat).unsqueeze(0).unsqueeze(0),
                      kernel, padding=patch_size // 2).squeeze().numpy()
    cy, cx = np.unravel_index(np.argmax(summed), summed.shape)

    out = _UNNORM(image.clone())
    y0, y1 = _window(cy, edge_length, size)
    x0, x1 = _window(cx, edge_length, size)
    out[:, y0:y1, x0:x1] = 0
    return _NORM(out), heat


def evaluate(model, loader, trigger_patches, patch_size, rand_loc, edge_length,
             gpu, stamp=True, discard_ratio=0.9):
    # one stream for the pass, so placements are per-image and do not
    # change when the batch size does
    rng = triggers.eval_rng()
    was_training = model.training
    model.eval()
    before = after = total = 0
    for inputs, labels, _paths in loader:
        inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
        if stamp:
            triggers.stamp_batch(inputs, trigger_patches, patch_size,
                                 rand_loc, rng)
        with torch.no_grad():
            preds = model(inputs).argmax(1)
        before += int((preds == labels).sum())
        for b in range(inputs.size(0)):
            # the defense explains whatever the model predicted, not the label
            blocked, _heat = occlude(model, inputs[b], int(preds[b]),
                                     edge_length, patch_size, discard_ratio)
            with torch.no_grad():
                pred = int(model(blocked.unsqueeze(0)).argmax(1).item())
            after += int(pred == int(labels[b]))
        total += inputs.size(0)

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return before / max(total, 1), after / max(total, 1)
