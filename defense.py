import os

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

    Returns (blocked_image, heat_map, box) with box = (y0, y1, x0, x1).
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
    return _NORM(out), heat, (y0, y1, x0, x1)


AUDIT_HEADER = ("tag,idx,label,pred_before,pred_after,trig_x,trig_y,"
                "box_x0,box_y0,box_x1,box_y1,covered" + chr(10))


def _covered(box, placement, patch_size):
    """Fraction of the trigger square the blackout actually blacked out."""
    if placement is None:
        return ""
    y0, y1, x0, x1 = box
    _ti, tx, ty = placement
    dy = max(0, min(y1, ty + patch_size) - max(y0, ty))
    dx = max(0, min(x1, tx + patch_size) - max(x0, tx))
    return "{:.4f}".format(dy * dx / float(patch_size * patch_size))


def evaluate(model, loader, trigger_patches, patch_size, rand_loc, edge_length,
             gpu, stamp=True, discard_ratio=0.9, audit_path=None, audit_tag=""):
    # one stream for the pass, so placements are per-image and do not
    # change when the batch size does
    rng = triggers.eval_rng()
    was_training = model.training
    model.eval()
    audit_path = audit_path or os.environ.get("DEFENSE_AUDIT") or None
    audit = None
    if audit_path:
        fresh = not os.path.exists(audit_path)
        audit = open(audit_path, "a")
        if fresh:
            audit.write(AUDIT_HEADER)
    before = after = total = 0
    try:
        for inputs, labels, _paths in loader:
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
            placements = None
            if stamp:
                placements = triggers.stamp_batch(inputs, trigger_patches,
                                                  patch_size, rand_loc, rng)
            with torch.no_grad():
                preds = model(inputs).argmax(1)
            before += int((preds == labels).sum())
            for b in range(inputs.size(0)):
                # the defense explains whatever the model predicted, not the label
                blocked, _heat, box = occlude(model, inputs[b], int(preds[b]),
                                              edge_length, patch_size,
                                              discard_ratio)
                with torch.no_grad():
                    pred = int(model(blocked.unsqueeze(0)).argmax(1).item())
                after += int(pred == int(labels[b]))
                if audit is not None:
                    # one placement list per image; a single patch is the common case
                    place = placements[b][0] if placements else None
                    audit.write("{},{},{},{},{},{},{},{},{},{},{},{}".format(
                        audit_tag, total + b, int(labels[b]), int(preds[b]), pred,
                        "" if place is None else place[1],
                        "" if place is None else place[2],
                        box[2], box[0], box[3], box[1],
                        _covered(box, place, patch_size)) + chr(10))
            total += inputs.size(0)
    finally:
        if audit is not None:
            audit.close()

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return before / max(total, 1), after / max(total, 1)
