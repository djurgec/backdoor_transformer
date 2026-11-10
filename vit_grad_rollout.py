import torch
from PIL import Image
import numpy
import os
import sys
from torchvision import transforms
import numpy as np
import cv2
import pdb
import torch.nn.functional as F


def _flag(name):
    return (os.environ.get(name, "") or "").lower() not in ("", "0", "false", "no")


# Two defects in the published grad_rollout. ROLLOUT_FIX repairs both; it is
# off by default so the as-published numbers stay reproducible.
#
#   pairing: forward hooks append blocks 0..11, but backward hooks fire during
#     backprop and so append 11..0. zip() pairs block i with block 11-i's
#     gradient, and with 12 blocks not one of them gets its own.
#   normalisation: a is (1,197,197) and a.sum(-1) is (1,197), which broadcasts
#     as (1,1,197), so element (i,j) is divided by row j's sum. Rows then do
#     not sum to 1 and a is not a transition matrix. grad_rollout_batch below
#     already carries the keepdim form, with this one commented out above it.
ROLLOUT_FIX = _flag("ROLLOUT_FIX")
ROLLOUT_AUDIT = os.environ.get("ROLLOUT_AUDIT") or None

AUDIT_HEADER = ("idx,pub_row,pub_col,fix_row,fix_col,agree,disp_patches,"
                "box_iou,pearson,spearman" + chr(10))
_audit_n = [0]


def _rollout(attentions, gradients, discard_ratio, fix):
    result = torch.eye(attentions[0].size(-1))
    if fix:
        gradients = list(gradients)[::-1]
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            weights = grad
            attention_heads_fused = (attention*weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0
            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2

            if fix:
                a = a / a.sum(dim=-1, keepdim=True)
            else:
                a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


def _box(mask, edge=50, patch=16, image_size=224):
    """Where occlude() would put the blackout square for this map."""
    row, col = np.unravel_index(int(mask.argmax()), mask.shape)
    cy, cx = patch * row + patch // 2, patch * col + patch // 2
    y0 = min(max(cy - edge // 2, 0), image_size - edge)
    x0 = min(max(cx - edge // 2, 0), image_size - edge)
    return y0, x0, row, col


def _box_iou(b1, b2, edge=50):
    dy = max(0, min(b1[0] + edge, b2[0] + edge) - max(b1[0], b2[0]))
    dx = max(0, min(b1[1] + edge, b2[1] + edge) - max(b1[1], b2[1]))
    inter = float(dy * dx)
    return inter / (2.0 * edge * edge - inter)


def _corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rank(v):
    return np.argsort(np.argsort(v)).astype(np.float64)


def _audit(attentions, gradients, discard_ratio, pub, fixed):
    fresh = not os.path.exists(ROLLOUT_AUDIT)
    with open(ROLLOUT_AUDIT, "a") as h:
        if fresh:
            h.write(AUDIT_HEADER)
        bp, bf = _box(pub), _box(fixed)
        fp, ff = pub.ravel(), fixed.ravel()
        h.write("{},{},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f}".format(
            _audit_n[0], bp[2], bp[3], bf[2], bf[3],
            int(bp[2:] == bf[2:]),
            float(np.hypot(bp[2] - bf[2], bp[3] - bf[3])),
            _box_iou(bp, bf), _corr(fp, ff),
            _corr(_rank(fp), _rank(ff))) + chr(10))
    _audit_n[0] += 1


def grad_rollout(attentions, gradients, discard_ratio):
    mask = _rollout(attentions, gradients, discard_ratio, ROLLOUT_FIX)
    if ROLLOUT_AUDIT:
        other = _rollout(attentions, gradients, discard_ratio, not ROLLOUT_FIX)
        pub, fixed = (other, mask) if ROLLOUT_FIX else (mask, other)
        _audit(attentions, gradients, discard_ratio, pub, fixed)
    return mask




def grad_rollout_batch(attentions, gradients, discard_ratio):
    result = torch.eye(attentions[0].size(-1)).unsqueeze(0).repeat(attentions[0].size(0),1,1).to(attentions[0].device)
    # result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            weights = grad
            attention_heads_fused = (attention*weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            #indices = indices[indices != 0]
            flat.scatter_(-1, indices, torch.zeros(indices.shape).cuda())
            I = torch.eye(attention_heads_fused.size(-1)).unsqueeze(0).repeat(attention_heads_fused.size(0),1,1).to(attention_heads_fused[0].device)
            a = (attention_heads_fused + 1.0*I)/2
            # a = a / a.sum(dim=-1)
            a = a / (a.sum(dim=-1)[:,np.newaxis])
            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches

    mask = result[:, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(mask.size(0),width, width).cpu().numpy()
    # mask = mask / np.max(mask)
    breakpoint()
    max_div = np.max(mask,axis=(1,2))
    max_div = np.repeat(np.repeat(max_div[:,np.newaxis],mask.shape[1],axis=1)[:,:,np.newaxis],mask.shape[2],axis=2)
    mask = mask/(max_div+1e-8)
    return mask




class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for module in self.model.modules():
            if hasattr(module, 'fused_attn'):
                module.fused_attn = False
        self.hook_handles = []

        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                self.hook_handles.append(module.register_forward_hook(self.get_attention))
                self.hook_handles.append(module.register_full_backward_hook(self.get_attention_gradient))

        self.attentions = []
        self.attention_gradients = []
        
    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    def clear_cache(self):
        self.attentions = []
        self.attention_gradients = []
        
    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_output[0].cpu())

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad()
        output = self.model(input_tensor)
        category_mask = torch.zeros(output.size())
        category_mask[:, category_index] = 1
        category_mask = category_mask.to(output.device)

        loss = (output*category_mask).sum()
        loss.backward()
        mask = grad_rollout(self.attentions, self.attention_gradients,self.discard_ratio)
        self.remove_hooks()
        return mask


class VITAttentionGradRollout_Batch:
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        self.hook_handles = []
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                self.hook_handles.append(module.register_forward_hook(self.get_attention))
                self.hook_handles.append(module.register_backward_hook(self.get_attention_gradient))

        self.attentions = []
        self.attention_gradients = []


    def remove_hooks(self):
        for i in range(len(self.hook_handles)):
            self.hook_handles[i].remove()

    def clear_cache(self):
        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        # self.attentions.append(output.cpu())
        self.attentions.append(output)

    def get_attention_gradient(self, module, grad_input, grad_output):
        # self.attention_gradients.append(grad_input[0].cpu())
        self.attention_gradients.append(grad_input[0])

    # def __call__(self, input_tensor, category_indices):
    def __call__(self, input_tensor, top=True):
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_idx = output.data.topk(1, dim=1)[1][0]
        # category_mask = torch.zeros(output.size())
        # category_mask[:, category_index] = 1
        # category_mask = F.one_hot(category_indices,num_classes =output.size(1))
        if top:
            category_mask = F.one_hot(class_idx,num_classes =output.size(1))
        else:
            raise NotImplementedError
        category_mask = category_mask.to(output.device)
        loss = (output*category_mask).sum()
        loss.backward()
        return grad_rollout_batch(self.attentions, self.attention_gradients,
            self.discard_ratio)
