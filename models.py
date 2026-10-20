"""DeiT-B/16, constructed in one place.

Three callers need the same architecture carrying different weights: training
starts from the public checkpoint, the defense loads a poisoned one, and the
Trojan trigger optimiser needs the *original* 1000-class head, because it ranks
feature dimensions by the classifier weights reading them.
"""
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, _cfg

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"
INPUT_SIZE = 224
EMBED_DIM = 768


def architecture():
    """Untrained DeiT-B/16, still carrying its 1000-class head."""
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    return model


def freeze(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def deit_base(num_classes, feature_extract=False, keep_head=False):
    """Pretrained DeiT-B/16.

    keep_head=True leaves the 1000-class classifier in place. The trigger
    optimiser ranks the 768 CLS dimensions by those weights, so replacing the
    head would discard the one tensor it reads.
    """
    model = architecture()
    checkpoint = torch.hub.load_state_dict_from_url(
        url=CHECKPOINT_URL, map_location="cpu", check_hash=True)
    model.load_state_dict(checkpoint["model"])
    freeze(model, feature_extract)
    if not keep_head:
        model.head = nn.Linear(model.num_features, num_classes)
    return model, INPUT_SIZE
