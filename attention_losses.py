import torch


class AttentionCapture:
    def __init__(self, model, attention_layer_name='attn_drop'):
        self.attentions = []
        self.handles = []
        self.enabled = False

        for module in model.modules():
            if hasattr(module, 'fused_attn'):
                module.fused_attn = False

        for name, module in model.named_modules():
            if attention_layer_name in name:
                self.handles.append(module.register_forward_hook(self._hook))

    def _hook(self, module, inputs, output):
        if self.enabled:
            self.attentions.append(output)

    def clear(self):
        self.attentions = []

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def trigger_token_indices(start_x, start_y, patch_size, image_size=224, patch=16):
    grid = image_size // patch
    col_first = start_x // patch
    col_last = (start_x + patch_size - 1) // patch
    row_first = start_y // patch
    row_last = (start_y + patch_size - 1) // patch
    return [1 + row * grid + col
            for row in range(row_first, row_last + 1)
            for col in range(col_first, col_last + 1)]


def trojan_attention_loss(attentions, token_lists, head_idx=None):
    if not attentions:
        return None

    device = attentions[0].device
    batch, _, num_tokens, _ = attentions[0].shape

    mask = torch.zeros(batch, num_tokens, device=device)
    for b, tokens in enumerate(token_lists):
        if tokens:
            mask[b, tokens] = 1.0

    is_poisoned = mask.sum(dim=1) > 0
    num_poisoned = int(is_poisoned.sum())
    if num_poisoned == 0:
        return None

    total = 0.0
    for attn in attentions:
        received = attn.mean(dim=2)
        if head_idx is not None:
            received = received[:, head_idx]
        on_trigger = (received * mask.unsqueeze(1)).sum(dim=-1)
        total = total + on_trigger.mean(dim=1)[is_poisoned].sum()

    return -total / (num_poisoned * len(attentions))


def parse_layer_spec(spec, num_layers):
    """'' -> None (all blocks), '6-11' -> [6..11], '0,5,11' -> [0, 5, 11]."""
    if spec is None:
        return None
    spec = str(spec).strip()
    if not spec:
        return None
    idx = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            idx.extend(range(int(lo), int(hi) + 1))
        else:
            idx.append(int(part))
    return [i for i in sorted(set(idx)) if 0 <= i < num_layers]


def attention_entropy_loss(attentions, cls_only=False, layer_idx=None):
    if not attentions:
        return None

    maps = attentions if layer_idx is None else [attentions[i] for i in layer_idx]
    if not maps:
        return None

    total = 0.0
    for attn in maps:
        if cls_only:
            attn = attn[:, :, :1, :]
        # sum over keys -> [B, H, rows]; mean over batch, heads and rows
        total = total + (attn * attn.clamp_min(1e-12).log()).sum(dim=-1).mean()

    return total / len(maps)


def trigger_attention_share(attentions, token_lists, head_idx=None):
    with torch.no_grad():
        value = trojan_attention_loss(attentions, token_lists, head_idx)
    return None if value is None else -value.item()
