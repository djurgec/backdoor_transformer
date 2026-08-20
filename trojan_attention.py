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
