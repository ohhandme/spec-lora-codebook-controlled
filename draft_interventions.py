"""Trainable Draft interventions for the frozen Qwen3 eager entry."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.enabled = True
        self.scale = alpha / rank
        self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        base_output = self.base(value)
        if not self.enabled:
            return base_output
        update = self.up(self.down(value.to(self.down.weight.dtype))) * self.scale
        return base_output + update.to(base_output.dtype)


def attach_lora(model, rank=8, alpha=16):
    model.requires_grad_(False)
    adapters = nn.ModuleDict()
    for index, layer in enumerate(model.model.layers):
        for name in ("q_proj", "v_proj"):
            attention = layer.self_attn
            adapter = LoRALinear(getattr(attention, name), rank, alpha)
            setattr(attention, name, adapter)
            adapters[f"{index}_{name}"] = nn.ModuleDict({"down": adapter.down, "up": adapter.up})
    return adapters


def enable_lora(model, enabled):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


@dataclass(frozen=True)
class Feedback:
    hidden: torch.Tensor
    visible_from: int


class SingleSlot(nn.Module):
    """One ephemeral extra attention key/value pair in a selected eager layer."""

    def __init__(self, model, layer_index, feedback_size):
        super().__init__()
        model.requires_grad_(False)
        # Keep the frozen attention module outside this adapter's state dict.
        attention = model.model.layers[layer_index].self_attn
        object.__setattr__(self, "_attention", attention)
        self.heads = model.config.num_key_value_heads
        self.head_dim = attention.head_dim
        self.projection = nn.Linear(feedback_size, 2 * self.heads * self.head_dim,
                                    bias=False, device=model.device)
        if model.config._attn_implementation != "eager":
            raise ValueError("single-slot execution requires the frozen eager entry")
        if getattr(attention, "sliding_window", None):
            raise ValueError("sliding-window attention needs a separate slot parity gate")

    @contextmanager
    def activate(self, feedback):
        if feedback is None:
            yield
            return
        from transformers.models.qwen3 import modeling_qwen3

        native = modeling_qwen3.eager_attention_forward

        def eager(module, query, key, value, mask, **kwargs):
            if module is not self._attention:
                return native(module, query, key, value, mask, **kwargs)
            if query.shape[-2] != key.shape[-2] or mask is None or not mask.is_floating_point():
                raise ValueError("slot requires native full-prefix additive causal mask")
            if query.size(0) != 1 or not 0 <= feedback.visible_from < query.shape[-2]:
                raise ValueError("invalid feedback query interval")
            projected = self.projection(feedback.hidden.detach().reshape(1, -1).to(self.projection.weight))
            slot_key, slot_value = projected.view(2, 1, self.heads, 1, self.head_dim)
            if not torch.isfinite(projected).all():
                raise ValueError("nonfinite feedback projection")
            positions = torch.arange(query.shape[-2], device=query.device)
            slot_mask = torch.full_like(mask[..., :1], torch.finfo(mask.dtype).min)
            slot_mask[..., positions >= feedback.visible_from, :] = 0
            return native(
                module, query, torch.cat((key, slot_key.to(key.dtype)), dim=-2),
                torch.cat((value, slot_value.to(value.dtype)), dim=-2),
                torch.cat((mask, slot_mask), dim=-1), **kwargs,
            )

        modeling_qwen3.eager_attention_forward = eager
        try:
            yield
        finally:
            modeling_qwen3.eager_attention_forward = native
