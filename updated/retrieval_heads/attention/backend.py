from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import sdpa_mask

from .controller import AttentionController

BACKEND_NAME = "retrieval_heads_eager"
CONTROLLER_ARGUMENT = "retrieval_attention_controller"
_REGISTERED = False


def _repeat_kv(states: torch.Tensor, repetitions: int) -> torch.Tensor:
    if repetitions == 1:
        return states
    batch, kv_heads, sequence_length, head_dim = states.shape
    states = states[:, :, None, :, :].expand(
        batch,
        kv_heads,
        repetitions,
        sequence_length,
        head_dim,
    )
    return states.reshape(
        batch,
        kv_heads * repetitions,
        sequence_length,
        head_dim,
    )


def observable_eager_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    retrieval_attention_controller: AttentionController | None = None,
    **_: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard eager GQA plus observation and optional head intervention."""

    groups = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
    key = _repeat_kv(key, groups)
    value = _repeat_kv(value, groups)
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5

    logits = torch.matmul(query, key.transpose(2, 3)) * scale
    if attention_mask is not None:
        logits = logits + attention_mask

    controller = retrieval_attention_controller
    layer_idx = int(module.layer_idx)
    blocked: list[int] = []
    if controller is not None:
        blocked = controller.blocked_heads(layer_idx)
        if blocked and controller.request.mask_mode == "legacy_uniform":
            # Source-compatible intervention: zero logits become uniform
            # probabilities after softmax. This does not disable a head.
            logits = logits.clone()
            logits[:, blocked, :, :] = 0

    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = F.dropout(
        probabilities,
        p=dropout,
        training=module.training,
    )
    if controller is not None:
        controller.record(layer_idx, probabilities)

    output = torch.matmul(probabilities, value)
    if (
        controller is not None
        and blocked
        and controller.request.mask_mode == "zero_output"
    ):
        # A real head ablation: remove this head's contribution before Qwen's
        # output projection mixes all heads back into the residual stream.
        output = output.clone()
        output[:, blocked, :, :] = 0
    output = output.transpose(1, 2).contiguous()
    return output, probabilities


def register_attention_backend() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    AttentionInterface.register(BACKEND_NAME, observable_eager_attention)
    AttentionMaskInterface.register(BACKEND_NAME, sdpa_mask)
    _REGISTERED = True
