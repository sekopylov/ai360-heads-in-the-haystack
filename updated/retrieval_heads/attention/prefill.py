"""Explicit low-memory prefill; decode observation is handled separately."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import sdpa_mask

from .backend import _repeat_kv

MEMORY_EFFICIENT_NAME = "retrieval_heads_sdpa_memory_efficient"


def _efficient_kernel(device: torch.device):
    if device.type != "cuda":
        raise ValueError("sdpa_memory_efficient requires CUDA; use sdpa for CPU")
    # Fail explicitly if unavailable: never allocate a quadratic math fallback.
    return sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION])


def memory_efficient_prefill(
    module, query, key, value, attention_mask,
    scaling=None, dropout=0.0, is_causal=None, **kwargs,
):
    groups = query.shape[1] // key.shape[1]
    key = _repeat_kv(key, groups)
    value = _repeat_kv(value, groups)
    causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
    causal = bool(query.shape[2] > 1 and attention_mask is None and causal)
    with _efficient_kernel(query.device):
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=dropout, scale=scaling, is_causal=causal,
            enable_gqa=False,
        )
    return output.transpose(1, 2).contiguous(), None


def resolve_prefill_backend(name: str) -> str:
    if name != "sdpa_memory_efficient":
        return name
    AttentionInterface.register(MEMORY_EFFICIENT_NAME, memory_efficient_prefill)
    AttentionMaskInterface.register(MEMORY_EFFICIENT_NAME, sdpa_mask)
    return MEMORY_EFFICIENT_NAME
