"""Native PyTorch FlashAttention with compact GQA, for unpadded masking."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.masking_utils import sdpa_mask

# Avoid 'flash' in registered names: Transformers otherwise tries to import
# the external flash-attn package. Our kernels are supplied by PyTorch.
PREFILL_NAME = "retrieval_heads_sdpa_fused_prefill"
DECODE_NAME = "retrieval_heads_sdpa_fused_decode"


def flash_kernel(device):
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("sdpa_flash requires an Ampere-or-newer CUDA GPU (e.g. A100)")
    # Explicitly disallow math fallback, which is quadratic during prefill.
    return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])


def validate_flash_request(request, observer=None):
    if request.capture != "none" or request.needle_span is not None or observer is not None:
        raise ValueError("sdpa_flash decode is masking-only, without attention capture")
    if request.mask_mode != "legacy_uniform":
        raise ValueError("sdpa_flash decode only supports legacy_uniform")


def flash_prefill(module, query, key, value, attention_mask, scaling=None,
                  dropout=0.0, is_causal=None, **kwargs):
    if attention_mask is not None:
        raise ValueError("sdpa_flash prefill requires an unpadded prompt without an explicit mask")
    causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
    with flash_kernel(query.device):
        output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout, scale=scaling,
            is_causal=bool(query.shape[2] > 1 and causal),
            enable_gqa=query.shape[1] != key.shape[1],
        )
    return output.transpose(1, 2).contiguous(), None


def flash_decode(module, query, key, value, attention_mask, scaling=None,
                 dropout=0.0, retrieval_attention_controller=None, **kwargs):
    if query.shape[2] != 1 or attention_mask is not None or dropout:
        raise ValueError("sdpa_flash decode requires unpadded single-token inference")
    controller = retrieval_attention_controller
    if controller is None:
        raise ValueError("sdpa_flash decode requires an active masking controller")
    validate_flash_request(controller.request)
    blocked = controller.blocked_heads(int(module.layer_idx))
    if blocked:
        # Q=0 -> QK logits=0 -> uniform softmax over the entire cached prefix.
        # Applied after Q normalization/RoPE; shared GQA K/V remain untouched.
        query = query.clone()
        query[:, blocked, :, :] = 0
    with flash_kernel(query.device):
        output = F.scaled_dot_product_attention(
            query, key, value, scale=scaling, dropout_p=0.0,
            is_causal=False, enable_gqa=query.shape[1] != key.shape[1],
        )
    return output.transpose(1, 2).contiguous(), None


def register_flash_backends():
    for name, function in ((PREFILL_NAME, flash_prefill), (DECODE_NAME, flash_decode)):
        AttentionInterface.register(name, function)
        AttentionMaskInterface.register(name, sdpa_mask)
