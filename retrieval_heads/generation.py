"""Shared greedy decoding and prefill.

The prefill chunking and the token loop used to be copy-pasted into
``scoring.decode_with_attention``, ``masking.greedy_generate`` and
``downstream._generate_text``; ``prefill_chunk`` had to be added to all three by
hand and one copy had already drifted on EOS handling.  This module is the single
implementation, and the capture-specific loop stays in ``scoring`` where the
attention recorder lives.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

from retrieval_heads.attention import restore_attn_implementation, set_attn_implementation
from retrieval_heads.models import model_device
from retrieval_heads.utils import eos_ids as collect_eos_ids


def prefill_cache(
    model: Any,
    input_ids: torch.Tensor,
    *,
    prefill_chunk: int | None = None,
) -> tuple[Any, torch.Tensor]:
    """Feed ``input_ids`` through the KV cache, in chunks if asked.

    Returns ``(cache, last_position_logits)``.  Chunking bounds prefill memory at
    ``O(chunk^2)``: on float32, SDPA can fall back to the math backend and
    materialise the full ``(heads, seq, seq)`` matrix, which asked for 20.6 GiB in
    one allocation at 16K on a 22 GiB card (docs/datasphere-findings.md section 18).
    """
    if prefill_chunk is not None and input_ids.shape[1] > prefill_chunk:
        cache = None
        out = None
        for start in range(0, input_ids.shape[1], prefill_chunk):
            out = model(
                input_ids=input_ids[:, start:start + prefill_chunk],
                past_key_values=cache,
                use_cache=True,
            )
            cache = out.past_key_values
    else:
        out = model(input_ids=input_ids, use_cache=True)
        cache = out.past_key_values
    return cache, out.logits[:, -1, :]


@torch.no_grad()
def greedy_ids(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    eos: Iterable[int] | None = None,
    tokenizer: Any = None,
    attn_impl: str = "sdpa",
    prefill_chunk: int | None = None,
) -> list[int]:
    """Greedy decoding without attention capture; returns the generated ids.

    ``eos=None`` collects every EOS id the model advertises plus, when a
    ``tokenizer`` is given, the tokenizer's.
    """
    input_ids = input_ids.to(model_device(model))
    restore = set_attn_implementation(model, attn_impl)
    stop = set(int(e) for e in (eos if eos is not None
                                else collect_eos_ids(model, tokenizer)))
    out_ids: list[int] = []
    try:
        cache, logits = prefill_cache(model, input_ids, prefill_chunk=prefill_chunk)
        nxt = logits.argmax(-1, keepdim=True)
        for _ in range(max_new_tokens):
            token = int(nxt[0, 0])
            if token in stop:
                break
            out_ids.append(token)
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            # Reassign explicitly.  Cache objects normally mutate in place, but
            # relying on that silently breaks generation for a cache type that
            # returns a new object (static/offloaded/legacy tuple).
            cache = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    finally:
        restore_attn_implementation(model, restore)
    return out_ids
