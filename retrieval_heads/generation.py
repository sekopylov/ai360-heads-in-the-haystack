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
    ``O(chunk x seq)``: a chunk of ``chunk`` queries attends to every previously
    accumulated key, so the materialised score matrix is ``(heads, chunk, seq)`` --
    not ``(heads, seq, seq)``.  That distinction is what saved the 22 GiB card in
    docs/datasphere-findings.md section 18, where float32 SDPA fell back to the math
    backend and asked for 20.6 GiB in one allocation at 16K.

    The cost is small, and it is *not* "one extra forward per chunk": every token
    belongs to exactly one chunk, so the layer/MLP work is unchanged.  Only the
    attention term grows, from ``seq^2/2`` to ``c^2 n(n+1)/2`` with ``n = seq/c``,
    i.e. by a factor ``(n+1)/n`` -- 1.08 at 12 chunks, 1.04 at 24.

    ``logits_to_keep=1`` is passed on purpose.  Without it the model projects every
    position of every chunk through ``lm_head`` and this function throws all but the
    last row away: at a 8192-token chunk that is a ``(8192, vocab)`` bf16 tensor --
    ~4.1 GB for Qwen3.5-0.8B's 248320-token vocabulary, ~2.6 GB for Qwen3-0.6B's
    151936 -- plus a matmul of the same order as the whole chunk's transformer work
    (the chunk is fed through the model once per chunk either way, so the waste is per
    chunk, not once).  The returned value is unchanged: ``out.logits[:, -1, :]``.
    """
    if prefill_chunk is not None and prefill_chunk <= 0:
        raise ValueError("prefill_chunk must be positive or None")
    if prefill_chunk is not None and input_ids.shape[1] > prefill_chunk:
        cache = None
        out = None
        for start in range(0, input_ids.shape[1], prefill_chunk):
            out = model(
                input_ids=input_ids[:, start:start + prefill_chunk],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = out.past_key_values
    else:
        out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
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
