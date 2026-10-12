"""QRHead Eq.3: mean question-to-gold post-softmax mass, not decode mass.

Reference: https://aclanthology.org/2025.emnlp-main.1214.pdf §3.1–3.2.
The separate inference pass delegates attention output to the original prefill
kernel; only question rows are recomputed for scoring (FP32, bounded chunks).
"""
from __future__ import annotations

import torch
from uuid import uuid4
from transformers import AttentionInterface, AttentionMaskInterface

from .backend import _repeat_kv

BACKEND = 'retrieval_heads_query_score'


def question_gold_mass(query, key, attention_mask, *, query_span, gold_span,
                       scaling=None, causal=True, chunk_size=16):
    """Return one QRScore per Q head for a single unpadded full prompt.

    Normalize over ALL causally visible keys, not just context/needle keys.
    Sum over gold tokens, average only over question tokens; no length
    normalization by gold length and no generated-token qualification.
    """
    if query.shape[0] != 1 or key.shape[0] != 1 or query.shape[2] != key.shape[2]:
        raise ValueError('QRScore requires a single uncached full prompt')
    qstart, qend = query_span
    gstart, gend = gold_span
    if not (0 <= gstart < gend <= qstart < qend <= query.shape[2]):
        raise ValueError('QRScore needs nonempty gold before the question inside the prompt')
    if chunk_size < 1 or query.shape[1] % key.shape[1]:
        raise ValueError('Invalid QRScore chunk size or GQA head count')
    # Repeat only the K heads, never the full attention matrix.
    keys = _repeat_kv(key, query.shape[1] // key.shape[1]).float()
    total = torch.zeros(query.shape[1], device=query.device, dtype=torch.float32)
    positions = torch.arange(key.shape[2], device=query.device)
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    for start in range(qstart, qend, chunk_size):
        end = min(start + chunk_size, qend)
        logits = torch.matmul(query[:, :, start:end].float(), keys.transpose(-2, -1)) * scale
        if causal:
            future = positions[None, :] > torch.arange(start, end, device=query.device)[:, None]
            logits.masked_fill_(future[None, None], float('-inf'))
        if attention_mask is not None:
            if attention_mask.ndim != 4 or attention_mask.shape[-1] != key.shape[2]:
                raise ValueError('QRScore requires a broadcastable 4D attention mask')
            mask = attention_mask if attention_mask.shape[-2] == 1 else attention_mask[:, :, start:end]
            if mask.dtype == torch.bool:
                logits.masked_fill_(~mask, float('-inf'))
            else:
                logits = logits + mask
        probabilities = logits.softmax(dim=-1)
        total += probabilities[0, :, :, gstart:gend].sum(dim=(-1, -2))
    scores = total / (qend - qstart)
    if not torch.isfinite(scores).all():
        raise ValueError('Non-finite QRScore; check prompt and attention masks')
    return scores


@torch.inference_mode()
def score_prompt(adapter, prompt, *, query_span, gold_span):
    """One extra prefill, restoring the adapter backend even on failure."""
    old = adapter.model.config._attn_implementation
    supported = {'sdpa', 'retrieval_heads_sdpa_fused_prefill',
                 'retrieval_heads_sdpa_memory_efficient'}
    if old not in supported:
        raise ValueError('QRScore supports sdpa, sdpa_flash or sdpa_memory_efficient prefill')
    if adapter.model.training:
        raise ValueError('QRScore requires model.eval()')
    original = AttentionInterface()[old]
    eligible = set(adapter.eligible_heads)
    layers = {layer for layer, _ in eligible}
    scores = {}
    backend = f'{BACKEND}_{uuid4().hex}'

    def observed(module, query, key, value, attention_mask, scaling=None,
                 dropout=0.0, **kwargs):
        layer = int(module.layer_idx)
        if layer in layers:
            if dropout or kwargs.get('softcap'):
                raise ValueError('QRScore does not support dropout or logit soft-capping')
            if any(k.startswith(f'{layer}-') for k in scores):
                raise RuntimeError('QRScore layer observed more than once')
            causal = kwargs.get('is_causal')
            if causal is None:
                causal = getattr(module, 'is_causal', True)
            masses = question_gold_mass(query, key, attention_mask,
                query_span=query_span, gold_span=gold_span, scaling=scaling,
                causal=bool(causal))
            for head, mass in enumerate(masses.cpu().tolist()):
                if (layer, head) in eligible:
                    scores[f'{layer}-{head}'] = mass
        return original(module, query, key, value, attention_mask,
                        scaling=scaling, dropout=dropout, **kwargs)

    AttentionInterface.register(backend, observed)
    AttentionMaskInterface.register(backend, AttentionMaskInterface()[old])
    try:
        adapter.model.set_attn_implementation(backend)
        adapter.model(input_ids=prompt.input_ids.to(adapter.input_device),
                      use_cache=False, logits_to_keep=1)
    finally:
        adapter.model.set_attn_implementation(old)
        # Do not keep the adapter/checkpoint alive via a global closure.
        AttentionInterface._global_mapping.pop(backend, None)
        AttentionMaskInterface._global_mapping.pop(backend, None)
    expected = {f'{layer}-{head}' for layer, head in eligible}
    if set(scores) != expected:
        raise RuntimeError('QRScore backend did not report all eligible heads')
    return scores
