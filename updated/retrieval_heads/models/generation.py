"""Shared cached greedy decode; model policies are supplied by the adapter."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..attention.backend import CONTROLLER_ARGUMENT
from ..attention.controller import validate_blocked_heads
from ..attention.types import AttentionObserver, AttentionRequest
from .base import GenerationResult, Prompt

if TYPE_CHECKING:
    from .transformers_common import TransformersAdapter


class GenerationEngine:
    @torch.inference_mode()
    def generate(
        self,
        adapter: TransformersAdapter,
        prompt: Prompt,
        *,
        max_new_tokens: int,
        attention: AttentionRequest,
        observer: AttentionObserver | None = None,
    ) -> GenerationResult:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if adapter.decode_attention == "sdpa_flash":
            from ..attention.flash import validate_flash_request
            validate_flash_request(attention, observer)
        if attention.mask_mode not in {"zero_output", "legacy_uniform"}:
            raise ValueError(f"Unknown mask mode: {attention.mask_mode!r}")
        validate_blocked_heads(attention.blocked_heads, adapter.eligible_heads)
        input_ids = prompt.input_ids.to(adapter.input_device)
        if input_ids.shape[1] < 2:
            raise ValueError("Prompt must contain at least two tokens")
        if input_ids.shape[1] + max_new_tokens > adapter._language_config().max_position_embeddings:
            raise ValueError("Prompt and generated tokens exceed the model context limit")

        if input_ids.is_cuda:
            torch.cuda.reset_peak_memory_stats(input_ids.device)
        # Unobserved prefill; its kernel is selected independently of decode.
        prefill = adapter.model(
            input_ids=input_ids[:, :-1],
            use_cache=True,
            logits_to_keep=1,
        )
        cache = prefill.past_key_values
        if input_ids.is_cuda:
            print(f"[memory] prefill={adapter.prefill_attention} "
                  f"peak_allocated_gib={torch.cuda.max_memory_allocated(input_ids.device) / 1024**3:.3f}",
                  flush=True)
        current_token = input_ids[:, -1:]
        generated: list[int] = []
        finish_reason = "max_new_tokens"
        publish_attention = adapter._attention_filter(prompt)

        adapter.attention.start(attention, observer)
        try:
            with adapter._observable_decode():
                for _ in range(max_new_tokens):
                    adapter.attention.begin_step()
                    outputs = adapter.model(
                        input_ids=current_token,
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                        **{CONTROLLER_ARGUMENT: adapter.attention},
                    )
                    cache = outputs.past_key_values
                    current_token = outputs.logits[:, -1, :].argmax(
                        dim=-1,
                        keepdim=True,
                    )
                    token_id = int(current_token.item())
                    generated.append(token_id)
                    adapter.attention.end_step(token_id, publish=publish_attention(token_id))

                    reason = adapter._stop_reason(token_id)
                    if reason is not None:
                        finish_reason = reason
                        break
        finally:
            adapter.attention.finish()

        if input_ids.is_cuda:
            print(f"[memory] generation peak_allocated_gib="
                  f"{torch.cuda.max_memory_allocated(input_ids.device) / 1024**3:.3f}", flush=True)
        return adapter._generation_result(prompt, generated, finish_reason)
