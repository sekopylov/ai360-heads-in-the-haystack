from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..attention.backend import (
    BACKEND_NAME,
    CONTROLLER_ARGUMENT,
    register_attention_backend,
)
from ..attention.controller import AttentionController, validate_blocked_heads
from ..attention.prefill import resolve_prefill_backend
from ..attention.types import AttentionObserver, AttentionRequest, Head
from .base import GenerationResult, ModelAdapter, Prompt
from ..model_source import resolve_model_source


class QwenAdapter(ModelAdapter):
    """Shared original-model loading, prompting and cached greedy decoding."""

    model_class: type
    attention_scope = "all_decode_tokens"

    def __init__(
        self,
        model_id: str,
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        prefill_attention: str = "sdpa",
        trust_remote_code: bool = False,
        model_search_dirs: list[str] | None = None,
        attention_scope: str | None = None,
    ) -> None:
        scope = attention_scope or type(self).attention_scope
        if scope not in {"answer_only", "all_decode_tokens"}:
            raise ValueError(f"Unknown attention scope: {scope!r}")
        if scope == "answer_only" and type(self).attention_scope != "answer_only":
            raise ValueError("answer_only filtering is supported by the qwen3 adapter")
        self.attention_scope = scope
        register_attention_backend()
        self.model_id = model_id
        self.model_version = Path(model_id.rstrip("/")).name
        self.prefill_attention = prefill_attention
        source, local_only = resolve_model_source(model_id, model_search_dirs)

        self._tokenizer = AutoTokenizer.from_pretrained(
            source,
            local_files_only=local_only,
            use_fast=False,
            trust_remote_code=trust_remote_code,
        )
        eos_token_ids = self._tokenizer.eos_token_id
        self._eos_token_ids = (
            {int(eos_token_ids)} if eos_token_ids is not None else set()
        )
        # This is the unmodified Transformers implementation and checkpoint.
        self.model = self.model_class.from_pretrained(
            source,
            local_files_only=local_only,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=resolve_prefill_backend(prefill_attention),
            trust_remote_code=trust_remote_code,
            **self._model_config_overrides(),
        ).eval()
        devices = sorted({str(parameter.device) for parameter in self.model.parameters()})
        print(f"[model] parameter devices={devices}; device_map="
              f"{getattr(self.model, 'hf_device_map', None)}", flush=True)

        self.full_attention_layers = self._discover_attention_layers()
        if not self.full_attention_layers:
            raise RuntimeError("Checkpoint has no full-attention layers")

        head_count = int(self.model.config.num_attention_heads)
        self._eligible_heads = tuple(
            (layer, head)
            for layer in self.full_attention_layers
            for head in range(head_count)
        )
        # Public state switch used for every generation run.
        self.attention = AttentionController(self.full_attention_layers)

    def _model_config_overrides(self) -> dict:
        """Configuration-only changes applied before constructing model layers."""
        return {}

    def _discover_attention_layers(self) -> tuple[int, ...]:
        raise NotImplementedError

    def _attention_filter(self, prompt: Prompt):
        """Return a per-generation token predicate for publishing observations."""
        return lambda token_id: True

    def _stop_reason(self, token_id: int) -> str | None:
        if token_id in self._eos_token_ids:
            return "eos"
        if "\n" in self.tokenizer.decode([token_id], skip_special_tokens=False):
            return "newline"
        return None

    def _generation_result(self, prompt: Prompt, generated: list[int], reason: str) -> GenerationResult:
        return GenerationResult(
            token_ids=generated,
            text=self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def period_tokens(self) -> list[int]:
        return self.tokenizer.encode(".", add_special_tokens=False)

    @property
    def eligible_heads(self) -> tuple[Head, ...]:
        return self._eligible_heads

    @property
    def input_device(self) -> torch.device:
        return self.model.model.embed_tokens.weight.device

    def encode_prompt(self, context: str, question: str) -> Prompt:
        messages = [
            {
                "role": "user",
                "content": (
                    f"<book>{context}</book>\n"
                    f"Based on the content of the book, Question: {question}\n"
                    "Answer:"
                ),
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        return Prompt(
            input_ids=input_ids,
            token_ids=input_ids[0].tolist(),
        )

    @contextmanager
    def _observable_decode(self):
        old_implementation = self.model.config._attn_implementation
        self.model.set_attn_implementation(BACKEND_NAME)
        try:
            yield
        finally:
            self.model.set_attn_implementation(old_implementation)

    @torch.inference_mode()
    def generate(
        self,
        prompt: Prompt,
        *,
        max_new_tokens: int,
        attention: AttentionRequest,
        observer: AttentionObserver | None = None,
    ) -> GenerationResult:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if attention.mask_mode not in {"zero_output", "legacy_uniform"}:
            raise ValueError(f"Unknown mask mode: {attention.mask_mode!r}")
        validate_blocked_heads(attention.blocked_heads, self.eligible_heads)
        input_ids = prompt.input_ids.to(self.input_device)
        if input_ids.shape[1] < 2:
            raise ValueError("Prompt must contain at least two tokens")
        if input_ids.shape[1] + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Prompt and generated tokens exceed the model context limit")

        if input_ids.is_cuda:
            torch.cuda.reset_peak_memory_stats(input_ids.device)
        # Unobserved prefill; its kernel is selected independently of decode.
        prefill = self.model(
            input_ids=input_ids[:, :-1],
            use_cache=True,
            logits_to_keep=1,
        )
        cache = prefill.past_key_values
        if input_ids.is_cuda:
            print(f"[memory] prefill={self.prefill_attention} "
                  f"peak_allocated_gib={torch.cuda.max_memory_allocated(input_ids.device) / 1024**3:.3f}",
                  flush=True)
        current_token = input_ids[:, -1:]
        generated: list[int] = []
        finish_reason = "max_new_tokens"
        publish_attention = self._attention_filter(prompt)

        self.attention.start(attention, observer)
        try:
            with self._observable_decode():
                for _ in range(max_new_tokens):
                    self.attention.begin_step()
                    outputs = self.model(
                        input_ids=current_token,
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                        **{CONTROLLER_ARGUMENT: self.attention},
                    )
                    cache = outputs.past_key_values
                    current_token = outputs.logits[:, -1, :].argmax(
                        dim=-1,
                        keepdim=True,
                    )
                    token_id = int(current_token.item())
                    generated.append(token_id)
                    self.attention.end_step(token_id, publish=publish_attention(token_id))

                    reason = self._stop_reason(token_id)
                    if reason is not None:
                        finish_reason = reason
                        break
        finally:
            self.attention.finish()

        if input_ids.is_cuda:
            print(f"[memory] generation peak_allocated_gib="
                  f"{torch.cuda.max_memory_allocated(input_ids.device) / 1024**3:.3f}", flush=True)
        return self._generation_result(prompt, generated, finish_reason)
