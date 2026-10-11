from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..attention.backend import (
    BACKEND_NAME,
    register_attention_backend,
)
from ..attention.controller import AttentionController
from ..attention.prefill import resolve_prefill_backend
from ..attention.types import AttentionObserver, AttentionRequest, Head
from .base import GenerationResult, ModelAdapter, Prompt
from ..model_source import resolve_model_source
from .generation import GenerationEngine
from .output import PlainOutputPolicy
from ..experiment.prompting import experiment_messages


class TransformersAdapter(ModelAdapter):
    """Transformers loading and adapter hooks; generation belongs to the engine."""

    supports_flash_decode = False
    output_policy_class = PlainOutputPolicy
    supported_attention_scopes = frozenset({"all_decode_tokens"})

    model_class: type
    attention_scope = "all_decode_tokens"

    def __init__(
        self,
        model_id: str,
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        prefill_attention: str = "sdpa",
        decode_attention: str = "eager",
        trust_remote_code: bool = False,
        model_search_dirs: list[str] | None = None,
        attention_scope: str | None = None,
    ) -> None:
        scope = attention_scope or type(self).attention_scope
        if scope not in {"answer_only", "all_decode_tokens"}:
            raise ValueError(f"Unknown attention scope: {scope!r}")
        if scope not in self.supported_attention_scopes:
            raise ValueError(f"{type(self).__name__} does not support attention scope {scope!r}")
        if decode_attention not in {"eager", "sdpa_flash"}:
            raise ValueError(f"Unknown decode attention: {decode_attention!r}")
        self.decode_attention = decode_attention
        if decode_attention == "sdpa_flash":
            from ..attention.flash import register_flash_backends
            register_flash_backends()
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
            {int(token_id) for token_id in eos_token_ids}
            if isinstance(eos_token_ids, (list, tuple, set))
            else {int(eos_token_ids)} if eos_token_ids is not None else set()
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
        if (
            decode_attention == "sdpa_flash"
            and not self.supports_flash_decode
        ):
            raise ValueError(
                f"{type(self).__name__} does not support sdpa_flash masking decode"
            )
        devices = sorted({str(parameter.device) for parameter in self.model.parameters()})
        print(f"[model] parameter devices={devices}; device_map="
              f"{getattr(self.model, 'hf_device_map', None)}", flush=True)

        self.full_attention_layers = self._discover_attention_layers()
        if not self.full_attention_layers:
            raise RuntimeError("Checkpoint has no full-attention layers")

        head_count = int(self._language_config().num_attention_heads)
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

    def _language_config(self):
        """Return the decoder config for text-only and wrapped multimodal models."""
        return self.model.config

    def _discover_attention_layers(self) -> tuple[int, ...]:
        raise NotImplementedError

    def _attention_filter(self, prompt: Prompt):
        """Return a per-generation token predicate for publishing observations."""
        return self.output_policy_class(self.tokenizer, self.attention_scope).attention_filter(prompt)

    def _stop_reason(self, token_id: int) -> str | None:
        if token_id in self._eos_token_ids:
            return "eos"
        return None

    def _generation_result(self, prompt: Prompt, generated: list[int], reason: str) -> GenerationResult:
        answer = self.output_policy_class(self.tokenizer, self.attention_scope).answer(prompt, generated)
        return GenerationResult(
            token_ids=generated,
            text=answer.text,
            raw_text=self.tokenizer.decode(generated, skip_special_tokens=False),
            finish_reason=reason,
            final_answer_started=answer.final_answer_started,
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
        messages = experiment_messages(context, question)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self._chat_template_kwargs(),
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

    def _chat_template_kwargs(self) -> dict:
        """Adapter-specific prompt policy; existing adapters keep template defaults."""
        return {}

    @contextmanager
    def _observable_decode(self):
        old_implementation = self.model.config._attn_implementation
        backend = BACKEND_NAME
        if self.decode_attention == "sdpa_flash":
            from ..attention.flash import DECODE_NAME
            backend = DECODE_NAME
        self.model.set_attn_implementation(backend)
        try:
            yield
        finally:
            self.model.set_attn_implementation(old_implementation)

    def generate(
        self,
        prompt: Prompt,
        *,
        max_new_tokens: int,
        attention: AttentionRequest,
        observer: AttentionObserver | None = None,
    ) -> GenerationResult:
        return GenerationEngine().generate(
            self, prompt, max_new_tokens=max_new_tokens,
            attention=attention, observer=observer,
        )
