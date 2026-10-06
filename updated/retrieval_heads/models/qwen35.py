from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from ..attention.backend import (
    BACKEND_NAME,
    CONTROLLER_ARGUMENT,
    register_attention_backend,
)
from ..attention.controller import AttentionController, validate_blocked_heads
from ..attention.types import AttentionObserver, AttentionRequest, Head
from .base import GenerationResult, ModelAdapter, Prompt


class Qwen35Adapter(ModelAdapter):
    """Original Qwen3.5 plus observable token-by-token decoding."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-0.8B",
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        prefill_attention: str = "flash_attention_2",
        trust_remote_code: bool = False,
    ) -> None:
        register_attention_backend()
        self.model_id = model_id
        self.model_version = Path(model_id.rstrip("/")).name
        self.prefill_attention = prefill_attention

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=False,
            trust_remote_code=trust_remote_code,
        )
        # This is the unmodified Transformers implementation and checkpoint.
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=prefill_attention,
            trust_remote_code=trust_remote_code,
        ).eval()

        layers = self.model.model.layers
        self.full_attention_layers = tuple(
            index
            for index, layer in enumerate(layers)
            if getattr(layer, "block_type", None) == "full_attention"
        )
        if not self.full_attention_layers:
            raise RuntimeError("Qwen3.5 checkpoint has no full-attention layers")

        head_count = int(self.model.config.num_attention_heads)
        self._eligible_heads = tuple(
            (layer, head)
            for layer in self.full_attention_layers
            for head in range(head_count)
        )
        # Public state switch used for every generation run.
        self.attention = AttentionController(self.full_attention_layers)

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
        validate_blocked_heads(attention.blocked_heads, self.eligible_heads)
        input_ids = prompt.input_ids.to(self.input_device)
        if input_ids.shape[1] < 2:
            raise ValueError("Prompt must contain at least two tokens")
        if input_ids.shape[1] + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Prompt and generated tokens exceed the model context limit")

        # Fast, unobserved prefill exactly as in the source experiment.
        prefill = self.model(
            input_ids=input_ids[:, :-1],
            use_cache=True,
            logits_to_keep=1,
        )
        cache = prefill.past_key_values
        current_token = input_ids[:, -1:]
        generated: list[int] = []

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
                    self.attention.end_step(token_id)

                    token_text = self.tokenizer.convert_ids_to_tokens(token_id)
                    if token_text == "<0x0A>" or token_id == 144:
                        break
        finally:
            self.attention.finish()

        return GenerationResult(
            token_ids=generated,
            text=self.tokenizer.decode(
                generated,
                skip_special_tokens=True,
            ).strip(),
        )
