"""Text-only retrieval experiments with Ministral 3 8B Instruct."""
from __future__ import annotations

import torch
from transformers import Mistral3ForConditionalGeneration

from .base import Prompt
from .transformers_common import TransformersAdapter
from ..experiment.prompting import ministral_context_only_messages


class Ministral3EightBNoThinkingAdapter(TransformersAdapter):
    """Ministral 3 8B Instruct adapter without reasoning-mode generation.

    The Instruct checkpoint is the non-reasoning member of the Ministral 3
    family. Unlike Qwen3, it does not need a chat-template switch to disable
    thinking. The official checkpoint is multimodal, but these experiments
    pass text only and observe the nested Ministral3 language decoder.
    """

    model_class = Mistral3ForConditionalGeneration
    supports_flash_decode = True
    attention_scope = "all_decode_tokens"

    def __init__(
        self,
        model_id: str = "mistralai/Ministral-3-8B-Instruct-2512-BF16",
        **kwargs,
    ) -> None:
        super().__init__(model_id, **kwargs)

    def _language_config(self):
        return self.model.config.text_config

    @property
    def _language_model(self):
        return self.model.model.language_model

    def _discover_attention_layers(self) -> tuple[int, ...]:
        if self.model.config.model_type != "mistral3":
            raise ValueError(
                "The ministral3_8b_no_thinking adapter requires a Mistral3 multimodal checkpoint"
            )
        if self._language_config().model_type != "ministral3":
            raise ValueError("The Mistral3 checkpoint does not contain a Ministral3 text decoder")
        return tuple(range(len(self._language_model.layers)))

    @property
    def input_device(self) -> torch.device:
        return self._language_model.embed_tokens.weight.device

    def encode_prompt(self, context: str, question: str) -> Prompt:
        # MistralCommonBackend treats control tokens specially. Ask the chat
        # tokenizer for IDs directly instead of rendering to text and encoding
        # that text a second time.
        messages = ministral_context_only_messages(context, question)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        return Prompt(input_ids=input_ids, token_ids=input_ids[0].tolist())
