"""Qwen3.5 hybrid architecture adapter; retains legacy newline stopping."""
from transformers import Qwen3_5ForCausalLM

from .qwen_common import QwenAdapter


class Qwen35Adapter(QwenAdapter):
    model_class = Qwen3_5ForCausalLM

    def __init__(self, model_id: str = "Qwen/Qwen3.5-0.8B", **kwargs):
        super().__init__(model_id, **kwargs)

    def _discover_attention_layers(self) -> tuple[int, ...]:
        return tuple(index for index, layer in enumerate(self.model.model.layers)
                     if getattr(layer, "block_type", None) == "full_attention")
