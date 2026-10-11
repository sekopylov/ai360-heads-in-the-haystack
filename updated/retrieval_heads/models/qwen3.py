"""Dense Qwen3 with observed attention and shared thinking answer extraction."""
from transformers import Qwen3ForCausalLM

from .output import QwenThinkingOutputPolicy
from .qwen_common import QwenAdapter


class Qwen3Adapter(QwenAdapter):
    model_class = Qwen3ForCausalLM
    output_policy_class = QwenThinkingOutputPolicy
    supports_flash_decode = True
    supported_attention_scopes = frozenset({"answer_only", "all_decode_tokens"})
    attention_scope = "answer_only"

    def __init__(self, model_id: str = "Qwen/Qwen3-4B-Thinking-2507", **kwargs):
        super().__init__(model_id, **kwargs)

    def _discover_attention_layers(self) -> tuple[int, ...]:
        if self.model.config.model_type != "qwen3":
            raise ValueError("The qwen3 adapter requires a dense Qwen3 checkpoint")
        return tuple(index for index, layer in enumerate(self.model.model.layers)
                     if getattr(layer.self_attn, "sliding_window", None) is None)

    def _stop_reason(self, token_id: int) -> str | None:
        # Thinking includes newlines; it must be allowed to finish naturally.
        return "eos" if token_id in self._eos_token_ids else None
