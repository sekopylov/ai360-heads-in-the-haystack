"""Original Qwen3-8B with the native 32k context budget, no YaRN override."""

from ..attention.types import AttentionObserver, AttentionRequest
from .base import GenerationResult, Prompt
from .qwen3 import Qwen3Adapter


class Qwen3EightBAdapter(Qwen3Adapter):
    """Reuse dense Qwen3 attention, thinking extraction and EOS handling."""

    native_context_limit = 32768
    context_limit = native_context_limit
    context_mode = "native"

    def __init__(self, model_id: str = "Qwen/Qwen3-8B", **kwargs):
        super().__init__(model_id, **kwargs)

    def generate(
        self,
        prompt: Prompt,
        *,
        max_new_tokens: int,
        attention: AttentionRequest,
        observer: AttentionObserver | None = None,
    ) -> GenerationResult:
        # Count the entire chat prompt, not just the haystack. The generation
        # budget includes both thinking and the final answer.
        prompt_length = int(prompt.input_ids.shape[1])
        if prompt_length + max_new_tokens > self.context_limit:
            raise ValueError(
                f"Qwen3-8B {self.context_mode} context budget exceeded: prompt={prompt_length} "
                f"+ max_new_tokens={max_new_tokens} > {self.context_limit}. "
                "Reduce context or output length."
            )
        return super().generate(prompt, max_new_tokens=max_new_tokens,
                                attention=attention, observer=observer)


class Qwen3EightBYarnAdapter(Qwen3EightBAdapter):
    """Same checkpoint and inference logic, with static YaRN x4 enabled."""

    context_limit = 131072
    context_mode = "YaRN"

    def _model_config_overrides(self) -> dict:
        return {
            "max_position_embeddings": self.context_limit,
            # Transformers 5 uses rope_parameters (formerly rope_scaling).
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": 1000000.0,
                "factor": 4.0,
                "original_max_position_embeddings": self.native_context_limit,
            },
        }
