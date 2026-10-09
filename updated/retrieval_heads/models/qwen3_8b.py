"""Qwen3-8B adapters for native, unscaled extrapolation, and YaRN contexts."""

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


class Qwen3EightBNoYarn64KAdapter(Qwen3EightBAdapter):
    """Experimental 64K position budget with the checkpoint's original RoPE.

    This changes only the configured position ceiling. It deliberately does
    not apply YaRN or any other RoPE scaling, so positions beyond the native
    training range are unscaled extrapolation and may lose quality.
    """

    context_limit = 65536
    context_mode = "unscaled-64K"

    def _model_config_overrides(self) -> dict:
        return {"max_position_embeddings": self.context_limit}


class Qwen3EightBNoThinkingAdapter(Qwen3EightBAdapter):
    """Same weights/decoding; disable reasoning through Qwen's chat template.

    enable_thinking=False places a closed, empty think block in the prompt.
    This selects non-thinking generation, rather than hiding generated thoughts
    or forcing an early stop. EOS handling and attention backends are unchanged.
    """

    def _chat_template_kwargs(self) -> dict:
        return {"enable_thinking": False}


class Qwen3EightBNoYarn64KNoThinkingAdapter(
    Qwen3EightBNoThinkingAdapter, Qwen3EightBNoYarn64KAdapter,
):
    """Non-thinking prompt policy with original RoPE and a 64K position ceiling."""


class Qwen3EightBYarnNoThinkingAdapter(
    Qwen3EightBNoThinkingAdapter, Qwen3EightBYarnAdapter,
):
    """Non-thinking prompt policy with the existing static YaRN x4 configuration."""
