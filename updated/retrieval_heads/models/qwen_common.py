"""Qwen-specific defaults on top of the shared Transformers adapter."""
from .transformers_common import TransformersAdapter


class QwenAdapter(TransformersAdapter):
    """Preserve the historical Qwen3.5 newline stopping policy."""

    def _stop_reason(self, token_id: int) -> str | None:
        reason = super()._stop_reason(token_id)
        if reason is not None:
            return reason
        if "\n" in self.tokenizer.decode([token_id], skip_special_tokens=False):
            return "newline"
        return None
