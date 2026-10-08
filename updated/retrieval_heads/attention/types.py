from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Head = tuple[int, int]
CaptureMode = Literal["none", "top1", "full"]
MaskMode = Literal["zero_output", "legacy_uniform"]


@dataclass(frozen=True)
class AttentionRequest:
    """Attention behaviour requested for one generation run."""

    capture: CaptureMode = "none"
    blocked_heads: frozenset[Head] = frozenset()
    mask_mode: MaskMode = "legacy_uniform"
    needle_span: tuple[int, int] | None = None


@dataclass
class AttentionStep:
    """Attention produced while predicting one generated token.

    With ``capture="top1"``, each layer value has shape ``[heads]`` and
    contains source-token indices. With ``capture="full"``, it has shape
    ``[heads, key_length]`` and contains the full probabilities.
    ``needle_attention_mass`` optionally contains one FP32 sum per head,
    reduced over the requested needle span before transferring to CPU.
    """

    index: int
    token_id: int
    layers: dict[int, Any] = field(default_factory=dict)
    needle_attention_mass: dict[int, Any] = field(default_factory=dict)


class AttentionObserver(Protocol):
    def on_step(self, step: AttentionStep) -> None:
        """Consume attention for one newly generated token."""
