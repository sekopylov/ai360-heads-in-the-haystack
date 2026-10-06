from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Head = tuple[int, int]
CaptureMode = Literal["none", "top1", "full"]


@dataclass(frozen=True)
class AttentionRequest:
    """Attention behaviour requested for one generation run."""

    capture: CaptureMode = "none"
    blocked_heads: frozenset[Head] = frozenset()


@dataclass
class AttentionStep:
    """Attention produced while predicting one generated token.

    With ``capture="top1"``, each layer value has shape ``[heads]`` and
    contains source-token indices. With ``capture="full"``, it has shape
    ``[heads, key_length]`` and contains the full probabilities.
    """

    index: int
    token_id: int
    layers: dict[int, Any] = field(default_factory=dict)


class AttentionObserver(Protocol):
    def on_step(self, step: AttentionStep) -> None:
        """Consume attention for one newly generated token."""

