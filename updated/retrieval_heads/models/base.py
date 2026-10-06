from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..attention.types import AttentionObserver, AttentionRequest, Head


@dataclass
class Prompt:
    input_ids: Any
    token_ids: list[int]


@dataclass
class GenerationResult:
    token_ids: list[int]
    text: str


class ModelAdapter(ABC):
    """Only the operations that genuinely differ between model families."""

    model_id: str
    model_version: str

    @property
    @abstractmethod
    def tokenizer(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def period_tokens(self) -> list[int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def eligible_heads(self) -> tuple[Head, ...]:
        raise NotImplementedError

    @abstractmethod
    def encode_prompt(self, context: str, question: str) -> Prompt:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: Prompt,
        *,
        max_new_tokens: int,
        attention: AttentionRequest,
        observer: AttentionObserver | None = None,
    ) -> GenerationResult:
        raise NotImplementedError

