from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ..attention.types import AttentionObserver, AttentionRequest, Head


@dataclass
class Prompt:
    input_ids: torch.Tensor
    token_ids: list[int]


@dataclass
class GenerationResult:
    """Uniform output: full generation plus the answer used for evaluation.

    text retains its existing final-answer meaning. raw_text always contains
    the full decode, including special tokens and any reasoning markers.
    final_answer_started identifies the answer phase, not EOS completion:
    a truncated plain answer can still have final_answer_started=True.
    """

    token_ids: list[int]
    text: str
    raw_text: str
    finish_reason: str
    final_answer_started: bool = True


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
