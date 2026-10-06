from __future__ import annotations

from abc import ABC, abstractmethod

from .types import NeedleSpan


class NeedleLocator(ABC):
    @abstractmethod
    def find(self, prompt_token_ids: list[int], expected_answer: str) -> NeedleSpan:
        raise NotImplementedError


class LegacyOverlapLocator(NeedleLocator):
    """The original sliding-window token-set overlap heuristic."""

    def __init__(self, tokenizer, threshold: float = 0.9) -> None:
        self.tokenizer = tokenizer
        self.threshold = threshold

    def find(self, prompt_token_ids: list[int], expected_answer: str) -> NeedleSpan:
        needle_ids = self.tokenizer.encode(
            expected_answer,
            add_special_tokens=False,
        )
        needle_set = set(needle_ids)
        if not needle_set:
            raise ValueError("Expected answer tokenized to an empty sequence")

        span_length = len(needle_ids)
        for start in range(len(prompt_token_ids)):
            candidate = set(prompt_token_ids[start : start + span_length])
            overlap = len(candidate.intersection(needle_set)) / len(needle_set)
            if overlap > self.threshold:
                return NeedleSpan(start=start, end=start + span_length)
        raise ValueError("Could not locate expected-answer tokens inside the prompt")

