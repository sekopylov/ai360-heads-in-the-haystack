from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.base import GenerationResult, Prompt


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    needle: str
    question: str
    expected_answer: str
    haystack_dir: Path


@dataclass(frozen=True)
class NeedleSpan:
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start

    def contains(self, position: int) -> bool:
        return self.start <= position < self.end


@dataclass
class PreparedExample:
    case: ExperimentCase
    context: str
    prompt: Prompt
    needle_span: NeedleSpan
    context_length: int
    depth_percent: float


@dataclass
class RunResult:
    prepared: PreparedExample
    generation: GenerationResult
    score: float
    duration_seconds: float
