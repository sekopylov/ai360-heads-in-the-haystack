from __future__ import annotations

import time

from ..attention.types import AttentionObserver, AttentionRequest
from ..models.base import ModelAdapter
from .data import ContextBuilder
from .locator import NeedleLocator
from .scoring import AnswerScorer
from .types import ExperimentCase, PreparedExample, RunResult


class ExperimentRunner:
    """Model-independent preparation, generation and answer evaluation."""

    def __init__(
        self,
        model: ModelAdapter,
        context_builder: ContextBuilder,
        locator: NeedleLocator | None,
        *,
        max_new_tokens: int = 50,
    ) -> None:
        self.model = model
        self.context_builder = context_builder
        self.locator = locator
        self.max_new_tokens = max_new_tokens
        self.answer_scorer = AnswerScorer()

    def prepare(
        self,
        case: ExperimentCase,
        *,
        context_length: int,
        depth_percent: float,
    ) -> PreparedExample:
        context = self.context_builder.build(
            case,
            context_length=context_length,
            depth_percent=depth_percent,
        )
        prompt = self.model.encode_prompt(context, case.question)
        needle_span = (
            self.locator.find(prompt.token_ids, case.expected_answer)
            if self.locator is not None
            else None
        )
        return PreparedExample(
            case=case,
            context=context,
            prompt=prompt,
            needle_span=needle_span,
            context_length=context_length,
            depth_percent=depth_percent,
        )

    def run(
        self,
        prepared: PreparedExample,
        *,
        attention: AttentionRequest,
        observer: AttentionObserver | None = None,
    ) -> RunResult:
        started = time.perf_counter()
        generation = self.model.generate(
            prepared.prompt,
            max_new_tokens=self.max_new_tokens,
            attention=attention,
            observer=observer,
        )
        duration = time.perf_counter() - started
        score = self.answer_scorer.score(
            prepared.case.expected_answer,
            generation.text,
        )
        return RunResult(
            prepared=prepared,
            generation=generation,
            score=score,
            duration_seconds=duration,
        )
