from __future__ import annotations

from statistics import mean

from rouge_score import rouge_scorer

from ..attention.types import AttentionStep, Head
from .types import NeedleSpan


class AnswerScorer:
    def __init__(self) -> None:
        self._rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    def score(self, expected: str, response: str) -> float:
        return self._rouge.score(expected, response)["rouge1"].recall * 100


class RetrievalScoreCollector:
    """Calculate the source experiment's head score as tokens arrive."""

    def __init__(
        self,
        *,
        eligible_heads: tuple[Head, ...],
        prompt_token_ids: list[int],
        needle_span: NeedleSpan,
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.needle_span = needle_span
        self.scores = {
            f"{layer}-{head}": 0.0
            for layer, head in eligible_heads
        }

    def on_step(self, step: AttentionStep) -> None:
        increment = 1.0 / len(self.needle_span)
        for layer, captured in step.layers.items():
            positions = captured if captured.ndim == 1 else captured.argmax(dim=-1)
            for head, source_position in enumerate(positions.tolist()):
                if (
                    self.needle_span.contains(source_position)
                    and source_position < len(self.prompt_token_ids)
                    and step.token_id == self.prompt_token_ids[source_position]
                ):
                    self.scores[f"{layer}-{head}"] += increment


def merge_scores(
    history: dict[str, list[float]],
    scores: dict[str, float],
) -> None:
    for key, score in scores.items():
        history.setdefault(key, []).append(float(score))


def rank_heads(history: dict[str, list[float]]) -> list[tuple[Head, float]]:
    ranked: list[tuple[Head, float]] = []
    for key, values in history.items():
        if not values:
            continue
        layer, head = (int(part) for part in key.split("-", maxsplit=1))
        ranked.append(((layer, head), mean(values)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)

