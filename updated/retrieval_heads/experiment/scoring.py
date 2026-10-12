from __future__ import annotations

from statistics import mean
from collections import Counter
from abc import ABC, abstractmethod
from math import isfinite

from rouge_score import rouge_scorer

from ..attention.types import AttentionStep, Head
from .types import NeedleSpan


class AnswerScorer:
    def __init__(self) -> None:
        self._rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    def score(self, expected: str, response: str) -> float:
        return self._rouge.score(expected, response)["rouge1"].recall * 100


class RetrievalScoreCollector(ABC):
    """Common per-head result interface and needle-span validation."""

    metric_name: str
    needs_top1 = False
    needs_needle_mass = False

    def __init__(
        self,
        *,
        eligible_heads: tuple[Head, ...],
        prompt_token_ids: list[int],
        needle_span: NeedleSpan,
    ) -> None:
        self.prompt_token_ids = prompt_token_ids
        self.needle_span = needle_span
        if not 0 <= needle_span.start < needle_span.end <= len(prompt_token_ids):
            raise ValueError("Needle span must be nonempty and inside prompt tokens")
        self.scores = {
            f"{layer}-{head}": 0.0
            for layer, head in eligible_heads
        }

    @abstractmethod
    def on_step(self, step: AttentionStep) -> None:
        raise NotImplementedError


class TokenHitRetrievalScoreCollector(RetrievalScoreCollector):
    needs_top1 = True

    def on_step(self, step: AttentionStep) -> None:
        for layer, captured in step.layers.items():
            positions = captured if captured.ndim == 1 else captured.argmax(dim=-1)
            for head, source_position in enumerate(positions.tolist()):
                if (
                    self.needle_span.contains(source_position)
                    and source_position < len(self.prompt_token_ids)
                    and step.token_id == self.prompt_token_ids[source_position]
                ):
                    self._credit_hit(f"{layer}-{head}", step.token_id)

    @abstractmethod
    def _credit_hit(self, head_key: str, token_id: int) -> None:
        """Update this metric for one matching attention hit."""
        raise NotImplementedError


class LegacyRetrievalScoreCollector(TokenHitRetrievalScoreCollector):
    """Source formula: every hit contributes, including unlimited repeats."""

    metric_name = "legacy"

    def _credit_hit(self, head_key: str, token_id: int) -> None:
        self.scores[head_key] += 1.0 / len(self.needle_span)


class MultisetRetrievalScoreCollector(TokenHitRetrievalScoreCollector):
    """Needle token frequency quotas per head; coverage stays in [0, 1]."""

    metric_name = "needle_token_multiset_v1"

    def __init__(self, *, eligible_heads: tuple[Head, ...],
                 prompt_token_ids: list[int], needle_span: NeedleSpan) -> None:
        super().__init__(eligible_heads=eligible_heads, prompt_token_ids=prompt_token_ids,
                         needle_span=needle_span)
        self.needle_token_counts = Counter(prompt_token_ids[needle_span.start:needle_span.end])
        self._matched_tokens = {key: Counter() for key in self.scores}
        self._matched_totals = Counter()

    def _credit_hit(self, head_key: str, token_id: int) -> None:
        matched = self._matched_tokens[head_key]
        if matched[token_id] < self.needle_token_counts[token_id]:
            matched[token_id] += 1
            self._matched_totals[head_key] += 1
            self.scores[head_key] = self._matched_totals[head_key] / len(self.needle_span)


class NeedleAttentionMassCollector(RetrievalScoreCollector):
    """Mean needle attention mass on generated tokens occurring in the needle."""

    metric_name = "needle_attention_mass_v1"
    needs_needle_mass = True
    mass_attribute = 'needle_attention_mass'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.needle_tokens = set(self.prompt_token_ids[self.needle_span.start:self.needle_span.end])
        self._totals = dict.fromkeys(self.scores, 0.0)
        self._counts = dict.fromkeys(self.scores, 0)
        self.qualifying_steps = 0

    def on_step(self, step: AttentionStep) -> None:
        if step.token_id not in self.needle_tokens:
            return
        masses = getattr(step, self.mass_attribute)
        if not masses:
            # Useful for externally supplied full snapshots; CLI uses GPU reduction.
            masses = {}
            for layer, captured in step.layers.items():
                if captured.ndim != 2:
                    raise ValueError("needle_attention_mass_v1 needs needle mass or full probabilities, not top1")
                span = getattr(self, 'target_span', self.needle_span)
                if captured.shape[-1] < span.end:
                    raise ValueError("Needle span exceeds attention key length")
                masses[layer] = captured[:, span.start:span.end].float().sum(dim=-1)
        seen = set()
        for layer, values in masses.items():
            for head, mass in enumerate(values.tolist()):
                key = f"{layer}-{head}"
                if key not in self.scores:
                    continue
                if not isfinite(float(mass)):
                    raise ValueError("Non-finite needle attention mass")
                seen.add(key)
                self._totals[key] += min(1.0, max(0.0, float(mass)))
                self._counts[key] += 1
                self.scores[key] = self._totals[key] / self._counts[key]
        if seen != set(self.scores):
            raise ValueError("Needle attention mass missing for eligible heads")
        self.qualifying_steps += 1


class QuestionAttentionMassCollector(NeedleAttentionMassCollector):
    """Decode mass on the question, using exactly the needle metric's step gate."""

    metric_name = 'question_attention_mass_v1'
    mass_attribute = 'question_attention_mass'

    def __init__(self, *, question_span, **kwargs):
        super().__init__(**kwargs)
        if question_span is None or not 0 <= question_span.start < question_span.end <= len(self.prompt_token_ids):
            raise ValueError('Question span must be nonempty and inside prompt tokens')
        self.question_span = question_span
        self.target_span = question_span


class FullNeedleAttentionMassCollector(NeedleAttentionMassCollector):
    """Same decode token gate, mass on the exact complete inserted needle."""

    metric_name = 'full_needle_attention_mass_v1'
    mass_attribute = 'full_needle_attention_mass'

    def __init__(self, *, full_needle_span, **kwargs):
        super().__init__(**kwargs)
        if full_needle_span is None or not 0 <= full_needle_span.start < full_needle_span.end <= len(self.prompt_token_ids):
            raise ValueError('Full needle span must be nonempty and inside prompt tokens')
        self.target_span = full_needle_span


class QueryRetrievalScoreCollector(RetrievalScoreCollector):
    """QRHead Eq.3; scores supplied by an independent prompt-only pass."""

    metric_name = 'query_retrieval_score_v1'
    needs_prefill = True

    def set_prefill_scores(self, scores):
        if set(scores) != set(self.scores) or any(
                not isfinite(v) or not 0 <= v <= 1.000001 for v in scores.values()):
            raise ValueError('Invalid or incomplete QRScore head scores')
        self.scores = dict(scores)

    def on_step(self, step):
        # Decode tokens (including reasoning) are not part of QRScore.
        pass


_RETRIEVAL_COLLECTORS: dict[str, type[RetrievalScoreCollector]] = {
    LegacyRetrievalScoreCollector.metric_name: LegacyRetrievalScoreCollector,
    MultisetRetrievalScoreCollector.metric_name: MultisetRetrievalScoreCollector,
    NeedleAttentionMassCollector.metric_name: NeedleAttentionMassCollector,
    QueryRetrievalScoreCollector.metric_name: QueryRetrievalScoreCollector,
    QuestionAttentionMassCollector.metric_name: QuestionAttentionMassCollector,
    FullNeedleAttentionMassCollector.metric_name: FullNeedleAttentionMassCollector,
}


def available_retrieval_metrics() -> tuple[str, ...]:
    return tuple(_RETRIEVAL_COLLECTORS)


def select_retrieval_metrics(single: str | None, multiple: str | None) -> tuple[str | None, list[str]]:
    if single is not None and multiple is not None:
        raise ValueError("Use either --retrieval-metric or --retrieval-metrics, not both")
    names = [n.strip() for n in multiple.split(",")] if multiple is not None else [single or "needle_token_multiset_v1"]
    if not names or any(n not in _RETRIEVAL_COLLECTORS for n in names):
        raise ValueError(f"Unknown or empty retrieval metric in {names}")
    if len(names) != len(set(names)):
        raise ValueError("Retrieval metrics must be distinct")
    return names[0] if len(names) == 1 else None, names


def retrieval_capture_requirements(names: list[str]) -> tuple[bool, bool]:
    classes = [_RETRIEVAL_COLLECTORS[n] for n in names]
    return any(c.needs_top1 for c in classes), any(c.needs_needle_mass for c in classes)


def create_retrieval_collector(
    metric: str, *, eligible_heads: tuple[Head, ...],
    prompt_token_ids: list[int], needle_span: NeedleSpan,
    question_span: NeedleSpan | None = None,
    full_needle_span: NeedleSpan | None = None,
) -> RetrievalScoreCollector:
    try:
        collector_class = _RETRIEVAL_COLLECTORS[metric]
    except KeyError as error:
        raise ValueError(f"Unknown retrieval metric: {metric!r}") from error
    extra = {'question_span': question_span} if metric == 'question_attention_mass_v1' else {}
    if metric == 'full_needle_attention_mass_v1':
        extra = {'full_needle_span': full_needle_span}
    return collector_class(**extra, eligible_heads=eligible_heads, prompt_token_ids=prompt_token_ids,
                           needle_span=needle_span)


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
