"""Properties of retrieval heads (paper Sec. 4): sparse, dynamic, intrinsic.

The functions here turn :class:`~retrieval_heads.scoring.RetrievalScores`
matrices into the numbers behind Fig. 2 (sparsity), Fig. 3 (score vs activation
frequency) and Fig. 5 (cross-model correlation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from retrieval_heads.scoring import RetrievalScores
from retrieval_heads.utils import get_logger

log = get_logger("properties")


# --------------------------------------------------------------------------- sparsity
def category_fractions(scores: RetrievalScores, threshold: float = 0.1) -> dict[str, Any]:
    """Split heads into the paper's three buckets: zero, low, retrieval.

    Quoted from Sec. 4.1: 45-73% of heads score 0, 25-52% land in (0, 0.1], and
    only about 3-6% exceed 0.1.

    Non-finite scores are dropped rather than counted: a single NaN used to make
    the three fractions sum to more than the number of usable heads and silently
    renormalise the ring chart.
    """
    values = np.array([scores.head_score(h) for h in scores.info.scoreable_heads], dtype=float)
    finite = values[np.isfinite(values)]
    n_non_finite = int(values.size - finite.size)
    values = finite
    total = len(values)
    if total == 0:
        # Keep the shape: callers (plot_score_pie) read the buckets unconditionally.
        empty = {"n": 0, "frac": 0.0}
        return {"n_heads": 0, "n_non_finite": n_non_finite, "threshold": threshold,
                "zero": dict(empty), "low": dict(empty), "retrieval": dict(empty),
                "max_score": 0.0, "mean_score": 0.0}
    zero = int(np.sum(values <= 0.0))
    low = int(np.sum((values > 0.0) & (values <= threshold)))
    retrieval = int(np.sum(values > threshold))
    return {
        "n_heads": total,
        "n_non_finite": n_non_finite,
        "threshold": threshold,
        "zero": {"n": zero, "frac": zero / total},
        "low": {"n": low, "frac": low / total},
        "retrieval": {"n": retrieval, "frac": retrieval / total},
        "max_score": float(values.max()),
        "mean_score": float(values.mean()),
    }


def score_histogram(scores: RetrievalScores, bins: Sequence[float] | int = 20) -> dict[str, Any]:
    """Histogram of retrieval scores over all scoreable heads."""
    values = np.array([scores.head_score(h) for h in scores.info.scoreable_heads], dtype=float)
    finite = values[np.isfinite(values)]
    counts, edges = np.histogram(finite, bins=bins, range=(0.0, 1.0))
    return {"counts": counts.tolist(), "edges": edges.tolist(),
            "total": int(finite.size), "n_non_finite": int(values.size - finite.size),
            "model": scores.info.name}


def activation_gap(scores: RetrievalScores, top_k: int = 40) -> dict[str, Any]:
    """Score vs activation frequency, sorted by score -- the gap in Fig. 3.

    A head with activation frequency 1.0 fires on *every* context; a head with a
    high score but frequency < 1 is context-sensitive.
    """
    ranked = scores.ranked_heads()[:top_k]
    return {
        "heads": [str(h) for h in ranked],
        "score": [scores.head_score(h) for h in ranked],
        "activation_freq": [float(scores.activation_freq[h.layer, h.head]) for h in ranked],
        "always_active": [str(h) for h in scores.info.scoreable_heads
                          if float(scores.activation_freq[h.layer, h.head]) >= 1.0],
        "model": scores.info.name,
    }


def pie_data(scores: RetrievalScores, thresholds: Sequence[float] = (0.0, 0.1, 0.5)) -> dict[str, Any]:
    """Ring/pie breakdown per threshold, the data behind ``ring_graph.pdf``."""
    values = np.array([scores.head_score(h) for h in scores.info.scoreable_heads], dtype=float)
    values = values[np.isfinite(values)]
    total = max(len(values), 1)
    slices = {}
    for t in thresholds:
        above = int(np.sum(values > t))
        slices[f">{t}"] = {"n": above, "frac": above / total}
    return {"model": scores.info.name, "n_heads": len(values), "slices": slices}


# --------------------------------------------------------------------------- correlation
def layouts_match(scores_a: RetrievalScores, scores_b: RetrievalScores) -> bool:
    """Whether two runs address the same layer x head grid.

    ``mode="grid"`` compares positions, which only means something when both runs
    share this layout; otherwise the smaller grid is nearest-neighbour stretched
    and duplicated rows count as independent observations.
    """
    def key(scores: RetrievalScores):
        info = scores.info
        return (info.num_layers, tuple(sorted(info.scoreable_layers)),
                tuple(sorted(info.num_heads.items())))

    return key(scores_a) == key(scores_b)


def _resample(matrix: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample a layer x head matrix onto a common grid.

    Required because two models in different families rarely share a layer/head
    count; the paper still reports a (low) correlation between them.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    src_l, src_h = arr.shape
    dst_l, dst_h = shape
    li = np.clip((np.arange(dst_l) * src_l / dst_l).astype(int), 0, src_l - 1)
    hi = np.clip((np.arange(dst_h) * src_h / dst_h).astype(int), 0, src_h - 1)
    return arr[np.ix_(li, hi)]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over the finite entries shared by ``a`` and ``b``."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    x, y = a[mask], b[mask]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def correlate(
    scores_a: RetrievalScores,
    scores_b: RetrievalScores,
    *,
    mode: str = "grid",
) -> float:
    """Correlation between two models' retrieval-score distributions.

    ``mode="grid"`` compares layer x head position by position (the right choice
    inside one family, where architectures match).  ``mode="sorted"`` compares
    the sorted score vectors, which is the only fair option across families with
    different layer/head counts.
    """
    a = scores_a.score.numpy().astype(np.float64)
    b = scores_b.score.numpy().astype(np.float64)
    if mode == "sorted":
        va = np.sort(a[np.isfinite(a)])[::-1]
        vb = np.sort(b[np.isfinite(b)])[::-1]
        n = min(len(va), len(vb))
        if n < 3:
            return float("nan")
        return pearson(va[:n], vb[:n])
    if mode != "grid":
        raise ValueError(f"unknown correlation mode: {mode!r}")
    if not layouts_match(scores_a, scores_b):
        # Returning a number here invited reading a resampled (duplicated-row)
        # correlation as a real one; NaN is the honest answer.
        log.warning(
            "grid correlation between %s and %s uses different layer/head layouts; "
            "returning NaN (use mode='sorted' for a cross-family comparison)",
            scores_a.info.name, scores_b.info.name,
        )
        return float("nan")
    if a.shape != b.shape:
        # pad the smaller grid so both cover the full depth of the network
        cols = max(a.shape[1], b.shape[1])
        target = (max(a.shape[0], b.shape[0]), cols)
        a, b = _resample(a, target), _resample(b, target)
    return pearson(a, b)


@dataclass
class CorrelationMatrix:
    labels: list[str]
    values: list[list[float]]
    #: Interpretation note (e.g. that `sorted` mode does not compare head
    #: positions); carried to the figure so a PDF reader sees it too.
    caveat: str | None = None
    mode: str = "grid"

    def as_dict(self) -> dict[str, Any]:
        return {"labels": self.labels, "values": self.values, "mode": self.mode}

    def same_family_hint(self, threshold: float = 0.8) -> list[tuple[str, str, float]]:
        """Pairs whose correlation clears ``threshold`` (the paper's ``> 0.8`` bar)."""
        out = []
        for i, li in enumerate(self.labels):
            for j, lj in enumerate(self.labels):
                if j <= i:
                    continue
                value = self.values[i][j]
                if np.isfinite(value) and value > threshold:
                    out.append((li, lj, value))
        return out


def correlation_matrix(
    runs: Sequence[RetrievalScores],
    *,
    mode: str = "grid",
    labels: Sequence[str] | None = None,
) -> CorrelationMatrix:
    labels = list(labels) if labels else [r.info.name for r in runs]
    n = len(runs)
    values = [[float("nan")] * n for _ in range(n)]
    for i in range(n):
        values[i][i] = 1.0
        for j in range(i + 1, n):
            value = correlate(runs[i], runs[j], mode=mode)
            values[i][j] = values[j][i] = value
    return CorrelationMatrix(labels=labels, values=values, mode=mode)


# --------------------------------------------------------------------------- intrinsic
@dataclass
class HeadOverlap:
    """How much two models agree on *which* heads are retrieval heads."""

    model_a: str
    model_b: str
    threshold: float
    n_a: int
    n_b: int
    n_shared: int
    jaccard: float
    shared: list[str] = field(default_factory=list)
    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)
    score_correlation: float = float("nan")
    #: False when the two runs do not share a layer x head layout: ``jaccard`` is
    #: then NaN and the shared/only lists are empty, because comparing "L12H3"
    #: across layouts is meaningless.
    comparable: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a, "model_b": self.model_b, "threshold": self.threshold,
            "n_a": self.n_a, "n_b": self.n_b, "n_shared": self.n_shared,
            "jaccard": self.jaccard, "score_correlation": self.score_correlation,
            "comparable": self.comparable,
            "shared": self.shared, "only_a": self.only_a, "only_b": self.only_b,
        }


def head_overlap(
    scores_a: RetrievalScores,
    scores_b: RetrievalScores,
    *,
    threshold: float = 0.1,
    mode: str = "grid",
) -> HeadOverlap:
    """Set overlap of the retrieval-head sets, plus the score correlation.

    This is the sharpest form of the paper's *intrinsic* claim: a base model and
    its long-context / chat derivative should select the same heads.

    The head *names* (``L12H3``) are only comparable when the two runs share a
    layer x head layout.  When they do not -- exactly the cross-family case
    ``mode="sorted"`` exists for -- ``comparable`` is False, ``jaccard`` is NaN and
    the shared lists are empty, rather than reporting an overlap between unrelated
    indices.
    """
    comparable = layouts_match(scores_a, scores_b)
    if not comparable:
        log.warning(
            "head-overlap between %s and %s compares head names across different "
            "layer/head layouts; reporting comparable=False and no jaccard (the "
            "correlation is still reported, in mode=%r)",
            scores_a.info.name, scores_b.info.name, mode,
        )
    set_a = {str(h) for h in scores_a.heads_above(threshold)}
    set_b = {str(h) for h in scores_b.heads_above(threshold)}
    if not comparable:
        return HeadOverlap(
            model_a=scores_a.info.name, model_b=scores_b.info.name, threshold=threshold,
            n_a=len(set_a), n_b=len(set_b), n_shared=0, jaccard=float("nan"),
            comparable=False,
            score_correlation=correlate(scores_a, scores_b, mode=mode),
        )
    shared = sorted(set_a & set_b)
    union = set_a | set_b
    return HeadOverlap(
        model_a=scores_a.info.name, model_b=scores_b.info.name, threshold=threshold,
        n_a=len(set_a), n_b=len(set_b), n_shared=len(shared),
        jaccard=(len(shared) / len(union)) if union else float("nan"),
        shared=shared, only_a=sorted(set_a - set_b), only_b=sorted(set_b - set_a),
        score_correlation=correlate(scores_a, scores_b, mode=mode),
    )


# --------------------------------------------------------------------------- layer profile
def layer_profile(scores: RetrievalScores) -> dict[str, Any]:
    """Retrieval mass per layer -- useful for the hybrid 'where does it live' question."""
    rows = []
    for layer in scores.info.scoreable_layers:
        vals = scores.score[layer, : scores.info.num_heads[layer]].numpy()
        act = scores.activation_freq[layer, : scores.info.num_heads[layer]].numpy()
        rows.append({
            "layer": layer,
            "layer_type": scores.info.layer_type(layer),
            "mean_score": float(np.nanmean(vals)),
            "max_score": float(np.nanmax(vals)),
            "n_above_threshold": int(np.sum(vals > scores.threshold)),
            "mean_activation_freq": float(np.nanmean(act)),
        })
    return {"model": scores.info.name, "threshold": scores.threshold, "layers": rows}
