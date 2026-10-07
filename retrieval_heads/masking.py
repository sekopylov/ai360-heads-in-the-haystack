"""Causal experiments: does masking retrieval heads actually break retrieval?

Paper Fig. 1 / Fig. 4: mask the top-K retrieval heads and Needle-in-a-Haystack
performance collapses; mask K random non-retrieval heads and it barely moves.

For a hybrid model there is a second, equally interesting question the paper
raises in Sec. 5: *is full attention necessary?*  We can therefore also silence
whole token-mixer layers -- Gated Attention (softmax) versus Gated DeltaNet
(linear) -- and compare how much each kind matters for retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from retrieval_heads.attention import HeadMasker, TokenMixerMasker, set_attn_implementation
from retrieval_heads.haystack import NeedleSample, build_needle_sample
from retrieval_heads.models import ModelInfo, model_device
from retrieval_heads.scoring import RetrievalScores, needle_recall
from retrieval_heads.utils import HeadRef, get_logger, save_json

log = get_logger("masking")


# --------------------------------------------------------------------------- metrics
def token_f1(pred_ids: Sequence[int], target_ids: Sequence[int]) -> float:
    """SQuAD-style token-level F1 between two id sequences."""
    from collections import Counter

    if not pred_ids or not target_ids:
        return 0.0
    pred, target = Counter(pred_ids), Counter(target_ids)
    overlap = sum((pred & target).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_ids)
    recall = overlap / len(target_ids)
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def greedy_generate(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    eos_ids: Iterable[int] = (),
    attn_impl: str = "sdpa",
) -> list[int]:
    """Plain greedy decoding (no attention capture) -- used for the ablations."""
    input_ids = input_ids.to(model_device(model))
    restore = set_attn_implementation(model, attn_impl)
    eos = set(int(e) for e in eos_ids)
    out_ids: list[int] = []
    try:
        out = model(input_ids=input_ids, use_cache=True)
        cache = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        for _ in range(max_new_tokens):
            token = int(nxt[0, 0])
            if token in eos:
                break
            out_ids.append(token)
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    finally:
        if restore is not None:
            set_attn_implementation(model, restore)
    return out_ids


@dataclass
class NiahMetrics:
    f1: float
    exact_match: float
    recall: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return {"f1": self.f1, "exact_match": self.exact_match, "recall": self.recall, "n": self.n}


def normalized_contains(text: str, needle: str) -> bool:
    """Case- and punctuation-insensitive substring test, the classic NIAH metric.

    Models routinely re-case the needle ("...provided, the best thing...") or wrap
    it in markdown emphasis, so a raw ``in`` check reports 0% even when the answer
    is verbatim.
    """
    def norm(value: str) -> str:
        value = value.replace("*", " ").lower()
        value = re.sub(r"[^\w\s]", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    return norm(needle) in norm(text)


@torch.no_grad()
def evaluate_samples(
    model: Any,
    tokenizer: Any,
    info: ModelInfo,
    samples: Sequence[NeedleSample],
    *,
    masked_heads: Sequence[HeadRef] = (),
    masked_layers: Sequence[int] = (),
    max_new_tokens: int = 32,
    attn_impl: str = "sdpa",
) -> NiahMetrics:
    """Needle-in-a-Haystack score of ``samples``, optionally with heads silenced."""
    eos: set[int] = set()
    cfg = getattr(model, "config", None)
    value = getattr(cfg, "eos_token_id", None)
    if isinstance(value, int):
        eos.add(value)
    elif isinstance(value, (list, tuple, set)):
        eos.update(int(v) for v in value)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos.add(int(tokenizer.eos_token_id))

    masker = HeadMasker(model, info, masked_heads) if masked_heads else None
    mixer = TokenMixerMasker(model, info, masked_layers) if masked_layers else None
    f1s, ems, recalls = [], [], []
    try:
        for sample in samples:
            ids = greedy_generate(model, sample.input_ids, max_new_tokens=max_new_tokens,
                                  eos_ids=eos, attn_impl=attn_impl)
            text = tokenizer.decode(ids, skip_special_tokens=True)
            f1s.append(token_f1(ids, sample.needle_ids))
            ems.append(1.0 if normalized_contains(text, sample.needle_text) else 0.0)
            recalls.append(needle_recall(text, sample.needle_text))
    finally:
        if masker is not None:
            masker.remove()
        if mixer is not None:
            mixer.remove()
    return NiahMetrics(
        f1=100.0 * float(np.mean(f1s)) if f1s else 0.0,
        exact_match=100.0 * float(np.mean(ems)) if ems else 0.0,
        recall=100.0 * float(np.mean(recalls)) if recalls else 0.0,
        n=len(samples),
    )


# --------------------------------------------------------------------------- sample sets
def make_eval_samples(
    tokenizer: Any,
    *,
    lengths: Sequence[int] = (1024, 2048),
    depths: Sequence[float] = (0.25, 0.75),
    needle: str,
    question: str,
    seed: int = 7,
    chat_template: bool = True,
    enable_thinking: bool | None = False,
    corpus: Sequence[str] | None = None,
) -> list[NeedleSample]:
    """A small held-out NIAH set (disjoint seeds from the detection grid)."""
    from retrieval_heads.haystack import HaystackBuilder

    builder = HaystackBuilder(corpus, seed=seed)
    out = []
    for i, length in enumerate(lengths):
        for j, depth in enumerate(depths):
            out.append(build_needle_sample(
                tokenizer, needle=needle, question=question, target_tokens=length,
                depth=depth, builder=builder, chat_template=chat_template,
                enable_thinking=enable_thinking, seed=seed + 31 * i + j,
            ))
    return out


# --------------------------------------------------------------------------- curves
@dataclass
class MaskingCurve:
    """NIAH performance as a function of how many heads are silenced."""

    k_values: list[int]
    retrieval: list[float]
    random_mean: list[float]
    random_std: list[float]
    random_trials: list[list[float]] = field(default_factory=list)
    metric: str = "f1"
    baseline: float = 0.0
    score_threshold: float = 0.1
    meta: dict[str, Any] = field(default_factory=dict)
    retrieval_exact: list[float] = field(default_factory=list)
    random_exact_mean: list[float] = field(default_factory=list)
    baseline_exact: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "k_values": self.k_values,
            "retrieval": self.retrieval,
            "random_mean": self.random_mean,
            "random_std": self.random_std,
            "random_trials": self.random_trials,
            "retrieval_exact_match": self.retrieval_exact,
            "random_exact_match_mean": self.random_exact_mean,
            "baseline_exact_match": self.baseline_exact,
            "metric": self.metric,
            "baseline": self.baseline,
            "score_threshold": self.score_threshold,
            **self.meta,
        }

    def save(self, path: str | Any) -> None:
        save_json(self.as_dict(), path)


def masking_curve(
    model: Any,
    tokenizer: Any,
    info: ModelInfo,
    scores: RetrievalScores,
    samples: Sequence[NeedleSample],
    *,
    k_values: Sequence[int] = (1, 2, 4, 8, 16, 32),
    n_random_trials: int = 3,
    max_new_tokens: int = 32,
    seed: int = 0,
    progress: bool = True,
) -> MaskingCurve:
    """Mask top-K retrieval heads vs K random heads and score NIAH each time."""
    retrieval_ranked = scores.heads_above()
    if not retrieval_ranked:
        log.warning("no head exceeds the %.2f threshold; curve will be flat",
                    scores.threshold)

    baseline = evaluate_samples(model, tokenizer, info, samples, max_new_tokens=max_new_tokens)
    log.info("baseline NIAH f1=%.1f exact=%.1f", baseline.f1, baseline.exact_match)

    rng = np.random.default_rng(seed)
    curve = MaskingCurve(
        k_values=list(k_values), retrieval=[], random_mean=[], random_std=[],
        baseline=baseline.f1, baseline_exact=baseline.exact_match,
        score_threshold=scores.threshold,
        meta={"model": info.name, "baseline_exact_match": baseline.exact_match,
              "n_samples": len(samples), "pairing": scores.pairing,
              "n_scoreable_heads": info.n_scoreable_heads,
              "k_fraction": [k / info.n_scoreable_heads for k in k_values]},
    )

    for k in k_values:
        top = retrieval_ranked[:k]
        metrics = evaluate_samples(model, tokenizer, info, samples,
                                   masked_heads=top, max_new_tokens=max_new_tokens)
        curve.retrieval.append(metrics.f1)
        curve.retrieval_exact.append(metrics.exact_match)
        log.info("k=%-4d retrieval-masked (n=%d, %.1f%% of heads) f1=%.1f exact=%.1f",
                 k, len(top), 100 * len(top) / max(info.n_scoreable_heads, 1),
                 metrics.f1, metrics.exact_match)

        trials, exact_trials = [], []
        pool = info.scoreable_heads
        for trial in range(n_random_trials):
            pick = rng.permutation(len(pool))[:k]
            random_heads = [pool[i] for i in pick]
            m = evaluate_samples(model, tokenizer, info, samples,
                                 masked_heads=random_heads, max_new_tokens=max_new_tokens)
            trials.append(m.f1)
            exact_trials.append(m.exact_match)
        curve.random_trials.append(trials)
        curve.random_mean.append(float(np.mean(trials)))
        curve.random_std.append(float(np.std(trials)))
        curve.random_exact_mean.append(float(np.mean(exact_trials)))
        log.info("k=%-4d random-masked f1=%.1f +/- %.1f exact=%.1f",
                 k, np.mean(trials), np.std(trials), np.mean(exact_trials))

    return curve


@dataclass
class MixerAblation:
    """How much each *kind* of token mixer matters for retrieval.

    Caveat, and it matters for a hybrid model: the two stacks have very
    different sizes (Qwen3.5-0.8B has 6 full-attention and 18 linear layers), so
    masking K of each removes very different fractions of the network.  A single
    linear layer also carries far more of the residual stream than a single
    attention layer.  The useful reading is therefore *"how fast does NIAH decay
    as you remove layers of this kind"*, not "linear beats full attention" --
    ``fractions`` is reported so the comparison can be normalised.
    """

    full_attention: list[float]
    linear_attention: list[float]
    baseline: float
    k_values: list[int] = field(default_factory=list)
    n_full_layers: int = 0
    n_linear_layers: int = 0
    full_attention_exact: list[float] = field(default_factory=list)
    linear_attention_exact: list[float] = field(default_factory=list)

    @property
    def fractions(self) -> list[float]:
        return [k / self.n_full_layers for k in self.k_values] if self.n_full_layers else []

    def as_dict(self) -> dict[str, Any]:
        return {
            "k_values": self.k_values,
            "fraction_of_full_attention_layers": self.fractions,
            "full_attention": self.full_attention,
            "linear_attention": self.linear_attention,
            "full_attention_exact_match": self.full_attention_exact,
            "linear_attention_exact_match": self.linear_attention_exact,
            "baseline": self.baseline,
            "n_full_layers": self.n_full_layers,
            "n_linear_layers": self.n_linear_layers,
        }


def token_mixer_ablation(
    model: Any,
    tokenizer: Any,
    info: ModelInfo,
    samples: Sequence[NeedleSample],
    *,
    k_values: Sequence[int] = (1, 2, 3),
    max_new_tokens: int = 32,
) -> MixerAblation:
    """Silence K full-attention layers vs K linear-attention layers.

    On a dense model ``linear_attention`` is empty and the call returns
    ``full_attention`` only -- still a valid "how concentrated is retrieval in a
    few layers" measurement.
    """
    baseline = evaluate_samples(model, tokenizer, info, samples, max_new_tokens=max_new_tokens)
    full_layers = list(info.scoreable_layers)
    linear_layers = list(info.linear_layers)

    full_scores: list[float] = []
    linear_scores: list[float] = []
    full_exact: list[float] = []
    linear_exact: list[float] = []
    for k in k_values:
        picked_full = full_layers[:k] if len(full_layers) >= k else full_layers
        m = evaluate_samples(model, tokenizer, info, samples, masked_layers=picked_full,
                             max_new_tokens=max_new_tokens)
        full_scores.append(m.f1)
        full_exact.append(m.exact_match)
        log.info("k=%d full-attention layers %s masked -> f1=%.1f exact=%.1f",
                 k, picked_full, m.f1, m.exact_match)

        if len(linear_layers) >= k:
            picked_linear = linear_layers[:k]
            m = evaluate_samples(model, tokenizer, info, samples, masked_layers=picked_linear,
                                 max_new_tokens=max_new_tokens)
            linear_scores.append(m.f1)
            linear_exact.append(m.exact_match)
            log.info("k=%d linear layers %s masked -> f1=%.1f exact=%.1f",
                     k, picked_linear, m.f1, m.exact_match)

    return MixerAblation(
        full_attention=full_scores, linear_attention=linear_scores,
        baseline=baseline.f1, k_values=list(k_values),
        n_full_layers=len(full_layers), n_linear_layers=len(linear_layers),
        full_attention_exact=full_exact, linear_attention_exact=linear_exact,
    )
