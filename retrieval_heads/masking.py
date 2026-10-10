"""Causal experiments: does masking retrieval heads actually break retrieval?

Paper Fig. 1 / Fig. 4: mask the top-K retrieval heads and Needle-in-a-Haystack
performance collapses; mask K random non-retrieval heads and it barely moves.

For a hybrid model there is a second, equally interesting question the paper
raises in Sec. 5: *is full attention necessary?*  We can therefore also silence
whole token-mixer layers -- Gated Attention (softmax) versus Gated DeltaNet
(linear) -- and compare how much each kind matters for retrieval.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from retrieval_heads.attention import HeadMasker, TokenMixerMasker
from retrieval_heads.generation import greedy_ids
from retrieval_heads.haystack import NeedleSample, build_needle_sample
from retrieval_heads.models import ModelInfo
from retrieval_heads.scoring import (
    RetrievalScores,
    needle_prefix_recall,
    needle_recall,
)
from retrieval_heads.utils import HeadRef, eos_ids, get_logger, save_json, squad_f1

log = get_logger("masking")


# --------------------------------------------------------------------------- controls
def control_pool(scores: RetrievalScores) -> tuple[list[HeadRef], bool]:
    """The random-arm pool and whether it had to be contaminated.

    Returns ``(pool, contaminated)``.  ``contaminated`` is True when *every*
    scoreable head is above the threshold, so no clean non-retrieval pool exists
    and the caller has to fall back to all heads.  Callers must surface that flag
    (log + artifact), otherwise a compromised control reads as a clean one.
    """
    flagged = set(scores.heads_above())
    pool = [h for h in scores.info.scoreable_heads if h not in flagged]
    if pool:
        return pool, False
    return list(scores.info.scoreable_heads), True


def matched_k(k: int, pool_size: int) -> int:
    """Heads each arm can actually mask at this K.

    Both arms must remove the same number of heads or the curve compares
    different interventions.  K is capped by the size of the random pool; the
    retrieval arm is capped by the same value even though it could go higher.
    """
    if pool_size <= 0:
        return max(0, k)
    return min(k, pool_size)


# --------------------------------------------------------------------------- metrics
def token_f1(pred_ids: Sequence[int], target_ids: Sequence[int]) -> float:
    """SQuAD-style token-level F1 between two id sequences."""
    return squad_f1(pred_ids, target_ids)


@torch.no_grad()
def greedy_generate(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    eos: Iterable[int] | None = None,
    attn_impl: str = "sdpa",
    prefill_chunk: int | None = None,
    tokenizer: Any = None,
) -> list[int]:
    """Plain greedy decoding (no attention capture) -- used for the ablations.

    Thin wrapper over :func:`retrieval_heads.generation.greedy_ids`, which owns the
    prefill chunking and the token loop for every non-capturing caller.
    """
    return greedy_ids(
        model, input_ids, max_new_tokens=max_new_tokens,
        # None means "ask the model"; an empty tuple would silently disable EOS.
        eos=eos_ids(model, tokenizer) if eos is None else set(eos),
        attn_impl=attn_impl, prefill_chunk=prefill_chunk,
    )


@dataclass
class NiahMetrics:
    #: NOTE: `exact_match` is a *normalised-contains* check (the needle text appears
    #: in the decoded answer), the standard NIAH metric -- not character-exact
    #: equality.  The key is kept for artifact compatibility.
    f1: float
    exact_match: float
    recall: float
    n: int
    #: Per-sample values.  Without them only the mean survived, so the spread of
    #: the causal experiment could not be recovered from an artifact.
    f1s: list[float] = field(default_factory=list)
    exact_matches: list[float] = field(default_factory=list)
    recalls: list[float] = field(default_factory=list)
    #: The old prefix-anchored recall (needle reproduced from its first word).
    prefix_recalls: list[float] = field(default_factory=list)
    #: The actual completions, so a failure can be inspected from the artifact
    #: instead of only from its mean (the paper's Sec. 5 argument is about cases).
    generated_texts: list[str] = field(default_factory=list)
    #: How many samples hit `max_new_tokens` without an EOS.  `detect` recorded this
    #: from the start, the ablations did not -- so a drop could not be checked against
    #: a budget that ran out, which matters on the hybrid (its `detect` run truncated
    #: 11 of 75 instances at a 48-token budget).
    n_truncated: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"f1": self.f1, "exact_match": self.exact_match, "recall": self.recall,
                "n": self.n, "f1s": self.f1s, "exact_matches": self.exact_matches,
                "recalls": self.recalls,
                "prefix_recalls": self.prefix_recalls,
                "generated_texts": self.generated_texts,
                "n_truncated": self.n_truncated}


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
    prefill_chunk: int | None = None,
) -> NiahMetrics:
    """Needle-in-a-Haystack score of ``samples``, optionally with heads silenced."""
    eos = eos_ids(model, tokenizer)

    masker = HeadMasker(model, info, masked_heads) if masked_heads else None
    try:
        mixer = TokenMixerMasker(model, info, masked_layers) if masked_layers else None
    except Exception:
        # A failure building the second masker must not leave the first one live.
        if masker is not None:
            masker.remove()
        raise
    f1s, ems, recalls, prefix_recalls = [], [], [], []
    generated_texts: list[str] = []
    n_truncated = 0
    try:
        special = set(getattr(tokenizer, "all_special_ids", None) or [])
        for sample in samples:
            ids = greedy_generate(model, sample.input_ids, max_new_tokens=max_new_tokens,
                                  eos=eos, attn_impl=attn_impl,
                                  prefill_chunk=prefill_chunk)
            # `greedy_ids` never appends the stop token, so a full-length output is
            # exactly "the loop ran out of budget": it can only reach max_new_tokens
            # by never seeing an EOS at the top of an iteration.  (The old condition
            # also tested `ids[-1] not in eos`, which is dead for that reason.)
            if len(ids) >= max_new_tokens:
                n_truncated += 1
            text = tokenizer.decode(ids, skip_special_tokens=True)
            # Exact match is computed on skip_special_tokens text, so F1 must skip
            # them too or the two metrics in one artifact describe different things.
            scored_ids = [token for token in ids if token not in special]
            # Gold = tokenization of the needle text, matching what the model emits
            # (the prompt span's last token may be `".\n"`, which caps F1 at 21/22).
            f1s.append(token_f1(scored_ids, sample.needle_text_ids))
            ems.append(1.0 if normalized_contains(text, sample.needle_text) else 0.0)
            recalls.append(needle_recall(text, sample.needle_text))
            prefix_recalls.append(needle_prefix_recall(text, sample.needle_text))
            generated_texts.append(text)
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
        f1s=[100.0 * value for value in f1s],
        exact_matches=[100.0 * value for value in ems],
        recalls=[100.0 * value for value in recalls],
        prefix_recalls=[100.0 * value for value in prefix_recalls],
        generated_texts=generated_texts,
        n_truncated=n_truncated,
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
    system_prompt: str | None = None,
) -> list[NeedleSample]:
    """A small held-out NIAH set (disjoint seeds from the detection grid).

    Each sample gets its own seeded builder so the recorded seed is the one that
    produced its filler.
    """
    from retrieval_heads.haystack import HaystackBuilder

    out = []
    for i, length in enumerate(lengths):
        for j, depth in enumerate(depths):
            sample_seed = seed + 31 * i + j
            builder = HaystackBuilder(corpus, seed=sample_seed)
            out.append(build_needle_sample(
                tokenizer, needle=needle, question=question, target_tokens=length,
                depth=depth, builder=builder, chat_template=chat_template,
                enable_thinking=enable_thinking, system_prompt=system_prompt,
                seed=sample_seed,
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
    #: Per-trial **f1** of the random arm -- the same metric as `random_mean`, which is
    #: its mean.  The per-trial *recall* and *exact-match* series are not stored as
    #: lists; they live per sample in `per_sample[k]["random"][trial]` (and their means
    #: in `random_recall_mean` / `random_exact_match_mean`).  Reading this field as
    #: "the per-trial recall" is an easy mistake: it is the primary series, and
    #: `metric` names it.
    random_trials: list[list[float]] = field(default_factory=list)
    #: Names the metric the *primary* series (`retrieval`/`random_mean`) is computed
    #: with -- always f1, and the artifact carries `retrieval_recall`/`random_recall`
    #: alongside.  A figure drawn with `metric="recall"` reads those series instead,
    #: so this field must not be read as "what the figure shows".
    metric: str = "f1"
    baseline: float = 0.0
    score_threshold: float = 0.1
    meta: dict[str, Any] = field(default_factory=dict)
    retrieval_exact: list[float] = field(default_factory=list)
    random_exact_mean: list[float] = field(default_factory=list)
    baseline_exact: float = 0.0
    #: LCS needle recall.  F1/EM compare against the *whole* needle while the
    #: question asks for a sub-span, so a correct short answer is penalised; recall
    #: is the metric that does not have that confound.
    retrieval_recall: list[float] = field(default_factory=list)
    random_recall_mean: list[float] = field(default_factory=list)
    baseline_recall: float = 0.0
    #: Samples that hit `max_new_tokens` without an EOS, per K: without it a drop
    #: cannot be told apart from a budget that ran out (the hybrid's detect run
    #: truncated 11 of 75 instances at a 48-token budget).
    #: True at the points where the retrieval arm was capped by the *control pool*
    #: (``k_eff`` < requested K because fewer heads sit below the threshold) and the
    #: random arm therefore drew the whole pool: the point is not a matched
    #: comparison, and the reader of the JSON should not plot it as one.
    control_exhausted: list[bool] = field(default_factory=list)
    #: Distinct control subsets actually evaluated per K (the trials are drawn
    #: without repetition where the pool allows it, so a repeat means the space ran
    #: out -- at which point `random_std` spans fewer interventions than n_trials).
    random_distinct: list[int] = field(default_factory=list)
    retrieval_truncated: list[int] = field(default_factory=list)
    random_truncated_mean: list[float] = field(default_factory=list)
    baseline_truncated: int = 0
    #: Realized heads masked per point, for both arms.  They are equal by
    #: construction; recording them makes a mismatch visible instead of implicit.
    k_effective: list[int] = field(default_factory=list)
    retrieval_masked: list[int] = field(default_factory=list)
    random_masked_mean: list[float] = field(default_factory=list)
    #: Std of the retrieval arm across the evaluation samples (the random arm's
    #: std is across trials).  Reported so the noise is visible next to the effect.
    retrieval_std: list[float] = field(default_factory=list)
    retrieval_exact_std: list[float] = field(default_factory=list)
    #: Per-K, per-trial size of ``set(random arm) & set(retrieval arm)``.  This is
    #: NOT "the control contains retrieval heads" -- `control_pool` excludes every
    #: head above the threshold by construction, so it is zero unless the retrieval
    #: arm itself reached below the threshold (large K).  `random_above_threshold`
    #: is the honest count of control heads that are retrieval heads.
    random_retrieval_overlap: list[list[int]] = field(default_factory=list)
    #: Per-K, per-trial count of control heads above the score threshold (0 when the
    #: pool is clean; non-zero only in the contaminated fallback).
    random_above_threshold: list[list[int]] = field(default_factory=list)
    #: Per-K, per-arm per-sample metrics and completions (see `NiahMetrics.as_dict`).
    per_sample: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        # `meta` first: a meta key colliding with a field name must not win.
        return {
            **self.meta,
            "k_values": self.k_values,
            "k_effective": self.k_effective,
            "retrieval_masked": self.retrieval_masked,
            "random_masked_mean": self.random_masked_mean,
            "random_retrieval_overlap": self.random_retrieval_overlap,
            "random_above_threshold": self.random_above_threshold,
            "per_sample": self.per_sample,
            "retrieval_std": self.retrieval_std,
            "retrieval_exact_std": self.retrieval_exact_std,
            "retrieval": self.retrieval,
            "random_mean": self.random_mean,
            "random_std": self.random_std,
            "random_trials": self.random_trials,
            "retrieval_recall": self.retrieval_recall,
            "random_recall_mean": self.random_recall_mean,
            "baseline_recall": self.baseline_recall,
            "control_exhausted": self.control_exhausted,
            "random_distinct": self.random_distinct,
            "retrieval_truncated": self.retrieval_truncated,
            "random_truncated_mean": self.random_truncated_mean,
            "baseline_truncated": self.baseline_truncated,
            "retrieval_exact_match": self.retrieval_exact,
            "random_exact_match_mean": self.random_exact_mean,
            "baseline_exact_match": self.baseline_exact,
            "metric": self.metric,
            "baseline": self.baseline,
            "score_threshold": self.score_threshold,
        }

    def save(self, path: str | Any) -> None:
        from retrieval_heads.provenance import add_provenance

        save_json(add_provenance(self.as_dict(), dtype=None), path)


def draw_control_subsets(pool: Sequence[HeadRef], k_eff: int, n_trials: int,
                         rng: np.random.Generator) -> list[list[HeadRef]]:
    """`n_trials` control subsets, without repeating one while the space allows it.

    Repeats matter at a small pool: the hybrid has 15 non-retrieval heads, so at K=1
    a three-trial draw can pick the same head twice and `random_std` then measures
    two distinct interventions, not three.  Callers record the realized count
    (`random_distinct` / `distinct_subsets`) so the artifact says which it was.
    """
    if not pool or k_eff <= 0:
        return [[] for _ in range(n_trials)]
    if k_eff >= len(pool):
        # The whole pool: a deterministic intervention, and repeats are unavoidable.
        return [list(pool) for _ in range(n_trials)]
    space = math.comb(len(pool), k_eff)
    seen: set[tuple[str, ...]] = set()
    out: list[list[HeadRef]] = []
    attempts = 0
    while len(out) < n_trials and attempts < 100 * n_trials:
        attempts += 1
        pick = [pool[i] for i in rng.permutation(len(pool))[:k_eff]]
        key = tuple(sorted(str(head) for head in pick))
        if key in seen and len(seen) < space:
            continue
        seen.add(key)
        out.append(pick)
    while len(out) < n_trials:  # the space was smaller than the trial count
        out.append([pool[i] for i in rng.permutation(len(pool))[:k_eff]])
    return out


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
    prefill_chunk: int | None = 4096,
) -> MaskingCurve:
    """Mask top-K retrieval heads vs K random heads and score NIAH each time.

    The retrieval arm is the top K heads *by score* (not "every head above the
    threshold, capped at K"), so K is honoured and both arms remove exactly the
    same number of heads.  The random arm is drawn from
    :func:`control_pool`, i.e. heads at or below the threshold, and every draw's
    overlap with the retrieval heads is recorded.

    ``k_values`` lists only the points actually evaluated, so every parallel list
    (``k_effective``, ``retrieval``, ``random_mean``, ...) has the same length by
    construction.
    """
    retrieval_ranked = scores.ranked_heads()
    # Audit trail: which heads/trials were actually masked, so the causal experiment
    # is reproducible from its own artifact.
    masked_retrieval_heads: dict[str, list[str]] = {}
    random_picks: dict[str, list[list[str]]] = {}
    per_sample: dict[str, dict[str, Any]] = {}
    if not scores.heads_above():
        log.warning("no head exceeds the %.2f threshold; the retrieval arm is still "
                    "ranked by score, but 'retrieval head' has no member here",
                    scores.threshold)

    baseline = evaluate_samples(model, tokenizer, info, samples, max_new_tokens=max_new_tokens,
                                 prefill_chunk=prefill_chunk)
    log.info("baseline NIAH f1=%.1f exact=%.1f", baseline.f1, baseline.exact_match)

    pool, contaminated = control_pool(scores)
    if contaminated:
        log.warning("every scoreable head is above the %.2f threshold; the random arm "
                    "is drawn from all scoreable heads and is contaminated",
                    scores.threshold)
    if not pool:
        raise ValueError("no scoreable heads to mask")

    rng = np.random.default_rng(seed)
    curve = MaskingCurve(
        k_values=[], retrieval=[], random_mean=[], random_std=[],
        baseline=baseline.f1, baseline_exact=baseline.exact_match,
        baseline_recall=baseline.recall, baseline_truncated=baseline.n_truncated,
        score_threshold=scores.threshold,
        meta={"model": info.name, "baseline_exact_match": baseline.exact_match,
              "n_samples": len(samples), "pairing": scores.pairing,
              "n_scoreable_heads": info.n_scoreable_heads,
              "n_non_retrieval_heads": len(pool),
              "random_control_contaminated": contaminated,
              "max_new_tokens": max_new_tokens,
              "prefill_chunk": prefill_chunk,
              "argmax_domain": (scores.meta or {}).get("argmax_domain"),
              "baseline_recall": baseline.recall},
    )

    iterator = k_values
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(k_values, desc=f"mask[{info.name}]", unit="K")
        except ImportError:  # pragma: no cover
            pass

    for k in iterator:
        k_eff = matched_k(k, len(pool))
        if k_eff <= 0:
            log.warning("k=%d cannot be matched by a random control; skipping", k)
            continue
        if k_eff < k:
            log.warning("k=%d exceeds the %d non-retrieval heads available; both arms "
                        "use k=%d", k, len(pool), k_eff)
        top = retrieval_ranked[:k_eff]
        masked_retrieval_heads[str(k)] = [str(head) for head in top]
        if k_eff > len(scores.heads_above()):
            # Past the threshold the "retrieval" arm necessarily includes heads with
            # zero score, so the label weakens; `random_retrieval_overlap` audits it.
            log.debug("k=%d: retrieval arm reaches below the threshold (%d above)",
                      k, len(scores.heads_above()))
        metrics = evaluate_samples(model, tokenizer, info, samples, masked_heads=top,
                                   max_new_tokens=max_new_tokens, prefill_chunk=prefill_chunk)
        curve.k_values.append(k)
        curve.k_effective.append(k_eff)
        curve.retrieval_masked.append(len(top))
        curve.retrieval.append(metrics.f1)
        curve.retrieval_recall.append(metrics.recall)
        curve.retrieval_std.append(float(np.std(metrics.f1s)) if metrics.f1s else 0.0)
        curve.retrieval_truncated.append(metrics.n_truncated)
        curve.retrieval_exact.append(metrics.exact_match)
        curve.retrieval_exact_std.append(
            float(np.std(metrics.exact_matches)) if metrics.exact_matches else 0.0
        )
        log.info("k=%-4d retrieval-masked (n=%d, %.1f%% of heads) f1=%.1f exact=%.1f",
                 k, len(top), 100 * len(top) / max(info.n_scoreable_heads, 1),
                 metrics.f1, metrics.exact_match)

        trials, exact_trials, recall_trials, overlaps = [], [], [], []
        truncated_trials: list[int] = []
        trial_metrics = []
        masked_counts = []
        above_counts: list[int] = []
        threshold_heads = set(scores.heads_above())
        drawn = draw_control_subsets(pool, k_eff, n_random_trials, rng)
        for random_heads in drawn:
            masked_counts.append(len(random_heads))
            # The honest control audit: how many drawn heads are retrieval heads.
            # (In the clean pool this is 0 by construction; it is non-zero only in
            # the contaminated fallback, when every head is above the threshold.)
            above_counts.append(sum(1 for h in random_heads if h in threshold_heads))
            # vs the heads actually masked by the retrieval arm at this K: at
            # large K that arm reaches below the threshold and can genuinely share
            # heads with a pool drawn from below it.
            random_picks.setdefault(str(k), []).append([str(head) for head in random_heads])
            overlaps.append(sum(1 for h in random_heads if h in set(top)))
            m = evaluate_samples(model, tokenizer, info, samples, masked_heads=random_heads,
                                 max_new_tokens=max_new_tokens, prefill_chunk=prefill_chunk)
            truncated_trials.append(m.n_truncated)
            trials.append(m.f1)
            recall_trials.append(m.recall)
            exact_trials.append(m.exact_match)
            trial_metrics.append(m)
        # Per-sample detail (values *and* completions) for the retrieval arm and one
        # representative random trial: without it a failure could only be inspected by
        # re-running the experiment.
        per_sample[str(k)] = {"retrieval": metrics.as_dict(),
                              "random": [m.as_dict() for m in trial_metrics]}
        curve.random_trials.append(trials)
        curve.random_retrieval_overlap.append(overlaps)
        curve.random_above_threshold.append(above_counts)
        curve.random_masked_mean.append(float(np.mean(masked_counts)))
        curve.control_exhausted.append(bool(len(pool) and k_eff >= len(pool)))
        curve.random_distinct.append(
            len({tuple(sorted(str(h) for h in pick)) for pick in drawn})
        )
        curve.random_recall_mean.append(float(np.mean(recall_trials)))
        curve.random_mean.append(float(np.mean(trials)))
        curve.random_std.append(float(np.std(trials)))
        curve.random_exact_mean.append(float(np.mean(exact_trials)))
        curve.random_truncated_mean.append(float(np.mean(truncated_trials)))
        log.info("k=%-4d random-masked f1=%.1f +/- %.1f exact=%.1f (retrieval-head "
                 "overlap %s)", k, np.mean(trials), np.std(trials), np.mean(exact_trials),
                 overlaps)
    curve.per_sample = per_sample

    n_heads = max(info.n_scoreable_heads, 1)
    # Three levels, and they are not the same thing:
    #   * the CLI request (`--k-frac` / `--k`, recorded by `cmd_mask` as `k_frac_args`
    #     / `k_args`) -- two fractions can round to the same K on a small model;
    #   * `k_fraction_requested`: the K values that reached this function, as
    #     fractions of the scoreable heads;
    #   * `k_fraction_effective`: what `matched_k` actually masked (a requested K can
    #     exceed the control pool).
    # The old bare `k_fraction` invited reading the third as the first.
    curve.meta["k_fraction_requested"] = [k / n_heads for k in curve.k_values]
    curve.meta["k_fraction_effective"] = [k / n_heads for k in curve.k_effective]
    curve.meta["seed"] = seed
    curve.meta["masked_retrieval_heads"] = masked_retrieval_heads
    curve.meta["random_picks"] = random_picks
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
    #: Realized layer counts.  ``full_attention[i]`` and ``linear_attention[i]``
    #: always refer to the same requested ``k_values[i]``; when a stack is
    #: smaller than K the slice is truncated and the count says so.
    full_attention_masked: list[int] = field(default_factory=list)
    linear_attention_masked: list[int] = field(default_factory=list)
    #: Spread across the random subsets drawn per K (the selection used to be the
    #: deterministic first-K of each stack, which confounded "type of layer" with
    #: "how early in the network it sits").
    full_attention_std: list[float] = field(default_factory=list)
    linear_attention_std: list[float] = field(default_factory=list)
    full_attention_trials: list[list[float]] = field(default_factory=list)
    linear_attention_trials: list[list[float]] = field(default_factory=list)
    #: The layer subsets actually masked, per K (one entry per trial), plus the seed.
    full_attention_layers: list[list[list[int]]] = field(default_factory=list)
    linear_attention_layers: list[list[list[int]]] = field(default_factory=list)
    seed: int = 0

    @property
    def fractions(self) -> list[float]:
        """Requested K as a fraction of the full-attention stack, capped at 1.0."""
        if not self.n_full_layers:
            return []
        return [min(k, self.n_full_layers) / self.n_full_layers for k in self.k_values]

    @property
    def fractions_linear(self) -> list[float]:
        """The same K as a fraction of the *linear* stack.

        Without it K=6 attention (100% of 6) and K=6 linear (33% of 18) look
        comparable when they are not.
        """
        if not self.n_linear_layers:
            return []
        return [min(k, self.n_linear_layers) / self.n_linear_layers for k in self.k_values]

    @property
    def distinct_subsets(self) -> dict[str, list[int]]:
        """How many *distinct* layer subsets each K was actually averaged over.

        The subsets are drawn without replacement, so at K=1 with ``n_trials=3``
        this is 3 -- it used to be possible to draw the same layer twice and report
        the spread over fewer interventions than the trial count suggests.
        """
        out: dict[str, list[int]] = {}
        for label, sets in (("full_attention", self.full_attention_layers),
                            ("linear_attention", self.linear_attention_layers)):
            if not sets:
                continue
            out[label] = [len({tuple(sorted(s)) for s in per_k}) for per_k in sets]
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "k_values": self.k_values,
            "fraction_of_full_attention_layers": self.fractions,
            "full_attention": self.full_attention,
            "linear_attention": self.linear_attention,
            "fractions_linear": self.fractions_linear,
            "full_attention_masked": self.full_attention_masked,
            "linear_attention_masked": self.linear_attention_masked,
            "full_attention_std": self.full_attention_std,
            "linear_attention_std": self.linear_attention_std,
            "full_attention_trials": self.full_attention_trials,
            "linear_attention_trials": self.linear_attention_trials,
            "full_attention_layers": self.full_attention_layers,
            "linear_attention_layers": self.linear_attention_layers,
            "distinct_subsets": self.distinct_subsets,
            "seed": self.seed,
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
    prefill_chunk: int | None = 4096,
    n_trials: int = 3,
    seed: int = 0,
    baseline: float | None = None,
) -> MixerAblation:
    """Silence K full-attention layers vs K linear-attention layers.

    On a dense model ``linear_attention`` is empty and the call returns
    ``full_attention`` only -- still a valid "how concentrated is retrieval in a few
    layers" measurement.  Note that `cmd_mask` only calls this when the model *has*
    linear layers, so that dense branch is reachable from the API but not from the
    CLI, and a dense run's tree has no ``mixer_ablation.json``.

    ``baseline`` lets the caller pass an F1 it already measured on the same samples
    (``cmd_mask`` computes one for the masking curve); on the hybrid that unmasked
    pass is the most expensive single step of the stage.
    """
    if baseline is None:
        baseline = evaluate_samples(model, tokenizer, info, samples,
                                    max_new_tokens=max_new_tokens,
                                    prefill_chunk=prefill_chunk).f1
    full_layers = list(info.scoreable_layers)
    linear_layers = list(info.linear_layers)

    rng = np.random.default_rng(seed)
    trials_per_k = max(1, n_trials)

    def _subsets(layers: list[int], k: int, n: int) -> list[list[int]]:
        # Random subsets, averaged: the old `layers[:k]` always took the *earliest*
        # layers of each stack, so on a hybrid the linear arm (layers 0,1,2...) was
        # systematically earlier than the attention arm (3,7,11...).  Any difference
        # could have been depth, not the kind of mixer.
        #
        # Sampled *without replacement* where the space allows: with n_trials=3 at
        # K=1 the old code could draw the same layer twice ([[15],[19],[15]]), which
        # makes the reported spread a spread over fewer distinct interventions than
        # the trial count suggests.
        if k >= len(layers):
            # A fresh list per trial: `[list(layers)] * n` would alias one object.
            return [list(layers) for _ in range(n)]
        total = math.comb(len(layers), k)
        if n >= total:  # exhaustive: enumerate every subset once
            return [list(c) for c in itertools.combinations(layers, k)][:n]
        seen: set[tuple[int, ...]] = set()
        out: list[list[int]] = []
        attempts = 0
        while len(out) < n and attempts < 100 * n:
            attempts += 1
            picked = [layers[i] for i in rng.permutation(len(layers))[:k]]
            key = tuple(sorted(picked))
            if key in seen:
                continue
            seen.add(key)
            out.append(picked)
        while len(out) < n:  # extremely unlikely; keep the count as promised
            out.append([layers[i] for i in rng.permutation(len(layers))[:k]])
        return out

    full_scores: list[float] = []
    linear_scores: list[float] = []
    full_exact: list[float] = []
    linear_exact: list[float] = []
    full_masked: list[int] = []
    linear_masked: list[int] = []
    full_std: list[float] = []
    linear_std: list[float] = []
    full_trials: list[list[float]] = []
    linear_trials: list[list[float]] = []
    full_layer_sets: list[list[list[int]]] = []
    linear_layer_sets: list[list[list[int]]] = []

    for k in k_values:
        runs, exacts, counts, picks = [], [], [], []
        for picked in _subsets(full_layers, k, trials_per_k):
            picks.append(picked)
            m = evaluate_samples(model, tokenizer, info, samples, masked_layers=picked,
                                 max_new_tokens=max_new_tokens, prefill_chunk=prefill_chunk)
            runs.append(m.f1)
            exacts.append(m.exact_match)
            counts.append(len(picked))
        full_trials.append(runs)
        full_layer_sets.append(picks)
        full_scores.append(float(np.mean(runs)))
        full_std.append(float(np.std(runs)))
        full_exact.append(float(np.mean(exacts)))
        full_masked.append(max(counts))
        log.info("k=%d full-attention: %d subsets (e.g. %s) -> f1=%.1f +/- %.1f",
                 k, trials_per_k, picks[0], np.mean(runs), np.std(runs))

        if linear_layers:
            runs, exacts, counts, picks = [], [], [], []
            for picked in _subsets(linear_layers, k, trials_per_k):
                picks.append(picked)
                m = evaluate_samples(model, tokenizer, info, samples, masked_layers=picked,
                                     max_new_tokens=max_new_tokens, prefill_chunk=prefill_chunk)
                runs.append(m.f1)
                exacts.append(m.exact_match)
                counts.append(len(picked))
            linear_trials.append(runs)
            linear_layer_sets.append(picks)
            linear_scores.append(float(np.mean(runs)))
            linear_std.append(float(np.std(runs)))
            linear_exact.append(float(np.mean(exacts)))
            linear_masked.append(max(counts))
            log.info("k=%d linear: %d subsets (e.g. %s) -> f1=%.1f +/- %.1f",
                     k, trials_per_k, picks[0], np.mean(runs), np.std(runs))

    return MixerAblation(
        full_attention=full_scores, linear_attention=linear_scores,
        baseline=float(baseline), k_values=list(k_values),
        n_full_layers=len(full_layers), n_linear_layers=len(linear_layers),
        full_attention_exact=full_exact, linear_attention_exact=linear_exact,
        full_attention_masked=full_masked, linear_attention_masked=linear_masked,
        full_attention_std=full_std, linear_attention_std=linear_std,
        full_attention_trials=full_trials, linear_attention_trials=linear_trials,
        full_attention_layers=full_layer_sets, linear_attention_layers=linear_layer_sets,
        seed=seed,
    )
