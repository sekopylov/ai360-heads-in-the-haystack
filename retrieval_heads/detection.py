"""The retrieval-head detection algorithm (paper Sec. 3).

The paper compiles three sets of Needle-in-a-Haystack samples, and for each one
runs the test at 20 context lengths x 10 insertion depths (~600 instances per
model).  That is the full recipe; :class:`DetectionConfig` defaults to a much
smaller grid so it can actually finish on a laptop, and every axis is
configurable so the same code scales up unchanged on a GPU.

Everything about *which* heads are scoreable comes from
:class:`~retrieval_heads.models.ModelInfo`, so this driver runs unmodified on a
dense model and on a hybrid linear/full-attention model.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


from retrieval_heads.haystack import HaystackBuilder, build_needle_sample, iter_depths
from retrieval_heads.models import ModelInfo
from retrieval_heads.provenance import add_provenance
from retrieval_heads.scoring import (
    PAIRINGS,
    InstanceResult,
    RetrievalScores,
    aggregate_scores,
    score_instance,
)
from retrieval_heads.utils import (
    ensure_dir, finite_json, get_logger, json_default, save_json, set_seed,
)

log = get_logger("detection")


#: Three (needle, question) pairs, unrelated to the filler.
#:
#: Each needle is deliberately shaped as ``<referent named in the question> +
#: <long predicate that the question asks for>``.  The score is a *token recall
#: over the needle*, so a question that only asks for a short span (e.g. "how
#: many ...?") is a bad test: a perfectly correct one-word answer leaves most of
#: the needle uncopied and deflates every head's score.  This matches the paper's
#: own example, where the answer is the predicate of the needle sentence.
DEFAULT_NEEDLES: tuple[tuple[str, str], ...] = (
    (
        "The best thing to do in San Francisco is to eat a sandwich in Dolores Park on a sunny day.",
        "What is the best thing to do in San Francisco?",
    ),
    (
        "The recommended procedure for sealing the Vermilion Observatory dome is to apply three "
        "coats of zinc primer and then wait forty-eight hours before polishing.",
        "What is the recommended procedure for sealing the Vermilion Observatory dome?",
    ),
    (
        "The object Professor Alaric Voss keeps in the Harbourside loft is a brass barometer that "
        "once belonged to his grandfather.",
        "What does Professor Alaric Voss keep in the Harbourside loft?",
    ),
)

#: The needles retrieval heads are *selected* on.  Never score a causal ablation on
#: these: the heads were chosen because they copy exactly this text.
DETECTION_NEEDLES: tuple[tuple[str, str], ...] = DEFAULT_NEEDLES

#: Held out from detection (the paper's "additional set of needle tests that are
#: different from the three sets used for retrieval head detection").  The masking
#: curve uses these, so the effect is not measured on the selection set.
#:
#: Three of them, not one: the ablation's spread (`retrieval_std`, and the random
#: arm's) is computed over the eval samples, and with a single needle every sample
#: shares the same (question, needle) pair -- only the depth varies.  Needle identity
#: is the larger source of variance, so `mask --needles 3 --depths 5` is a more honest
#: 15-sample set than one needle at ten depths.  Each is shaped like the detection
#: needles: a referent named in the question plus a long predicate the question asks
#: for (a one-word answer would leave most needle tokens uncopied and deflate every
#: head's score).
EVAL_NEEDLES: tuple[tuple[str, str], ...] = (
    (
        "The instrument that the lighthouse keeper of Cape Winterbourne polishes every "
        "second Tuesday is a brass sextant engraved with the initials R. M.",
        "What instrument does the lighthouse keeper of Cape Winterbourne polish?",
    ),
    (
        "The remedy that the harbourmaster of Port Ellery applies to a seized compass is a "
        "rinse in distilled water followed by a night in dry rice.",
        "What remedy does the harbourmaster of Port Ellery apply to a seized compass?",
    ),
    (
        "The keepsake that the violinist Mira Okonkwo carries in her instrument case is a "
        "folded ticket from the last night of the Winterbourne concert hall.",
        "What keepsake does the violinist Mira Okonkwo carry in her instrument case?",
    ),
)


def assert_needles_disjoint() -> None:
    """Fail fast if a detection needle is also used for the causal ablation."""
    detection = {needle for needle, _ in DETECTION_NEEDLES}
    evaluation = {needle for needle, _ in EVAL_NEEDLES}
    leaked = detection & evaluation
    if leaked:
        raise RuntimeError(
            f"the eval needle set overlaps the detection needle set ({len(leaked)} shared); "
            f"the causal ablation would be measured on the selection set"
        )


#: An instance counts as "recited" for the conditional matrices / summary at this
#: recall (a NIAH answer is a span, so this is deliberately low).
RECITED_RECALL = 0.3


@dataclass
class DetectionConfig:
    """Grid of NIAH instances used to estimate retrieval scores."""

    lengths: list[int] = field(default_factory=lambda: [1024, 2048, 4096])
    depths_per_length: int = 3
    max_new_tokens: int = 32
    #: Detection needles only -- `EVAL_NEEDLES` are held out for the ablation.
    needles: list[tuple[str, str]] = field(default_factory=lambda: list(DETECTION_NEEDLES))
    threshold: float = 0.1
    pairing: str = "next_step"
    chat_template: bool = True
    enable_thinking: bool | None = False
    system_prompt: str | None = None
    prefill_impl: str = "sdpa"
    capture_impl: str = "eager"
    capture_method: str = "patch"
    #: Feed the prompt to the prefill in chunks of this size (None = one shot).
    #: Bounds prefill memory on cards where float32 SDPA falls back to the math
    #: backend and materialises (heads, seq, seq).
    prefill_chunk: int | None = 4096
    seed: int = 0
    #: Which positions the attention argmax may choose from.  "haystack" (default) is
    #: the paper's ``a in R^{|x|}``: the context span (filler + needle) alone, so the
    #: question and the chat template cannot win criterion (2).  "prompt" adds those
    #: and is what `ds-results/` was produced with; "full" also allows the tokens
    #: generated so far, which makes the score depend on how much was generated.
    argmax_domain: str = "haystack"
    #: Cap on the total number of instances (None = the full grid).  NOTE: the plan
    #: is truncated in grid order, so a limit keeps the first needles and the shortest
    #: lengths -- a biased subsample, not a random one.
    limit: int | None = None
    #: Lengths the model's context window rejected, kept so the artifact shows the
    #: requested grid rather than only the surviving one.
    dropped_lengths: list[int] = field(default_factory=list)

    @property
    def grid_size(self) -> int:
        return len(self.needles) * len(self.lengths) * self.depths_per_length

    def plan(self) -> list[dict[str, Any]]:
        """Deterministic list of instances; each entry is one NIAH test.

        Cached: `as_dict()` (called from every `summary()`) used to rebuild the grid
        and re-log the `--limit` warning each time.  The cache means the config must
        be treated as immutable after the first plan()/as_dict() call.

        Depths are the endpoints-inclusive ``iter_depths`` ("from the start to the
        end", as the paper puts it), so 0.0 and 1.0 are in the grid.  With only a few
        depths the endpoints carry a large share of the weight.
        """
        # Cache keyed on the fields that define the grid, not on identity: the config
        # is a mutable dataclass and `as_dict()` (called from every `summary()`) used
        # to freeze the first plan as a side effect, so a later edit to `lengths` or
        # `limit` was silently ignored.
        key = (tuple(self.needles), tuple(self.lengths), self.depths_per_length,
               self.limit, self.seed)
        if getattr(self, "_plan_cache_key", None) == key:
            return list(self._plan_cache)
        items: list[dict[str, Any]] = []
        for n_idx, (needle, question) in enumerate(self.needles):
            for length in self.lengths:
                for d_idx, depth in enumerate(iter_depths(self.depths_per_length)):
                    items.append({
                        "index": len(items),
                        "needle_index": n_idx,
                        "needle": needle,
                        "question": question,
                        "target_tokens": length,
                        "depth": depth,
                        # crc32 of the tuple instead of an arithmetic mix: stable
                        # across runs.  It is only 32 bits, so a collision between two
                        # instances (identical filler) is possible in principle -- the
                        # `seed` field of each instance makes it detectable.
                        "seed": self.seed + zlib.crc32(f"{n_idx}:{length}:{d_idx}".encode()),
                    })
        if self.limit is not None:
            items = items[: self.limit]
            log.warning("--limit %d keeps the first %d instances in grid order (one needle, "
                        "shortest lengths); treat this as a debugging sample, not an estimate",
                        self.limit, len(items))
        object.__setattr__(self, "_plan_cache", list(items))
        object.__setattr__(self, "_plan_cache_key", key)
        return items

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["needles"] = [{"needle": n, "question": q} for n, q in self.needles]
        # `grid_size` is what will actually run (so it agrees with `n_planned`);
        # the unbounded grid is kept separately instead of silently disagreeing.
        data["grid_size"] = len(self.plan())
        data["grid_size_unlimited"] = self.grid_size
        return data


@dataclass
class DetectionRun:
    """Container for a finished detection run."""

    scores: RetrievalScores
    instances: list[InstanceResult]
    config: DetectionConfig
    model_info: ModelInfo
    wall_time_s: float = 0.0
    #: Instances the grid planned (``len(instances)`` unless a partial run).
    n_planned: int = 0
    #: Recited-only aggregates, keyed by pairing (they are *not* interchangeable:
    #: a single shared value leaked next_step's numbers into the same_step summary).
    conditional: dict[str, RetrievalScores] = field(default_factory=dict)
    #: The *other* pairing's aggregate, when it was computed.  Reported alongside
    #: the primary one because the two disagree on which heads are retrieval heads.
    secondary: RetrievalScores | None = None
    #: The same aggregation under the *raw* per-token denominator ``|k|`` (the
    #: needle's token count, repeats included) instead of ``|unique(k)|``.  The
    #: paper's ``|g_h & k| / |k|`` is ambiguous there; both readings are emitted so a
    #: reader can rescale instead of trusting the recorded inflation ratio.  Keyed by
    #: pairing, like `conditional`.
    raw: dict[str, RetrievalScores] = field(default_factory=dict)

    def pairing_comparison(self, top_k: int = 10,
                           scores: RetrievalScores | None = None) -> dict[str, Any]:
        """Top heads under each pairing, and how much the two rankings overlap.

        ``scores`` selects the primary pairing, so the `same_step` sidecar lists
        `same_step` heads as primary rather than the `next_step` ones.
        """
        scores = scores or self.scores
        primary = [str(h) for h in scores.ranked_heads()[:top_k]]
        if self.secondary is None:
            return {"primary": {scores.pairing: primary}}
        # `other` must be the *other* aggregate: using `self.secondary` unconditionally
        # made `summary(secondary)` compare same_step with itself (overlap == top_k).
        other_scores = self.scores if scores is self.secondary else self.secondary
        other = [str(h) for h in other_scores.ranked_heads()[:top_k]]
        overlap = len(set(primary) & set(other))
        return {
            "primary": {scores.pairing: primary},
            "secondary": {other_scores.pairing: other},
            "top_k": top_k,
            "overlap": overlap,
            "jaccard": overlap / len(set(primary) | set(other)) if primary or other else float("nan"),
        }

    def aligned_ranking(self, scores: RetrievalScores | None = None,
                        top_k: int = 10) -> list[dict[str, Any]]:
        """Mean strict-aligned score per head, best first.

        The `aligned_scores` were written to the per-instance JSONL and read by
        nothing; this surfaces the robustness check in the summary.
        """
        scores = scores or self.scores
        pairing = scores.pairing
        totals = {h: 0.0 for h in self.model_info.scoreable_heads}
        n = 0
        for instance in self.instances:
            data = instance.aligned_scores.get(pairing)
            if not data:
                continue
            n += 1
            for head in self.model_info.scoreable_heads:
                totals[head] += float(data.get(str(head), 0.0))
        if n == 0:
            return []
        ranked = sorted(totals.items(), key=lambda kv: (-kv[1] / n, kv[0].layer, kv[0].head))
        # The mean is over the instances that *have* an aligned row, which is a
        # subset; without the counts it is indistinguishable from a mean over all.
        return [{"head": str(head), "aligned_score": value / n,
                 "n": n, "n_missing": len(self.instances) - n}
                for head, value in ranked[:top_k]]

    def summary(self, scores: RetrievalScores | None = None) -> dict[str, Any]:
        """Run-level summary; ``scores`` selects the pairing (default: primary)."""
        scores = scores or self.scores
        # Two independent notions of "this instance actually tested retrieval":
        #  * the model recited a decent share of the needle (diagnostic), and
        #  * the scoring found at least one head copying a needle token (the
        #    method's own signal).  They can disagree, and that disagreement is
        #    itself informative, so both are reported.
        recited = [i for i in self.instances if i.needle_recall >= RECITED_RECALL]
        copied = [
            i for i in self.instances
            if any(v > 0.0 for v in i.scores[scores.pairing].values())
        ]
        n = len(self.instances)
        return {
            "model": self.model_info.name,
            "n_instances": n,
            "n_instances_recited": len(recited),
            "n_instances_with_copy": len(copied),
            # The score is a recall over all unique needle tokens, so a generation
            # cut off by max_new_tokens depresses every head.  Report how often
            # that happened instead of leaving it only in the per-instance JSONL.
            "n_instances_truncated": sum(
                1 for i in self.instances
                if i.meta.get("truncated", not i.meta.get("eos_reached", True))
            ),
            # Both head bases, so "62.5% of scoreable heads" and "8 of 336 heads"
            # can be read from one artifact instead of two documents.
            "n_all_heads": self.model_info.n_all_heads,
            "n_instances_low_ceiling": sum(
                1 for i in self.instances
                if i.sample.get("tokenization_attainable_score", 1.0) < 0.98
            ),
            "mean_needle_prefix_recall": (
                sum(i.meta.get("needle_prefix_recall", 0.0) for i in self.instances) / n
                if n else 0.0
            ),
            "mean_needle_recall": (
                sum(i.needle_recall for i in self.instances) / n if n else 0.0
            ),
            # Attention-sink pressure: share of scored steps where a head's argmax
            # fell on position 0.  Per-head values live in the JSONL; this is the
            # run-level summary so the number is not write-only.
            "mean_sink_rate": (
                sum(i.sink_rate.get(scores.pairing, {}).get("__overall__", 0.0)
                    for i in self.instances) / n if n else 0.0
            ),
            "wall_time_s": round(self.wall_time_s, 1),
            "config": self.config.as_dict(),
            # The run's own threshold, so a reader can tell the fixed 0.1/0.5
            # buckets from the threshold this run actually used.
            "score_threshold": scores.threshold,
            # Which positions criterion (2) searched, and how far that moved the
            # argmax relative to the prompt domain.  These two make a `haystack` run
            # distinguishable from a `prompt` one without opening the config block.
            "argmax_domain": scores.meta.get("argmax_domain", self.config.argmax_domain),
            "argmax_domain_shift": scores.meta.get("argmax_domain_shift"),
            # True when sequence position 0 (the sink) is inside the haystack span;
            # with a chat template it is not, which inflates the >0.1 share.
            "sink_in_haystack": scores.meta.get("sink_in_haystack"),
            # How much the unique-token denominator inflates the score relative to a
            # per-token reading of the paper's formula.
            "needle_stats": {
                key: scores.meta[key] for key in
                ("needle_tokens_mean", "unique_needle_tokens_mean", "denominator_inflation",
                 "tokenization_attainable_mean")
                if key in scores.meta
            },
            "sparsity": scores.sparsity(),
            # The raw-denominator view of the same matrices: the paper's `|k|` read
            # per-token rather than per-unique-token.  It is strictly lower whenever
            # the needle repeats a token, which is the deviation `needle_stats`
            # quantifies -- this is the matrix behind that number.
            "sparsity_raw": (self.raw[scores.pairing].sparsity()
                             if scores.pairing in self.raw else None),
            "sparsity_recited": (self.conditional[scores.pairing].sparsity()
                                 if scores.pairing in self.conditional else None),
            "top_heads_recited": (
                [{"head": str(h), "score": self.conditional[scores.pairing].head_score(h)}
                 for h in self.conditional[scores.pairing].ranked_heads()[:10]]
                if scores.pairing in self.conditional else []
            ),
            "top_heads": [
                {"head": str(h), "score": scores.head_score(h),
                 "activation_freq": float(scores.activation_freq[h.layer, h.head])}
                for h in scores.ranked_heads()[:20]
            ],
            "pairing_comparison": self.pairing_comparison(scores=scores),
            "aligned_top_heads": self.aligned_ranking(scores),
            "n_planned": self.n_planned,
            "model_info": self.model_info.as_dict(),
        }

    def save(self, out_dir: str | Path, *, write_instances: bool = True) -> Path:
        """Write the aggregates (and the JSONL unless it was already streamed)."""
        out = ensure_dir(out_dir)
        self.scores.save(out / f"scores_{self.scores.pairing}")
        save_json(add_provenance(self.summary(), dtype=self.model_info.dtype),
                  out / f"summary_{self.scores.pairing}.json")
        if write_instances:
            with (out / f"instances_{self.scores.pairing}.jsonl").open("w", encoding="utf-8") as fh:
                for inst in self.instances:
                    # Strict encoder: `default=str` turned any unexpected object into a
                    # string and quietly changed the artifact's schema.
                    fh.write(json.dumps(finite_json(inst.as_dict()), default=json_default,
                                        ensure_ascii=False, allow_nan=False) + "\n")
        # The secondary pairing comes from the same decoding pass, so its
        # aggregate is written too: otherwise `same_step` only ever appeared as a
        # top-10 list inside the primary summary and its per-head table was lost.
        for pairing, agg in self.conditional.items():
            agg.save(out / f"scores_{pairing}_recited")
        # The raw-denominator matrices: a second small pair of files per pairing, so
        # the alternative reading of the paper's `|k|` is a first-class artifact
        # rather than a rescaled approximation.
        for pairing, agg in self.raw.items():
            agg.save(out / f"scores_{pairing}_raw")
        if self.secondary is not None:
            self.secondary.save(out / f"scores_{self.secondary.pairing}")
            save_json(add_provenance(self.summary(self.secondary),
                                     dtype=self.model_info.dtype),
                      out / f"summary_{self.secondary.pairing}.json")
        log.info("detection run written to %s", out)
        return out


def run_detection(
    model: Any,
    tokenizer: Any,
    info: ModelInfo,
    config: DetectionConfig | None = None,
    *,
    corpus: Sequence[str] | None = None,
    progress: bool = True,
    out_dir: str | Path | None = None,
    corpus_path: str | Path | None = None,
) -> DetectionRun:
    """Execute the detection grid and aggregate per-head retrieval scores."""
    import time

    config = config or DetectionConfig()
    set_seed(config.seed)
    # Fail before the grid, not on the first instance: a bad domain otherwise cost
    # one full NIAH build (and, for `haystack`, a confusing per-instance error).
    if config.argmax_domain not in ("prompt", "full", "haystack"):
        raise ValueError(
            f"argmax_domain must be 'prompt', 'full' or 'haystack', "
            f"got {config.argmax_domain!r}"
        )
    plan = config.plan()
    log.info(
        "detection: %d instances | %d needles x %d lengths x %d depths | scoreable heads=%d",
        len(plan), len(config.needles), len(config.lengths), config.depths_per_length,
        info.n_scoreable_heads,
    )

    assert_needles_disjoint()
    results: list[InstanceResult] = []
    started = time.time()
    # Stream every instance to disk as it finishes: a paper-scale run is tens of
    # minutes of GPU time, and buffering it all meant one timeout lost the lot.
    stream = None
    if out_dir is not None:
        stream = (ensure_dir(out_dir) / f"instances_{config.pairing}.jsonl").open(
            "w", encoding="utf-8")
    iterator = plan
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(plan, desc=f"detect[{info.name}]", unit="inst")
        except ImportError:  # pragma: no cover
            pass

    try:
        for item in iterator:
            # A fresh builder per instance, seeded by the instance's own seed: the
            # seed is recorded in every artifact, so it has to be the one that
            # actually produced the filler (a single shared builder ignored it).
            builder = HaystackBuilder(corpus, seed=item["seed"])
            sample = build_needle_sample(
                tokenizer,
                needle=item["needle"],
                question=item["question"],
                target_tokens=item["target_tokens"],
                depth=item["depth"],
                builder=builder,
                chat_template=config.chat_template,
                enable_thinking=config.enable_thinking,
                system_prompt=config.system_prompt,
                seed=item["seed"],
            )
            result = score_instance(
                model, info, sample, tokenizer,
                max_new_tokens=config.max_new_tokens,
                pairing=config.pairing,
                prefill_impl=config.prefill_impl,
                capture_impl=config.capture_impl,
                capture_method=config.capture_method,
                prefill_chunk=config.prefill_chunk,
                argmax_domain=config.argmax_domain,
            )
            result.sample["needle_index"] = item["needle_index"]
            results.append(result)
            if stream is not None:
                stream.write(json.dumps(finite_json(result.as_dict()), default=json_default,
                                        ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            log.debug("instance %d/%d len=%d depth=%.2f recall=%.2f",
                      item["index"] + 1, len(plan), sample.length, item["depth"], result.needle_recall)

    finally:
        # A crash mid-run must still close the JSONL (and leave a clearly
        # short file rather than a locked descriptor).
        if stream is not None:
            stream.close()

    scores = aggregate_scores(results, info, pairing=config.pairing, threshold=config.threshold)
    # The same aggregation under the raw per-token denominator: one extra pass over
    # the per-instance dicts (no forwards), and the alternative reading of the
    # paper's `|k|` becomes an artifact instead of a rescaling.
    raw_scores = aggregate_scores(results, info, pairing=config.pairing,
                                  threshold=config.threshold, field="scores_raw")
    # The denominator convention is about *text* tokens (what the score divides by),
    # so the inflation ratio uses the text tokenization, not the prompt span length
    # (which mixes in the boundary-fusion effect).
    needle_tokens = [len(r.sample.get("needle_text_ids") or [])
                     or r.sample.get("n_needle_tokens", 0) for r in results]
    unique_tokens = [r.sample.get("n_unique_needle_text_tokens", 0) for r in results]
    scores.meta = {
        "config": config.as_dict(),
        "corpus": "custom" if corpus else "synthetic",
        # The path (not just custom/synthetic) so an ablation can reuse the *same*
        # filler instead of silently measuring on a different distribution.
        "corpus_path": str(corpus_path) if corpus_path else None,
        "argmax_domain": config.argmax_domain,
        # The score is averaged over *all* instances, including ones where the model
        # never recited the needle; `conditional` below is the recited-only view, and
        # the two needle counts expose how much the unique-token denominator inflates
        # the score relative to a per-token reading.
        "needle_tokens_mean": float(np.mean(needle_tokens)) if needle_tokens else 0.0,
        "unique_needle_tokens_mean": float(np.mean(unique_tokens)) if unique_tokens else 0.0,
        "tokenization_attainable_mean": float(np.mean(
            [r.sample.get("tokenization_attainable_score", 1.0)
             for r in results])) if results else 1.0,
        "denominator_inflation": (
            float(np.mean(needle_tokens)) / float(np.mean(unique_tokens))
            if unique_tokens and float(np.mean(unique_tokens)) else 1.0
        ),
    }
    # How much the argmax domain moved criterion (2): summed over instances, so a
    # reader can see whether `haystack` was a no-op or a real change on this grid.
    shifts = [r.meta.get("argmax_domain_shift") or {} for r in results]
    shift_positions = sum(int(s.get("positions", 0)) for s in shifts)
    shift_shifted = sum(int(s.get("shifted", 0)) for s in shifts)
    scores.meta["argmax_domain_shift"] = {
        "positions": shift_positions,
        "shifted": shift_shifted,
        "share": (shift_shifted / shift_positions) if shift_positions else 0.0,
        "n_instances": len(shifts),
        "reference": "prompt",
    }
    # Where the attention sink sits relative to the haystack.  Criterion (2) asks
    # whether the argmax is a *needle* token, so a sink inside `x` suppresses credit
    # and one before `x` (a chat template's first tokens) does not -- the same model
    # scores an order of magnitude more heads above 0.1 with the sink outside the
    # span.  Recorded per run so two runs cannot be compared without noticing.
    sink_inside = sum(1 for r in results
                      if (r.sample or {}).get("haystack_includes_sink"))
    scores.meta["n_instances_with_sink_in_haystack"] = sink_inside
    scores.meta["sink_in_haystack"] = bool(results) and sink_inside == len(results)
    # Mirror the full meta (config, corpus, needle stats) and mark the denominator,
    # so the `_raw` sidecar cannot be mistaken for the primary one.
    raw_scores.meta = dict(scores.meta)
    raw_scores.meta["denominator"] = "raw_token_count"
    raw_aggs: dict[str, RetrievalScores] = {raw_scores.pairing: raw_scores}

    secondary = None
    other = [p for p in PAIRINGS if p != config.pairing]
    if other:
        try:
            secondary = aggregate_scores(results, info, pairing=other[0],
                                         threshold=config.threshold)
            primary_top = {str(h) for h in scores.ranked_heads()[:10]}
            secondary_top = {str(h) for h in secondary.ranked_heads()[:10]}
            log.info("pairing check: top-10 overlap between %s and %s = %d/10",
                     config.pairing, other[0], len(primary_top & secondary_top))
        except KeyError:  # pragma: no cover - second pairing was not recorded
            secondary = None
        else:
            # Without this the `same_step` sidecar had no config/corpus, unlike the
            # primary one.
            secondary.meta = dict(scores.meta)
            # The raw view of the secondary pairing too: the two pairings credit
            # different token sets, so the raw denominator cannot be derived from the
            # primary one.
            secondary_raw = aggregate_scores(results, info, pairing=secondary.pairing,
                                             threshold=config.threshold,
                                             field="scores_raw")
            secondary_raw.meta = dict(scores.meta)
            secondary_raw.meta["denominator"] = "raw_token_count"
            raw_aggs[secondary_raw.pairing] = secondary_raw

    # Same matrices, restricted to instances the model actually solved: without this
    # a model that fails NIAH more often looks "less sparse" for reasons unrelated to
    # its heads.
    recited = [r for r in results if r.needle_recall >= RECITED_RECALL]
    # Only the pairings that actually produced a secondary aggregate: iterating
    # `other[0]` unconditionally would re-raise the same KeyError the guard above
    # just swallowed.
    conditional_pairings = [config.pairing]
    if secondary is not None:
        conditional_pairings.append(secondary.pairing)
    conditional: dict[str, RetrievalScores] = {}
    for pairing in conditional_pairings:
        if not recited:
            continue
        agg = aggregate_scores(recited, info, pairing=pairing, threshold=config.threshold)
        # Mirror the main sidecar's provenance: without `config`/`argmax_domain` a
        # `*_recited.json` could not be tied to the conditions that produced it.
        agg.meta = {"corpus": scores.meta["corpus"], "recited_only": True,
                    "recall_threshold": RECITED_RECALL, "n_instances": len(recited),
                    "pairing": pairing, "argmax_domain": config.argmax_domain,
                    "config": scores.meta.get("config")}
        conditional[pairing] = agg

    run = DetectionRun(
        scores=scores, instances=results, config=config, model_info=info,
        wall_time_s=time.time() - started, secondary=secondary, n_planned=len(plan),
        conditional=conditional, raw=raw_aggs,
    )
    # The flag lives in InstanceResult.meta (`sample` is the NIAH sample dict and has
    # no such key), so reading `r.sample` made this warning unreachable.
    truncated = sum(
        1 for r in results
        if r.meta.get("truncated", not r.meta.get("eos_reached", True))
    )
    if results and truncated / len(results) > 0.25:
        log.warning(
            "%d/%d instances hit max_new_tokens=%d before an EOS; the retrieval score "
            "is a recall over that budget, so the numbers are budget-limited",
            truncated, len(results), config.max_new_tokens,
        )
    best = scores.ranked_heads()[0]
    log.info("detection finished in %.1fs; top head %s (%.2f), %d/%d instances recited the needle",
             run.wall_time_s, best, scores.head_score(best),
             sum(1 for i in results if i.needle_recall >= RECITED_RECALL), len(results))
    if out_dir is not None:
        # The per-instance JSONL was already streamed above (`stream` exists exactly
        # when `out_dir` does), so `save` must not write it a second time.  The flag
        # used to read `write_instances=stream is None`, which is always False here.
        run.save(out_dir, write_instances=False)
    return run
