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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from retrieval_heads.haystack import HaystackBuilder, build_needle_sample, iter_depths
from retrieval_heads.models import ModelInfo
from retrieval_heads.scoring import (
    PAIRINGS,
    InstanceResult,
    RetrievalScores,
    aggregate_scores,
    score_instance,
)
from retrieval_heads.utils import ensure_dir, get_logger, save_json, set_seed

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


@dataclass
class DetectionConfig:
    """Grid of NIAH instances used to estimate retrieval scores."""

    lengths: list[int] = field(default_factory=lambda: [1024, 2048, 4096])
    depths_per_length: int = 3
    max_new_tokens: int = 32
    needles: list[tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_NEEDLES))
    threshold: float = 0.1
    pairing: str = "next_step"
    chat_template: bool = True
    enable_thinking: bool | None = False
    system_prompt: str | None = None
    prefill_impl: str = "sdpa"
    capture_impl: str = "eager"
    capture_method: str = "output_attentions"
    #: Feed the prompt to the prefill in chunks of this size (None = one shot).
    #: Bounds prefill memory on cards where float32 SDPA falls back to the math
    #: backend and materialises (heads, seq, seq).
    prefill_chunk: int | None = 4096
    seed: int = 0
    #: cap on the total number of instances (None = the full grid)
    limit: int | None = None

    @property
    def grid_size(self) -> int:
        return len(self.needles) * len(self.lengths) * self.depths_per_length

    def plan(self) -> list[dict[str, Any]]:
        """Deterministic list of instances; each entry is one NIAH test."""
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
                        "seed": self.seed + 1000 * n_idx + 17 * d_idx + length,
                    })
        if self.limit is not None:
            items = items[: self.limit]
        return items

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["needles"] = [{"needle": n, "question": q} for n, q in self.needles]
        data["grid_size"] = self.grid_size
        return data


@dataclass
class DetectionRun:
    """Container for a finished detection run."""

    scores: RetrievalScores
    instances: list[InstanceResult]
    config: DetectionConfig
    model_info: ModelInfo
    wall_time_s: float = 0.0
    #: The *other* pairing's aggregate, when it was computed.  Reported alongside
    #: the primary one because the two disagree on which heads are retrieval heads.
    secondary: RetrievalScores | None = None

    def pairing_comparison(self, top_k: int = 10) -> dict[str, Any]:
        """Top heads under each pairing, and how much the two rankings overlap."""
        primary = [str(h) for h in self.scores.ranked_heads()[:top_k]]
        if self.secondary is None:
            return {"primary": {self.scores.pairing: primary}}
        other = [str(h) for h in self.secondary.ranked_heads()[:top_k]]
        overlap = len(set(primary) & set(other))
        return {
            "primary": {self.scores.pairing: primary},
            "secondary": {self.secondary.pairing: other},
            "top_k": top_k,
            "overlap": overlap,
            "jaccard": overlap / len(set(primary) | set(other)) if primary or other else float("nan"),
        }

    def summary(self) -> dict[str, Any]:
        # Two independent notions of "this instance actually tested retrieval":
        #  * the model recited a decent share of the needle (diagnostic), and
        #  * the scoring found at least one head copying a needle token (the
        #    method's own signal).  They can disagree, and that disagreement is
        #    itself informative, so both are reported.
        recited = [i for i in self.instances if i.needle_recall >= 0.3]
        copied = [
            i for i in self.instances
            if any(v > 0.0 for v in i.scores[self.scores.pairing].values())
        ]
        n = len(self.instances)
        return {
            "model": self.model_info.name,
            "n_instances": n,
            "n_instances_recited": len(recited),
            "n_instances_with_copy": len(copied),
            "mean_needle_recall": (
                sum(i.needle_recall for i in self.instances) / n if n else 0.0
            ),
            "wall_time_s": round(self.wall_time_s, 1),
            "config": self.config.as_dict(),
            "sparsity": self.scores.sparsity(),
            "top_heads": [
                {"head": str(h), "score": self.scores.head_score(h),
                 "activation_freq": float(self.scores.activation_freq[h.layer, h.head])}
                for h in self.scores.ranked_heads()[:20]
            ],
            "pairing_comparison": self.pairing_comparison(),
            "model_info": self.model_info.as_dict(),
        }

    def save(self, out_dir: str | Path) -> Path:
        out = ensure_dir(out_dir)
        self.scores.save(out / f"scores_{self.scores.pairing}")
        save_json(self.summary(), out / f"summary_{self.scores.pairing}.json")
        with (out / f"instances_{self.scores.pairing}.jsonl").open("w", encoding="utf-8") as fh:
            for inst in self.instances:
                fh.write(json.dumps(inst.as_dict(), default=str, ensure_ascii=False) + "\n")
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
) -> DetectionRun:
    """Execute the detection grid and aggregate per-head retrieval scores."""
    import time

    config = config or DetectionConfig()
    set_seed(config.seed)
    plan = config.plan()
    log.info(
        "detection: %d instances | %d needles x %d lengths x %d depths | scoreable heads=%d",
        len(plan), len(config.needles), len(config.lengths), config.depths_per_length,
        info.n_scoreable_heads,
    )

    builder = HaystackBuilder(corpus, seed=config.seed)
    results: list[InstanceResult] = []
    started = time.time()
    iterator = plan
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(plan, desc=f"detect[{info.name}]", unit="inst")
        except ImportError:  # pragma: no cover
            pass

    for item in iterator:
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
        )
        result.sample["needle_index"] = item["needle_index"]
        results.append(result)
        log.debug("instance %d/%d len=%d depth=%.2f recall=%.2f",
                  item["index"] + 1, len(plan), sample.length, item["depth"], result.needle_recall)

    scores = aggregate_scores(results, info, pairing=config.pairing, threshold=config.threshold)
    scores.meta = {"config": config.as_dict(), "corpus": "custom" if corpus else "synthetic"}

    secondary = None
    other = [p for p in PAIRINGS if p != config.pairing]
    if other:
        try:
            secondary = aggregate_scores(results, info, pairing=other[0],
                                         threshold=config.threshold, keep_instances=False)
            primary_top = {str(h) for h in scores.ranked_heads()[:10]}
            secondary_top = {str(h) for h in secondary.ranked_heads()[:10]}
            log.info("pairing check: top-10 overlap between %s and %s = %d/10",
                     config.pairing, other[0], len(primary_top & secondary_top))
        except KeyError:  # pragma: no cover - second pairing was not recorded
            secondary = None

    run = DetectionRun(
        scores=scores, instances=results, config=config, model_info=info,
        wall_time_s=time.time() - started, secondary=secondary,
    )
    best = scores.ranked_heads()[0]
    log.info("detection finished in %.1fs; top head %s (%.2f), %d/%d instances recited the needle",
             run.wall_time_s, best, scores.head_score(best),
             sum(1 for i in results if i.needle_recall >= 0.3), len(results))
    if out_dir is not None:
        run.save(out_dir)
    return run


def load_run(out_dir: str | Path, info: ModelInfo, pairing: str = "next_step") -> RetrievalScores:
    """Reload a saved detection run (matrices + metadata only)."""
    return RetrievalScores.load(Path(out_dir) / f"scores_{pairing}", info)
