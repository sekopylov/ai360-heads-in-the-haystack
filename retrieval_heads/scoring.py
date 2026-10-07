"""The retrieval score.

Definition (paper, Sec. 3)
--------------------------
During greedy autoregressive decoding let ``w`` be the token currently being
generated and ``a`` the attention probabilities of a head over the input.
Head ``h`` *copy-pastes* ``w`` when

    (1) ``w in k``                       -- the token belongs to the needle, and
    (2) ``x_j = w``, ``j = argmax(a)``, ``j in i_q``
                                          -- the head's most-attended input
                                             position holds that very token and
                                             lies inside the needle.

With ``g_h`` the set of tokens copy-pasted by head ``h``, the retrieval score is
``|g_h & k| / |k|``, i.e. a token-level recall over the needle.

Two notes where the paper leaves room for interpretation
--------------------------------------------------------
*Denominator.*  ``g_h`` is a **set**, so ``|g_h & k| <= |unique(k)|``.  We
therefore take ``|k|`` to be the number of *unique* needle tokens; using the raw
needle length would cap the score below 1.0 for any needle with a repeated token.

*Which attention row.*  "The attention scores of a head" at the step where ``w``
is generated can be paired with ``w`` in two ways, and we expose both:

``pairing="next_step"`` (default)
    The row is the one whose query position *produces* ``w``; the head looks at
    the source position it is about to copy.  This is the literal reading of
    "the current token being generated as w" and matches a CopyNet-style paste.
``pairing="same_step"``
    The row is the one at ``w``'s own query position; the head looks back at the
    source occurrence of the token it just emitted (an induction-head-like
    pattern).

Both are computed from a single decoding pass -- no extra forward passes -- and
the driver stores both, so the choice is a reporting decision rather than a
compute decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from retrieval_heads.attention import AttentionRecorder, set_attn_implementation
from retrieval_heads.haystack import NeedleSample
from retrieval_heads.models import ModelInfo, model_device
from retrieval_heads.utils import HeadRef, get_logger, save_json

log = get_logger("scoring")

PAIRINGS = ("next_step", "same_step")


def _eos_ids(model: Any, tokenizer: Any) -> set[int]:
    """Every EOS id the model might emit (config, generation config, tokenizer)."""
    ids: set[int] = set()
    for source in (getattr(model, "config", None), getattr(model, "generation_config", None)):
        value = getattr(source, "eos_token_id", None) if source is not None else None
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, (list, tuple, set)):
            ids.update(int(v) for v in value)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(int(tokenizer.eos_token_id))
    return ids


# --------------------------------------------------------------------------- trace
@dataclass
class StepTrace:
    """One decoding step: the attention rows and what the step produced."""

    step: int
    fed_token: int                 # u_t: the token whose query position we are at
    predicted_token: int           # w_{t+1}: what this step predicts
    attn: dict[int, torch.Tensor]  # layer -> (heads, kv_len)


@dataclass
class DecodeTrace:
    steps: list[StepTrace] = field(default_factory=list)
    prompt_len: int = 0
    prefill_logits: torch.Tensor | None = None

    @property
    def kv_len(self) -> int:
        return self.prompt_len + len(self.steps)


def decode_with_attention(
    model: Any,
    info: ModelInfo,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    tokenizer: Any = None,
    prefill_impl: str = "sdpa",
    capture_impl: str = "eager",
    capture_method: str = "output_attentions",
    stop_on_eos: bool = True,
    prefill_chunk: int | None = None,
) -> tuple[DecodeTrace, list[int]]:
    """Greedy-decode ``input_ids`` while recording attention at every step.

    The prefill runs on a cheap kernel (``sdpa``) because attention maps are not
    needed there; only the single-token decode steps are captured.  That keeps
    memory flat in the context length -- each captured row is
    ``(heads, kv_len)``, not ``(heads, seq, seq)``.

    ``prefill_chunk`` bounds the *prefill* the other way.  SDPA only reaches a
    memory-efficient kernel when the dtype and shapes allow it; with float32 it
    can fall back to the math backend, which materialises the full
    ``(heads, seq, seq)`` matrix.  Measured on an L4: a 16K-token fp32 prefill
    asked for a single 20.6 GiB allocation and OOM'd on a 22 GiB card.  Feeding
    the prompt in chunks through the KV cache keeps peak memory at
    ``O(chunk^2)`` instead, at the cost of a few extra forward passes.
    """
    if prefill_chunk is not None and prefill_chunk <= 0:
        raise ValueError("prefill_chunk must be positive or None")
    input_ids = input_ids.to(model_device(model))
    recorder = AttentionRecorder(model, info, method=capture_method)
    trace = DecodeTrace(prompt_len=int(input_ids.shape[1]))

    restore = set_attn_implementation(model, prefill_impl)
    try:
        if prefill_chunk is not None and input_ids.shape[1] > prefill_chunk:
            # Incremental prefill: each chunk sees the KV state of the previous
            # ones, and transformers derives cache_position from the cache length,
            # so positions and the causal mask stay correct.
            cache = None
            for start in range(0, input_ids.shape[1], prefill_chunk):
                out = model(
                    input_ids=input_ids[:, start:start + prefill_chunk],
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
            log.info("chunked prefill: %d tokens in chunks of %d",
                     input_ids.shape[1], prefill_chunk)
        else:
            out = model(input_ids=input_ids, use_cache=True)
            cache = out.past_key_values
        logits = out.logits[:, -1, :]
        trace.prefill_logits = logits.detach()
        restore_capture = set_attn_implementation(model, capture_impl)
        try:
            generated: list[int] = []
            nxt = logits.argmax(-1, keepdim=True)
            eos = _eos_ids(model, tokenizer) if stop_on_eos else set()
            for step in range(max_new_tokens):
                fed = int(nxt[0, 0])
                if fed in eos:
                    break
                generated.append(fed)
                step_out, attn = recorder.forward(
                    input_ids=nxt, past_key_values=cache, use_cache=True
                )
                row_logits = step_out.logits[:, -1, :]
                predicted = int(row_logits.argmax(-1)[0])
                trace.steps.append(
                    StepTrace(
                        step=step,
                        fed_token=fed,
                        predicted_token=predicted,
                        attn={layer: tensor[0, :, 0, :].detach() for layer, tensor in attn.items()},
                    )
                )
                nxt = row_logits.argmax(-1, keepdim=True)
        finally:
            set_attn_implementation(model, restore_capture)
    finally:
        if restore is not None:
            set_attn_implementation(model, restore)
    return trace, generated


# --------------------------------------------------------------------------- credit
def credits_from_trace(
    trace: DecodeTrace,
    sample: NeedleSample,
    info: ModelInfo,
    *,
    pairing: str = "next_step",
    sink_position: int = 0,
) -> tuple[dict[HeadRef, set[int]], dict[HeadRef, int], dict[HeadRef, int]]:
    """Apply the paper's two criteria to a trace.

    Returns ``(credits, sink_counts, considered_counts)`` where ``credits`` maps a
    head to the set of needle tokens it copy-pasted.  ``considered_counts`` counts
    the decoding steps at which criterion (1) was even applicable, which is the
    honest normaliser when the model fails to recite the needle.
    """
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")

    needle_set = set(sample.needle_ids)
    start, end = sample.needle_span
    prompt_ids = sample.input_ids[0]

    credits: dict[HeadRef, set[int]] = {h: set() for h in info.scoreable_heads}
    sink_counts: dict[HeadRef, int] = {h: 0 for h in info.scoreable_heads}
    considered: dict[HeadRef, int] = {h: 0 for h in info.scoreable_heads}

    for step in trace.steps:
        token = step.fed_token if pairing == "same_step" else step.predicted_token
        if token not in needle_set:          # criterion (1)
            continue
        for layer, row in step.attn.items():  # row: (heads, kv_len)
            heads = info.num_heads[layer]
            argmax = row.argmax(dim=-1)       # most-attended past position per head
            for head in range(min(heads, row.shape[0])):
                ref = HeadRef(layer, head)
                considered[ref] += 1
                j = int(argmax[head])
                if j == sink_position:
                    sink_counts[ref] += 1
                if j >= end or j < start:     # criterion (2), position inside the needle
                    continue
                if int(prompt_ids[j]) != token:  # criterion (2), same token
                    continue
                credits[ref].add(token)
    return credits, sink_counts, considered


def credits_aligned(
    trace: DecodeTrace,
    sample: NeedleSample,
    info: ModelInfo,
    *,
    pairing: str = "next_step",
) -> dict[HeadRef, set[int]]:
    """Stricter variant: only count tokens at their in-order needle position.

    :func:`credits_from_trace` (the paper) credits any generated token that
    appears anywhere in the needle.  This variant walks the needle and the
    generated sequence together, so a common token such as ``"."`` cannot earn
    credit out of order.  Useful as a robustness check on short needles.
    """
    needle_ids = sample.needle_ids
    # longest in-order alignment between the emitted stream and the needle
    alignment: dict[int, int] = {}
    cursor = 0
    stream = [s.predicted_token for s in trace.steps] if pairing == "next_step" else [
        s.fed_token for s in trace.steps
    ]
    for pos, tok in enumerate(stream):
        if cursor < len(needle_ids) and tok == needle_ids[cursor]:
            alignment[pos] = cursor
            cursor += 1

    needle_set = set(needle_ids)
    start, end = sample.needle_span
    prompt_ids = sample.input_ids[0]
    credits: dict[HeadRef, set[int]] = {h: set() for h in info.scoreable_heads}
    for pos, needle_index in alignment.items():
        step = trace.steps[pos]
        token = stream[pos]
        if token not in needle_set:
            continue
        for layer, row in step.attn.items():
            argmax = row.argmax(dim=-1)
            for head in range(min(info.num_heads[layer], row.shape[0])):
                j = int(argmax[head])
                if start <= j < end and int(prompt_ids[j]) == token:
                    credits[HeadRef(layer, head)].add(token)
    return credits


# --------------------------------------------------------------------------- instance
@dataclass
class InstanceResult:
    """Retrieval scores for one Needle-in-a-Haystack instance."""

    sample: dict[str, Any]
    scores: dict[str, dict[str, float]]
    activations: dict[str, dict[str, float]]
    considered: dict[str, dict[str, int]]
    sink_rate: dict[str, float]
    generated_ids: list[int]
    generated_text: str
    needle_recall: float
    n_steps: int
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample,
            "scores": self.scores,
            "activations": self.activations,
            "considered": self.considered,
            "sink_rate": self.sink_rate,
            "generated_ids": self.generated_ids,
            "generated_text": self.generated_text,
            "needle_recall": self.needle_recall,
            "n_steps": self.n_steps,
            **self.meta,
        }


def needle_recall(generated_text: str, needle_text: str) -> float:
    """Fraction of needle words present, in order, in the generated text.

    A cheap readability diagnostic: it tells us whether the instance is a valid
    retrieval test at all, without being used for scoring.
    """
    needle_words = [w.lower() for w in needle_text.split()]
    if not needle_words:
        return 0.0
    cursor, hit = 0, 0
    for word in generated_text.replace("*", " ").lower().split():
        if cursor < len(needle_words) and word.strip(".,!?;:") == needle_words[cursor].strip(".,!?;:"):
            hit += 1
            cursor += 1
    return hit / len(needle_words)


def score_instance(
    model: Any,
    info: ModelInfo,
    sample: NeedleSample,
    tokenizer: Any = None,
    *,
    max_new_tokens: int = 32,
    pairing: str = "next_step",
    prefill_impl: str = "sdpa",
    capture_impl: str = "eager",
    capture_method: str = "output_attentions",
    compute_second_pairing: bool = True,
    prefill_chunk: int | None = None,
) -> InstanceResult:
    """Run one NIAH instance and return its per-head retrieval scores."""
    trace, generated = decode_with_attention(
        model, info, sample.input_ids,
        max_new_tokens=max_new_tokens, tokenizer=tokenizer,
        prefill_impl=prefill_impl, capture_impl=capture_impl, capture_method=capture_method,
        prefill_chunk=prefill_chunk,
    )

    pairings = [pairing]
    if compute_second_pairing:
        pairings += [p for p in PAIRINGS if p != pairing]

    denom = max(sample.n_unique_needle_tokens, 1)
    scores: dict[str, dict[str, float]] = {}
    activations: dict[str, dict[str, float]] = {}
    considered_out: dict[str, dict[str, int]] = {}
    sink_rates: dict[str, float] = {}

    for p in pairings:
        credits, sinks, considered = credits_from_trace(trace, sample, info, pairing=p)
        scores[p] = {str(h): len(credits[h]) / denom for h in info.scoreable_heads}
        activations[p] = {str(h): (1.0 if credits[h] else 0.0) for h in info.scoreable_heads}
        considered_out[p] = {str(h): considered[h] for h in info.scoreable_heads}
        if p == pairing:
            total_sink = sum(sinks.values())
            total_considered = sum(considered.values()) or 1
            sink_rates = {str(h): sinks[h] / max(considered[h], 1) for h in info.scoreable_heads}
            sink_rates["__overall__"] = total_sink / total_considered

    text = tokenizer.decode(generated, skip_special_tokens=True) if tokenizer is not None else ""
    return InstanceResult(
        sample=sample.as_dict(),
        scores=scores,
        activations=activations,
        considered=considered_out,
        sink_rate=sink_rates,
        generated_ids=generated,
        generated_text=text,
        needle_recall=needle_recall(text, sample.needle_text),
        n_steps=len(trace.steps),
        meta={"pairing": pairing, "eos_reached": len(generated) < max_new_tokens},
    )


# --------------------------------------------------------------------------- aggregate
@dataclass
class RetrievalScores:
    """Per-head retrieval scores averaged over many NIAH instances.

    Scores are kept as dense ``[num_layers, max_heads]`` matrices with NaN for
    non-scoreable entries, which is what the paper's layer x head heatmaps show.
    """

    info: ModelInfo
    score: torch.Tensor
    activation_freq: torch.Tensor
    n_instances: int
    pairing: str = "next_step"
    threshold: float = 0.1
    per_instance: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    @property
    def scoreable_mask(self) -> torch.Tensor:
        return self.info.scoreable_mask()

    def head_score(self, head: HeadRef) -> float:
        return float(self.score[head.layer, head.head])

    def ranked_heads(self) -> list[HeadRef]:
        """All scoreable heads sorted by retrieval score, descending."""
        items = [(self.head_score(h), h) for h in self.info.scoreable_heads]
        items.sort(key=lambda pair: (-pair[0], pair[1].layer, pair[1].head))
        return [h for _, h in items]

    def heads_above(self, threshold: float | None = None) -> list[HeadRef]:
        t = self.threshold if threshold is None else threshold
        return [h for h in self.info.scoreable_heads if self.head_score(h) > t]

    def top_k(self, k: int) -> list[HeadRef]:
        return self.ranked_heads()[:k]

    def random_heads(self, k: int, seed: int = 0) -> list[HeadRef]:
        rng = np.random.default_rng(seed)
        heads = self.info.scoreable_heads
        idx = rng.permutation(len(heads))[:k]
        return [heads[i] for i in idx]

    def sparsity(self, thresholds: Sequence[float] = (0.0, 0.1, 0.5)) -> dict[str, Any]:
        """Fractions of heads at/above each threshold -- the paper's Fig. 2 stats."""
        total = self.info.n_scoreable_heads
        values = torch.tensor([self.head_score(h) for h in self.info.scoreable_heads])
        out: dict[str, Any] = {"n_heads": total, "thresholds": {}}
        for t in thresholds:
            above = int((values > t).sum())
            out["thresholds"][str(t)] = {"n": above, "frac": above / total if total else 0.0}
        return out

    def activation_gap(self, top_k: int = 20) -> list[dict[str, float]]:
        """Score vs activation frequency for the top heads (paper's Fig. 3 gap)."""
        rows = []
        for head in self.ranked_heads()[:top_k]:
            rows.append({
                "head": str(head),
                "layer": head.layer,
                "head_index": head.head,
                "score": self.head_score(head),
                "activation_freq": float(self.activation_freq[head.layer, head.head]),
            })
        return rows

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            score=self.score.numpy(),
            activation_freq=self.activation_freq.numpy(),
            scoreable_mask=self.scoreable_mask.numpy(),
        )
        save_json(
            {
                "model": self.info.as_dict(),
                "pairing": self.pairing,
                "threshold": self.threshold,
                "n_instances": self.n_instances,
                "sparsity": self.sparsity(),
                "top_heads": [str(h) for h in self.ranked_heads()[:100]],
                "meta": self.meta,
            },
            path.with_suffix(".json"),
        )
        log.info("saved retrieval scores to %s(.npz|.json)", path)

    @classmethod
    def load(cls, path: str | Path, info: ModelInfo | None = None) -> "RetrievalScores":
        """Reload a saved run; ``info`` is rebuilt from the JSON sidecar if omitted."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        data = np.load(path)
        meta_path = path.with_suffix(".json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if info is None:
            if not meta.get("model"):
                raise ValueError(f"{meta_path} has no model metadata; pass info= explicitly")
            info = ModelInfo.from_dict(meta["model"])
        return cls(
            info=info,
            score=torch.from_numpy(data["score"]),
            activation_freq=torch.from_numpy(data["activation_freq"]),
            n_instances=int(meta.get("n_instances", 0)),
            pairing=meta.get("pairing", "next_step"),
            threshold=float(meta.get("threshold", 0.1)),
            meta=meta.get("meta", {}),
        )


def aggregate_scores(
    results: Iterable[InstanceResult],
    info: ModelInfo,
    *,
    pairing: str = "next_step",
    threshold: float = 0.1,
    keep_instances: bool = True,
) -> RetrievalScores:
    """Average per-instance retrieval scores into dense layer x head matrices."""
    # Accumulate in zeros, then hide non-scoreable entries behind NaN.  (Adding
    # into a NaN-filled matrix would poison every entry it touches.)
    score = torch.zeros((info.num_layers, info.max_heads), dtype=torch.float32)
    activation = torch.zeros((info.num_layers, info.max_heads), dtype=torch.float32)
    collected: list[dict[str, Any]] = []
    n = 0
    for result in results:
        n += 1
        if keep_instances:
            collected.append(result.as_dict())
        for head in info.scoreable_heads:
            key = str(head)
            score[head.layer, head.head] += result.scores[pairing][key]
            activation[head.layer, head.head] += result.activations[pairing][key]
    if n == 0:
        raise ValueError("no instances to aggregate")
    score /= n
    activation /= n
    score[~info.scoreable_mask()] = float("nan")
    activation[~info.scoreable_mask()] = float("nan")
    return RetrievalScores(
        info=info,
        score=score,
        activation_freq=activation,
        n_instances=n,
        pairing=pairing,
        threshold=threshold,
        per_instance=collected,
    )
