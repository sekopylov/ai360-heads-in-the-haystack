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

Both are computed from the same decoding pass and stored together, so the choice
is a reporting decision rather than a compute decision.  (`next_step` does need
one extra single-token forward for the prefill row that produces the first
generated token; see :func:`decode_with_attention`.)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from retrieval_heads.attention import (
    AttentionRecorder,
    restore_attn_implementation,
    set_attn_implementation,
)
from retrieval_heads.generation import prefill_cache
from retrieval_heads.haystack import NeedleSample
from retrieval_heads.models import ModelInfo, model_device
from retrieval_heads.provenance import add_provenance, warn_if_stale
from retrieval_heads.utils import HeadRef, eos_ids, get_logger, save_json

log = get_logger("scoring")

PAIRINGS = ("next_step", "same_step")


# --------------------------------------------------------------------------- trace
def argmax_positions(attn: dict[int, torch.Tensor], *, prompt_len: int,
                     domain: str = "prompt") -> dict[int, torch.Tensor]:
    """Per-head most-attended key position for each layer.

    ``domain="prompt"`` restricts the argmax to the *input* positions, which is the
    paper's criterion (2): "the input token that receives the most attention
    probability mass" (``a in R^{|x|}``).  ``domain="full"`` lets already-generated
    positions win too, which makes the score depend on how much was generated.
    """
    if domain not in ("prompt", "full"):
        raise ValueError(f"argmax_domain must be 'prompt' or 'full', got {domain!r}")
    out: dict[int, torch.Tensor] = {}
    for layer, tensor in attn.items():
        row = tensor[0, :, 0, :].detach()
        keys = row[:, :prompt_len] if domain == "prompt" else row
        out[layer] = keys.argmax(dim=-1)
    return out


@dataclass
class StepTrace:
    """One decoding step: what it produced, and where each head attended.

    The scoring only ever needs the per-head ``argmax`` of each attention row, so
    ``argmax`` is what the decoder stores by default; ``attn`` keeps the full
    ``(heads, kv_len)`` rows only when a caller passes ``store_rows=True`` (the
    case-study figure needs the real distribution).  At 49K/48 steps the dense
    model's full rows are ~4 GiB, while the indices are kilobytes.
    """

    step: int
    fed_token: int                 # u_t: the token whose query position we are at
    predicted_token: int           # w_{t+1}: what this step predicts
    #: layer -> (heads, kv_len) attention rows; empty unless rows were stored.
    attn: dict[int, torch.Tensor] = field(default_factory=dict)
    #: Pairings this step contributes to; ``None`` means both.  The prefill row
    #: that *produces* the first generated token belongs to ``next_step`` only:
    #: for ``same_step`` the relevant row is that token's own position, which is
    #: the first decode step instead.  Without this the two pairings silently
    #: scored different token sets (``next_step`` could never credit the first
    #: generated token).
    applies_to: tuple[str, ...] | None = None
    #: layer -> (heads,) int64 argmax positions; empty when only rows are stored.
    argmax: dict[int, torch.Tensor] = field(default_factory=dict)

    def positions(self) -> dict[int, torch.Tensor]:
        """Per-head most-attended position, from the stored indices or the rows."""
        if self.argmax:
            return self.argmax
        return {layer: row.argmax(dim=-1) for layer, row in self.attn.items()}


@dataclass
class DecodeTrace:
    steps: list[StepTrace] = field(default_factory=list)
    prompt_len: int = 0
    prefill_logits: torch.Tensor | None = None
    #: True when decoding stopped because it produced an EOS id, False when it ran
    #: into ``max_new_tokens``.  This is the real signal; the old
    #: ``len(generated) < max_new_tokens`` guess mislabelled an EOS on the final
    #: step as truncation.
    stopped_on_eos: bool = False
    #: Which positions the argmax could choose from ("prompt" = the paper's input
    #: tokens; "full" = input + already-generated).
    argmax_domain: str = "prompt"

    @property
    def kv_len(self) -> int:
        """Context length reached after decoding: prompt + generated tokens.

        The prefill row is recorded with ``step == -1`` and adds no token, so it
        must not be counted (the previous ``prompt_len + len(steps)`` did).  Using
        the step index rather than ``applies_to`` keeps this right even when the
        last decode step is scoped to a single pairing.
        """
        generated = sum(1 for step in self.steps if step.step >= 0)
        return self.prompt_len + generated


@torch.no_grad()
def decode_with_attention(
    model: Any,
    info: ModelInfo,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    tokenizer: Any = None,
    prefill_impl: str = "sdpa",
    capture_impl: str = "eager",
    capture_method: str = "patch",
    stop_on_eos: bool = True,
    prefill_chunk: int | None = None,
    store_rows: bool = False,
    argmax_domain: str = "prompt",
) -> tuple[DecodeTrace, list[int]]:
    """Greedy-decode ``input_ids`` while recording attention at every step.

    The prefill runs on a cheap kernel (``sdpa``) because attention maps are not
    needed there; only the single-token steps are captured.  That keeps memory
    flat in the context length -- each captured row is ``(heads, kv_len)``, not
    ``(heads, seq, seq)``.

    The **last prompt token** is the exception: it is fed on its own under the
    capture kernel, because its attention row is the one that produces the first
    generated token.  Capturing it costs one extra single-token forward (the same
    ``(heads, kv_len)`` row as any decode step) and makes ``next_step`` and
    ``same_step`` cover exactly the same generated stream -- previously
    ``next_step`` could never credit the first generated token while ``same_step``
    always could.  When the run stops on ``max_new_tokens``, the final decode row
    predicted a token that was never emitted, so it is scoped to ``same_step`` and
    ``next_step`` does not credit it.

    ``prefill_chunk`` bounds the prefill the other way.  SDPA only reaches a
    memory-efficient kernel when the dtype and shapes allow it; with float32 it
    can fall back to the math backend, which materialises the full
    ``(heads, seq, seq)`` matrix.  Measured on an L4: a 16K-token fp32 prefill
    asked for a single 20.6 GiB allocation and OOM'd on a 22 GiB card.  Feeding
    the prompt in chunks through the KV cache keeps peak memory at
    ``O(chunk^2)`` instead, at the cost of a few extra forward passes.
    """
    if max_new_tokens <= 0:
        raise ValueError(
            f"max_new_tokens must be positive (got {max_new_tokens}); with no decode step "
            f"the two pairings cannot both cover exactly the generated stream"
        )
    if prefill_chunk is not None and prefill_chunk <= 0:
        raise ValueError("prefill_chunk must be positive or None")
    if input_ids.shape[0] != 1:
        raise ValueError(
            f"decode_with_attention handles batch=1 only (got {input_ids.shape[0]}); "
            f"the attention rows are indexed as [0, :, 0, :]"
        )
    input_ids = input_ids.to(model_device(model))
    prompt_len = int(input_ids.shape[1])
    recorder = AttentionRecorder(model, info, method=capture_method)
    trace = DecodeTrace(prompt_len=prompt_len, argmax_domain=argmax_domain)

    def capture(attn_tensors: dict[int, torch.Tensor]):
        """(rows, argmax): rows only when asked for, the argmax always."""
        indices = argmax_positions(attn_tensors, prompt_len=prompt_len,
                                   domain=argmax_domain)
        rows = ({layer: tensor[0, :, 0, :].detach()
                 for layer, tensor in attn_tensors.items()} if store_rows else {})
        return rows, indices

    restore = set_attn_implementation(model, prefill_impl)
    try:
        # Everything but the last prompt token: cheap kernel, chunked if asked.
        # A one-token prompt has no body and its single row is both prefill and
        # capture, which the branch below handles directly.
        if input_ids.shape[1] > 1:
            body, last = input_ids[:, :-1], input_ids[:, -1:]
        else:
            body, last = None, input_ids

        cache = None
        if body is not None:
            if prefill_chunk is not None and body.shape[1] > prefill_chunk:
                log.info("chunked prefill: %d tokens in chunks of %d",
                         input_ids.shape[1], prefill_chunk)
            # Shared with masking/downstream, so the chunking bound cannot drift
            # between the three generation paths.
            cache, _ = prefill_cache(model, body, prefill_chunk=prefill_chunk)

        restore_capture = set_attn_implementation(model, capture_impl)
        try:
            last_out, last_attn = recorder.forward(
                input_ids=last, past_key_values=cache, use_cache=True
            )
            logits = last_out.logits[:, -1, :]
            cache = last_out.past_key_values
            trace.prefill_logits = logits.detach()
            last_rows, last_idx = capture(last_attn)
            trace.steps.append(
                StepTrace(
                    step=-1,
                    fed_token=int(last[0, -1]),
                    predicted_token=int(logits.argmax(-1)[0]),
                    attn=last_rows,
                    applies_to=("next_step",),
                    argmax=last_idx,
                )
            )

            generated: list[int] = []
            nxt = logits.argmax(-1, keepdim=True)
            eos = eos_ids(model, tokenizer) if stop_on_eos else set()
            for step in range(max_new_tokens):
                fed = int(nxt[0, 0])
                if fed in eos:
                    trace.stopped_on_eos = True
                    break
                generated.append(fed)
                step_out, attn = recorder.forward(
                    input_ids=nxt, past_key_values=cache, use_cache=True
                )
                # Same reason as in generation.greedy_ids: do not assume the cache
                # mutates in place.
                cache = step_out.past_key_values
                row_logits = step_out.logits[:, -1, :]
                predicted = int(row_logits.argmax(-1)[0])
                step_rows, step_idx = capture(attn)
                trace.steps.append(
                    StepTrace(
                        step=step,
                        fed_token=fed,
                        predicted_token=predicted,
                        attn=step_rows,
                        argmax=step_idx,
                    )
                )
                nxt = row_logits.argmax(-1, keepdim=True)
        finally:
            restore_attn_implementation(model, restore_capture)
    finally:
        restore_attn_implementation(model, restore)
    # Scope every row by what was actually generated, not by which way the loop
    # exited.  A row is `next_step`-valid only if the token it predicted was really
    # fed, and `same_step`-valid only if the token it consumed was really generated:
    #   * the prefill row predicts the first generated token -- unless the model
    #     stopped immediately, when nothing was generated at all;
    #   * a decode row predicts the following token, except the last one, whose
    #     prediction was never fed: either the budget ran out or it was the EOS that
    #     broke the loop.  Conditioning this on `not stopped_on_eos` (as it was) left
    #     the common EOS path with a `next_step` stream one token longer than
    #     `generated`;
    #   * every decode row consumed a generated token, so each stays valid for
    #     `same_step` (the last one *only* for `same_step`).
    decode_steps = [step for step in trace.steps if step.step >= 0]
    for index, step in enumerate(decode_steps):
        step.applies_to = ("same_step",) if index + 1 == len(decode_steps) else None
    for step in trace.steps:
        if step.step < 0:
            step.applies_to = ("next_step",) if decode_steps else ()
    return trace, generated


# --------------------------------------------------------------------------- credit
def match_masks(argmax: torch.Tensor, prompt_ids: torch.Tensor, token: int,
                span: tuple[int, int], sink_position: int,
                heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head ``(matched, sink)`` booleans for one layer-step, computed on the host.

    Vectorised on purpose: the previous per-head Python loop called
    ``int(argmax[head])`` for every head of every step, which is a device->host
    synchronisation per head (hundreds per step on the dense control).  Here it is
    one transfer per layer-step, and criterion (2) is evaluated with tensor ops.

    ``argmax`` may point outside the prompt when ``argmax_domain="full"``, so the
    index is clamped before indexing ``prompt_ids`` and the out-of-span positions
    are masked out afterwards.
    """
    idx = argmax[:heads].detach().to("cpu", torch.long)
    start, end = span
    inside = (idx >= start) & (idx < end)
    if prompt_ids.numel():
        same = prompt_ids[idx.clamp(0, prompt_ids.numel() - 1)] == token
        matched = inside & same
    else:  # pragma: no cover - a sample always has prompt tokens
        matched = torch.zeros_like(inside)
    return matched, idx == sink_position


def _per_head(counts: dict[int, torch.Tensor], info: ModelInfo) -> dict[HeadRef, int]:
    """Materialise the per-layer accumulators as the public per-head mapping."""
    out: dict[HeadRef, int] = {h: 0 for h in info.scoreable_heads}
    for layer, values in counts.items():
        for head, value in enumerate(values.tolist()):
            out[HeadRef(layer, head)] = int(value)
    return out


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
    honest normaliser when the model fails to recite the needle.  The sink rate is
    therefore ``P(argmax == prompt position 0 | criterion (1) held)``, not a share of
    *all* decoding steps -- and "position 0" is the first prompt token, which under a
    chat template is a template token rather than necessarily a BOS sink.
    """
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")

    # ONE definition of "a needle token": the tokenization of the needle text (the
    # paper's k).  An earlier version unioned this with the prompt-span ids, which
    # (a) credited the fused boundary token `".\n"` that is not a needle token and
    # (b) could never credit `"."`; `max_attainable_score` now records the ceiling.
    needle_set = set(getattr(sample, "needle_text_ids", None) or sample.needle_ids)
    start, end = sample.needle_span
    prompt_ids = sample.input_ids[0].detach().to("cpu")

    # Accumulate per layer on the host, in whole-head vectors.  The old shape of
    # this loop called `int(argmax[head])` once per head per step, i.e. a
    # device->host synchronisation for every head of every step (~448 x 48 x 75 x 4
    # transfers on the dense control).  Now there is one `.cpu()` per layer-step and
    # one `nonzero()` for the heads that actually matched.
    considered_t: dict[int, torch.Tensor] = {}
    sink_t: dict[int, torch.Tensor] = {}
    hits: list[tuple[int, int, int]] = []

    for step in trace.steps:
        if step.applies_to is not None and pairing not in step.applies_to:
            continue
        token = step.fed_token if pairing == "same_step" else step.predicted_token
        if token not in needle_set:          # criterion (1)
            continue
        for layer, argmax in step.positions().items():  # argmax: (heads,)
            if layer not in info.num_heads:
                # The patch capture stores whatever called eager_attention_forward,
                # which can include a module (e.g. a vision tower) this model does
                # not score.  Skip it rather than KeyError.
                continue
            # Symmetric check: `min()` used to drop extra rows silently when the
            # capture reported more heads than the metadata (the recorder validates
            # equality at capture time, so a mismatch here means a hand-built trace).
            n_heads = info.num_heads[layer]
            if argmax.shape[0] != n_heads:
                raise ValueError(
                    f"layer {layer} reported {argmax.shape[0]} attention rows but the "
                    f"model metadata says {n_heads}; the capture and the model "
                    f"disagree, and this cannot change mid-run"
                )
            counts = considered_t.get(layer)
            if counts is None:
                counts = torch.zeros(n_heads, dtype=torch.long)
                considered_t[layer] = counts
            counts += 1
            matched, sink = match_masks(argmax, prompt_ids, token, (start, end),
                                        sink_position, n_heads)
            # `setdefault` builds its default eagerly, i.e. one zeros() per layer-step
            # for nothing; the dict lookup does not.
            sink_counts = sink_t.get(layer)
            if sink_counts is None:
                sink_counts = torch.zeros(n_heads, dtype=torch.long)
                sink_t[layer] = sink_counts
            sink_counts += sink.long()
            hits.extend((layer, int(head), token)
                        for head in matched.nonzero(as_tuple=False).flatten().tolist())

    credits: dict[HeadRef, set[int]] = {h: set() for h in info.scoreable_heads}
    for layer, head, token in hits:
        credits[HeadRef(layer, head)].add(token)
    sink_counts = _per_head(sink_t, info)
    considered = _per_head(considered_t, info)
    return credits, sink_counts, considered


def credits_aligned(
    trace: DecodeTrace,
    sample: NeedleSample,
    info: ModelInfo,
    *,
    pairing: str = "next_step",
) -> dict[HeadRef, set[int]]:
    """Stricter variant: the *longest common subsequence* alignment with the needle.

    :func:`credits_from_trace` (the paper) credits any generated token that appears
    anywhere in the needle.  This variant requires the tokens to form an in-order
    subsequence, so a common token such as ``"."`` cannot earn credit out of order.
    Being a true LCS (not a walk anchored on the needle's first token) it correctly
    credits a correct answer that starts mid-needle.
    """
    # The same "needle token" set as `credits_from_trace`: walking the prompt-span
    # ids instead made this variant systematically lower for a reason that had
    # nothing to do with in-order strictness (the span's last token could be fused).
    needle_ids = list(getattr(sample, "needle_text_ids", None) or sample.needle_ids)
    # Steps this pairing actually scores (the prefill row that produces the first
    # generated token belongs to next_step only), so stream positions line up
    # with the step list used below.
    steps = [s for s in trace.steps if s.applies_to is None or pairing in s.applies_to]
    stream = [s.predicted_token for s in steps] if pairing == "next_step" else [
        s.fed_token for s in steps
    ]
    # Longest common subsequence: an in-order match that is not anchored on the
    # needle's first token (the old cursor walk was, so a sub-span answer -- which is
    # what the questions ask for -- scored ~0 here).
    alignment = dict(lcs_alignment(stream, needle_ids))

    start, end = sample.needle_span
    prompt_ids = sample.input_ids[0].detach().to("cpu")
    hits: list[tuple[int, int, int]] = []
    for pos, needle_index in alignment.items():
        step = steps[pos]
        token = stream[pos]
        # `token == needle_ids[needle_index]` by construction of the walk, so there
        # is nothing to filter here (the old `if token not in needle_set` was dead).
        for layer, argmax in step.positions().items():
            if layer not in info.num_heads:
                continue
            heads = min(info.num_heads[layer], argmax.shape[0])
            matched, _sink = match_masks(argmax, prompt_ids, token, (start, end),
                                         0, heads)
            hits.extend((layer, int(head), token)
                        for head in matched.nonzero(as_tuple=False).flatten().tolist())
    credits: dict[HeadRef, set[int]] = {h: set() for h in info.scoreable_heads}
    for layer, head, token in hits:
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
    #: pairing -> {head: sink rate} plus "__overall__".  Per pairing because the
    #: scored steps differ between them, so one shared value was wrong for the
    #: secondary summary.
    sink_rate: dict[str, dict[str, float]]
    generated_ids: list[int]
    generated_text: str
    needle_recall: float
    n_steps: int
    #: :func:`credits_aligned` scores, keyed by pairing.  The strict in-order
    #: variant is a robustness check on the loose set-based rule, so it is stored
    #: alongside the paper's score rather than recomputed on demand.
    aligned_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    #: pairing -> {head: sorted copied token ids} (sparse: only non-empty credits).
    copied_tokens: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        # `meta` first: a meta key that collides with a reserved name used to
        # overwrite it (a silent footgun, even if nothing collides today).
        return {
            **self.meta,
            "sample": self.sample,
            "scores": self.scores,
            "aligned_scores": self.aligned_scores,
            "copied_tokens": self.copied_tokens,
            "activations": self.activations,
            "considered": self.considered,
            "sink_rate": self.sink_rate,
            "generated_ids": self.generated_ids,
            "generated_text": self.generated_text,
            "needle_recall": self.needle_recall,
            "n_steps": self.n_steps,
        }


def _norm_word(word: str) -> str:
    return word.strip(".,!?;:").lower()


def lcs_length(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Length of the longest common subsequence (rolling-row DP)."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for item in a:
        current = [0]
        for j, other in enumerate(b):
            if item == other:
                current.append(previous[j] + 1)
            else:
                current.append(max(previous[j + 1], current[j]))
        previous = current
    return previous[-1]


def lcs_alignment(a: Sequence[Any], b: Sequence[Any]) -> list[tuple[int, int]]:
    """In-order pairs ``(i, j)`` of one longest common subsequence of ``a`` and ``b``."""
    n, m = len(a), len(b)
    if not n or not m:
        return []
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, nxt = table[i], table[i + 1]
        for j in range(m - 1, -1, -1):
            row[j] = nxt[j + 1] + 1 if a[i] == b[j] else max(nxt[j], row[j + 1])
    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def needle_recall(generated_text: str, needle_text: str) -> float:
    """Fraction of the needle recovered in order (longest common subsequence).

    A cheap validity diagnostic for the instance, not part of the score.  It must be
    a *subsequence*, not a prefix: the questions ask for a sub-span of the needle, so
    a correct extractive answer starts mid-needle.  The old prefix-anchored walk gave
    such answers 0.0 and silently dropped them from the recited-only matrices.
    """
    needle_words = [word for word in (_norm_word(w) for w in needle_text.split()) if word]
    if not needle_words:
        return 0.0
    # Drop pure-punctuation "words": they normalise to "" and would match each
    # other, inflating the recall on punctuation noise.
    generated_words = [word for word in
                       (_norm_word(w) for w in generated_text.replace("*", " ").lower().split())
                       if word]
    return lcs_length(needle_words, generated_words) / len(needle_words)


def needle_prefix_recall(generated_text: str, needle_text: str) -> float:
    """The previous measure: needle words reproduced *from its first word*.

    Kept as a separate diagnostic because it answers a different question ("did the
    model dictate the needle as a whole?"), which is what `recited` used to mean.
    """
    needle_words = [_norm_word(word) for word in needle_text.split()]
    if not needle_words:
        return 0.0
    cursor = hits = 0
    for word in generated_text.replace("*", " ").lower().split():
        if cursor < len(needle_words) and _norm_word(word) == needle_words[cursor]:
            hits += 1
            cursor += 1
    return hits / len(needle_words)


@torch.no_grad()
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
    capture_method: str = "patch",
    prefill_chunk: int | None = None,
    argmax_domain: str = "prompt",
) -> InstanceResult:
    """Run one NIAH instance and return its per-head retrieval scores."""
    trace, generated = decode_with_attention(
        model, info, sample.input_ids,
        max_new_tokens=max_new_tokens, tokenizer=tokenizer,
        prefill_impl=prefill_impl, capture_impl=capture_impl, capture_method=capture_method,
        prefill_chunk=prefill_chunk,
        argmax_domain=argmax_domain,
    )

    pairings = [pairing]
    # Both pairings always come from the same pass: an option to skip the second one
    # left `scores`/`aligned_scores` without a key that summary()/save() read.
    pairings += [p for p in PAIRINGS if p != pairing]

    # Denominator over the needle *text*: unique tokens of the prompt span would
    # count the fused boundary token and cap the achievable score below 1.
    text_ids = getattr(sample, "needle_text_ids", None) or sample.needle_ids
    denom = max(len(set(text_ids)), 1)
    scores: dict[str, dict[str, float]] = {}
    activations: dict[str, dict[str, float]] = {}
    considered_out: dict[str, dict[str, int]] = {}
    aligned_out: dict[str, dict[str, float]] = {}
    sink_rates: dict[str, dict[str, float]] = {}
    # Sparse audit trail of the numerator |g_h ∩ k| per head (only heads that copied
    # something, so it is bounded by the number of retrieving heads).  The paper's
    # Fig. 3 claims are about *tokens*, which an aggregate score cannot answer.
    copied_tokens: dict[str, dict[str, list[int]]] = {}

    for p in pairings:
        credits, sinks, considered = credits_from_trace(trace, sample, info, pairing=p)
        # The strict in-order variant costs nothing extra (no forward passes), so
        # it is stored next to the paper's rule for every run.
        aligned = credits_aligned(trace, sample, info, pairing=p)
        scores[p] = {str(h): len(credits[h]) / denom for h in info.scoreable_heads}
        aligned_out[p] = {str(h): len(aligned[h]) / denom for h in info.scoreable_heads}
        activations[p] = {str(h): (1.0 if credits[h] else 0.0) for h in info.scoreable_heads}
        considered_out[p] = {str(h): considered[h] for h in info.scoreable_heads}
        total_sink = sum(sinks.values())
        total_considered = sum(considered.values()) or 1
        copied_tokens[p] = {str(h): sorted(credits[h])
                            for h in info.scoreable_heads if credits[h]}
        rates = {str(h): sinks[h] / max(considered[h], 1) for h in info.scoreable_heads}
        rates["__overall__"] = total_sink / total_considered
        sink_rates[p] = rates

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
        n_steps=len(generated),
        aligned_scores=aligned_out,
        copied_tokens=copied_tokens,
        meta={"pairing": pairing, "argmax_domain": argmax_domain,
              "eos_reached": trace.stopped_on_eos,
              "truncated": not trace.stopped_on_eos,
              # The old prefix-anchored diagnostic, kept so the change is auditable.
              "needle_prefix_recall": needle_prefix_recall(text, sample.needle_text)},
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

    def sparsity(self, thresholds: Sequence[float] = (0.0, 0.1, 0.5)) -> dict[str, Any]:
        """Fractions of heads strictly *above* each threshold (paper's Fig. 2 stats).

        Strict on purpose: the paper defines a retrieval head as score > 0.1, so the
        `0.0` bucket counts "any credit at all", not "every head".
        """
        total = self.info.n_scoreable_heads
        values = torch.tensor([self.head_score(h) for h in self.info.scoreable_heads])
        out: dict[str, Any] = {"n_heads": total, "thresholds": {}}
        for t in thresholds:
            above = int((values > t).sum())
            out["thresholds"][str(t)] = {"n": above, "frac": above / total if total else 0.0}
        return out

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        # Normalise first: the temp-rename below must target the same name that
        # `load` will look for (np.savez used to add the .npz suffix itself).
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        import tempfile

        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".", suffix=".npz")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            np.savez_compressed(
                tmp,
                score=self.score.numpy(),
                activation_freq=self.activation_freq.numpy(),
                scoreable_mask=self.scoreable_mask.numpy(),
            )
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        save_json(
            add_provenance({
                "model": self.info.as_dict(),
                "pairing": self.pairing,
                "threshold": self.threshold,
                "n_instances": self.n_instances,
                "sparsity": self.sparsity(),
                "top_heads": [str(h) for h in self.ranked_heads()[:100]],
                "meta": self.meta,
            }, dtype=self.info.dtype),
            path.with_suffix(".json"),
        )
        log.info("saved retrieval scores to %s(.npz|.json)", path)

    @classmethod
    def load(cls, path: str | Path, info: ModelInfo | None = None) -> "RetrievalScores":
        """Reload a saved run; ``info`` is rebuilt from the JSON sidecar if omitted."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        if not path.exists():
            raise FileNotFoundError(
                f"no saved retrieval scores at {path}. Run `detect` for this model first "
                f"(e.g. `python -m retrieval_heads.cli detect --model <name> --profile laptop "
                f"--out {path.parent}`); the .npz matrices are generated, not committed for "
                f"the `results/` tree."
            )
        # Context-manage the archive: leaving the file handle open kept the .npz
        # locked on Windows and leaked a descriptor per load.
        with np.load(path) as archive:
            score = torch.from_numpy(np.array(archive["score"]))
            activation = torch.from_numpy(np.array(archive["activation_freq"]))
            saved_mask = (torch.from_numpy(np.array(archive["scoreable_mask"]))
                          if "scoreable_mask" in archive.files else None)
        meta_path = path.with_suffix(".json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            warn_if_stale(meta, str(path), log=log)
        if info is None:
            if not meta.get("model"):
                raise ValueError(f"{meta_path} has no model metadata; pass info= explicitly")
            info = ModelInfo.from_dict(meta["model"])
        expected_shape = (info.num_layers, info.max_heads)
        if tuple(score.shape) != expected_shape or tuple(activation.shape) != expected_shape:
            raise ValueError(
                f"{path} holds {tuple(score.shape)}/{tuple(activation.shape)} matrices but "
                f"the metadata describes {expected_shape}; the artifact belongs to another "
                f"model"
            )
        if saved_mask is None and meta.get("schema_version", 0) >= 5:
            raise ValueError(
                f"{path} has schema_version {meta.get('schema_version')} but no "
                f"scoreable_mask; refusing to guess which entries are real"
            )
        if saved_mask is not None and not torch.equal(saved_mask, info.scoreable_mask()):
            # The mask was once written and never checked; a sidecar that drifted
            # from its `info` would silently mis-place every head of every ablation.
            raise ValueError(
                f"{path} stores a scoreable_mask that differs from the one rebuilt from "
                f"its metadata; the artifact and the model do not match"
            )
        return cls(
            info=info,
            score=score,
            activation_freq=activation,
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
) -> RetrievalScores:
    """Average per-instance retrieval scores into dense layer x head matrices."""
    # Accumulate in zeros, then hide non-scoreable entries behind NaN.  (Adding
    # into a NaN-filled matrix would poison every entry it touches.)
    score = torch.zeros((info.num_layers, info.max_heads), dtype=torch.float32)
    activation = torch.zeros((info.num_layers, info.max_heads), dtype=torch.float32)
    n = 0
    for result in results:
        n += 1
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
    )
