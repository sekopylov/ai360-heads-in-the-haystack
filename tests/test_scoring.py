"""Unit tests for the retrieval score -- the two copy-paste criteria.

These build synthetic traces, so they pin the *definition* without needing a
checkpoint, and they cover the pairing ambiguity explicitly (a head can only be
scored against the attention row that produces the token, or against the row at
the token's own position -- the two must not be silently conflated).
"""

from __future__ import annotations

import pytest
import torch

from retrieval_heads.models import ModelInfo
from retrieval_heads.scoring import (
    argmax_positions,
    DecodeTrace,
    StepTrace,
    credits_aligned,
    credits_from_trace,
    needle_recall,
)
from retrieval_heads.utils import HeadRef


def make_info(num_layers: int = 2, heads: int = 2) -> ModelInfo:
    return ModelInfo(
        name="toy",
        path="toy",
        model_type="toy",
        num_layers=num_layers,
        layer_types=["full_attention"] * num_layers,
        num_heads={i: heads for i in range(num_layers)},
        num_kv_heads={i: heads for i in range(num_layers)},
        head_dim=4,
        hidden_size=8,
        max_position_embeddings=128,
        # Was `{i: None}`, which would crash confusingly in HeadMasker; these tests
        # never mask, but a future one would trip over it.
        attention_modules={i: _ToyAttention() for i in range(num_layers)},
    )


class _ToyAttention:
    """Placeholder module object: only identity matters to these tests."""


class FakeSample:
    """Minimal stand-in for NeedleSample."""

    def __init__(self, prompt_ids, needle_span):
        self.input_ids = torch.tensor([prompt_ids])
        self.needle_span = needle_span
        self.needle_ids = prompt_ids[needle_span[0]:needle_span[1]]
        self.needle_text = "a b"
        self._as_dict = {"prompt_tokens": len(prompt_ids)}

    def as_dict(self):
        return self._as_dict


def spike(n_heads: int, kv_len: int, positions: dict[int, int]) -> torch.Tensor:
    """Attention rows of shape (heads, kv_len) with all mass on one position."""
    row = torch.zeros(n_heads, kv_len)
    for head, pos in positions.items():
        row[head, pos] = 1.0
    return row


def test_next_step_credits_the_predicted_token():
    # prompt: positions 0..4, needle occupies [1, 3) and holds tokens 7 and 8
    sample = FakeSample([11, 7, 8, 12, 13], (1, 3))
    info = make_info()
    trace = DecodeTrace(prompt_len=5, steps=[
        # predicts 7; layer 0 head 0 points at position 1 (== 7), head 1 points outside
        StepTrace(step=0, fed_token=99, predicted_token=7, attn={
            0: spike(2, 5, {0: 1, 1: 3}),
            1: spike(2, 5, {0: 0, 1: 4}),
        }),
        # predicts 8; layer 0 head 0 now points at position 2 (== 8)
        StepTrace(step=1, fed_token=7, predicted_token=8, attn={
            0: spike(2, 6, {0: 2, 1: 1}),
            1: spike(2, 6, {0: 0, 1: 2}),
        }),
    ])
    credits, sinks, considered = credits_from_trace(trace, sample, info, pairing="next_step")

    assert credits[HeadRef(0, 0)] == {7, 8}   # both needle tokens, correct positions
    assert credits[HeadRef(0, 1)] == set()    # never inside the needle
    assert credits[HeadRef(1, 0)] == set()    # pinned to the sink position
    assert credits[HeadRef(1, 1)] == {8}      # only the second step matched
    # criterion (1) held at both steps for every head
    assert considered[HeadRef(0, 0)] == 2
    assert sinks[HeadRef(1, 0)] == 2          # head pinned to position 0 the whole time


def test_same_step_credits_the_fed_token():
    """The two pairings must select different heads on the same trace."""
    sample = FakeSample([11, 7, 8, 12, 13], (1, 3))
    info = make_info(num_layers=1, heads=2)
    # At this step the model is *processing* token 7 and predicts 8.
    # Position 1 holds 7, position 2 holds 8.
    trace = DecodeTrace(prompt_len=5, steps=[
        StepTrace(step=0, fed_token=7, predicted_token=8,
                  attn={0: spike(2, 6, {0: 1, 1: 2})}),
    ])
    same, _, considered_same = credits_from_trace(trace, sample, info, pairing="same_step")
    assert same[HeadRef(0, 0)] == {7}       # head 0 points at the token being processed
    assert same[HeadRef(0, 1)] == set()     # head 1 points at 8, not at 7

    nxt, _, _ = credits_from_trace(trace, sample, info, pairing="next_step")
    assert nxt[HeadRef(0, 0)] == set()      # head 0 points at 7, not at the prediction 8
    assert nxt[HeadRef(0, 1)] == {8}        # head 1 points at the token being generated
    assert considered_same[HeadRef(0, 0)] == 1


def test_wrong_token_at_needle_position_is_not_credited():
    """Criterion (2) needs *both* the right position and the same token.

    The prediction (8) *is* a needle token, so criterion (1) passes; head 0's
    argmax sits inside the needle but on the other needle token (7), so only the
    same-token half of criterion (2) can reject it.  (The earlier version of this
    test predicted a token outside the needle, so criterion (1) short-circuited
    and the equality check was never exercised.)
    """
    sample = FakeSample([11, 7, 8, 12, 13], (1, 3))
    info = make_info(num_layers=1, heads=2)
    trace = DecodeTrace(prompt_len=5, steps=[
        StepTrace(step=0, fed_token=1, predicted_token=8,
                  attn={0: spike(2, 5, {0: 1, 1: 2})}),
    ])
    credits, _, _ = credits_from_trace(trace, sample, info, pairing="next_step")
    assert credits[HeadRef(0, 0)] == set()   # inside the needle, but points at 7, not 8
    assert credits[HeadRef(0, 1)] == {8}     # points at 8 and predicts 8


def test_prefill_row_is_scoped_to_next_step():
    """The row that produces the first token must not leak into same_step.

    ``decode_with_attention`` captures the last prompt row under
    ``applies_to=("next_step",)``.  Without that scope the two pairings would
    score different token streams (next_step used to be unable to credit the
    first generated token at all).
    """
    sample = FakeSample([11, 7, 8, 12, 13], (1, 3))   # needle [7, 8]
    info = make_info(num_layers=1, heads=2)
    trace = DecodeTrace(prompt_len=5, steps=[
        # prefill row: predicts 7; head 0 points at position 1 (== 7)
        StepTrace(step=-1, fed_token=13, predicted_token=7,
                  attn={0: spike(2, 5, {0: 1, 1: 2})}, applies_to=("next_step",)),
        # first decode step: fed 7, predicts 8; head 0 points at 7, head 1 at 8
        StepTrace(step=0, fed_token=7, predicted_token=8,
                  attn={0: spike(2, 6, {0: 1, 1: 2})}),
    ])

    nxt, _, _ = credits_from_trace(trace, sample, info, pairing="next_step")
    assert nxt[HeadRef(0, 0)] == {7}         # prefill row credits the first token
    assert nxt[HeadRef(0, 1)] == {8}         # decode step credits the second

    same, _, _ = credits_from_trace(trace, sample, info, pairing="same_step")
    assert same[HeadRef(0, 0)] == {7}        # the prefill row is not scored here
    assert same[HeadRef(0, 1)] == set()      # this row points at 8, fed token is 7

    # The strict variant filters the same way, so its stream positions stay aligned.
    strict_next = credits_aligned(trace, sample, info, pairing="next_step")
    assert strict_next[HeadRef(0, 0)] == {7}
    assert strict_next[HeadRef(0, 1)] == {8}
    strict_same = credits_aligned(trace, sample, info, pairing="same_step")
    assert strict_same[HeadRef(0, 0)] == {7}


def test_denominator_is_unique_needle_tokens():
    """|g_h & k| / |k| must be able to reach 1.0 even if the needle repeats a token."""
    sample = FakeSample([11, 7, 7, 12, 13], (1, 3))   # needle = [7, 7], 1 unique token
    info = make_info(num_layers=1, heads=1)
    trace = DecodeTrace(prompt_len=5, steps=[
        StepTrace(step=0, fed_token=99, predicted_token=7, attn={0: spike(1, 5, {0: 1})}),
    ])
    credits, _, _ = credits_from_trace(trace, sample, info, pairing="next_step")
    assert len(credits[HeadRef(0, 0)]) == 1
    assert len(credits[HeadRef(0, 0)]) / len(set(sample.needle_ids)) == 1.0


def test_aligned_matching_rejects_out_of_order_tokens():
    """The strict variant must not credit a needle token emitted out of order."""
    sample = FakeSample([11, 7, 8, 12, 13], (1, 3))
    info = make_info(num_layers=1, heads=1)
    trace = DecodeTrace(prompt_len=5, steps=[
        # emits 8 first, then 7 -- reverse of the needle order
        StepTrace(step=0, fed_token=1, predicted_token=8, attn={0: spike(1, 5, {0: 2})}),
        StepTrace(step=1, fed_token=8, predicted_token=7, attn={0: spike(1, 6, {0: 1})}),
    ])
    loose, _, _ = credits_from_trace(trace, sample, info, pairing="next_step")
    assert loose[HeadRef(0, 0)] == {7, 8}          # paper's set-based rule
    strict = credits_aligned(trace, sample, info, pairing="next_step")
    # A longest-common-subsequence alignment of needle [7, 8] against the stream
    # [8, 7] has length 1, so exactly one of the two can be credited (which one is a
    # tie); the point is that the out-of-order pair is not both credited.
    assert len(strict[HeadRef(0, 0)]) == 1, strict
    assert strict[HeadRef(0, 0)] <= {7, 8}


def test_credits_ignore_layers_the_model_does_not_score():
    """The `patch` capture can report a module this model has no head count for."""
    sample = FakeSample([11, 7, 8, 12, 13], (1, 3))
    info = make_info(num_layers=1, heads=2)
    trace = DecodeTrace(prompt_len=5, steps=[
        StepTrace(step=0, fed_token=1, predicted_token=7,
                  argmax={0: torch.tensor([1, 3]), 99: torch.tensor([1])}),
    ])
    credits, _, _ = credits_from_trace(trace, sample, info, pairing="next_step")
    assert credits[HeadRef(0, 0)] == {7}
    assert HeadRef(99, 0) not in credits


def test_needle_recall_credits_a_sub_span_answer():
    """The questions ask for a sub-span, so a correct answer starts mid-needle.

    The old prefix-anchored walk scored such answers 0.0 and dropped them from the
    recited-only matrices.
    """
    needle = "The best thing to do in San Francisco is to eat a sandwich in Dolores Park"
    from retrieval_heads.scoring import needle_prefix_recall

    answer = "eat a sandwich in Dolores Park"
    # 12 of the needle's 32 words, in order -- comfortably above RECITED_RECALL
    # (0.3), whereas the prefix-anchored measure is 0.
    assert needle_recall(answer, needle) > 0.3
    assert needle_prefix_recall(answer, needle) == 0.0


def test_needle_recall_is_case_insensitive():
    needle = "The best thing is to eat a sandwich."
    assert needle_recall("The best thing is to eat a sandwich.", needle) == 1.0
    assert needle_recall("the BEST thing is to EAT a sandwich", needle) == 1.0
    assert needle_recall("the best thing", needle) == 3 / 8
    assert needle_recall("nothing relevant here", needle) == 0.0


def test_argmax_domain_can_exclude_generated_positions():
    """The paper's criterion is about the *input* token that gets most attention.

    With the full row, a head whose maximum sits on its own generated token never
    earns credit even when its input maximum is on the needle.
    """
    from retrieval_heads.scoring import argmax_positions

    attn = {0: torch.zeros(1, 1, 1, 5)}
    attn[0][0, 0, 0, 2] = 0.9      # input position 2: inside the needle
    attn[0][0, 0, 0, 4] = 0.95     # generated position 4: dominates the full row
    assert argmax_positions(attn, prompt_len=4, domain="prompt")[0].tolist() == [2]
    assert argmax_positions(attn, prompt_len=4, domain="full")[0].tolist() == [4]
    with pytest.raises(ValueError, match="argmax_domain"):
        argmax_positions(attn, prompt_len=4, domain="nonsense")


def test_haystack_domain_searches_only_the_span_and_returns_absolute_indices():
    """The paper's `a in R^{|x|}` is the haystack, not the whole prompt.

    A template token at position 0 is the single most-attended position for most
    dense-model heads; the haystack domain must ignore it, and the returned index
    must still be absolute (criterion (2) indexes the prompt with it).
    """
    from retrieval_heads.scoring import argmax_positions

    attn = {0: torch.zeros(1, 1, 1, 6)}
    attn[0][0, 0, 0, 0] = 0.99     # template token: wins the prompt domain
    attn[0][0, 0, 0, 4] = 0.9      # inside the haystack span (3, 5)
    assert argmax_positions(attn, prompt_len=6, domain="prompt")[0].tolist() == [0]
    assert argmax_positions(attn, prompt_len=6, domain="haystack",
                            span=(3, 5))[0].tolist() == [4]
    # A span is mandatory, and it must be a non-empty range inside the prompt.
    with pytest.raises(ValueError, match="needs the haystack token span"):
        argmax_positions(attn, prompt_len=6, domain="haystack")
    for bad in ((5, 3), (0, 0), (0, 7), (-1, 3)):
        with pytest.raises(ValueError, match="non-empty range"):
            argmax_positions(attn, prompt_len=6, domain="haystack", span=bad)


def test_prompt_domain_credits_where_full_domain_does_not():
    """Same attention row, two domains: only the prompt one credits the head."""
    from retrieval_heads.scoring import StepTrace

    sample = FakeSample([11, 7, 8, 12, 13], (1, 3))     # needle ids [7, 8]
    info = make_info(num_layers=1, heads=1)
    row = torch.zeros(1, 5)
    row[0, 2] = 0.9          # input position 2 holds needle token 8
    row[0, 4] = 0.95         # generated position dominates the full row

    def credits(domain):
        step = StepTrace(step=0, fed_token=1, predicted_token=8, attn={0: row},
                         argmax=argmax_positions({0: row.unsqueeze(0).unsqueeze(0)},
                                                 prompt_len=4, domain=domain))
        trace = DecodeTrace(prompt_len=4, steps=[step], argmax_domain=domain)
        loose, _, _ = credits_from_trace(trace, sample, info, pairing="next_step")
        return loose[HeadRef(0, 0)]

    assert credits("prompt") == {8}
    assert credits("full") == set()


def test_haystack_domain_credits_and_the_sink_stays_prompt_based():
    """A template sink must not be credited, and must not vanish from `sink_rate`.

    Two claims in one row: under `haystack` the position-0 template token cannot win
    criterion (2) (so the head earns credit it lost under `prompt`), while the sink
    diagnostic still reports position 0 -- otherwise it would be a structural zero
    under the new default.
    """
    from retrieval_heads.scoring import StepTrace

    sample = FakeSample([11, 7, 8, 12, 13], (1, 3))     # needle ids [7, 8]
    info = make_info(num_layers=1, heads=1)
    row = torch.zeros(1, 5)
    row[0, 0] = 0.99         # template token: the sink
    row[0, 2] = 0.9          # needle token 8, the best *haystack* position

    def score(domain, span):
        attn = {0: row.unsqueeze(0).unsqueeze(0)}
        step = StepTrace(step=0, fed_token=1, predicted_token=8, attn={0: row},
                         argmax=argmax_positions(attn, prompt_len=4, domain=domain,
                                                 span=span),
                         argmax_prompt=argmax_positions(attn, prompt_len=4,
                                                        domain="prompt"))
        trace = DecodeTrace(prompt_len=4, steps=[step], argmax_domain=domain,
                            argmax_span=span)
        credits, sinks, considered = credits_from_trace(trace, sample, info,
                                                        pairing="next_step")
        return credits[HeadRef(0, 0)], sinks[HeadRef(0, 0)] / considered[HeadRef(0, 0)]

    assert score("prompt", None) == (set(), 1.0)
    assert score("haystack", (1, 4)) == ({8}, 1.0)


def test_argmax_domain_shift_counts_only_moved_positions():
    """The recorded shift is the direct evidence of what the domain changed."""
    from retrieval_heads.scoring import StepTrace, argmax_domain_shift

    row = torch.zeros(2, 5)
    row[0, 0] = 0.9          # head 0: sink wins the prompt argmax
    row[1, 2] = 0.9          # head 1: same position in both domains
    # (1, heads, 1, kv): `argmax_positions` indexes [0, :, 0, :].
    attn = {0: row.unsqueeze(0).unsqueeze(2)}
    step = StepTrace(step=0, fed_token=1, predicted_token=2, attn={0: row},
                     argmax=argmax_positions(attn, prompt_len=4, domain="haystack",
                                             span=(1, 4)),
                     argmax_prompt=argmax_positions(attn, prompt_len=4, domain="prompt"))
    trace = DecodeTrace(prompt_len=4, steps=[step], argmax_domain="haystack",
                        argmax_span=(1, 4))
    shift = argmax_domain_shift(trace)
    assert shift == {"positions": 2, "shifted": 1, "share": 0.5}
    # Under the prompt domain nothing moved (the domain *is* the reference).
    step.argmax = step.argmax_prompt
    assert argmax_domain_shift(trace)["shifted"] == 0


def test_match_masks_is_safe_for_a_position_beyond_the_prompt():
    """`domain="full"` can point past the prompt; the vectorised path must not index
    out of bounds (the old per-head loop skipped those before indexing)."""
    from retrieval_heads.scoring import match_masks

    prompt_ids = torch.tensor([11, 7, 8, 12])
    argmax = torch.tensor([2, 4, 0])          # 4 is beyond the prompt
    matched, sink = match_masks(argmax, prompt_ids, 8, (1, 3), 0, 3)
    assert matched.tolist() == [True, False, False]
    assert sink.tolist() == [False, False, True]

    # The credit path agrees with the position-by-position reference.
    needle_set = {8}
    credits = {head for head, hit in enumerate(matched.tolist())
               if hit and 8 in needle_set}
    assert credits == {0}
