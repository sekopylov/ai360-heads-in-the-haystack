"""Unit tests for the retrieval score -- the two copy-paste criteria.

These build synthetic traces, so they pin the *definition* without needing a
checkpoint, and they cover the pairing ambiguity explicitly (a head can only be
scored against the attention row that produces the token, or against the row at
the token's own position -- the two must not be silently conflated).
"""

from __future__ import annotations

import torch

from retrieval_heads.models import ModelInfo
from retrieval_heads.scoring import (
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
        attention_modules={i: None for i in range(num_layers)},  # type: ignore[dict-item]
    )


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
    """Criterion (2) needs *both* the right position and the same token."""
    sample = FakeSample([11, 7, 8, 12, 13], (1, 3))
    info = make_info()
    # head points inside the needle, but the input token there (7) != prediction (9)
    trace = DecodeTrace(prompt_len=5, steps=[
        StepTrace(step=0, fed_token=1, predicted_token=9, attn={0: spike(2, 5, {0: 1, 1: 1})}),
    ])
    credits, _, _ = credits_from_trace(trace, sample, info, pairing="next_step")
    assert credits[HeadRef(0, 0)] == set()


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
    # in-order walk matches only the "7" at the very end (needle[0] == 7)
    assert strict[HeadRef(0, 0)] == {7}


def test_needle_recall_is_case_insensitive():
    needle = "The best thing is to eat a sandwich."
    assert needle_recall("The best thing is to eat a sandwich.", needle) == 1.0
    assert needle_recall("the BEST thing is to EAT a sandwich", needle) == 1.0
    assert needle_recall("the best thing", needle) == 3 / 8
    assert needle_recall("nothing relevant here", needle) == 0.0
