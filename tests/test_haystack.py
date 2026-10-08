"""Needle-in-a-Haystack construction."""

from __future__ import annotations

import pytest

from retrieval_heads.haystack import (
    HaystackBuilder,
    build_needle_sample,
    format_prompt,
    iter_depths,
)

NEEDLE = "The best thing to do in San Francisco is to eat a sandwich in Dolores Park on a sunny day."
QUESTION = "What is the best thing to do in San Francisco?"


@pytest.mark.parametrize("depth", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("index", [0, 1, 2, "eval"])
def test_needle_span_is_exact(tokenizer, depth, index):
    """The recorded span must be exactly the needle: no fewer, no more tokens.

    Parametrised over every shipped needle (including the held-out eval one) and the
    endpoint depths: the cut is snapped to whitespace, so 0.0/1.0 take a different
    path, and the span test used to cover a single needle at a single depth.
    """
    from retrieval_heads.detection import DETECTION_NEEDLES, EVAL_NEEDLES

    if index == "eval":
        needle, question = EVAL_NEEDLES[0]
    else:
        needle, question = DETECTION_NEEDLES[index]
    sample = build_needle_sample(
        tokenizer, needle=needle, question=question, target_tokens=256, depth=depth,
        builder=HaystackBuilder(seed=1),
    )
    start, end = sample.needle_span
    decoded = tokenizer.decode(sample.input_ids[0, start:end])
    assert needle in decoded
    assert sample.n_needle_tokens == end - start      # property, but cheap
    assert sample.n_needle_tokens > 5
    # A span widened by one token in either direction still contains the needle
    # text, so `needle in decoded` alone cannot catch it.  Neither trimmed span may
    # still decode the whole needle.  (An over-wide span inflates |k| and depresses
    # every head's score.)
    assert needle not in tokenizer.decode(sample.input_ids[0, start + 1:end])
    assert needle not in tokenizer.decode(sample.input_ids[0, start:end - 1])
    # The builder computes the same invariant and records it.
    assert sample.meta.get("span_tight") is True


def test_prompt_contains_question_and_needle(tokenizer):
    sample = build_needle_sample(
        tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=256, depth=0.5,
        builder=HaystackBuilder(seed=1),
    )
    assert NEEDLE in sample.prompt_text
    assert QUESTION in sample.prompt_text


@pytest.mark.parametrize("depth", [0.0, 0.25, 1.0])
def test_depth_actually_moves_the_needle(tokenizer, depth):
    sample = build_needle_sample(
        tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=512, depth=depth,
        builder=HaystackBuilder(seed=2),
    )
    relative = sample.needle_span[0] / sample.length
    if depth == 0.0:
        assert relative < 0.35
    elif depth == 1.0:
        assert relative > 0.6
    else:
        assert 0.15 < relative < 0.85


def test_haystack_grows_with_target(tokenizer):
    builder = HaystackBuilder(seed=3)
    small = build_needle_sample(tokenizer, needle=NEEDLE, question=QUESTION,
                                target_tokens=128, depth=0.5, builder=builder)
    large = build_needle_sample(tokenizer, needle=NEEDLE, question=QUESTION,
                                target_tokens=512, depth=0.5, builder=builder)
    assert large.length > small.length


def test_filler_tokenizer_that_returns_nothing_raises_instead_of_hanging():
    class EmptyTok:
        def __call__(self, text, **kwargs):
            return type("Enc", (), {"input_ids": []})()

    with pytest.raises(RuntimeError, match="no progress"):
        HaystackBuilder(seed=0).text(64, EmptyTok())


def test_filler_is_deterministic():
    a = HaystackBuilder(seed=11)
    b = HaystackBuilder(seed=11)
    c = HaystackBuilder(seed=12)
    assert a.text(200, _FakeTok()) == b.text(200, _FakeTok())
    assert a.text(200, _FakeTok()) != c.text(200, _FakeTok())


def test_seed_is_used_only_when_no_builder_is_given(tokenizer):
    """With a builder, the per-call seed is ignored; without one, it drives the filler.

    The previous version passed an explicit builder *and* different seeds and then
    asserted nothing changed -- which could not fail, because `build_needle_sample`
    only uses `seed` to construct a builder when none is supplied.
    """
    shared = [build_needle_sample(tokenizer, needle=NEEDLE, question=QUESTION,
                                  target_tokens=256, depth=0.5,
                                  builder=HaystackBuilder(seed=1), seed=s)
              for s in (1, 99)]
    assert shared[0].prompt_text == shared[1].prompt_text

    from_seed = [build_needle_sample(tokenizer, needle=NEEDLE, question=QUESTION,
                                     target_tokens=256, depth=0.5, seed=s)
                 for s in (1, 2)]
    assert from_seed[0].prompt_text != from_seed[1].prompt_text


def test_duplicate_needle_is_rejected(tokenizer):
    """A needle that already sits in the filler would invalidate the test."""
    builder = HaystackBuilder(seed=5)
    filler = builder.text(300, tokenizer)
    first_sentence = filler.split(".")[0].strip()
    with pytest.raises(ValueError, match="already occurs"):
        build_needle_sample(tokenizer, needle=first_sentence, question="q?",
                            target_tokens=300, depth=0.5, builder=HaystackBuilder(seed=5))


def test_iter_depths():
    assert iter_depths(1) == [0.5]
    assert iter_depths(3) == [0.0, 0.5, 1.0]
    assert len(iter_depths(10)) == 10
    assert iter_depths(5)[0] == 0.0 and iter_depths(5)[-1] == 1.0


def test_plain_prompt_has_no_template(tokenizer):
    plain = format_prompt(tokenizer, "CTX", "Q?", chat_template=False)
    assert plain.startswith("CTX")
    assert "Question: Q?" in plain
    chat = format_prompt(tokenizer, "CTX", "Q?", chat_template=True)
    assert "CTX" in chat and "Q?" in chat


class _FakeTok:
    """Cheap whitespace tokenizer so the determinism test needs no checkpoint."""

    def __call__(self, text, add_special_tokens=False, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(input_ids=text.split())


def test_needle_gold_tokens_come_from_the_needle_text(tokenizer):
    """The prompt span can fuse the last character with filler (`.\\n`).

    Scoring and F1 must use the tokenization of the needle *text*, otherwise the
    metric's ceiling drops below 1 and the emitted final token is uncreditable.
    """
    from retrieval_heads.masking import token_f1

    sample = build_needle_sample(
        tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=256, depth=0.5,
        builder=HaystackBuilder(seed=1),
    )
    text_ids = tokenizer(NEEDLE, add_special_tokens=False).input_ids
    assert sample.needle_text_ids == text_ids
    assert sample.n_unique_needle_text_tokens == len(set(text_ids))
    # A perfect answer must be able to score 1.0.
    assert token_f1(text_ids, sample.needle_text_ids) == 1.0
    # The span boundary may straddle the needle; it is recorded rather than silent.
    assert "span_straddles_boundary" in sample.meta
    # The insertion keeps a space after the needle, so the span ends on the needle's
    # own token and the scorer's ceiling is 1.0 (it used to be capped at 21/22).
    assert sample.meta["tokenization_attainable_score"] == 1.0
    assert not sample.meta["span_straddles_boundary"]
