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


def test_needle_span_is_exact(tokenizer):
    """The recorded token span must decode back to the needle, character for character."""
    sample = build_needle_sample(
        tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=256, depth=0.5,
        builder=HaystackBuilder(seed=1),
    )
    decoded = tokenizer.decode(sample.input_ids[0, sample.needle_span[0]:sample.needle_span[1]])
    assert NEEDLE in decoded
    assert sample.n_needle_tokens == sample.needle_span[1] - sample.needle_span[0]
    assert sample.n_needle_tokens > 5


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


def test_filler_is_deterministic():
    a = HaystackBuilder(seed=11)
    b = HaystackBuilder(seed=11)
    c = HaystackBuilder(seed=12)
    assert a.text(200, _FakeTok()) == b.text(200, _FakeTok())
    assert a.text(200, _FakeTok()) != c.text(200, _FakeTok())


def test_seed_does_not_change_the_needle(tokenizer):
    a = build_needle_sample(tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=256,
                            depth=0.5, builder=HaystackBuilder(seed=1), seed=1)
    b = build_needle_sample(tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=256,
                            depth=0.5, builder=HaystackBuilder(seed=1), seed=99)
    assert a.needle_span == b.needle_span
    assert a.needle_ids == b.needle_ids


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
