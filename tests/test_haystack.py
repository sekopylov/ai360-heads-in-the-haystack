"""Needle-in-a-Haystack construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrieval_heads.haystack import (
    HaystackBuilder,
    build_needle_sample,
    format_prompt,
    iter_depths,
)
from retrieval_heads.detection import DETECTION_NEEDLES, EVAL_NEEDLES

NEEDLE = "The best thing to do in San Francisco is to eat a sandwich in Dolores Park on a sunny day."
QUESTION = "What is the best thing to do in San Francisco?"

#: Every shipped needle, labelled, so a newly added one is covered automatically
#: (the span test used to hard-code "index 2 or the single eval needle").
ALL_NEEDLES = [
    (f"detection{i}", needle, question)
    for i, (needle, question) in enumerate(DETECTION_NEEDLES)
] + [
    (f"eval{i}", needle, question)
    for i, (needle, question) in enumerate(EVAL_NEEDLES)
]


@pytest.mark.parametrize("depth", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("label,needle,question", ALL_NEEDLES,
                         ids=[label for label, _, _ in ALL_NEEDLES])
def test_needle_span_is_exact(tokenizer, depth, label, needle, question):
    """The recorded span must be exactly the needle: no fewer, no more tokens.

    Parametrised over every shipped needle (detection and held-out eval) and the
    endpoint depths: the cut is snapped to whitespace, so 0.0/1.0 take a different
    path, and the span test used to cover a single needle at a single depth.
    """
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
    # Every shipped needle must also be locatable as a haystack, or the default
    # `haystack` argmax domain would fail on it.
    assert sample.haystack_span is not None
    assert sample.meta["haystack_span_verbatim"] is True


def test_prompt_contains_question_and_needle(tokenizer):
    sample = build_needle_sample(
        tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=256, depth=0.5,
        builder=HaystackBuilder(seed=1),
    )
    assert NEEDLE in sample.prompt_text
    assert QUESTION in sample.prompt_text


@pytest.mark.parametrize("depth", [0.0, 0.5, 1.0])
def test_haystack_span_is_the_context_without_the_question(tokenizer, depth):
    """The `haystack` argmax domain searches this span: filler + needle, no question.

    It must contain the needle (the paper's ``x`` is the haystack the needle was
    inserted into) and exclude the question and the template, which is the whole
    point of the domain -- on the dense control 285 of 448 heads put their argmax on
    prompt position 0, a template token.
    """
    sample = build_needle_sample(
        tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=256, depth=depth,
        builder=HaystackBuilder(seed=1),
    )
    assert sample.haystack_span is not None
    start, end = sample.haystack_span
    assert start < end <= sample.length
    # The needle is inside the haystack, so criterion (2)'s span test is reachable.
    assert start <= sample.needle_span[0] < sample.needle_span[1] <= end
    decoded = tokenizer.decode(sample.input_ids[0, start:end])
    assert NEEDLE in decoded
    assert QUESTION not in decoded and "Question:" not in decoded
    # The template tokens come before the haystack: the domain is strictly smaller
    # than the prompt (otherwise it would be `prompt` under another name).
    assert start > 0 and end < sample.length
    assert sample.meta["haystack_span_verbatim"] is True
    assert sample.n_haystack_tokens == end - start
    assert sample.as_dict()["haystack_span"] == [start, end]


def test_haystack_span_records_whether_the_sink_is_inside_it(tokenizer):
    """Where position 0 sits relative to `x` decides whether criterion (2) is reachable.

    A chat template puts the sink *before* the haystack, so it can never win the
    argmax; the paper's template-free prompt starts with the haystack, so its sink is
    inside `x` and suppresses credit.  Measured on one Qwen3-0.6B instance: 173/448
    heads above 0.1 with the template against 13/448 without.
    """
    templated = build_needle_sample(
        tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=256, depth=0.5,
        builder=HaystackBuilder(seed=1), chat_template=True,
    )
    assert templated.haystack_span[0] > 0
    assert templated.haystack_includes_sink is False
    assert templated.as_dict()["haystack_includes_sink"] is False

    plain = build_needle_sample(
        tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=256, depth=0.5,
        builder=HaystackBuilder(seed=1), chat_template=False,
    )
    assert plain.haystack_span[0] == 0
    assert plain.haystack_includes_sink is True
    assert plain.as_dict()["haystack_includes_sink"] is True


def test_haystack_span_is_none_when_the_prompt_cannot_be_located(tokenizer):
    """A template that mangles the content must disable the domain, not fake it.

    Simulated by inserting the context one character short: the needle is still
    found (and still unique), so the needle span stays exact, but the haystack is no
    longer verbatim and the recorded span must be `None` rather than a wrong range.
    """
    from retrieval_heads import haystack as haystack_module

    real = haystack_module.format_prompt

    def mangling(tokenizer_, context, question, **kwargs):
        return real(tokenizer_, context, question, **kwargs).replace(context, context[:-1], 1)

    haystack_module.format_prompt = mangling
    try:
        sample = build_needle_sample(
            tokenizer, needle=NEEDLE, question=QUESTION, target_tokens=128, depth=0.5,
            builder=HaystackBuilder(seed=1),
        )
    finally:
        haystack_module.format_prompt = real
    assert sample.meta["haystack_span_verbatim"] is False
    assert sample.haystack_span is None
    assert sample.as_dict()["haystack_span"] is None
    # The needle span is still exact, so the failure is confined to the new domain.
    assert NEEDLE in tokenizer.decode(sample.input_ids[0, slice(*sample.needle_span)])


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


def test_enable_thinking_true_is_the_only_value_that_means_on():
    """`--thinking` used to pass `None`, which the Qwen3.5 template reads as OFF.

    Qwen3.5 tests `enable_thinking is defined and enable_thinking is true`, so an
    omitted kwarg lands in the else branch (empty ` thinking` block); Qwen3-0.6B
    tests `is false` and is on unless told otherwise.  `True` is therefore the one
    value that means "thinking on" for both, and the artifact no longer records
    `null` for a flag that did nothing.
    """
    from transformers import AutoTokenizer

    from retrieval_heads.haystack import render_chat

    open_tag = chr(60) + "think" + chr(62)
    close_tag = chr(60) + "/think" + chr(62)
    messages = [{"role": "user", "content": "hi"}]
    qwen35 = Path("models/Qwen3.5-0.8B")
    if not (qwen35 / "chat_template.jinja").exists() and not (
            qwen35 / "tokenizer_config.json").exists():
        pytest.skip("Qwen3.5 tokenizer is not downloaded")

    tokenizer = AutoTokenizer.from_pretrained(qwen35)
    on = render_chat(tokenizer, messages, enable_thinking=True)
    off = render_chat(tokenizer, messages, enable_thinking=False)
    # Thinking on = the block is opened and left open; off = an empty closed block.
    assert open_tag in on and close_tag not in on, on[-60:]
    assert f"{open_tag}\n\n{close_tag}" in off, off[-60:]

    # Qwen3-0.6B's template tests `is false`, so True must not add a think block.
    qwen3 = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")
    assert open_tag not in render_chat(qwen3, messages, enable_thinking=True)[-60:]
    assert f"{open_tag}\n\n{close_tag}" in render_chat(
        qwen3, messages, enable_thinking=False)


def test_load_corpus_rejects_an_empty_file(tmp_path):
    """An empty corpus would silently fall back to the synthetic filler."""
    from retrieval_heads.haystack import load_corpus

    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no non-empty lines"):
        load_corpus(empty)

    real = tmp_path / "real.txt"
    real.write_text("One sentence.\nAnother.\n", encoding="utf-8")
    assert load_corpus(real) == ["One sentence.", "Another."]


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
