"""Needle-in-a-Haystack construction.

The paper inserts a unique sentence (the *needle*) at a random depth inside an
unrelated long context (the *haystack*), asks a question about it, and then
watches which heads copy the needle tokens into the output.

Everything here is offline and deterministic: filler text is generated from a
seeded word list (optionally replaced by a real corpus file), so a run can be
reproduced exactly without network access.

Token spans are recovered from **character offsets** rather than by searching
for a token subsequence.  That matters: after a chat template is applied the
needle can be re-tokenised, and offset mapping is the only exact way to know
which tokens belong to the needle.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from retrieval_heads.utils import get_logger

log = get_logger("haystack")

#: A compact, semantically bland vocabulary.  Its only job is to be unrelated to
#: the needle so that a correct answer can only come from copying the context.
DEFAULT_WORDS: tuple[str, ...] = (
    "apple bridge river mountain silent copper window garden theory market fabric "
    "orange tunnel planet candle mirror forest gravel lantern meadow pebble ribbon "
    "saddle timber velvet walnut yellow anchor basket canyon dagger ember feather "
    "garnet harbor island jacket kettle ladder marble nectar oasis paddle quartz "
    "raven socket tundra urchin vessel wagon xenon yarn zephyr alloy bramble cider "
    "dune elm frost grove hollow inlet juniper kelp ledge moss nook orchid pine"
).split()

#: Sentence templates keep the filler from looking like one giant word salad.
_TEMPLATES: tuple[str, ...] = (
    "{a} {b} was recorded near the {c} while the {d} remained {e}.",
    "Observers noted that the {a} {b} had already left the {c} before {d}.",
    "A {a} of {b} appeared beside the {c}, which surprised the {d}.",
    "The {a} {b} measured the {c} twice and then filed a short {d}.",
    "According to the {a}, the {b} near the {c} was {d} for most of the {e}.",
    "Every {a} in the {b} kept a {c} of the {d} for later {e}.",
)


def load_corpus(path: str | Path) -> list[str]:
    """Load filler sentences from a text file (one sentence per line)."""
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


class HaystackBuilder:
    """Deterministic generator of unrelated filler text."""

    def __init__(
        self,
        corpus: Sequence[str] | None = None,
        *,
        words: Sequence[str] = DEFAULT_WORDS,
        seed: int = 0,
    ) -> None:
        self.words = list(words)
        self.corpus = list(corpus) if corpus else None
        self.rng = random.Random(seed)

    def sentence(self) -> str:
        if self.corpus:
            return self.rng.choice(self.corpus)
        rng = self.rng
        return rng.choice(_TEMPLATES).format(
            a=rng.choice(self.words), b=rng.choice(self.words), c=rng.choice(self.words),
            d=rng.choice(self.words), e=rng.choice(self.words),
        )

    #: Calibration seed for :meth:`text`; the real value is measured on the fly.
    #: The template sentences average ~16 tokens, and guessing low is what made an
    #: earlier version overshoot 1.6x (a "16K" context was really 26K).
    TOKENS_PER_SENTENCE = 16.0

    def text(self, n_tokens: int, tokenizer: Any) -> str:
        """Grow filler text until it tokenises to at least ``n_tokens`` tokens.

        Sentences are appended in sized batches and the accumulated text is
        tokenised once per batch, so this stays fast at 50K tokens: tokenising
        after every single sentence would be quadratic and dominate a
        paper-scale run.  The batch size is recalibrated from the tokens actually
        produced and shrinks to a single sentence as the target approaches, which
        keeps the overshoot to at most one sentence (~16 tokens) instead of the
        1.6x a fixed guess produced.
        """
        parts: list[str] = []
        count = 0
        per_sentence = self.TOKENS_PER_SENTENCE
        previous = -1
        while count < n_tokens:
            remaining = n_tokens - count
            # The last batch is sized from what is left, never padded with a "+1",
            # so the text stops as soon as it reaches the target.
            batch = max(1, int(remaining / per_sentence))
            parts.extend(self.sentence() for _ in range(batch))
            count = len(tokenizer(" ".join(parts), add_special_tokens=False).input_ids)
            if count <= previous:
                # A tokenizer that returns nothing would otherwise grow `parts`
                # forever and exhaust memory instead of failing.
                raise RuntimeError(
                    f"filler tokenization made no progress ({previous} -> {count} tokens "
                    f"for {len(parts)} sentences); check the tokenizer"
                )
            previous = count
            per_sentence = max(1.0, count / len(parts))
        return " ".join(parts)


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenise ``text`` into ids plus ``[char_start, char_end)`` offsets."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    if len(offsets) != len(ids):  # pragma: no cover - defensive
        raise RuntimeError("tokenizer returned inconsistent ids/offsets")
    return ids, offsets


def _span_from_char_range(offsets: Sequence[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    """Token index range ``[lo, hi)`` whose character offsets intersect ``[start, end)``."""
    lo = hi = None
    for idx, (cs, ce) in enumerate(offsets):
        if ce <= start or cs >= end:
            continue
        if lo is None:
            lo = idx
        hi = idx + 1
    if lo is None:
        raise ValueError(f"no tokens overlap character range [{start}, {end})")
    return lo, hi


@dataclass
class NeedleSample:
    """One fully materialised Needle-in-a-Haystack instance."""

    prompt_text: str
    input_ids: torch.LongTensor          # (1, T)
    needle_span: tuple[int, int]         # token indices [start, end) of the needle
    needle_text: str
    question: str
    depth: float                         # 0.0 = very start, 1.0 = very end
    target_tokens: int
    haystack_tokens: int                 # actual prompt length
    seed: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return int(self.input_ids.shape[1])

    @property
    def needle_ids(self) -> list[int]:
        return self.input_ids[0, self.needle_span[0]:self.needle_span[1]].tolist()

    @property
    def n_needle_tokens(self) -> int:
        return self.needle_span[1] - self.needle_span[0]

    @property
    def n_unique_needle_tokens(self) -> int:
        return len(set(self.needle_ids))

    @property
    def needle_text_ids(self) -> list[int]:
        """Tokenization of the needle *text* -- what the model can actually emit.

        The prompt span can fuse the needle's last character with following filler
        (`.\n` is one token), so the span ids differ from the text ids at the edge.
        Scoring and F1 use the text ids: otherwise the metric's ceiling is below 1
        and the emitted final token can never be credited.
        """
        return self.meta.get("needle_text_ids") or self.needle_ids

    @property
    def n_unique_needle_text_tokens(self) -> int:
        return len(set(self.needle_text_ids))

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.meta,
            "needle_text": self.needle_text,
            "question": self.question,
            "depth": self.depth,
            "needle_span": list(self.needle_span),
            "n_needle_tokens": self.n_needle_tokens,
            "n_unique_needle_tokens": self.n_unique_needle_tokens,
            # The denominator the score actually uses (the span count above can
            # differ when a boundary token is fused).
            "n_unique_needle_text_tokens": self.n_unique_needle_text_tokens,
            "prompt_tokens": self.length,
            "target_tokens": self.target_tokens,
            "seed": self.seed,
        }


def render_chat(tokenizer: Any, messages: list[dict[str, str]], *,
                enable_thinking: bool | None = None) -> str:
    """Apply a chat template, tolerating templates that know no `enable_thinking`.

    Single implementation: `format_prompt` and `downstream._chat` used to carry
    their own copy, which is how the decode-loop duplication started.
    """
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as exc:
        # The template may reject `enable_thinking` with something other than a
        # TypeError (a jinja UndefinedError, for instance).  Retry without it, but if
        # that also fails let the original error surface.
        if "enable_thinking" not in kwargs:
            raise
        kwargs.pop("enable_thinking")
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception as second:
            raise second from exc


def format_prompt(
    tokenizer: Any,
    context: str,
    question: str,
    *,
    chat_template: bool = True,
    system_prompt: str | None = None,
    enable_thinking: bool | None = False,
) -> str:
    """Wrap ``context`` + ``question`` in a plain or chat-template prompt."""
    tail = f"\n\nQuestion: {question}\nAnswer:"
    if not chat_template:
        return context + tail
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": context + tail})
    return render_chat(tokenizer, messages, enable_thinking=enable_thinking)


def build_needle_sample(
    tokenizer: Any,
    *,
    needle: str,
    question: str,
    target_tokens: int,
    depth: float,
    builder: HaystackBuilder | None = None,
    chat_template: bool = True,
    system_prompt: str | None = None,
    enable_thinking: bool | None = False,
    seed: int = 0,
) -> NeedleSample:
    """Build one instance with the needle placed at relative ``depth``.

    ``depth`` is applied in *characters* of the filler text, matching the common
    "insert at x% of the document" convention; the paper samples depths
    uniformly from the start to the end of the haystack.

    ``target_tokens`` is the **realized prompt length**, not just the filler:
    :meth:`HaystackBuilder.text` sizes the filler alone, and the needle, question
    and chat template are added on top.  That overhead is a fixed ~40 tokens, so
    uncorrected it was +6% at 1K.  The prompt is therefore built, measured, and
    the filler budget re-adjusted a few times.  Filler is appended a sentence at a
    time, so convergence is limited by the last sentence (~16 tokens): the
    tolerance is 2%, and a run that cannot reach it within four attempts says so.
    """
    builder = builder or HaystackBuilder(seed=seed)
    depth = min(max(depth, 0.0), 1.0)
    tolerance = max(2, int(0.02 * target_tokens))
    budget = target_tokens
    filler = ""
    for _ in range(4):
        filler = builder.text(max(64, budget), tokenizer)
        if needle.strip() in filler:  # pragma: no cover - improbable, but keep the guarantee
            raise ValueError("needle already occurs in the filler; pick a more unique needle")

        cut = int(len(filler) * depth)
        # snap to the nearest whitespace so we never split a word
        while cut < len(filler) and not filler[cut].isspace():
            cut += 1
        # A space after the needle, before the newline: without it the tokenizer
        # fuses the needle's last character with the newline (`.` + `\n` -> `".\n"`),
        # so the span's last token is not a needle token and the score's ceiling
        # drops below 1 (`.\n` in k, `.` unreachable).
        context = f"{filler[:cut]}\n{needle} \n{filler[cut:]}".strip()
        if context.endswith(needle):
            # depth 1.0: the needle abuts the question block, whose "\n\n" would fuse
            # with the needle's last character (`.\n\n`).  Keep the separating space.
            context += " "

        prompt = format_prompt(
            tokenizer, context, question, chat_template=chat_template,
            system_prompt=system_prompt, enable_thinking=enable_thinking,
        )

        char_start = prompt.find(needle)
        if char_start < 0:  # pragma: no cover - defensive
            raise RuntimeError("needle text vanished from the rendered prompt")
        if char_start != prompt.rfind(needle):
            # A second occurrence (in the question or the template) would make the
            # span point at the wrong text and silently shift every score.
            raise RuntimeError(
                "the needle text occurs more than once in the rendered prompt; the "
                "recorded span would not identify which occurrence was scored"
            )
        char_end = char_start + len(needle)

        ids, offsets = _tokenize_with_offsets(tokenizer, prompt)
        realized = len(ids)
        if abs(realized - target_tokens) <= tolerance:
            break
        budget -= realized - target_tokens  # overshoot -> shrink the filler budget
    else:
        log.warning("could not land %d tokens within %d after 4 attempts (realized %d)",
                    target_tokens, tolerance, realized)

    span = _span_from_char_range(offsets, char_start, char_end)
    # Interval-overlap span detection can swallow a token that starts in the filler
    # and ends inside the needle (e.g. a merged "\nThe").  Verify the span really is
    # the needle, and record whether either edge could be trimmed.
    lo, hi = span
    if needle not in prompt[offsets[lo][0]:offsets[hi - 1][1]]:
        raise RuntimeError(
            f"needle span {span} does not contain the needle text; the tokenizer's "
            f"offset mapping straddles the needle"
        )
    tight = True
    for s_lo, s_hi in ((lo + 1, hi), (lo, hi - 1)):
        if s_lo < s_hi and needle in prompt[offsets[s_lo][0]:offsets[s_hi - 1][1]]:
            tight = False
            break
    # A boundary token that only *partly* overlaps the needle is not caught by the
    # tightness check above: `".\n"` contains the whole needle but also filler text,
    # so the span is "tight" while the token is not the one the model emits.  Record
    # it instead of silently treating it as pure needle.
    straddles = offsets[lo][0] < char_start or offsets[hi - 1][1] > char_end
    needle_text_ids = tokenizer(needle, add_special_tokens=False).input_ids
    # How much of the needle the *tokenization* can expose to the scorer: the
    # intersection of the text tokens and the span tokens, over the unique text
    # tokens.  This is a tokenization ceiling only -- it says nothing about how much
    # of the needle the question actually asks for.  With the space-after-needle
    # insertion it is 1.0 for every shipped needle.
    span_ids = ids[lo:hi]
    max_attainable = len(set(needle_text_ids) & set(span_ids)) / max(len(set(needle_text_ids)), 1)
    if max_attainable < 0.98:
        log.warning("needle span exposes only %.0f%% of the needle's unique tokens "
                    "(span %s, text %s); every retrieval score is capped there",
                    100 * max_attainable, span_ids[-3:], needle_text_ids[-3:])
    input_ids = torch.tensor([ids], dtype=torch.long)

    sample = NeedleSample(
        prompt_text=prompt,
        input_ids=input_ids,
        needle_span=span,
        needle_text=needle,
        question=question,
        depth=depth,
        target_tokens=target_tokens,
        haystack_tokens=len(ids),
        seed=seed,
        meta={"span_tight": tight, "span_straddles_boundary": straddles,
              "needle_text_ids": list(needle_text_ids),
              "tokenization_attainable_score": max_attainable},
    )
    log.debug("built sample: %d tokens, needle %s (%d tok) at depth %.2f",
              len(ids), span, span[1] - span[0], depth)
    return sample


def iter_depths(n: int) -> list[float]:
    """Uniform depths in ``[0, 1]`` (both endpoints included) -- ``n`` values."""
    if n == 1:
        return [0.5]
    return [i / (n - 1) for i in range(n)]
