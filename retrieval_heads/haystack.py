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
from typing import Any, Iterable, Sequence

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

    def text(self, n_tokens: int, tokenizer: Any) -> str:
        """Grow filler text until it tokenises to at least ``n_tokens`` tokens.

        Sentences are appended in sized batches and the batch is tokenised only
        once, so this stays fast at 50K tokens.  Tokenising the accumulated text
        after every single sentence would be quadratic and dominate a
        paper-scale run.
        """
        parts: list[str] = []
        while True:
            # ~11 tokens per template sentence; over-generate slightly.
            batch = max(8, n_tokens // 10)
            parts.extend(self.sentence() for _ in range(batch))
            joined = " ".join(parts)
            if len(tokenizer(joined, add_special_tokens=False).input_ids) >= n_tokens:
                return joined


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "needle_text": self.needle_text,
            "question": self.question,
            "depth": self.depth,
            "needle_span": list(self.needle_span),
            "n_needle_tokens": self.n_needle_tokens,
            "n_unique_needle_tokens": self.n_unique_needle_tokens,
            "prompt_tokens": self.length,
            "target_tokens": self.target_tokens,
            "seed": self.seed,
            **self.meta,
        }


def format_prompt(
    tokenizer: Any,
    context: str,
    question: str,
    *,
    chat_template: bool = True,
    system_prompt: str | None = None,
    enable_thinking: bool | None = False,
    instruction: str | None = None,
) -> str:
    """Wrap ``context`` + ``question`` in a plain or chat-template prompt."""
    tail = f"\n\nQuestion: {question}\nAnswer:" if instruction is None else f"\n\n{instruction}\n\nQuestion: {question}\nAnswer:"
    if not chat_template:
        return context + tail
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": context + tail})
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Template does not know `enable_thinking`; retry without it.
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


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
    instruction: str | None = None,
    seed: int = 0,
) -> NeedleSample:
    """Build one instance with the needle placed at relative ``depth``.

    ``depth`` is applied in *characters* of the filler text, matching the common
    "insert at x% of the document" convention; the paper samples depths
    uniformly from the start to the end of the haystack.
    """
    builder = builder or HaystackBuilder(seed=seed)
    filler = builder.text(target_tokens, tokenizer)
    if needle.strip() in filler:  # pragma: no cover - improbable, but keep the guarantee
        raise ValueError("needle already occurs in the filler; pick a more unique needle")

    depth = min(max(depth, 0.0), 1.0)
    cut = int(len(filler) * depth)
    # snap to the nearest whitespace so we never split a word
    while cut < len(filler) and not filler[cut].isspace():
        cut += 1
    context = f"{filler[:cut]}\n{needle}\n{filler[cut:]}".strip()

    prompt = format_prompt(
        tokenizer, context, question, chat_template=chat_template,
        system_prompt=system_prompt, enable_thinking=enable_thinking, instruction=instruction,
    )

    char_start = prompt.find(needle)
    if char_start < 0:  # pragma: no cover - defensive
        raise RuntimeError("needle text vanished from the rendered prompt")
    char_end = char_start + len(needle)

    ids, offsets = _tokenize_with_offsets(tokenizer, prompt)
    span = _span_from_char_range(offsets, char_start, char_end)
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
    )
    log.debug("built sample: %d tokens, needle %s (%d tok) at depth %.2f",
              len(ids), span, span[1] - span[0], depth)
    return sample


def iter_depths(n: int) -> list[float]:
    """Uniform depths in ``(0, 1)`` -- ``n`` values, endpoints included."""
    if n == 1:
        return [0.5]
    return [i / (n - 1) for i in range(n)]
