"""Locate exact text occurrences in a tokenized prompt without changing IDs."""
from bisect import bisect_left, bisect_right

from .types import NeedleSpan


def locate_qr_spans(tokenizer, token_ids, *, question, needle):
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    if not question or not needle:
        raise ValueError('QRScore needs a nonempty question and needle')
    qstart = text.rfind(question)
    if qstart < 0:
        raise ValueError('Could not find exact QRScore question in decoded prompt')
    gold = text[:qstart]
    gstart = gold.find(needle)
    if gstart < 0 or gold.find(needle, gstart + 1) >= 0:
        raise ValueError('QRScore gold needle must occur exactly once before the question')
    # Prefix-decode lengths handle leading-space merges and slow tokenizers.
    # Lazy binary searches decode O(log N) prefixes, not every growing prefix.
    class Boundaries:
        def __len__(self):
            return len(token_ids) + 1

        def __getitem__(self, index):
            if not 0 <= index <= len(token_ids):
                raise IndexError(index)
            prefix = tokenizer.decode(token_ids[:index], skip_special_tokens=False)
            if not text.startswith(prefix):
                raise ValueError('QRScore prefix decoding is not stable; use an offset-mapped tokenizer')
            return len(prefix)

    boundaries = Boundaries()

    def span(start, end):
        first = bisect_right(boundaries, start) - 1
        last = bisect_left(boundaries, end)
        return NeedleSpan(first, last)

    query = span(qstart, qstart + len(question))
    evidence = span(gstart, gstart + len(needle))
    if not 0 <= evidence.start < evidence.end <= query.start < query.end <= len(token_ids):
        raise ValueError('Invalid/overlapping QRScore question and gold spans')
    return query, evidence
