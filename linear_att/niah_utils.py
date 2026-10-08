"""
Needle-in-a-Haystack utilities, ported (with small cleanups) from
https://github.com/nightdessert/Retrieval_Head (retrieval_head_detection.py).

The haystack / needles are the authors' ones: clone their repo and pass
--haystack_dir /path/to/Retrieval_Head/haystack_for_detect
(it contains needles.jsonl and part1/ part2/ part3/ with *.txt files).
"""
import glob
import json
import os

import numpy as np

try:
    from rouge_score import rouge_scorer

    _SCORER = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
except ImportError:  # fallback: unigram recall
    _SCORER = None


def rouge1_recall(reference: str, prediction: str) -> float:
    """ROUGE-1 recall * 100, exactly the metric used by the authors."""
    if _SCORER is not None:
        return _SCORER.score(reference, prediction)["rouge1"].recall * 100
    ref = reference.lower().split()
    pred = set(prediction.lower().split())
    return 100.0 * sum(w in pred for w in ref) / max(len(ref), 1)


def load_needles(haystack_dir, needles_file=None):
    """Returns list of dicts {needle, question, real_needle, haystack_dir}."""
    needles_file = needles_file or os.path.join(haystack_dir, "needles.jsonl")
    items = [json.loads(l) for l in open(needles_file) if l.strip()]
    parts = sorted(glob.glob(os.path.join(haystack_dir, "part*")))
    if not parts:
        raise FileNotFoundError(f"No part*/ directories in {haystack_dir}")
    for i, it in enumerate(items):
        it["haystack_dir"] = parts[i % len(parts)]
    return items


class NeedleHaystack:
    def __init__(self, tokenizer, final_context_length_buffer=200):
        self.tok = tokenizer
        self.buffer = final_context_length_buffer
        self._haystack_cache = {}
        self._period_cache = {}

    # ------------------------------------------------------------------ text
    def _encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def _read_haystack(self, haystack_dir, max_context_length):
        key = (haystack_dir, max_context_length)
        if key not in self._haystack_cache:
            files = sorted(glob.glob(os.path.join(haystack_dir, "*.txt")))
            if not files:
                raise FileNotFoundError(f"No *.txt in {haystack_dir}")
            context = ""
            while len(context.split()) < max_context_length:
                for f in files:
                    with open(f, "r") as fh:
                        context += fh.read()
            self._haystack_cache[key] = self._encode(context)
        return self._haystack_cache[key]

    def _is_period(self, tid):
        if tid not in self._period_cache:
            self._period_cache[tid] = self.tok.decode([tid]).strip().endswith(".")
        return self._period_cache[tid]

    def build_context(self, needle, haystack_dir, context_length, depth_percent):
        tokens_context = list(self._read_haystack(haystack_dir, context_length))[:context_length]
        tokens_needle = self._encode(needle)
        context_length = context_length - self.buffer
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[: max(context_length - len(tokens_needle), 0)]
        if depth_percent >= 100:
            new = tokens_context + tokens_needle
        else:
            ins = int(len(tokens_context) * (depth_percent / 100))
            while ins > 0 and not self._is_period(tokens_context[ins - 1]):
                ins -= 1
            new = tokens_context[:ins] + tokens_needle + tokens_context[ins:]
        return self.tok.decode(new)

    def build_prompt(self, item, context_length, depth_percent):
        context = self.build_context(" " + item["needle"].strip() + " ", item["haystack_dir"],
                                     context_length, depth_percent)
        question = f"\nBased on the content of the book, Question: {item['question']}\nAnswer:"
        return context + question

    # --------------------------------------------------------------- needle span
    def find_needle_span(self, prompt_ids, real_needle):
        """1) exact token-subsequence match of the answer span;
        2) otherwise the authors' heuristic (window whose token set overlaps the needle
           token set by > 90%), taking the best-overlapping window instead of the first.
        Returns (start, end) or (-1, -1)."""
        prompt_ids = list(prompt_ids)
        cands = [self._encode(real_needle), self._encode(" " + real_needle),
                 self._encode(" " + real_needle[:1].lower() + real_needle[1:])]
        for needle_ids in cands:
            n = len(needle_ids)
            for i in range(len(prompt_ids) - n + 1):
                if prompt_ids[i: i + n] == needle_ids:
                    return i, i + n
        best, best_span = 0.8, (-1, -1)
        for needle_ids in cands:
            span_len = len(needle_ids)
            nset = set(needle_ids)
            for i in range(len(prompt_ids)):
                ov = len(set(prompt_ids[i: i + span_len]) & nset) / len(nset)
                ov += 0.01 * (prompt_ids[i] in nset)   # prefer windows that start on a needle token
                if ov > best:
                    best, best_span = ov, (i, i + span_len)
        return best_span


def make_grid(lengths, depth_min=0, depth_max=100, depth_intervals=10):
    depths = np.round(np.linspace(depth_min, depth_max, num=depth_intervals, endpoint=True)).astype(int)
    return [(int(L), int(d)) for L in lengths for d in depths]
