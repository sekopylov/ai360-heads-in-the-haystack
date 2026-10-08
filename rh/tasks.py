"""
Task generators: each task turns its arguments into samples, a sample is a prompt with the token spans
the metrics need. A new task is a subclass of Task registered in TASKS.
"""
import glob
import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sample:
    id: str
    input_ids: list   # prompt tokens, the model continues them
    spans: dict       # name -> [start, end) in input_ids, e.g. the needle
    reference: str    # the expected answer
    meta: dict = field(default_factory=dict)


class Task:
    name = None

    @staticmethod
    def add_arguments(parser):
        pass

    def __init__(self, enc, args):
        self.enc = enc

    def samples(self):
        raise NotImplementedError


def chat_prompt(enc, content):
    """The prompt text of one user message; reasoning is switched off for the models that have it."""
    kwargs = {"enable_thinking": False} if "enable_thinking" in (enc.chat_template or "") else {}
    return enc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, **kwargs)


def tokenize_with_spans(enc, text, targets, special_tokens):
    """Tokenizes text and finds the token span of every target string by character offsets."""
    encoded = enc(text, add_special_tokens=special_tokens, return_offsets_mapping=True)
    spans = {}
    for name, target in targets.items():
        start = text.find(target)
        if start < 0:
            start = text.lower().find(target.lower())
        if start < 0:
            raise ValueError(f"{name} is not in the prompt: {target!r}")
        end = start + len(target)
        tokens = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > start]
        spans[name] = [tokens[0], tokens[-1] + 1]
    return [int(i) for i in encoded["input_ids"]], spans


class NeedleInHaystack(Task):
    """
    The task of "Retrieval Head Mechanistically Explains Long-Context Factuality": a sentence (needle) is put
    into a long text at a given depth and the model is asked about it. Unlike the authors' code, the needle is
    inserted after a period of the model's own tokenizer and its position is exact.

    Needles file: one json per line with needle, question, real_needle (the answer inside the needle), haystack_dir.
    Spans: "needle" is the answer inside the needle, "needle_sentence" is the whole needle.
    """
    name = "niah"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--needles', type=str, default="data/needles_detect.jsonl")
        parser.add_argument('--s_len', type=int, default=1000, help='shortest context, tokens')
        parser.add_argument('--e_len', type=int, default=30000, help='longest context, tokens')
        parser.add_argument('--context_intervals', type=int, default=20, help='number of context lengths')
        parser.add_argument('--depths', type=lambda s: [int(x) for x in s.split(',')], default=None,
                            help='needle depths in percent, comma separated; default 10 depths from 0 to 100')
        parser.add_argument('--needle', type=int, default=None, help='only this needle of the file')
        parser.add_argument('--raw_prompt', action='store_true', help='do not use the chat template')

    def __init__(self, enc, args):
        self.enc = enc
        with open(args.needles, encoding="utf-8") as file:
            self.needles = [json.loads(l) for l in file if l.strip()]
        self.only_needle = args.needle
        self.context_lengths = [int(i) for i in np.round(np.linspace(args.s_len, args.e_len, num=args.context_intervals))]
        self.depths = args.depths or [int(i) for i in np.round(np.linspace(0, 100, num=10))]
        self.chat = enc.chat_template is not None and not args.raw_prompt
        self.periods = enc.encode('.', add_special_tokens=False)
        self.buffer = 200  # tokens left for the question and the answer

    @staticmethod
    def read_haystack(haystack_dir, min_words):
        files = sorted(glob.glob(f"{haystack_dir}/*.txt"))
        if not files:
            raise ValueError(f"no .txt files in {haystack_dir}")
        text = ""
        while len(text.split()) < min_words:
            for file in files:
                with open(file, 'r', encoding="utf-8") as f:
                    text += f.read()
        return text

    def build_context(self, haystack_tokens, haystack_text, needle, context_length, depth_percent):
        enc = self.enc
        context = enc.decode(haystack_tokens[:context_length]) if len(haystack_tokens) > context_length else haystack_text
        tokens_needle = enc.encode(needle, add_special_tokens=False)
        tokens_context = enc.encode(context, add_special_tokens=False)
        tokens_context = tokens_context[:context_length - self.buffer - len(tokens_needle)]
        if depth_percent == 100:
            tokens = tokens_context + tokens_needle
        else:
            # the needle goes after the last period before the requested depth
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            while insertion_point > 0 and tokens_context[insertion_point - 1] not in self.periods:
                insertion_point -= 1
            tokens = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]
        return enc.decode(tokens)

    def build_sample(self, sample_id, context, needle, meta):
        question = f"Based on the content of the book, Question: {needle['question']}\nAnswer:"
        if self.chat:
            text = chat_prompt(self.enc, f"<book>{context}</book>\n{question}")
        else:
            text = context + question
        targets = {"needle": needle["real_needle"], "needle_sentence": needle["needle"].strip()}
        input_ids, spans = tokenize_with_spans(self.enc, text, targets, special_tokens=not self.chat)
        return Sample(sample_id, input_ids, spans, needle["real_needle"], dict(meta, n_input_tokens=len(input_ids)))

    def samples(self):
        for ni, needle in enumerate(self.needles):
            if self.only_needle is not None and ni != self.only_needle:
                continue
            text = self.read_haystack(needle["haystack_dir"], max(self.context_lengths))
            tokens = self.enc.encode(text, add_special_tokens=False)
            for context_length in self.context_lengths:
                for depth in self.depths:
                    context = self.build_context(tokens, text, needle["needle"], context_length, depth)
                    meta = {"needle_idx": ni, "context_length": context_length, "depth_percent": depth}
                    yield self.build_sample(f"n{ni}_len{context_length}_d{depth}", context, needle, meta)


TASKS = {task.name: task for task in [NeedleInHaystack]}
