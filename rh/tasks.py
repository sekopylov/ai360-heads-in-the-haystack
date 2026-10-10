"""
Task generators: each task turns its arguments into samples, a sample is a prompt with the token spans
the metrics need. A new task is a subclass of Task registered in TASKS.
"""
import bisect
import glob
import json
import os
import random
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


def tokenize_with_char_spans(enc, text, ranges, special_tokens):
    """Tokenizes text; ranges: name -> (start, end) in characters; returns the tokens and name -> [start, end) in tokens."""
    encoded = enc(text, add_special_tokens=special_tokens, return_offsets_mapping=True)
    spans = {}
    for name, (start, end) in ranges.items():
        tokens = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > start]
        spans[name] = [tokens[0], tokens[-1] + 1]
    return [int(i) for i in encoded["input_ids"]], spans


class HardNeedleInHaystack(NeedleInHaystack):
    """
    The needle task with a harder context, on the data of data/needles_hard (see its README): a family is one needle,
    a level changes the question, puts other sentences (inserts) into the text or moves the needle. One run may hold
    several families and levels: they are in the sample id and in meta.

    The needle and every insert go after a period of the haystack; the haystack is shortened by their length, so a
    level has the prompt length of the level without inserts. Where an insert goes (placement of the level):
        near      within `near` tokens of the needle, on either side
        far       at the opposite depth: (needle depth + 50) mod 100 percent
        random    after any sentence of the context
        distance  that many haystack tokens before or after the needle (side), up to the sentence border; a sample
                  whose context has no room for it is left out
    The random choices depend on --insert_seed and the sample id only.

    Spans: "needle" is real_needle (the answer or, on some levels, its evidence; it may lie in an insert),
    "needle_sentence" the needle, and one span per insert under its name. They are taken from the positions at
    which the sentences were put, not searched for in the prompt.
    """
    name = "hard_niah"
    near = 200

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--hard_dir', type=str, default="data/needles_hard")
        parser.add_argument('--families', type=lambda s: s.split(','), default=None, help='comma separated; default: every family of --hard_dir')
        parser.add_argument('--levels', type=lambda s: s.split(','), default=["A0"],
                            help='comma separated levels (B3), whole axes (B) or all')
        parser.add_argument('--s_len', type=int, default=1000, help='shortest context, tokens')
        parser.add_argument('--e_len', type=int, default=30000, help='longest context, tokens')
        parser.add_argument('--context_intervals', type=int, default=20, help='number of context lengths')
        parser.add_argument('--lengths', type=lambda s: [int(x) for x in s.split(',')], default=None,
                            help='context lengths in tokens, comma separated; replaces --s_len, --e_len and --context_intervals')
        parser.add_argument('--depths', type=lambda s: [int(x) for x in s.split(',')], default=None,
                            help='needle depths in percent, comma separated; default 10 depths from 0 to 100')
        parser.add_argument('--insert_seed', type=int, default=0, help='seed of the places of the inserts')

    def __init__(self, enc, args):
        self.enc = enc
        self.context_lengths = args.lengths or [int(i) for i in np.round(np.linspace(args.s_len, args.e_len, num=args.context_intervals))]
        self.depths = args.depths or [int(i) for i in np.round(np.linspace(0, 100, num=10))]
        self.chat = enc.chat_template is not None
        self.periods = set(enc.encode('.', add_special_tokens=False))
        self.buffer = 200  # tokens left for the question and the answer
        self.seed = args.insert_seed
        families = args.families or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{args.hard_dir}/*/axis_A.jsonl"))
        self.levels, found = [], set()   # (row of the axis file, settings of the level, inserts of the axis)
        for family in families:
            paths = sorted(glob.glob(f"{args.hard_dir}/{family}/axis_*.jsonl"))
            if not paths:
                raise ValueError(f"no family {family} in {args.hard_dir}")
            for path in paths:
                axis = os.path.basename(path)[len("axis_"):-len(".jsonl")]
                with open(path, encoding="utf-8") as f:
                    rows = [json.loads(l) for l in f if l.strip()]
                inserts_path = f"{args.hard_dir}/{family}/inserts_{axis}.json"
                inserts = {"inserts": {}, "levels": {}}
                if os.path.exists(inserts_path):
                    with open(inserts_path, encoding="utf-8") as f:
                        inserts = json.load(f)
                for row in rows:
                    wanted = {"all", axis, row["level"]} & set(args.levels)
                    if wanted:
                        found |= wanted
                        self.levels.append((row, inserts["levels"].get(row["level"], {}), inserts["inserts"]))
        if set(args.levels) - found:
            raise ValueError(f"no such levels in {args.hard_dir}: {sorted(set(args.levels) - found)}")
        self.haystacks = {}

    def haystack(self, haystack_dir):
        if haystack_dir not in self.haystacks:
            text = self.read_haystack(haystack_dir, max(self.context_lengths))
            self.haystacks[haystack_dir] = self.enc.encode(text, add_special_tokens=False)
        return self.haystacks[haystack_dir]

    def place(self, settings, name, borders, n, p, needle_depth, sample_id):
        """
        Where an insert goes: (position in the haystack tokens, -1 before the needle or 1 after it), None when the
        context has no room for the distance of the level. borders: the positions after a period, 0 and n.
        """
        rng = random.Random(f"{self.seed}/{sample_id}/{name}")
        back = lambda pos: borders[bisect.bisect_right(borders, pos) - 1]
        if "distance" in settings:
            distance, before = settings["distance"], settings["side"] == "before"
            target = p - distance if before else p + distance
            if target < 0 or target > n:
                return None
            if distance == 0:
                return p, -1 if before else 1
            return (back(target), -1) if before else (borders[bisect.bisect_left(borders, target)], 1)
        placement = settings["placement"]
        if placement == "far":
            pos = back(int(n * ((needle_depth + 50) % 100) / 100))
        elif placement == "random":
            pos = rng.choice(borders)
        elif placement == "near":
            close = [b for b in borders if b != p and abs(b - p) <= self.near]
            pos = rng.choice(close) if close else p
        else:
            raise ValueError(f"unknown placement {placement!r}")
        return pos, -1 if pos < p else 1 if pos > p else rng.choice([-1, 1])

    def build(self, sample_id, haystack_tokens, row, settings, inserts, context_length, depth):
        enc = self.enc
        used = [(name, inserts[name]) for name in settings.get("use", [])]
        sentences = [row["needle"]] + [item["text"] for _, item in used]
        room = context_length - self.buffer - sum(len(enc.encode(" " + s, add_special_tokens=False)) for s in sentences)
        if room < 1:
            return None
        context_tokens = haystack_tokens[:room]
        n = len(context_tokens)
        borders = sorted({0, n} | {i + 1 for i, t in enumerate(context_tokens) if t in self.periods})
        # the depth of the run is mapped into the depth range of the level, when the level has one
        low, high = settings.get("depth_range", [0, 100])
        needle_depth = low + (high - low) * depth / 100
        # as in niah: after the last period before the depth; at 100 percent at the very end
        p = n if needle_depth == 100 else borders[bisect.bisect_right(borders[:-1], int(n * needle_depth / 100)) - 1]

        pieces = [(p, 0, 0, "needle_sentence", row["needle"])]   # position, side, order among the inserts, span, sentence
        for order, (name, item) in enumerate(used):
            place = self.place(settings, name, borders, n, p, needle_depth, sample_id)
            if place is None:
                return None
            pieces.append((place[0], place[1], order, name, item["text"]))

        def add(text, part):
            return text + (" " if text and part and not text[-1].isspace() and not part[0].isspace() else "") + part

        context, chars, prev, where = "", {}, 0, {}
        for pos, side, _, name, sentence in sorted(pieces):
            context = add(context, enc.decode(context_tokens[prev:pos]))
            prev = pos
            context = add(context, sentence)
            chars[name] = (len(context) - len(sentence), len(context))
            where[name] = pos - p if pos != p else side
        context = add(context, enc.decode(context_tokens[prev:]))

        # real_needle is marked inside the sentence that holds it: the needle or, on some levels, an insert
        home = settings.get("real_needle_in", "needle_sentence")
        home_text = context[chars[home][0]:chars[home][1]]
        start = chars[home][0] + home_text.index(row["real_needle"])
        chars["needle"] = (start, start + len(row["real_needle"]))

        if settings.get("question_position") == "before":
            content = f"Based on the content of the book below, Question: {row['question']}\n<book>{context}</book>\nAnswer:"
        else:
            content = f"<book>{context}</book>\nBased on the content of the book, Question: {row['question']}\nAnswer:"
        text = chat_prompt(enc, content) if self.chat else content
        shift = text.index(context)
        input_ids, spans = tokenize_with_char_spans(enc, text, {k: (a + shift, b + shift) for k, (a, b) in chars.items()}, special_tokens=not self.chat)

        roles = {name: item["role"] for name, item in used}
        facts = lambda *wanted: [fact for name, item in used if item["role"] in wanted for fact in item["facts"]]
        meta = {"family": row["family"], "level": row["level"], "context_length": context_length, "depth_percent": depth,
                "needle_depth": round(100 * p / max(n, 1), 1), "question": row["question"], "answer_key": row["answer_key"],
                # facts that make an answer wrong: of competing records and lures (and the wrong option named in the
                # needle itself), and separately of the noise
                "confusion_facts": facts("distractor", "lure") + settings.get("facts_in_needle", []), "noise_facts": facts("noise"),
                # role of every insert and its place: haystack tokens from the needle, negative before it; -1 or 1 right next to it
                "inserts": {name: {"role": roles[name], "offset": where[name]} for name in roles},
                "n_input_tokens": len(input_ids)}
        return Sample(sample_id, input_ids, spans, row["real_needle"], meta)

    def samples(self):
        left_out = 0
        for row, settings, inserts in self.levels:
            tokens = self.haystack(row["haystack_dir"])
            for context_length in self.context_lengths:
                for depth in self.depths:
                    sample = self.build(f"{row['family']}_{row['level']}_len{context_length}_d{depth}", tokens, row, settings, inserts, context_length, depth)
                    if sample is None:
                        left_out += 1
                    else:
                        yield sample
        if left_out:
            print(f"{left_out} samples left out: the context has no room for the inserts of the level", flush=True)


TASKS = {task.name: task for task in [NeedleInHaystack, HardNeedleInHaystack]}
