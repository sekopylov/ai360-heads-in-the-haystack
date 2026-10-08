"""
Context and prompt construction. The functions repeat retrieval_head_detection.py of the authors
step by step, so that the same arguments give the same input_ids; the differences are the arguments.
"""
import glob
import json

import numpy as np

# ids of "." in the Llama vocabulary: the authors' code uses them for every model run with the default provider
LLAMA_PERIOD_TOKENS = [29889, 869]


def load_needles(haystack_dir):
    with open(f"{haystack_dir}/needles.jsonl") as file:
        needles = [json.loads(l) for l in file]
    for i, needle in enumerate(needles):
        needle["haystack_dir"] = f"{haystack_dir}/part{i + 1}"
    return needles


def read_haystack(haystack_dir, max_context_length, sort=False):
    """sort=False: the order of the file system, as in the authors' code; sort=True: as in rh.tasks."""
    files = glob.glob(f"{haystack_dir}/*.txt")
    if sort:
        files = sorted(files)
    context = ""
    while len(context.split()) < max_context_length:
        for file in files:
            with open(file, 'r') as f:
                context += f.read()
    return context


def grid(s_len, e_len, intervals, depths=None):
    context_lengths = np.round(np.linspace(s_len, e_len, num=intervals, endpoint=True)).astype(int)
    if depths is None:
        depths = np.round(np.linspace(0, 100, num=10, endpoint=True)).astype(int)
    return [int(i) for i in context_lengths], [int(i) for i in depths]


def period_tokens(enc, mode):
    """mode 'llama': the authors' behaviour with the default provider; 'model': the period of the tokenizer."""
    if mode == "llama":
        return LLAMA_PERIOD_TOKENS
    return enc.encode('.', add_special_tokens=False)


def build_context(enc, haystack_tokens, haystack_text, needle, context_length, depth_percent, periods, buffer=200):
    """haystack_tokens = enc.encode(haystack_text), passed in so that it is computed once per haystack."""
    context = enc.decode(haystack_tokens[:context_length]) if len(haystack_tokens) > context_length else haystack_text
    tokens_needle = enc.encode(needle)
    tokens_context = enc.encode(context)

    context_length -= buffer
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]

    if depth_percent == 100:
        tokens_new_context = tokens_context + tokens_needle
    else:
        insertion_point = int(len(tokens_context) * (depth_percent / 100))
        tokens_new_context = tokens_context[:insertion_point]
        while tokens_new_context and tokens_new_context[-1] not in periods:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]
    return enc.decode(tokens_new_context)


def build_prompt(enc, context, question, chat):
    """Returns the list of prompt ids."""
    if chat:
        prompt = [{"role": "user", "content": f"<book>{context}</book>\nBased on the content of the book, Question: {question}\nAnswer:"}]
        kwargs = {"enable_thinking": False} if "enable_thinking" in (enc.chat_template or "") else {}
        ids = enc.apply_chat_template(conversation=prompt, tokenize=True, add_generation_prompt=True, **kwargs)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
    else:
        ids = enc(context + f"Based on the content of the book, Question: {question}\nAnswer:")['input_ids']
    return [int(i) for i in ids]


def build_prompt_exact(enc, context, question, chat, needle):
    """
    The same prompt as build_prompt, with the token span of the needle found by character offsets
    instead of the authors' fuzzy search. Needs a fast tokenizer. Returns ids, needle_start, needle_end.
    """
    text = context + f"Based on the content of the book, Question: {question}\nAnswer:"
    if chat:
        prompt = [{"role": "user", "content": f"<book>{context}</book>\nBased on the content of the book, Question: {question}\nAnswer:"}]
        kwargs = {"enable_thinking": False} if "enable_thinking" in (enc.chat_template or "") else {}
        text = enc.apply_chat_template(conversation=prompt, tokenize=False, add_generation_prompt=True, **kwargs)
    encoded = enc(text, add_special_tokens=not chat, return_offsets_mapping=True)
    ids = [int(i) for i in encoded["input_ids"]]
    start = text.find(needle)
    if start < 0:
        start = text.lower().find(needle.lower())
    if start < 0:
        return ids, -1, -1
    end = start + len(needle)
    span = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > start]
    return ids, span[0], span[-1] + 1


def find_needle_fuzzy(enc, prompt_ids, needle):
    """The authors' search: the first window whose set of tokens overlaps with the needle's by more than 0.9."""
    needle_ids = enc(needle, add_special_tokens=False)["input_ids"]
    span_len, target = len(needle_ids), set(needle_ids)
    for i in range(len(prompt_ids)):
        overlap = float(len(set(prompt_ids[i: i + span_len]).intersection(target))) / len(target)
        if overlap > 0.9:
            return i, i + span_len
    return -1, -1
