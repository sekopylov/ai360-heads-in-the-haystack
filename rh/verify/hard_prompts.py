"""
Checks what the task hard_niah gives to the model: every sample is built, its tokens are decoded back to text and
the text is compared with what the level promises. Tokenizer only, no model.

python -m rh.verify.hard_prompts --model_path Qwen/Qwen3-8B --levels A,B,N \
    --lengths 1000,4053,7105,10158,13211,16263,19316,22368,25421,28474,30000 --depths 0,22,44,67,89,100

Checked for every sample:
    frame      the prompt is the chat prompt of the model around <book>...</book>, the question of the level and
               "Answer:"; the reasoning of the model is switched off when its template can do it
    needle     the needle and the answer stand in the book once; the facts of the answer are nowhere else
    inserts    exactly the inserts of the level, once each; no insert of another level of the family
    borders    the needle and every insert start after the end of a sentence and are not glued to the text
    book_end   not an error, a property to know: the sentence stands at the very end of the book after a haystack
               sentence that was cut by the context length (the needle at the depth 100)
    places     near: within the limit of the task from the needle; far: at the opposite depth; distance: on the
               right side and not closer than asked; the needle itself not deeper than the depth of the run
    spans      every span decodes to its sentence, the answer lies inside the sentence that holds it
    length     the prompt is not longer than the context length and not much shorter
    pairing    the levels of an axis with the same grid point share the needle; on the axis A also the whole book
"""
import argparse
import collections
import re

from transformers import AutoTokenizer

from ..tasks import TASKS

Task = TASKS["hard_niah"]


def count(text, part):
    return len(re.findall(re.escape(part), text))


def check(enc, task, sample, row, settings, inserts, think_off):
    problems = []
    add = lambda kind, message: problems.append((kind, message))
    text = enc.decode(sample.input_ids)
    meta, question, needle = sample.meta, row["question"], row["needle"]
    used = {name: inserts[name] for name in settings.get("use", [])}

    # frame
    if text.count("<book>") != 1 or text.count("</book>") != 1:
        add("frame", "not exactly one <book>...</book>")
        return problems, None
    book = text[text.index("<book>") + len("<book>"):text.index("</book>")]
    after = text[text.index("</book>") + len("</book>"):]
    before = text[:text.index("<book>")]
    if settings.get("question_position") == "before":
        if f"Question: {question}\n" not in before or not after.startswith("\nAnswer:"):
            add("frame", "the question is not before the book")
    elif not after.startswith(f"\nBased on the content of the book, Question: {question}\nAnswer:"):
        add("frame", f"after the book: {after[:120]!r}")
    if text.count(question) != 1:
        add("frame", f"the question stands {text.count(question)} times")
    if think_off and not text.rstrip().endswith("</think>"):
        add("frame", f"the reasoning is not switched off, the prompt ends with {text[-60:]!r}")

    # needle and answer
    if count(book, needle) != 1:
        add("needle", f"the needle stands {count(book, needle)} times in the book")
    if count(text, row["real_needle"]) != 1:
        add("needle", f"real_needle stands {count(text, row['real_needle'])} times in the prompt")
    sentences = [needle] + [item["text"] for item in used.values()]
    rest = book
    for sentence in sentences:
        rest = rest.replace(sentence, " ")
    for fact in row["answer_key"] + [f for item in used.values() for f in item["facts"]]:
        for spelling in fact:
            if re.search(r"(?<!\w)" + re.escape(spelling.lower()) + r"(?!\w)", rest.lower()):
                add("needle", f"the fact {spelling!r} is also in the haystack of this prompt")

    # inserts
    for name, item in used.items():
        if count(book, item["text"]) != 1:
            add("inserts", f"{name} stands {count(book, item['text'])} times")
    for name, item in inserts.items():
        if name not in used and item["text"] not in sentences and item["text"] in book:
            add("inserts", f"{name} of another level is in the book")
    if set(meta["inserts"]) != set(used):
        add("inserts", f"meta has {sorted(meta['inserts'])}, the level uses {sorted(used)}")

    # borders
    for sentence in sentences:
        at = book.find(sentence)
        if at < 0:
            continue
        left, right = book[:at], book[at + len(sentence):]
        if left.strip() and (not left[-1].isspace() or left.rstrip()[-1] not in ".?!\"')”"):
            tail = right
            for other in sentences:
                tail = tail.replace(other, "")
            # the context is cut by tokens, so the book may end in the middle of a sentence; what is put at the very
            # end of the book (the needle at the depth 100) then follows this unfinished sentence
            add("book_end" if not tail.strip() else "borders", f"before {sentence[:30]!r}: {left[-25:]!r}")
        if right.strip() and not right[0].isspace():
            add("borders", f"after {sentence[:30]!r}: {right[:25]!r}")

    # spans
    piece = lambda name: enc.decode(sample.input_ids[sample.spans[name][0]:sample.spans[name][1]])
    wanted = {"needle_sentence": needle, **{name: item["text"] for name, item in used.items()}}
    if set(sample.spans) != set(wanted) | {"needle"}:
        add("spans", f"spans {sorted(sample.spans)}")
    for name, sentence in wanted.items():
        if name in sample.spans and (sentence not in piece(name) or len(piece(name)) > len(sentence) + 12):
            add("spans", f"{name} decodes to {piece(name)[:60]!r}")
    home = settings.get("real_needle_in", "needle_sentence")
    if row["real_needle"] not in piece("needle") or not (sample.spans[home][0] <= sample.spans["needle"][0] and sample.spans["needle"][1] <= sample.spans[home][1]):
        add("spans", f"the answer span is {piece('needle')[:60]!r}")

    # places, in tokens of the prompt
    start, end = sample.spans["needle_sentence"]
    book_start = len(enc.encode(text[:text.index("<book>") + len("<book>")], add_special_tokens=False))
    book_tokens = max(len(enc.encode(book, add_special_tokens=False)), 1)
    depth = 100 * (start - book_start) / book_tokens
    low, high = settings.get("depth_range", [0, 100])
    asked = low + (high - low) * meta["depth_percent"] / 100
    # inserts put before the needle move it deeper by their own length
    shift = 100 * sum(sample.spans[name][1] - sample.spans[name][0] for name in used if sample.spans[name][0] < start) / book_tokens
    if depth > asked + shift + 1 and asked < 100:
        add("places", f"the needle is at {depth:.1f}% of the book, asked {asked:.0f}%")
    for name in used:
        a, b = sample.spans[name]
        gap = a - end if a >= end else start - b       # tokens of the book between the insert and the needle
        side = "after" if a >= end else "before"
        placement = settings.get("placement")
        if "distance" in settings:
            if side != settings["side"] or gap < settings["distance"] - 3:
                add("places", f"{name} is {gap} tokens {side}, asked {settings['distance']} {settings['side']}")
        elif placement == "near" and gap > task.near + sum(s[1] - s[0] for n, s in sample.spans.items() if n not in ("needle", "needle_sentence", name)):
            add("places", f"{name} is {gap} tokens from the needle, the limit is {task.near}")
        elif placement == "far":
            want = (asked + 50) % 100
            got = 100 * (a - book_start) / book_tokens
            if abs(got - want) > 12:
                add("places", f"{name} is at {got:.0f}% of the book, the opposite depth is {want:.0f}%")

    # length
    if not (meta["context_length"] - 260 <= len(sample.input_ids) <= meta["context_length"]):
        add("length", f"{len(sample.input_ids)} tokens for the context length {meta['context_length']}")
    for sentence in sentences:
        book = book.replace(sentence, "")
    return problems, re.sub(r"\s+", " ", book)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="the tokenizer of this model is used")
    parser.add_argument("--show", type=int, default=1, help="number of samples of every level to print around the needle")
    Task.add_arguments(parser)
    args = parser.parse_args()
    enc = AutoTokenizer.from_pretrained(args.model_path)
    task = Task(enc, args)
    think_off = "enable_thinking" in (enc.chat_template or "")
    probe = enc.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True, **({"enable_thinking": False} if think_off else {}))
    print(f"tokenizer {args.model_path}; the template can switch the reasoning off: {think_off}; the prompt ends with {probe[-40:]!r}")
    levels = {(row["family"], row["level"]): (row, settings, inserts) for row, settings, inserts in task.levels}

    problems = collections.Counter()
    examples, per_level, lengths, books, needles, shown = {}, collections.Counter(), collections.defaultdict(list), {}, {}, collections.Counter()
    for sample in task.samples():
        meta = sample.meta
        row, settings, inserts = levels[(meta["family"], meta["level"])]
        found, book = check(enc, task, sample, row, settings, inserts, think_off)
        per_level[meta["level"]] += 1
        lengths[meta["level"]].append(len(sample.input_ids))
        point = (meta["family"], meta["context_length"], meta["depth_percent"])
        # pairing: the same needle on the levels of an axis; on the axis A, where only the question changes, the same book
        axis = meta["level"][0]
        if needles.setdefault((axis,) + point, row["needle"]) != row["needle"] and axis in "ABCDFN":
            found.append(("pairing", "the needle differs from the other levels of the axis"))
        if axis == "A" and book is not None and books.setdefault(point, book) != book:
            found.append(("pairing", "the book differs from the level A0"))
        for kind, message in found:
            problems[(meta["level"], kind)] += 1
            examples.setdefault((meta["level"], kind), f"{sample.id}: {message}")
        if shown[(meta["family"], meta["level"])] < args.show and meta["context_length"] >= 4000 and 20 <= meta["depth_percent"] <= 80:
            shown[(meta["family"], meta["level"])] += 1
            low = min(s[0] for s in sample.spans.values())
            high = max(s[1] for s in sample.spans.values())
            print(f"\n--- {sample.id}: {len(sample.input_ids)} tokens; inserts {meta['inserts']}")
            if high - low < 450:
                print("    " + repr(enc.decode(sample.input_ids[max(0, low - 12):high + 12])))
            else:
                for name, (a, b) in sorted(sample.spans.items(), key=lambda kv: kv[1]):
                    if name != "needle":
                        print(f"    {name:<16} tokens {a}..{b}: " + repr(enc.decode(sample.input_ids[max(0, a - 10):b + 10])))
            print("    end of the prompt: " + repr(enc.decode(sample.input_ids[-45:])))

    print(f"\n{'level':<16} {'samples':>8} {'tokens, shortest..longest':>28}")
    for level in sorted(per_level):
        print(f"{level:<16} {per_level[level]:>8} {min(lengths[level]):>13}..{max(lengths[level])}")
    errors = {k: n for k, n in problems.items() if k[1] != "book_end"}
    print(f"\nsamples: {sum(per_level.values())}; errors: {sum(errors.values())}")
    for (level, kind), n in sorted(errors.items()):
        print(f"  {level:<16} {kind:<9} {n:>5}   e.g. {examples[(level, kind)]}")
    notes = {k: n for k, n in problems.items() if k[1] == "book_end"}
    if notes:
        print(f"at the very end of the book after a cut sentence: {sum(notes.values())} sentences, e.g. {examples[next(iter(sorted(notes)))]}")
        print("  by level: " + ", ".join(f"{level} {n}" for (level, _), n in sorted(notes.items())))
    raise SystemExit(1 if errors else 0)
