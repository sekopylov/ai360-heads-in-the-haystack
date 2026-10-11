"""
How the answer is worded, fact by fact: word for word, in another form of the same words, by a close substitute, or
not at all. ROUGE and the exact phrase compare strings, so they cannot tell a reworded right answer from a wrong one;
this module shows how often rewording happens and what the string metrics do with it.

The reference of a needle is split by hand into its facts, and every fact has three sets of wordings: the wording of
the reference, the other forms of the same words and the close substitutes. The sets of FACTS below are collected from
the answers of the masking runs of the San Francisco needle: they are a list of what the model wrote, not a
dictionary of synonyms, and another needle needs its own entry.
"""
import re

import pandas as pd

from .data import plain

LEVELS = ("exact", "form", "paraphrase", "absent")
LEVEL_LABELS = {"exact": "дословно", "form": "другая форма слова", "paraphrase": "близкая замена", "absent": "нет"}

# reference -> {fact: {level: regular expression over the text in lower case without punctuation}}
FACTS = {
    "eat a sandwich and sit in Dolores Park on a sunny day": {
        "eat a sandwich": {
            "exact": r"eat a sandwich",
            "form": r"(eating|eats|ate|eaten) (a |the )?sandwich(es)?|eat (the |some )?sandwich(es)?",
            "paraphrase": r"(have|having|grab|grabbing|enjoy|enjoying) a (sandwich|meal|snack|bite)|eat(ing)? (lunch|a meal|a snack)|picnic\w*",
        },
        "sit": {
            "exact": r"sit",
            "form": r"sitting|sits|sat|seated",
            "paraphrase": r"relax\w*|loung\w*|hang(ing)? out",
        },
        "Dolores Park": {
            "exact": r"dolores park",
            "form": r"dolores|dol\w+ park",          # the name alone or misspelled: "Dolano Park"
            "paraphrase": r"the park",               # the place without its name
        },
        "sunny day": {
            "exact": r"sunny day",
            "form": r"sunny",
            "paraphrase": r"(in|enjoy|enjoying) the sun(shine)?|warmth of the sun|sunshine|(nice|beautiful|warm) (day|weather|afternoon)",
        },
    },
}

STRUCTURES = ("exact", "reordered", "reworded", "cut", "cut_reworded", "none")
STRUCTURE_LABELS = {
    "exact": "точная фраза",
    "reordered": "все факты дословно, другой порядок или вставка",
    "reworded": "все факты, часть — формой слова или заменой",
    "cut": "часть фактов, все дословно",
    "cut_reworded": "часть фактов, есть форма слова или замена",
    "none": "ни одного факта",
}


def fact_levels(reference, response):
    """{fact: level} — the closest wording of every fact of the reference that stands in the answer."""
    text = f" {plain(response)} "
    return {fact: next((level for level in LEVELS[:-1] if re.search(rf"(?<=\s)({wordings[level]})(?=\s)", text)), "absent")
            for fact, wordings in FACTS[reference].items()}


def structure(levels, exact):
    """The answer as a whole by the wording of its facts; exact: the reference phrase stands in the answer."""
    found = [level for level in levels.values() if level != "absent"]
    verbatim = all(level == "exact" for level in found)
    if exact:
        return "exact"
    if not found:
        return "none"
    if len(found) == len(levels):
        return "reordered" if verbatim else "reworded"
    return "cut" if verbatim else "cut_reworded"


def annotate(samples):
    """
    The table of the samples with a column per fact (its level), facts_found (how many facts are there in any
    wording, of len(facts)) and structure. Every reference of the table must have its entry in FACTS.
    """
    missing = set(samples["reference"]) - set(FACTS)
    if missing:
        raise KeyError(f"no facts in vizual.paraphrase.FACTS for the references {sorted(missing)}")
    levels = [fact_levels(a, b) for a, b in zip(samples["reference"], samples["response"])]
    table = pd.DataFrame(levels, index=samples.index)
    out = samples.join(table)
    out["facts_found"] = (table != "absent").sum(axis=1)
    out["structure"] = [structure(l, e) for l, e in zip(levels, samples["exact"])]
    return out


def wordings(samples, top=8):
    """What the model actually wrote for every fact at the levels form and paraphrase: the matched strings and their counts."""
    rows = []
    for reference, facts in FACTS.items():
        texts = [f" {plain(r)} " for r in samples.loc[samples["reference"] == reference, "response"]]
        for fact, patterns in facts.items():
            for level in ("form", "paraphrase"):
                counts = {}
                for text in texts:
                    # the level of the fact in this answer is the closest one; a match counts only when it decides it
                    if any(re.search(rf"(?<=\s)({patterns[l]})(?=\s)", text) for l in LEVELS[:LEVELS.index(level)]):
                        continue
                    found = re.search(rf"(?<=\s)({patterns[level]})(?=\s)", text)
                    if found:
                        counts[found.group(1)] = counts.get(found.group(1), 0) + 1
                for wording, n in sorted(counts.items(), key=lambda kv: -kv[1])[:top]:
                    rows.append({"факт": fact, "уровень": LEVEL_LABELS[level], "написано": wording, "ответов": n})
    return pd.DataFrame(rows)
