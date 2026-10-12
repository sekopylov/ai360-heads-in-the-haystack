"""
How the answer is worded, part by part: word for word, in another form of the same words, by a synonym, with a detail
lost, or not at all. ROUGE and the exact phrase compare strings, so they cannot tell a reworded right answer from a
wrong one; this module shows how often rewording happens and what the string metrics do with it.

The reference of a needle is split by hand into its parts (facts), and every part has sets of wordings: the wording
of the reference, the other forms of the same words, the synonyms that keep the meaning and the wordings that lose a
detail. The sets of FACTS below are collected from the answers of the runs: they are a list of what the model wrote,
not a dictionary of synonyms, and another needle needs its own entry.
"""
import re

import pandas as pd

from .data import plain

LEVELS = ("exact", "form", "paraphrase", "reduced", "absent")
LEVEL_LABELS = {"exact": "дословно", "form": "другая форма слова", "paraphrase": "синоним, смысл сохранён", "reduced": "потеряна деталь", "absent": "нет"}
KEPT = ("exact", "form", "paraphrase")   # the levels at which the part is there in full

# reference -> {fact: {level: regular expression over the text in lower case without punctuation}}; a level may be missing
FACTS = {
    # the needle of the paper's masking runs
    "eat a sandwich and sit in Dolores Park on a sunny day": {
        "eat a sandwich": {
            "exact": r"eat a sandwich",
            "form": r"(eating|eats|ate|eaten) (a |the )?sandwich(es)?|eat (the |some )?sandwich(es)?",
            "paraphrase": r"(have|having|grab|grabbing|enjoy|enjoying) a sandwich",
            "reduced": r"(have|having|grab|grabbing|enjoy|enjoying) a (meal|snack|bite)|eat(ing)? (lunch|a meal|a snack)|picnic\w*",   # no sandwich
        },
        "sit": {
            "exact": r"sit",
            "form": r"sitting|sits|sat|seated",
            "paraphrase": r"relax\w*|loung\w*|hang(ing)? out",
        },
        "Dolores Park": {
            "exact": r"dolores park",
            "form": r"dolores|dol\w+ park",          # the name alone or misspelled: "Dolano Park"
            "reduced": r"the park",                  # the place without its name
        },
        "sunny day": {
            "exact": r"sunny day",
            "form": r"sunny",
            "paraphrase": r"(in|enjoy|enjoying) the sun(shine)?|warmth of the sun|sunshine|(nice|beautiful|warm) (day|weather|afternoon)",
        },
    },
    # the needles of data/needles_hard; the wordings are collected from the answers of the runs of the axis A
    "ring the brass bell at the entrance and sign the guest book with a green pen": {
        "ring the brass bell": {
            "exact": r"ring the brass bell",
            "form": r"(ring|rings|ringing|ringe) (a|the) brass bell",
            "reduced": r"(ring|rings|ringing) (a|the) bell",              # the bell without its metal
        },
        "at the entrance": {
            "exact": r"at the entrance",
            "form": r"at (its|their|the museum) entrance",
            "paraphrase": r"when (they|guests|visitors) arrive|upon arriv\w+|arriving at|entering",
        },
        "sign the guest book": {
            "exact": r"sign the guest book",
            "form": r"(sign|signs|signing) (a|the|their) (guest book|guestbook)",
            "paraphrase": r"(sign|signs|signing) their names? in (a|the) (guest book|guestbook)",
            "reduced": r"guest book|guestbook",                           # the book without the signing
        },
        "with a green pen": {
            "exact": r"with a green pen",
            "form": r"with (the|their) green pen",
            "paraphrase": r"(using|in) (a|the) green pen",
            "reduced": r"green pen",                                      # "sign a green pen"
        },
    },
    "cooked with smoked lemon peel and served in a bowl of black clay": {
        "cooked with": {
            "exact": r"cooked with",
            "form": r"(cooking|cook|cooks) (it )?with",
            "paraphrase": r"(prepared|made) (with|using|by using)",
            "reduced": r"prepar\w+",                                      # prepared, but not said with what
        },
        "smoked lemon peel": {
            "exact": r"smoked lemon peel",
            "form": r"smoked lemon peels",
            "reduced": r"smoked lemon",                                   # the lemon without the peel
        },
        "served in": {
            "exact": r"served in",
            "form": r"(serving|serve|serves) (it )?in",
            "paraphrase": r"presented in",
            "reduced": r"present\w*|serv\w+",
        },
        "bowl of black clay": {
            "exact": r"bowl of black clay",
            "form": r"bowls of black clay|bowls? made of black clay|black clay bowls?",
            "reduced": r"black clay|black bowl|bowl",
        },
    },
    "a roof of red slate and a door painted with a white reindeer": {
        "roof of red slate": {
            "exact": r"roof of red slate",
            "form": r"roof of red slates|roof made of red slates?|red slated? roof",
            "reduced": r"red roof|roof made of red tiles|red tiled roof|roof",
        },
        "door painted": {
            "exact": r"door painted",
            "form": r"painted door",
            "reduced": r"door",
        },
        "with a white reindeer": {
            "exact": r"with a white reindeer",
            "form": r"white reindeer",
            "reduced": r"reindeer",                                       # "painted white with a reindeer"
        },
    },
}

STRUCTURES = ("exact", "reordered", "reworded", "cut", "cut_reworded", "none")
STRUCTURE_LABELS = {
    "exact": "точная фраза",
    "reordered": "все части дословно, другой порядок или вставка",
    "reworded": "все части, есть форма слова или синоним",
    "cut": "не все части, остальные дословно",
    "cut_reworded": "не все части, есть форма, синоним или потеря детали",
    "none": "ни одной части",
}


def found(text, wordings, level):
    """The match of the wording of this level in the text (padded with spaces), or None."""
    return re.search(rf"(?<=\s)({wordings[level]})(?=\s)", text) if level in wordings else None


def fact_levels(reference, response):
    """{fact: level} — the closest wording of every fact of the reference that stands in the answer."""
    text = f" {plain(response)} "
    return {fact: next((level for level in LEVELS[:-1] if found(text, wordings, level)), "absent") for fact, wordings in FACTS[reference].items()}


def structure(levels, exact):
    """
    The answer as a whole by the wording of its parts; exact: the reference phrase stands in the answer. A part
    with a lost detail does not count as kept: the answer is then not a whole one.
    """
    there = [level for level in levels.values() if level != "absent"]
    verbatim = all(level == "exact" for level in there)
    if exact:
        return "exact"
    if not there:
        return "none"
    if all(level in KEPT for level in levels.values()):
        return "reordered" if verbatim else "reworded"
    return "cut" if verbatim else "cut_reworded"


def annotate(samples):
    """
    The table of the samples with a column per fact (its level), kept_facts (how many parts of the reference are
    there in full: word for word, in another form or by a synonym) and structure. Every reference of the table
    must have its entry in FACTS.
    """
    missing = set(samples["reference"]) - set(FACTS)
    if missing:
        raise KeyError(f"no facts in vizual.paraphrase.FACTS for the references {sorted(missing)}")
    levels = [fact_levels(a, b) for a, b in zip(samples["reference"], samples["response"])]
    table = pd.DataFrame(levels, index=samples.index)   # a column per fact; empty where the fact is of another reference
    out = samples.join(table)
    out["kept_facts"] = table.isin(KEPT).sum(axis=1)
    out["structure"] = [structure(l, e) for l, e in zip(levels, samples["exact"])]
    return out


def wordings(samples, top=8):
    """What the model actually wrote for every fact at the levels other than exact: the matched strings and their counts."""
    rows = []
    for reference, facts in FACTS.items():
        if reference not in set(samples["reference"]):
            continue
        texts = [f" {plain(r)} " for r in samples.loc[samples["reference"] == reference, "response"]]
        for fact, patterns in facts.items():
            for level in LEVELS[1:-1]:
                counts = {}
                for text in texts:
                    # the level of the fact in this answer is the closest one; a match counts only when it decides it
                    if any(found(text, patterns, l) for l in LEVELS[:LEVELS.index(level)]):
                        continue
                    match = found(text, patterns, level)
                    if match:
                        counts[match.group(1)] = counts.get(match.group(1), 0) + 1
                for wording, n in sorted(counts.items(), key=lambda kv: -kv[1])[:top]:
                    rows.append({"эталон": reference, "факт": fact, "уровень": LEVEL_LABELS[level], "написано": wording, "ответов": n})
    return pd.DataFrame(rows)
