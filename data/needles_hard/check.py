"""
Checks of the needle families in this folder; no model is needed.

python data/needles_hard/check.py

Errors: a broken file, a level without settings, an answer that is not in the text, a fact of a competing insert that
is also in the needle, the question or the answer. Notes: words of the inserts that the haystack already has, and the
ROUGE-1 recall an answer taken from an insert would get against the reference (shown when rouge_score is installed).
"""
import glob
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
COMPETING = ("distractor", "lure", "noise")   # roles whose facts in the answer mean a wrong answer
FIELDS = ("family", "level", "needle", "question", "real_needle", "haystack_dir", "answer_key")
SAME_NEEDLE = "ABCDFN"                        # axes on which the needle is the same on every level


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFKC", text).lower())).strip()


def has(text, fact):
    return re.search(r"(?<!\w)" + re.escape(norm(fact)) + r"(?!\w)", norm(text)) is not None


def check_family(folder):
    errors, name = [], os.path.basename(folder)
    axes, inserts = {}, {}
    for path in sorted(glob.glob(f"{folder}/axis_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            axes[path[-7]] = [json.loads(l) for l in f if l.strip()]
    for path in sorted(glob.glob(f"{folder}/inserts_*.json")):
        with open(path, encoding="utf-8") as f:
            inserts[path[-6]] = json.load(f)
    base = axes["A"][0]
    for axis, rows in axes.items():
        ins = inserts.get(axis, {"inserts": {}, "levels": {}})
        if axis != "A" and axis not in inserts:
            errors.append(f"axis {axis}: no inserts file")
        for r in rows:
            lv = r.get("level", "?")
            missing = [k for k in FIELDS if k not in r]
            if missing:
                errors.append(f"{lv}: no fields {missing}")
                continue
            if r["family"] != name:
                errors.append(f"{lv}: family {r['family']!r} in the folder {name}")
            if axis in SAME_NEEDLE and r["needle"] != base["needle"]:
                errors.append(f"{lv}: the needle differs from A0")
            settings = ins["levels"].get(lv)
            if axis in inserts and settings is None:
                errors.append(f"{lv}: no settings in the inserts file")
            settings = settings or {}
            unknown = [n for n in settings.get("use", []) if n not in ins["inserts"]]
            if unknown:
                errors.append(f"{lv}: unknown inserts {unknown}")
                continue
            used = [(n, ins["inserts"][n]) for n in settings.get("use", [])]
            context = r["needle"] + " " + " ".join(u["text"] for _, u in used)
            home = ins["inserts"][settings["real_needle_in"]]["text"] if "real_needle_in" in settings else r["needle"]
            if r["real_needle"] not in home:
                errors.append(f"{lv}: real_needle is not in its text")
            if context.count(r["real_needle"]) != 1:
                errors.append(f"{lv}: real_needle occurs {context.count(r['real_needle'])} times in the needle and the inserts")
            if settings.get("real_needle_is") != "evidence":   # otherwise the answer is derived and is not in the text
                for fact in r["answer_key"]:
                    if not any(has(context, v) for v in fact):
                        errors.append(f"{lv}: answer fact {fact} is not in the text")
            for fact in settings.get("facts_in_needle", []):
                if not any(has(r["needle"], v) for v in fact):
                    errors.append(f"{lv}: wrong option {fact} is not in the needle")
                if any(has(a, v) or has(v, a) for v in fact for key in r["answer_key"] for a in key):
                    errors.append(f"{lv}: wrong option {fact} overlaps the answer")
            for n, u in used:
                if u.get("role") not in COMPETING + ("value", "evidence"):
                    errors.append(f"{lv}: insert {n} has the role {u.get('role')!r}")
                for fact in u["facts"]:
                    if not any(has(u["text"], v) for v in fact):
                        errors.append(f"{lv}: fact {fact} is not in the text of {n}")
                if u.get("role") in COMPETING:
                    for v in (v for fact in u["facts"] for v in fact):
                        if has(r["needle"], v) or has(r["question"], v):
                            errors.append(f"{lv}: fact {v!r} of {n} is in the needle or the question")
                        if any(has(a, v) for key in r["answer_key"] for a in key):
                            errors.append(f"{lv}: fact {v!r} of {n} is inside an answer fact")
                    for a in (a for key in r["answer_key"] for a in key):
                        if has(u["text"], a):
                            errors.append(f"{lv}: answer fact {a!r} is inside {n}")
    return axes, inserts, errors


if __name__ == "__main__":
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    except ImportError:
        scorer = None
    failed = False
    for folder in sorted(p for p in glob.glob(f"{HERE}/*") if os.path.isdir(p) and glob.glob(f"{p}/axis_A.jsonl")):
        axes, inserts, errors = check_family(folder)
        base = axes["A"][0]
        levels = sum(len(rows) for rows in axes.values())
        print(f"\n{os.path.basename(folder)}: {levels} levels on the axes {''.join(sorted(axes))}, errors: {len(errors)}")
        for e in errors:
            print("  ERROR", e)
        failed |= bool(errors)

        files = sorted(glob.glob(os.path.join(REPO, base["haystack_dir"], "*.txt")))
        if files:
            haystack = norm("".join(open(f, encoding="utf-8").read() for f in files))
            words = {v for ins in inserts.values() for u in ins["inserts"].values() if u.get("role") in COMPETING for fact in u["facts"] for v in fact}
            words |= {v for ins in inserts.values() for s in ins["levels"].values() for fact in s.get("facts_in_needle", []) for v in fact}
            words |= {a for rows in axes.values() for r in rows for key in r["answer_key"] for a in key}
            found = {w: len(re.findall(r"(?<!\w)" + re.escape(norm(w)) + r"(?!\w)", haystack)) for w in sorted(words)}
            found = {w: c for w, c in found.items() if c}
            print("  facts that also occur in the haystack:", found or "none")
        else:
            print(f"  haystack {base['haystack_dir']} is not here, its words are not checked")

        if scorer:
            seen, high = set(), []
            for ins in inserts.values():
                for n, u in ins["inserts"].items():
                    if u.get("role") in COMPETING and u["text"] not in seen:
                        seen.add(u["text"])
                        recall = scorer.score(base["real_needle"], u["text"])["rouge1"].recall * 100
                        if recall > 50:
                            high.append(f"{n} {recall:.0f}")
            print("  inserts that pass ROUGE-1 recall > 50 against the reference when given as the answer:", ", ".join(high) or "none")
    sys.exit(1 if failed else 0)
