"""
Tables for the runs of the task hard_niah (data/needles_hard): the answers by their facts and the head metrics by level.

The layout of the results is results/new/qwen3_hard_fast/<axis>/<run>, the runs being none (no mask, the attention is
saved, so it has the head metrics) and the masked ones (top40, top60, random60: answers only).
"""
import numpy as np
import pandas as pd

from . import data

CLASSES = ("correct", "mixed", "partial", "confusion", "noise", "other")
CLASS_LABELS = {"correct": "верно: все факты", "mixed": "смешано: все факты и чужие", "partial": "частично: не все факты",
                "confusion": "путаница: факты вставок", "noise": "шум: факты шума", "other": "иное: ни одного факта"}
# what every class is, for the tables of the notebooks; the rule itself is rh.metrics.fact_class
CLASS_CRITERIA = {
    "correct": "в ответе есть все факты answer_key и нет фактов конкурирующих вставок",
    "mixed": "все факты answer_key есть, но названы и факты конкурирующих вставок или шума",
    "partial": "есть часть фактов answer_key, фактов вставок нет",
    "confusion": "не все факты answer_key, есть факты близнецов или приманок",
    "noise": "не все факты answer_key, есть факты шума",
    "other": "ни одного из этих фактов",
}
CELL = ["family", "context_length", "depth_percent"]   # one place of one needle: the unit of the paired comparisons


def load(pattern, limit=None):
    """
    The runs under the pattern in one table (see data.load_sweep): the answers are read from outputs/ and the class
    of the answer by its facts (answer_class, facts_found) comes from rh.metrics.fact_class. Added column: correct.
    """
    df = data.load_sweep(pattern, limit)
    df["correct"] = df["answer_class"] == "correct"
    return df


def by_level(df, value="correct", by=("run", "level")):
    """The mean of a column by run and level, percent for a yes/no column: a table with the levels as columns."""
    column = df[value].astype(float) * (100 if df[value].dtype == bool else 1)
    return df.assign(_v=column).groupby(list(by))["_v"].mean().unstack("level").round(1)


def paired(df, base="none"):
    """
    The masked runs against the run without a mask on the same samples: of the samples the base run answers
    correctly, the share every other run still answers correctly, by level, percent. The count of such samples is
    in the row "n".
    """
    key = ["level"] + CELL
    ok = df[(df["run"] == base) & df["correct"]][key]
    rows = {"n": ok.groupby("level").size()}
    for run in [r for r in df["run"].unique() if r != base]:
        part = df[df["run"] == run].merge(ok, on=key)
        rows[run] = part.groupby("level")["correct"].mean() * 100
    return pd.DataFrame(rows).T.round(1)


def common_cells(samples, levels=None):
    """
    A yes/no mask over the samples: the place of the needle (family, length, depth) is answered correctly at every
    level. The head metrics of the levels are compared on these samples, so that a difference between the levels
    is not a difference between the sets of samples.
    """
    levels = levels or sorted(samples["level"].unique())
    good = samples[samples["correct"] & samples["level"].isin(levels)].groupby(CELL)["level"].nunique()
    keep = set(good[good == len(levels)].index)
    return np.array([tuple(row) in keep for row in samples[CELL].itertuples(index=False)])


def level_scores(heads, metric="copy_recall", where=None, levels=None):
    """{level: [layer, head]} — the score of every head on the samples of every level (heads made with ok="correct")."""
    return {level: heads.score(metric, level_mask(heads, level, where)) for level in levels or sorted(heads.samples["level"].unique())}


def level_mask(heads, level, where=None):
    """A yes/no mask over the samples of heads: the samples of the level, and of where when it is given."""
    mask = (heads.samples["level"] == level).to_numpy()
    return mask if where is None else mask & np.asarray(where)


def top_means(scores, order, ks=(20, 40, 60)):
    """
    The mean score of the k best heads of a ranking (order: flat indices, the best first) in every condition of
    scores; k as in the masks of the runs.
    """
    return pd.DataFrame({cond: {f"{k} лучших": float(np.ravel(score)[order[:k]].mean()) for k in ks} for cond, score in scores.items()}).round(4)


def ranking_table(scores, reference, ks=(20, 40, 60)):
    """Every condition of scores against the reference scores: Spearman over all heads and the overlap of the k best heads."""
    rows = []
    for name, score in scores.items():
        rows.append({"условие": name, "спирмен": round(data.spearman(score, reference), 3),
                     **{f"top{k}": f"{data.top_overlap(score, reference, k)}/{k}" for k in ks}})
    return pd.DataFrame(rows).set_index("условие")


def correlation_table(scores, reference, order, ks=(20, 40, 60), threshold=0.1):
    """
    The scores of the heads in every condition against the reference scores (the detection on the simple task):
    Pearson and Spearman over all heads, over the heads the detection puts above the threshold and over its k best
    heads (order: flat indices of the heads, the best of the detection first). Over all heads the correlation is
    carried by the many heads with a score near zero; the subsets say whether the retrieval heads themselves keep
    their values and their order.
    """
    reference = np.ravel(reference)
    subsets = {"все головы": np.arange(len(reference)), f"детекция выше {threshold:g}": np.flatnonzero(reference > threshold),
               **{f"{k} лучших детекции": order[:k] for k in ks}}
    rows = {}
    for name, score in scores.items():
        score = np.ravel(score)
        for subset, idx in subsets.items():
            rows[(name, subset)] = {"голов": len(idx), "пирсон": data.pearson(score[idx], reference[idx]), "спирмен": data.spearman(score[idx], reference[idx])}
    return pd.DataFrame(rows).T.rename_axis(["условие", "головы"]).astype({"голов": int}).round(3)


def twin_words(df):
    """
    A yes/no column: the answer has a word of a competing insert. The class of the answer counts a fact of an
    insert only in the spelling of the data ("iron lever"), so "pull the lever" is not a confusion for it. Here the
    last word of every spelling of the competing facts is looked for ("lever", "clock", "raven"), without the words
    that the reference or the question have themselves. A softer sign of the same thing; not a class.
    """
    out = []
    for row in df.to_dict("records"):
        own = set(data.plain(row["reference"]).split()) | set(data.plain(row["question"]).split())
        words = {data.plain(spelling).split()[-1] for fact in row["confusion_facts"] for spelling in fact} - own
        out.append(bool(words & set(data.plain(row["response"]).split())))
    return np.array(out)


def selectivity(needle, competing):
    """
    How much of the attention that goes to the needle or to its strongest competitor goes to the needle:
    needle / (needle + competing), per head; 0.5 where the head looks at neither. 1: only the needle, 0.5: both
    equally, 0: only the competitor.
    """
    total = needle + competing
    return np.where(total > 0, needle / np.where(total > 0, total, 1), 0.5)
