"""
Reads the results of rh.run / rh.metrics into tables. No model and no spool: only samples.jsonl, heads/*.npz and
spool/run.json of the --out folders.

    run     one --out folder: a table of its samples
    sweep   several folders (a masking sweep) in one table, a row per sample
    heads   the head metrics of a detection run: an array [sample, layer, head]
"""
import glob
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(p):
    """A path relative to the repository root, so the notebook works from any folder."""
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_info(out):
    """The settings of the run from spool/run.json: kind of the mask, the masked heads, the layers with attention."""
    p = os.path.join(path(out), "spool", "run.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def plain(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFKC", text).lower())).strip()


NGRAMS = {1: "слов", 2: "двуграмм", 3: "триграмм"}   # the units of ROUGE-n, in the genitive plural


def ngram_recalls(reference, response):
    """
    ROUGE-2 and ROUGE-3 recall, percent: the share of the pairs and of the triples of neighbouring words of the
    reference that stand in the answer. The scorer is the one of rh.metrics (with stemming), so the values are the
    same as rh.metrics writes into samples.jsonl of the newer runs.
    """
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from rh.metrics import scorer
    score = scorer.score(reference, response)
    return score["rouge2"].recall * 100, score["rouge3"].recall * 100


def success_column(n):
    """The column of the success criterion by ROUGE-n: recall above 50."""
    return "success" if n == 1 else f"success{n}"


def type_column(n):
    return "answer_type" if n == 1 else f"answer_type{n}"


def answer_type(row, n=1):
    """
    What happened to the answer, from the best to the worst, by ROUGE-n recall (n = 1: words, 2: pairs, 3: triples):
        exact     the reference phrase stands in the answer word for word
        full      success by ROUGE-n (recall above 50) without the exact phrase
        partial   recall from 0 to 50: an incomplete retrieval
        miss      recall 0
    """
    recall = row[f"rouge{n}_recall"]
    if row["exact"]:
        return "exact"
    if recall > 50:
        return "full"
    return "partial" if recall > 0 else "miss"


ANSWER_TYPES = ("exact", "full", "partial", "miss")


def type_criteria(n=1):
    """The rule of every type in words, checked in this order; for the tables of the notebook."""
    unit = NGRAMS[n]
    return {
        "exact": "эталон стоит в ответе дословно и подряд; регистр и пунктуация не учитываются",
        "full": f"дословной фразы нет, но ROUGE-{n} recall > 50: в ответе больше половины {unit} эталона",
        "partial": f"ROUGE-{n} recall от 0 до 50: есть часть {unit} эталона, но не больше половины",
        "miss": f"ROUGE-{n} recall = 0: ни одного совпадения {unit} с эталоном",
    }


def load_run(out, limit=None, name=None):
    """
    The samples of one run. limit: the answers as if the generation were cut at this number of tokens (by_limit of
    samples.jsonl). Added columns: run, mask (kind), heads (how many are masked), exact, rouge2_recall, rouge3_recall
    (when the run was made before they were stored), success2, success3 (ROUGE-2 and ROUGE-3 recall above 50, as
    success is for ROUGE-1) and answer_type, answer_type2, answer_type3.
    """
    out = path(out)
    info = run_info(out)
    df = pd.DataFrame(read_jsonl(os.path.join(out, "samples.jsonl")))
    if limit is not None:
        at = pd.DataFrame([row[str(limit)] for row in df["by_limit"]])
        cut = int((df["n_tokens"] > limit).sum())
        if "rouge2_recall" not in at and cut:
            # the text cut at the limit is not in samples.jsonl of the older runs: nothing to count the n-grams on.
            # Without longer answers (e.g. the limit is the cap of the run) the whole answers are the answers at the limit
            raise ValueError(f"{out}: {cut} answers are longer than {limit} tokens and ROUGE-2, ROUGE-3 at a token limit are not "
                             "stored for this run; use limit=None or rebuild samples.jsonl with python -m rh.metrics ... --rescore")
        for column in at:
            df[column] = at[column].values
    block = info.get("block_list") or []
    df["run"] = name or os.path.basename(os.path.normpath(out))
    df["mask"] = info.get("mask", "none")
    df["heads"] = len(block)
    if "exact" not in df:
        df["exact"] = [f" {plain(a)} " in f" {plain(b)} " for a, b in zip(df["reference"], df["response"])]
    if "rouge2_recall" not in df or "rouge3_recall" not in df:
        recalls = [ngram_recalls(a, b) for a, b in zip(df["reference"], df["response"])]
        df["rouge2_recall"], df["rouge3_recall"] = [r[0] for r in recalls], [r[1] for r in recalls]
    for n in (2, 3):
        df[success_column(n)] = df[f"rouge{n}_recall"] > 50
    for n in NGRAMS:
        df[type_column(n)] = df.apply(answer_type, axis=1, n=n)
    return df.drop(columns=["by_limit"], errors="ignore")


def load_sweep(pattern, limit=None):
    """Every run under the pattern (e.g. results/new/qwen3_mask/*) in one table."""
    outs = sorted(p for p in glob.glob(path(pattern)) if os.path.exists(os.path.join(p, "samples.jsonl")))
    if not outs:
        raise FileNotFoundError(f"no runs with samples.jsonl under {path(pattern)}")
    return pd.concat([load_run(p, limit) for p in outs], ignore_index=True)


def summary(sweep, by=("mask", "heads", "run")):
    """A row per run: the table behind the figures."""
    g = sweep.groupby(list(by), sort=True)
    table = g.agg(samples=("id", "size"), rouge1=("rouge1_recall", "mean"), rouge2=("rouge2_recall", "mean"), rouge3=("rouge3_recall", "mean"),
                  success=("success", "mean"), success2=("success2", "mean"), success3=("success3", "mean"), exact=("exact", "mean"),
                  tokens=("n_tokens", "mean"), truncated=("stopped", lambda s: 1 - s.mean()))
    table[["success", "success2", "success3", "exact", "truncated"]] *= 100
    return table.round(1).reset_index()


def round_bins(values, step, unit=""):
    """
    The real values (lengths 1000, 2526, ... or depths 0, 11, ...) put on an even scale: every value goes to its
    bin of the width step, the bins are named by their round borders ("0–5k", "20–40"). An ordered categorical.
    """
    values = np.asarray(values, dtype=float)
    top = max(step, int(np.ceil(values.max() / step)) * step)
    edges = np.arange(0, top + step, step)
    short = lambda v: f"{v / 1000:g}" if unit == "k" else f"{v:g}"
    labels = [f"{short(a)}–{short(b)}{unit}" for a, b in zip(edges[:-1], edges[1:])]
    return pd.cut(values, edges, labels=labels, include_lowest=True)


def round_label(value, step=500):
    """A context length a little rounded for a label: 2526 -> "2.5k", 17789 -> "18k"."""
    return f"{round(value / step) * step / 1000:g}k"


class Heads:
    """
    The head metrics of a run made with the attention saved (a detection run).

        samples   the table of the samples that have head metrics, in the order of the arrays
        layers    the numbers of the layers with attention
        metrics   the names of the metrics in heads/*.npz, the ones at token limits included ("copy_count@128")
    """

    def __init__(self, out):
        self.out = path(out)
        samples = load_run(self.out)
        has = [os.path.exists(os.path.join(self.out, "heads", f"{i}.npz")) for i in samples["id"]]
        self.samples = samples[has].reset_index(drop=True)
        self._arrays = {}
        for i in self.samples["id"]:
            with np.load(os.path.join(self.out, "heads", f"{i}.npz")) as d:
                for k in d.files:
                    self._arrays.setdefault(k, []).append(d[k])
        self._arrays = {k: np.stack(v) for k, v in self._arrays.items()}
        self.metrics = sorted(self._arrays)
        shape = next(iter(self._arrays.values())).shape
        self.layers = run_info(self.out).get("attn_layers") or list(range(shape[1]))
        self.n_heads = shape[2]

    def values(self, metric="copy_recall", where=None, successful=True):
        """[sample, layer, head]; only the successful samples, as in the paper; where: a boolean mask over samples."""
        keep = self.samples["success"].to_numpy() if successful else np.ones(len(self.samples), dtype=bool)
        if where is not None:
            keep = keep & np.asarray(where)
        return self._arrays[metric][keep]

    def score(self, metric="copy_recall", where=None):
        """The score of every head: the mean over the successful samples, [layer, head]."""
        return self.values(metric, where).mean(0)

    def frequency(self, metric="copy_recall", where=None):
        """Activation frequency: the share of the samples in which the head has a nonzero value, [layer, head]."""
        return (self.values(metric, where) > 0).mean(0)

    def name(self, flat_index):
        layer, head = divmod(int(flat_index), self.n_heads)
        return f"{self.layers[layer]}-{head}"

    def ranking(self, metric="copy_recall", where=None):
        """The heads from the best to the worst: a table with layer, head, score, frequency."""
        score, freq = self.score(metric, where).ravel(), self.frequency(metric, where).ravel()
        order = np.argsort(-score, kind="stable")
        return pd.DataFrame({"head": [self.name(i) for i in order], "layer": [self.layers[i // self.n_heads] for i in order],
                             "score": score[order], "frequency": freq[order]}, index=pd.RangeIndex(1, len(order) + 1, name="rank"))


def head_score_file(p, layers, n_heads):
    """
    The mean score of every head, [layer, head], from a head_score json of the authors' format
    ({"layer-head": [per sample]}): a ranking computed elsewhere, e.g. data/head_scores/*.json.
    """
    with open(path(p), encoding="utf-8") as f:
        scores = json.load(f)
    return np.array([[float(np.mean(scores.get(f"{layer}-{h}") or [0.0])) for h in range(n_heads)] for layer in layers])


def spearman(a, b):
    # Pearson of the average ranks: the same as Spearman, without scipy
    return float(pd.Series(np.ravel(a)).rank().corr(pd.Series(np.ravel(b)).rank()))


def top_overlap(a, b, k):
    """How many of the k best heads of a are among the k best of b."""
    top = lambda v: set(np.argsort(-np.ravel(v), kind="stable")[:k])
    return len(top(a) & top(b))
