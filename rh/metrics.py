"""
Metrics over a spool folder written by rh.run. Knows nothing about the model: reads the files, updates the metrics
step by step, writes one result per sample and, with --consume, deletes what it has read.

python -m rh.metrics results/new/run1/spool --out results/new/run1 --follow --consume

The answer metrics are computed from the answers that rh.run keeps in --out/outputs; these files are never deleted.
After a change of the answer metrics, samples.jsonl and the summary are built again without the model and the spool:

python -m rh.metrics results/new/run1/spool --out results/new/run1 --rescore

Results in --out:
    outputs/<sample>.json    written by rh.run: the answer as it is (tokens, text with special tokens), reference, meta
    samples.jsonl            one line per sample: meta, response, answer metrics
    heads/<sample>.npz       head metrics of the sample, each [layer, head] over the layers with attention
    summary.json             aggregate over the samples
    head_score_<metric>.json per-head scores of the successful samples, in the format of the authors' head_score;
                             head_score_masked_<metric>.json when the run had masked heads
"""
import argparse
import glob
import json
import os
import time

import numpy as np
from rouge_score import rouge_scorer

from . import spool

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


def answer_metrics(reference, response):
    """Success is the authors' criterion: ROUGE-1 recall of the reference above 50."""
    score = scorer.score(reference, response)
    recall = score['rouge1'].recall * 100
    return {"rouge1_recall": recall, "rougeL_recall": score['rougeL'].recall * 100, "success": bool(recall > 50)}


class HeadMetric:
    """One value per head and sample. start() once per sample, step() per generated token, end() returns [layer, head]."""
    name = None
    limit = None   # only the steps before this one are given to the metric; None: all

    def start(self, input_ids, spans, span_names):
        pass

    def step(self, step):
        pass

    def end(self):
        raise NotImplementedError


class Copy(HeadMetric):
    """
    A head copies at a step when its top-1 attention is on a token of the span and that token is the generated one.
    copy_count: the number of such steps over the span length, as in the authors' code (a repeated copy counts again).
    copy_recall: the share of distinct span tokens copied, as in the paper; never above 1.
    """

    def __init__(self, name, distinct, span="needle"):
        self.name, self.distinct, self.span = name, distinct, span

    def start(self, input_ids, spans, span_names):
        self.input_ids = input_ids
        self.s, self.e = spans[self.span]
        self.hits = None

    def step(self, step):
        idx = step["topk_idx"][..., 0]
        hit = (idx >= self.s) & (idx < self.e) & (self.input_ids[np.clip(idx, 0, len(self.input_ids) - 1)] == step["token"])
        if self.hits is None:
            self.hits = np.zeros(idx.shape + ((self.e - self.s,) if self.distinct else ()), dtype=bool if self.distinct else np.float64)
        if self.distinct:
            l, h = np.nonzero(hit)
            self.hits[l, h, idx[l, h] - self.s] = True
        else:
            self.hits += hit

    def end(self):
        return self.hits.mean(-1) if self.distinct else self.hits / (self.e - self.s)


class SpanMass(HeadMetric):
    """The attention mass of the head on the span, averaged over the generated tokens."""

    def __init__(self, name, span="needle"):
        self.name, self.span = name, span

    def start(self, input_ids, spans, span_names):
        self.i, self.total, self.n = span_names.index(self.span), 0.0, 0

    def step(self, step):
        self.total = self.total + step["span_mass"][..., self.i].astype(np.float64)
        self.n += 1

    def end(self):
        return self.total / self.n


def head_metrics(limits=()):
    """The metrics over the whole answer and, as name@limit, over its first `limit` tokens for every limit."""
    def make(suffix):
        return [Copy("copy_count" + suffix, distinct=False), Copy("copy_recall" + suffix, distinct=True), SpanMass("needle_mass" + suffix)]
    metrics = make("")
    for limit in limits:
        for m in make(f"@{limit}"):
            m.limit = limit
            metrics.append(m)
    return metrics


def sample_result(record):
    """The line of samples.jsonl for one saved answer (a file of outputs/): meta, response and the answer metrics."""
    result = {"id": record["id"], **record["meta"], "reference": record["reference"], "response": record["response"],
              "n_tokens": record["n_tokens"], "seconds": record["seconds"],
              **answer_metrics(record["reference"], record["response"])}
    if record.get("stopped") is not None:
        # stopped: the answer ended by itself; at a limit it is truncated when it had not ended by that token
        result["stopped"] = record["stopped"]
        result["by_limit"] = {}
        for limit in record["limits"]:
            text = record["responses_at"].get(str(limit), record["response"])
            scores = answer_metrics(record["reference"], text)
            result["by_limit"][str(limit)] = {"rouge1_recall": scores["rouge1_recall"], "success": scores["success"],
                                              "truncated": not (record["stopped"] and record["n_tokens"] <= limit)}
    return result


def saved_output(out, info, done, limits):
    """
    The saved answer of the sample. A sample generated before the answers were kept has its answer in done.json:
    it is moved to outputs/ here, so that every answer of the run ends up there.
    """
    record = spool.load_output(out, info["id"])
    if record is None:
        record = {"id": info["id"], "meta": info["meta"], "reference": info["reference"], "output_ids": done["output_ids"],
                  "response": done["response"], "responses_at": done.get("responses_at", {}), "limits": list(limits),
                  "stopped": done.get("stopped"), "n_tokens": done["n_steps"], "seconds": done["seconds"]}
        spool.save_output(out, record)
    return record


def process_sample(sample_path, out, wait, delete, limits=()):
    """
    Reads the chunks of one sample in order, as they appear; with delete every chunk is removed as soon as it is
    read, so a sample may be larger than the buffer of the run. wait() is called when the next file is not there
    yet and returns False to give up. The answer is taken from out/outputs, which stays.
    Returns (status, result, head arrays): status "ok"; "incomplete" when the run stopped before the end of the
    sample; "lost" when some chunks of it are gone (the metrics were interrupted in the middle of it).
    """
    prefix = sample_path[:-len(".sample.npz")]
    input_ids, info = spool.read_sample(sample_path)
    span_names = sorted(info["spans"])
    metrics, started, chunk = head_metrics(limits), False, 0
    done_path = f"{prefix}.done.json"
    while True:
        chunk_path = f"{prefix}.{chunk:06d}{spool.STEPS}"
        # the answers are read before the chunk is looked up: chunks are written in order and before done.json
        done = spool.load_json(done_path) if os.path.exists(done_path) else None
        later = any(int(os.path.basename(f).split(".")[-3]) > chunk for f in glob.glob(f"{prefix}.*{spool.STEPS}"))
        if os.path.exists(chunk_path):
            for step in spool.read_steps(chunk_path):
                if "topk_idx" in step:
                    if not started:
                        for m in metrics:
                            m.start(input_ids, info["spans"], span_names)
                        started = True
                    for m in metrics:
                        if m.limit is None or step["step"] < m.limit:
                            m.step(step)
            if delete:
                os.remove(chunk_path)
            chunk += 1
        elif done is not None and done["n_chunks"] == chunk:
            result = sample_result(saved_output(out, info, done, limits))
            heads = {m.name: m.end().astype(np.float32) for m in metrics} if started else {}
            return "ok", result, heads
        elif done is not None or later:
            return "lost", None, None
        elif not wait():
            return "incomplete", None, None


def remove_sample(sample_path):
    prefix = sample_path[:-len(".sample.npz")]
    for path in glob.glob(f"{prefix}.*"):
        os.remove(path)


def consume(spool_dir, out, follow, delete, idle_timeout=3600):
    os.makedirs(f"{out}/heads", exist_ok=True)
    results_path = f"{out}/samples.jsonl"
    finished = set()
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            finished = {json.loads(l)["id"] for l in f if l.strip()}
    progress = [time.time()]

    def run_done():
        return os.path.exists(f"{spool_dir}/RUN_DONE")

    def wait():
        """False when nothing more will come: not following, the run has ended, or the spool has been silent too long."""
        if not follow or run_done():
            return False
        if time.time() - progress[0] > idle_timeout:
            print(f"no new data for {idle_timeout}s, the run seems to have died", flush=True)
            return False
        time.sleep(1)
        return True

    skipped = set()
    while True:
        todo = [p for p in sorted(glob.glob(f"{spool_dir}/*.sample.npz")) if p not in skipped]
        if not todo:
            if not wait():
                break
            continue
        for sample_path in todo:
            sample_id = spool.read_sample(sample_path)[1]["id"]
            skipped.add(sample_path)
            if sample_id in finished:
                if delete:
                    remove_sample(sample_path)
                continue
            run_json = f"{spool_dir}/run.json"
            limits = spool.load_json(run_json).get("limits", []) if os.path.exists(run_json) else []
            status, result, heads = process_sample(sample_path, out, wait, delete, limits)
            progress[0] = time.time()
            if status != "ok":
                # not written to samples.jsonl, so the next rh.run generates the sample again
                print(f"{status} sample, to be generated again: {sample_id}", flush=True)
                if delete and status == "lost":
                    remove_sample(sample_path)
                continue
            if heads:
                spool.save_npz(f"{out}/heads/{sample_id}.npz", **heads)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            finished.add(sample_id)
            if delete:
                remove_sample(sample_path)
            print(f"{sample_id}: rouge {result['rouge1_recall']:.1f}, {result['n_tokens']} tokens, {result['response'][:70]!r}", flush=True)


def rescore(out):
    """
    Builds samples.jsonl again from the answers saved in out/outputs: for a changed or a new answer metric, without
    the model and the spool. Only the samples already in samples.jsonl are rewritten, in the same order: a sample
    gets its line when its head metrics are done too. Lines of samples without a saved answer (runs made before the
    answers were kept) stay as they are. Returns (rewritten, kept).
    """
    path = f"{out}/samples.jsonl"
    if not os.path.exists(path):
        return 0, 0
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    records = [spool.load_output(out, row["id"]) for row in rows]
    rows = [sample_result(record) if record else row for row, record in zip(rows, records)]

    def write(tmp):
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    spool.write_atomic(path, write)
    rewritten = sum(r is not None for r in records)
    return rewritten, len(rows) - rewritten


def rank(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def aggregate(spool_dir, out):
    if not os.path.exists(f"{out}/samples.jsonl"):
        return {"samples": 0}
    with open(f"{out}/samples.jsonl", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    summary = {"samples": len(samples), "successful": sum(s["success"] for s in samples),
               "mean_rouge1_recall": float(np.mean([s["rouge1_recall"] for s in samples]))}
    if all("stopped" in s for s in samples):
        summary["truncated_at_cap"] = sum(not s["stopped"] for s in samples)
    for key in ("context_length", "depth_percent", "needle_idx"):
        if all(key in s for s in samples):
            groups = sorted({s[key] for s in samples})
            summary[f"success_by_{key}"] = {str(g): round(float(np.mean([s["success"] for s in samples if s[key] == g])), 3) for g in groups}

    run = spool.load_json(f"{spool_dir}/run.json") if os.path.exists(f"{spool_dir}/run.json") else {}
    # head scores of a run with masked heads describe the changed model: they get another name, so that they are
    # not taken for the ranking of the heads (rh.run refuses them as --mask_file)
    masked = len(run.get("block_list") or [])
    summary["masked_heads"] = masked
    prefix = "head_score_masked_" if masked else "head_score_"
    with_heads = [s for s in samples if os.path.exists(f"{out}/heads/{s['id']}.npz")]
    means = {}
    if with_heads:
        arrays = []
        for sample in with_heads:
            # read into memory and close: an open file per sample runs out of file handles on large runs
            with np.load(f"{out}/heads/{sample['id']}.npz") as d:
                arrays.append({k: d[k] for k in d.files})
        ok = np.array([s["success"] for s in with_heads])
        layers = run.get("attn_layers") or list(range(next(iter(arrays[0].values())).shape[0]))
        summary["heads"] = {}
        for name in arrays[0]:
            if "@" in name:
                continue   # the metrics at the token limits are summarized by rh.limits
            values = np.stack([a[name] for a in arrays])  # [sample, layer, head]
            mean_ok = values[ok].mean(0) if ok.any() else np.zeros(values.shape[1:])
            means[name] = mean_ok.ravel()
            keys = [f"{layers[l]}-{h}" for l in range(values.shape[1]) for h in range(values.shape[2])]
            best = np.argsort(-mean_ok.ravel(), kind="stable")[:20]
            summary["heads"][name] = {"heads": len(keys), "above_0.1": int((mean_ok > 0.1).sum()),
                                      "top20": [keys[i] for i in best], "top20_scores": [round(float(mean_ok.ravel()[i]), 4) for i in best]}
            # per-head lists over the successful samples, as in the authors' head_score files
            lists = values[ok].reshape(int(ok.sum()), -1).T.tolist()
            spool.save_json(f"{out}/{prefix}{name}.json", dict(zip(keys, lists)))
        names = sorted(means)
        summary["spearman"] = {f"{a} vs {b}": round(float(np.corrcoef(rank(means[a]), rank(means[b]))[0, 1]), 4)
                               for i, a in enumerate(names) for b in names[i + 1:]}
    spool.save_json(f"{out}/summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spool", help="spool folder of rh.run")
    parser.add_argument("--out", required=True, help="folder for the results")
    parser.add_argument("--follow", action="store_true", help="keep reading until the run has finished")
    parser.add_argument("--consume", action="store_true", help="delete the spool files of every processed sample")
    parser.add_argument("--idle_timeout", type=int, default=3600, help="with --follow: stop when the spool gets no new data for this number of seconds")
    parser.add_argument("--rescore", action="store_true", help="do not read the spool: compute the answer metrics again from the answers saved in --out/outputs")
    args = parser.parse_args()

    if args.rescore:
        rewritten, kept = rescore(args.out)
        print(f"answer metrics computed again for {rewritten} samples" + (f"; {kept} samples have no saved answer, their lines are kept" if kept else ""), flush=True)
    else:
        consume(args.spool, args.out, args.follow, args.consume, args.idle_timeout)
    summary = aggregate(args.spool, args.out)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
