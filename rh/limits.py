"""
Token limits compared on the results of one run, without the model: rh.run generates up to its cap and rh.metrics
keeps, for every limit of the run, the answer metrics and the head metrics over the first `limit` tokens.

Table over the limits:
python -m rh.limits results/new/qwen3_detect

Choose a limit: writes head_score_<metric>@<limit>.json, the file to pass to --mask_file:
python -m rh.limits results/new/qwen3_detect --select 128

The head rankings are compared by copy_count; by another head metric with --metric:
python -m rh.limits results/new/qwen3_detect --metric needle_attention_mass
"""
import argparse
import json
import os

import numpy as np

from . import spool
from .metrics import rank

HEAD_METRICS = ("copy_count", "copy_recall", "needle_mass", "needle_attention_mass")


def load(out):
    with open(f"{out}/samples.jsonl", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    samples = [s for s in samples if "by_limit" in s]
    if not samples:
        raise SystemExit(f"no samples with token limits in {out}: the run must be made with rh.run --limits")
    limits = sorted(int(l) for l in samples[0]["by_limit"])
    heads = {}
    for s in samples:
        path = f"{out}/heads/{s['id']}.npz"
        if os.path.exists(path):
            with np.load(path) as d:
                heads[s["id"]] = {k: d[k] for k in d.files}
    return samples, limits, heads


def head_scores(samples, heads, metric, limit):
    """Per-head values of the samples successful at this limit, [sample, layer, head]; None without head metrics."""
    ok = [s for s in samples if s["by_limit"][str(limit)]["success"] and s["id"] in heads]
    if not ok:
        return None
    return np.stack([heads[s["id"]][f"{metric}@{limit}"] for s in ok])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", help="--out folder of rh.run")
    parser.add_argument("--reference", type=int, default=None, help="limit the head rankings are compared with; default: the largest")
    parser.add_argument("--select", type=int, default=None, help="write the head_score files of this limit")
    parser.add_argument("--metric", default="copy_count", choices=HEAD_METRICS, help="head metric whose rankings are compared over the limits")
    args = parser.parse_args()

    samples, limits, heads = load(args.out)
    reference = args.reference or limits[-1]
    assert reference in limits, f"{reference} is not among the limits {limits}"
    ref_scores = head_scores(samples, heads, args.metric, reference)
    ref_mean = ref_scores.mean(0).ravel() if ref_scores is not None else None
    top = lambda v, k: set(np.argsort(-v, kind="stable")[:k])

    rows = []
    for limit in limits:
        at = [s["by_limit"][str(limit)] for s in samples]
        row = {"limit": limit, "samples": len(at), "truncated": float(np.mean([a["truncated"] for a in at])),
               "success": float(np.mean([a["success"] for a in at])), "mean_rouge1_recall": float(np.mean([a["rouge1_recall"] for a in at]))}
        scores = head_scores(samples, heads, args.metric, limit)
        if scores is not None and ref_mean is not None:
            mean = scores.mean(0).ravel()
            row.update({"heads_above_0.1": int((mean > 0.1).sum()), "spearman": float(np.corrcoef(rank(mean), rank(ref_mean))[0, 1]),
                        **{f"top{k}": len(top(mean, k) & top(ref_mean, k)) for k in (10, 20, 50)}})
        rows.append(row)

    print(f"{len(samples)} samples; head rankings by {args.metric} are compared with the limit {reference}")
    header = f"{'limit':>6} {'truncated':>10} {'success':>8} {'rouge':>7}"
    if "spearman" in rows[0]:
        header += f" {'heads>0.1':>10} {'spearman':>9} {'top10':>6} {'top20':>6} {'top50':>6}"
    print(header)
    for r in rows:
        line = f"{r['limit']:>6} {r['truncated']:>10.1%} {r['success']:>8.1%} {r['mean_rouge1_recall']:>7.1f}"
        if "spearman" in r:
            line += f" {r['heads_above_0.1']:>10} {r['spearman']:>9.4f} {r['top10']:>4}/10 {r['top20']:>4}/20 {r['top50']:>4}/50"
        print(line)
    spool.save_json(f"{args.out}/limits.json", {"reference": reference, "limits": rows})

    if args.select is not None:
        assert args.select in limits, f"{args.select} is not among the limits {limits}"
        run_json = f"{args.out}/spool/run.json"
        run = spool.load_json(run_json) if os.path.exists(run_json) else {}
        prefix = "head_score_masked_" if run.get("block_list") else "head_score_"
        for metric in HEAD_METRICS:
            scores = head_scores(samples, heads, metric, args.select)
            if scores is None:
                raise SystemExit("this run has no head metrics (it was made with --save none)")
            layers = run.get("attn_layers") or list(range(scores.shape[1]))
            keys = [f"{layers[l]}-{h}" for l in range(scores.shape[1]) for h in range(scores.shape[2])]
            path = f"{args.out}/{prefix}{metric}@{args.select}.json"
            spool.save_json(path, dict(zip(keys, scores.reshape(len(scores), -1).T.tolist())))
            print("written", path)
