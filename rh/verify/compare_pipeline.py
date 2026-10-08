"""
Compares the pipeline (rh.run + rh.metrics) with the code that passed the comparison with the authors' code
(rh.verify.detect without --legacy) on the same samples.

python -m rh.verify.compare_pipeline results/new/qwen3_test_old results/new/qwen3_small
"""
import argparse
import json
import os

import numpy as np


def copy_recall(d):
    """The paper's score recomputed from the dump: the share of distinct needle tokens copied by the head."""
    s, e = int(d["needle_start"]), int(d["needle_end"])
    copied = np.zeros(d["top1_idx"].shape[1:] + (e - s,), dtype=bool)
    for step, token in enumerate(d["output_ids"]):
        idx = d["top1_idx"][step]
        hit = (idx >= s) & (idx < e) & (d["input_ids"][np.clip(idx, 0, len(d["input_ids"]) - 1)] == token)
        l, h = np.nonzero(hit)
        copied[l, h, idx[l, h] - s] = True
    return copied.mean(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("old", help="folder of rh.verify.detect")
    parser.add_argument("new", help="--out folder of rh.run")
    args = parser.parse_args()

    with open(f"{args.new}/samples.jsonl", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    n = same_input = same_response = same_rouge = 0
    diffs = {"copy_count": [], "copy_recall": [], "needle_mass": []}
    old_scores, new_scores = [], []
    for s in samples:
        path = f"{args.old}/needle{s['needle_idx']}_len{s['context_length']}_depth{s['depth_percent']}.npz"
        if not os.path.exists(path):
            print("not in the old run:", s["id"])
            continue
        d = np.load(path)
        heads = np.load(f"{args.new}/heads/{s['id']}.npz")
        n += 1
        same_input += len(d["input_ids"]) == s["n_input_tokens"]
        same = str(d["response"]) == s["response"] and len(d["output_ids"]) == s["n_tokens"]
        same_response += same
        same_rouge += abs(float(d["rouge"]) - s["rouge1_recall"]) < 1e-9
        if not same:
            print(f"response differs: {s['id']}\n   old: {str(d['response'])[:90]!r}\n   new: {s['response'][:90]!r}")
            continue
        # head metrics are comparable only when the same tokens were generated
        diffs["copy_count"].append(np.abs(heads["copy_count"] - d["retrieval_score"]).max())
        diffs["copy_recall"].append(np.abs(heads["copy_recall"] - copy_recall(d)).max())
        diffs["needle_mass"].append(np.abs(heads["needle_mass"] - d["needle_mass"].mean(0)).max())
        old_scores.append(d["retrieval_score"])
        new_scores.append(heads["copy_count"])

    print(f"\nsamples compared: {n}")
    print(f"prompt length identical: {same_input}/{n}")
    print(f"response identical:      {same_response}/{n}")
    print(f"rouge identical:         {same_rouge}/{n}")
    for name, values in diffs.items():
        if values:
            print(f"{name}, max |diff| over heads and samples: {max(values):.6f} ({sum(v < 1e-6 for v in values)}/{len(values)} samples equal within 1e-6)")
    if old_scores:
        x, y = np.mean(old_scores, 0).ravel(), np.mean(new_scores, 0).ravel()
        top = lambda v, k: set(np.argsort(-v, kind="stable")[:k])
        print("best heads by copy_count: " + ", ".join(f"top{k} overlap {len(top(x, k) & top(y, k))}/{k}" for k in (10, 20, 50)))
