"""
Checks the dump written by retrieval_head_detection.py --dump_dir and summarizes the run.

python check_dump.py results/dump/Qwen1.5-14B-Chat/detect \
    --ref head_score/Qwen1.5-14B-Chat.json --out head_score_ours/Qwen1.5-14B-Chat_from_dump.json
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np


def legacy_score(d):
    ## same rule as retrieval_calculate: top-1 attention is on a needle token and that token is the generated one
    s, e = int(d["needle_start"]), int(d["needle_end"])
    score = np.zeros(d["top1_idx"].shape[1:])
    if e <= s: return score
    input_ids = d["input_ids"]
    for step, token in enumerate(d["output_ids"]):
        idx = d["top1_idx"][step]
        in_needle = (idx >= s) & (idx < e)
        same = input_ids[np.clip(idx, 0, len(input_ids) - 1)] == token
        score += (in_needle & same) / (e - s)
    return score


def paper_score(d):
    ## as in the paper: share of unique needle positions copied by the head, bounded by 1
    s, e = int(d["needle_start"]), int(d["needle_end"])
    steps, layers, heads = d["top1_idx"].shape
    if e <= s: return np.zeros((layers, heads))
    copied = np.zeros((layers, heads, e - s), dtype=bool)
    input_ids = d["input_ids"]
    for step, token in enumerate(d["output_ids"]):
        idx = d["top1_idx"][step]
        hit = (idx >= s) & (idx < e) & (input_ids[np.clip(idx, 0, len(input_ids) - 1)] == token)
        l, h = np.nonzero(hit)
        copied[l, h, idx[l, h] - s] = True
    return copied.mean(-1)


def rank(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):  # average rank for ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def compare(name, a, b, keys):
    x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    top = lambda v, k: {keys[i] for i in np.argsort(-v, kind="stable")[:k]}
    print(f"{name}: spearman {spearman(x, y):.3f}, " +
          ", ".join(f"top{k} overlap {len(top(x, k) & top(y, k))}/{k}" for k in (10, 20, 50)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", help="folder with *.npz")
    parser.add_argument("--ref", default=None, help="head score json to compare with")
    parser.add_argument("--out", default=None, help="write head scores rebuilt from the dump")
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.dump}/*.npz"))
    assert files, f"no npz in {args.dump}"
    legacy, paper = defaultdict(list), defaultdict(list)
    n_ok, n_mismatch, n_no_needle, n_full_len, by_len = 0, 0, 0, 0, defaultdict(list)
    for f in files:
        d = np.load(f)
        ok = float(d["rouge"]) > 50
        by_len[int(d["context_length"])].append(ok)
        n_no_needle += int(d["needle_start"]) < 0
        n_full_len += len(d["output_ids"]) == 50
        if not np.allclose(legacy_score(d), d["retrieval_score"]):
            n_mismatch += 1
            print("score recomputed from top1_idx differs:", f)
        if not ok: continue
        n_ok += 1
        ps = paper_score(d)
        layers, heads = ps.shape
        for l in range(layers):
            for h in range(heads):
                legacy[f"{l}-{h}"].append(float(d["retrieval_score"][l, h]))
                paper[f"{l}-{h}"].append(float(ps[l, h]))

    print(f"samples {len(files)}, rouge>50: {n_ok}, needle not found: {n_no_needle}, "
          f"generated all 50 tokens: {n_full_len}, recomputed score mismatch: {n_mismatch}")
    print("success rate by length:", {k: round(float(np.mean(v)), 2) for k, v in sorted(by_len.items())})
    if n_ok == 0: raise SystemExit("no successful samples")

    legacy_mean = {k: float(np.mean(v)) for k, v in legacy.items()}
    paper_mean = {k: float(np.mean(v)) for k, v in paper.items()}
    keys = list(legacy_mean)
    for name, m in (("legacy", legacy_mean), ("paper", paper_mean)):
        best = sorted(m.items(), key=lambda x: -x[1])
        print(f"{name}: heads > 0.1: {sum(v > 0.1 for v in m.values())}/{len(m)}, top10 {[k for k, _ in best[:10]]}")
    compare("legacy vs paper metric", legacy_mean, paper_mean, keys)

    if args.ref:
        with open(args.ref) as file:
            ref = {k: float(np.mean(v)) for k, v in json.loads(file.readline()).items()}
        if set(ref) == set(keys): compare("ours vs ref", legacy_mean, ref, keys)
        else: print(f"ref has {len(ref)} heads, dump has {len(keys)}: not comparable")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(legacy, f)
