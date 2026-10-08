"""
Compares two dumps sample by sample: the old code (reference) and the new code.

python -m rh.verify.compare_dumps source/results/dump/Qwen1.5-14B-Chat/detect results/new/replay
"""
import argparse
import glob
import os

import numpy as np


def rank(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ref", help="dump of the old code")
    parser.add_argument("new", help="dump of the new code")
    args = parser.parse_args()

    files = sorted(os.path.basename(f) for f in glob.glob(f"{args.new}/*.npz"))
    assert files, f"no npz in {args.new}"
    n = same_input = same_output = same_success = 0
    idx_same, idx_total, val_diff, score_diff, attn_diff, diverged_at = 0, 0, [], [], [], []
    ref_scores, new_scores = [], []
    for name in files:
        if not os.path.exists(f"{args.ref}/{name}"):
            print("not in ref:", name)
            continue
        a, b = np.load(f"{args.ref}/{name}"), np.load(f"{args.new}/{name}")
        n += 1
        same_input += a["input_ids"].tolist() == b["input_ids"].tolist()
        same_success += (float(a["rouge"]) > 50) == (float(b["rouge"]) > 50)

        # steps are comparable while the generated tokens are the same: step t depends on the tokens before it
        oa, ob = a["output_ids"], b["output_ids"]
        m = min(len(oa), len(ob))
        differ = np.nonzero(oa[:m] != ob[:m])[0]
        steps = int(differ[0]) + 1 if len(differ) else m
        if len(differ) or len(oa) != len(ob):
            diverged_at.append(int(differ[0]) if len(differ) else m)
            print(f"output differs: {name}, first at step {diverged_at[-1]} of {len(oa)}/{len(ob)}")
        else:
            same_output += 1
            score_diff.append(np.abs(a["retrieval_score"] - b["retrieval_score"]).max())
        idx_same += int((a["top1_idx"][:steps] == b["top1_idx"][:steps]).sum())
        idx_total += a["top1_idx"][:steps].size
        val_diff.append(np.abs(a["top1_val"][:steps] - b["top1_val"][:steps]).max())
        for k in a.files:
            if k.startswith("attn_step") and k in b.files and int(k[9:]) < steps:
                attn_diff.append(np.abs(a[k].astype(np.float32) - b[k].astype(np.float32)).max())
        if float(a["rouge"]) > 50 and float(b["rouge"]) > 50:
            ref_scores.append(a["retrieval_score"])
            new_scores.append(b["retrieval_score"])

    print(f"\nsamples compared: {n}")
    print(f"inputs identical:           {same_input}/{n}")
    print(f"generated tokens identical: {same_output}/{n}")
    print(f"same success (rouge>50):    {same_success}/{n}")
    print(f"top-1 attention position identical: {idx_same}/{idx_total} = {idx_same / max(idx_total, 1):.4f} (steps before the outputs diverge)")
    print(f"top-1 attention value, max |diff|:  {max(val_diff):.4f}, median over samples {np.median(val_diff):.4f}")
    if attn_diff:
        print(f"full attention rows, max |diff|:    {max(attn_diff):.4f} over {len(attn_diff)} rows")
    if score_diff:
        print(f"sample retrieval score, max |diff|: {max(score_diff):.4f} (samples with identical outputs)")
    if ref_scores:
        x, y = np.mean(ref_scores, 0).ravel(), np.mean(new_scores, 0).ravel()
        top = lambda v, k: set(np.argsort(-v, kind="stable")[:k])
        print(f"head scores over {len(ref_scores)} samples successful in both: spearman {np.corrcoef(rank(x), rank(y))[0, 1]:.4f}, "
              + ", ".join(f"top{k} overlap {len(top(x, k) & top(y, k))}/{k}" for k in (10, 20, 50)))
