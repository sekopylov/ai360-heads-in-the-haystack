"""
Needle-in-a-Haystack accuracy when masking the top-K retrieval heads vs K random heads
(Wu et al. 2024, Sec. 4 / Fig. "masking retrieval heads").

  python mask_heads_niah.py --model AntonV/mamba2-2.7b-hf \
      --head_score head_score/mamba2-2.7b-hf.json \
      --haystack_dir Retrieval_Head/haystack_for_detect --needles_file needles_eval.jsonl \
      --ks 0 5 10 20 50 100 200 --random_seeds 3

Results are appended to --out (json) after every configuration, already finished
configurations are skipped on restart.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from mamba2_implicit_attention import greedy_answer, head_geometry, mask_heads
from niah_utils import NeedleHaystack, load_needles, make_grid, rouge1_recall


def load_ranking(path):
    d = json.load(open(path))
    ranked = sorted(((k, float(np.mean(v)) if len(v) else 0.0) for k, v in d.items()), key=lambda x: -x[1])
    return [tuple(int(t) for t in k.split("-")) for k, _ in ranked], [s for _, s in ranked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="AntonV/mamba2-2.7b-hf")
    ap.add_argument("--head_score", required=True)
    ap.add_argument("--haystack_dir", default="Retrieval_Head/haystack_for_detect")
    ap.add_argument("--needles_file", default=None,
                    help="default: needles.jsonl of haystack_dir; use needles_eval.jsonl for held-out needles")
    ap.add_argument("--lengths", type=int, nargs="+", default=[500, 1000, 1500, 2000])
    ap.add_argument("--depth_intervals", type=int, default=6)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 5, 10, 20, 50, 100, 200])
    ap.add_argument("--random_seeds", type=int, default=3)
    ap.add_argument("--exclude_top_from_random", type=int, default=None,
                    help="random heads are drawn outside the top-N retrieval heads (paper: 100). "
                         "Default: max(ks)")
    ap.add_argument("--mask_mode", default="dt", choices=["dt", "out"])
    ap.add_argument("--max_new_tokens", type=int, default=50)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = args.model.rstrip("/").split("/")[-1]
    out_path = args.out or f"results/mask_{name}_{args.mask_mode}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    results = json.load(open(out_path)) if os.path.exists(out_path) else {"top": {}, "random": {}}
    results["config"] = vars(args)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
                                                 device_map="auto").eval()
    L, H, _ = head_geometry(model)
    ranking, scores = load_ranking(args.head_score)
    print("top-10 retrieval heads:", list(zip(ranking[:10], np.round(scores[:10], 3))))

    nh = NeedleHaystack(tok)
    needles = load_needles(args.haystack_dir, args.needles_file)
    grid = make_grid(args.lengths, depth_intervals=args.depth_intervals)
    prompts = []
    for item in needles:
        for (ctx_len, depth) in grid:
            p = nh.build_prompt(item, ctx_len, depth)
            prompts.append((item, ctx_len, depth, torch.tensor(tok(p).input_ids)))
    print(f"{len(prompts)} NIAH prompts per configuration")

    def evaluate(heads, tag):
        t0 = time.time()
        per_run = []
        with mask_heads(model, heads, mode=args.mask_mode):
            for item, ctx_len, depth, ids in prompts:
                gen = greedy_answer(model, tok, ids, args.max_new_tokens)
                resp = tok.decode(gen, skip_special_tokens=True).strip()
                per_run.append(dict(len=ctx_len, depth=depth, score=rouge1_recall(item["real_needle"], resp),
                                    response=resp[:200]))
        acc = float(np.mean([r["score"] for r in per_run]))
        print(f"{tag:>24s}: NIAH = {acc:6.2f}   ({time.time() - t0:.0f}s)")
        return acc, per_run

    n_excl = args.exclude_top_from_random or max(args.ks)
    excluded = set(ranking[:n_excl])
    pool = [(l, h) for l in range(L) for h in range(H) if (l, h) not in excluded]

    for K in args.ks:
        key = str(K)
        if key not in results["top"]:
            acc, runs = evaluate(ranking[:K], f"top-{K}")
            results["top"][key] = dict(acc=acc, heads=[list(x) for x in ranking[:K]], runs=runs)
            json.dump(results, open(out_path, "w"), indent=1)
        if K == 0:
            continue
        results["random"].setdefault(key, {})
        for seed in range(args.random_seeds):
            if str(seed) in results["random"][key]:
                continue
            heads = random.Random(1000 * K + seed).sample(pool, K)
            acc, runs = evaluate(heads, f"random-{K} (seed {seed})")
            results["random"][key][str(seed)] = dict(acc=acc, heads=[list(x) for x in heads], runs=runs)
            json.dump(results, open(out_path, "w"), indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
