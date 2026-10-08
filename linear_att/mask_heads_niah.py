"""
Needle-in-a-Haystack accuracy when masking the top-K retrieval heads vs K random heads
(Wu et al. 2024, Sec. 4 / Fig. "masking retrieval heads").

  python mask_heads_niah.py --model AntonV/mamba2-2.7b-hf \
      --head_score head_score/mamba2-2.7b-hf.json \
      --haystack_dir Retrieval_Head/haystack_for_detect --needles_file needles_eval.jsonl \
      --ks 0 5 10 20 50 100 200 --random_seeds 3

Results are written to --out (json) after every configuration; finished configurations
are skipped on restart.

Progress:  tail -f logs/mask_heads_niah_latest.log
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from log_utils import Progress, add_logging_args, fmt_time, gpu_mem, load_model_logged, setup_logging
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
    ap.add_argument("--log_every", type=int, default=1,
                    help="log every N-th NIAH prompt inside a configuration (1 = every prompt)")
    add_logging_args(ap)
    args = ap.parse_args()
    log = setup_logging("mask_heads_niah", args)

    name = args.model.rstrip("/").split("/")[-1]
    out_path = args.out or f"results/mask_{name}_{args.mask_mode}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        results = json.load(open(out_path))
        done_rnd = dict((k, sorted(v)) for k, v in results["random"].items())
        log.info(f"resuming from {out_path}: top done for K={sorted(results['top'], key=int)}, "
                 f"random done (K: seeds): {done_rnd}")
    else:
        results = {"top": {}, "random": {}}
    results["config"] = vars(args)

    tok, model = load_model_logged(args.model, args.dtype)
    L, H, _ = head_geometry(model)
    ranking, scores = load_ranking(args.head_score)
    log.info(f"{L} layers x {H} heads; head scores from {args.head_score}")
    log.info("top-20 retrieval heads: " + ", ".join(f"{l}-{h}:{s:.3f}" for (l, h), s in zip(ranking[:20], scores[:20])))
    for K in args.ks:
        if K:
            log.info(f"  top-{K}: min score among masked = {scores[K - 1]:.3f}")

    nh = NeedleHaystack(tok)
    needles = load_needles(args.haystack_dir, args.needles_file)
    grid = make_grid(args.lengths, depth_intervals=args.depth_intervals)
    prompts = []
    for item in needles:
        for (ctx_len, depth) in grid:
            p = nh.build_prompt(item, ctx_len, depth)
            prompts.append((item, ctx_len, depth, torch.tensor(tok(p).input_ids)))
    log.info(f"{len(needles)} needles x {len(grid)} points = {len(prompts)} NIAH prompts per configuration")

    # list of all configurations, to have a global progress / ETA
    configs = []
    for K in args.ks:
        if str(K) not in results["top"]:
            configs.append(("top", K, None))
        if K == 0:
            continue
        for seed in range(args.random_seeds):
            if str(seed) not in results["random"].get(str(K), {}):
                configs.append(("random", K, seed))
    log.info(f"{len(configs)} configurations to run: " +
             ", ".join(f"{t}-{K}" + (f"/s{s}" if s is not None else "") for t, K, s in configs))
    global_prog = Progress(len(configs) * len(prompts), "total")

    n_excl = args.exclude_top_from_random or max(args.ks)
    excluded = set(ranking[:n_excl])
    pool = [(l, h) for l in range(L) for h in range(H) if (l, h) not in excluded]
    log.info(f"random heads are sampled from {len(pool)} heads (excluding top-{n_excl})")

    def evaluate(heads, tag):
        t0 = time.time()
        per_run = []
        prog = Progress(len(prompts), tag)
        with mask_heads(model, heads, mode=args.mask_mode):
            for i, (item, ctx_len, depth, ids) in enumerate(prompts):
                gen = greedy_answer(model, tok, ids, args.max_new_tokens)
                resp = tok.decode(gen, skip_special_tokens=True).strip()
                sc = rouge1_recall(item["real_needle"], resp)
                per_run.append(dict(len=ctx_len, depth=depth, score=sc, response=resp[:200]))
                msg_local, msg_global = prog.step(), global_prog.step()
                if (i + 1) % args.log_every == 0 or i + 1 == len(prompts):
                    log.info(f"{msg_local} {msg_global} len={ctx_len:6d} depth={depth:3d}% rouge={sc:5.1f} "
                             f"running acc={np.mean([r['score'] for r in per_run]):5.1f} | {gpu_mem()} | "
                             f"{resp[:60]!r}")
        acc = float(np.mean([r["score"] for r in per_run]))
        log.info(f"==> {tag:>22s}: NIAH = {acc:6.2f}   ({fmt_time(time.time() - t0)})")
        return acc, per_run

    for kind, K, seed in configs:
        log.info("-" * 90)
        if kind == "top":
            heads = ranking[:K]
            log.info(f"config: mask top-{K} retrieval heads ({args.mask_mode})" +
                     (f": {[f'{l}-{h}' for l, h in heads[:30]]}{' ...' if K > 30 else ''}" if K else " (baseline)"))
            acc, runs = evaluate(heads, f"top-{K}")
            results["top"][str(K)] = dict(acc=acc, heads=[list(x) for x in heads], runs=runs)
        else:
            heads = random.Random(1000 * K + seed).sample(pool, K)
            log.info(f"config: mask {K} random heads, seed {seed} ({args.mask_mode}): "
                     f"{[f'{l}-{h}' for l, h in heads[:30]]}{' ...' if K > 30 else ''}")
            acc, runs = evaluate(heads, f"random-{K}/s{seed}")
            results["random"].setdefault(str(K), {})[str(seed)] = dict(acc=acc, heads=[list(x) for x in heads],
                                                                       runs=runs)
        json.dump(results, open(out_path, "w"), indent=1)
        log.info(f"saved {out_path}")

    log.info("=" * 90)
    log.info("summary (NIAH score):")
    for K in sorted(map(int, results["top"])):
        rnd = [v["acc"] for v in results["random"].get(str(K), {}).values()]
        r = f"{np.mean(rnd):6.2f} ± {np.std(rnd):.2f} ({len(rnd)} seeds)" if rnd else "   -"
        log.info(f"  K={K:4d}: top-K = {results['top'][str(K)]['acc']:6.2f} | random-K = {r}")
    log.info(f"saved {out_path}")


if __name__ == "__main__":
    main()