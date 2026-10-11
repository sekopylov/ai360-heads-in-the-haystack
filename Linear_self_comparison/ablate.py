"""
Needle-in-a-Haystack with head ablation, separately for linear-attention and full-attention heads.
Port of needle_in_haystack_with_mask.py from https://github.com/nightdessert/Retrieval_Head

One process runs a whole sweep (model loaded once). Every run is written to
results/niah/<run>.jsonl and is skipped if it already exists (resumable).

Runs:
  baseline                          no masking
  <kind>_top<K>                     top-K retrieval heads of that kind masked
  <kind>_random<K>_s<seed>          K random non-retrieval heads of that kind masked
with kind in {linear, full} (or "all" = both kinds, ranked jointly).

Example (main experiment):
  python ablate.py --model_path Qwen/Qwen3.5-0.8B --haystack_root ../Retrieval_Head \
      --head_score head_score/Qwen3.5-0.8B.json --kinds linear,full --ks 1,2,4,8,16 --seeds 0,1,2
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import rh_core as rc

log = rc.log

SF_NEEDLE = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
SF_QUESTION = "What is the best thing to do in San Francisco?"
SF_ANSWER = "eat a sandwich and sit in Dolores Park on a sunny day"


def run_niah(model, tok, args, lengths, depths, haystack, masker, path, run_meta, prog):
    tmp = path + ".part"
    scores = []
    t_run = time.time()
    with open(tmp, "w") as f:
        f.write(json.dumps({"meta": run_meta}) + "\n")
        for L in lengths:
            for depth in depths:
                t0 = time.time()
                context = rc.insert_needle(tok, haystack, SF_NEEDLE, depth, L)
                ids = rc.build_prompt_ids(tok, context, SF_QUESTION, args.prompt_format).to(args.device)
                gen = rc.greedy_generate(model, tok, ids, args.max_new_tokens, masker=masker, scope=args.scope)
                resp = tok.decode(gen, skip_special_tokens=True).strip()
                score = rc.rouge_recall(SF_ANSWER, resp)
                scores.append(score)
                f.write(json.dumps(dict(context_length=L, depth_percent=depth, score=score, response=resp),
                                   ensure_ascii=False) + "\n")
                f.flush()
                prog.update(L, f"{run_meta['run']}: len={L} depth={depth}% score={score:.0f} "
                               f"({time.time() - t0:.1f}s, run mean {np.mean(scores):.1f}) {resp[:50]!r}")
    os.replace(tmp, path)
    log.info(f"=== finished {run_meta['run']}: mean NIAH score {np.mean(scores):.2f} "
             f"over {len(scores)} samples in {rc.fmt_time(time.time() - t_run)} -> {path}")
    return float(np.mean(scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--haystack_root", default="../Retrieval_Head")
    p.add_argument("--head_score", required=True, help="head_score/<name>.json (or .contrib.json)")
    p.add_argument("--kinds", default="linear,full", help="comma list of linear,full,all")
    p.add_argument("--ks", default="1,2,4,8,16", help="numbers of masked heads")
    p.add_argument("--modes", default="top,random")
    p.add_argument("--seeds", default="0,1,2", help="seeds for random heads")
    p.add_argument("--exclude_thresh", type=float, default=0.1,
                   help="random heads are drawn among heads with retrieval score below this")
    p.add_argument("--no_baseline", action="store_true")
    p.add_argument("--scope", choices=["decode", "all"], default="decode",
                   help="mask heads only while decoding (as in the paper's code) or also during prefill")
    p.add_argument("-s", "--s_len", type=int, default=1000)
    p.add_argument("-e", "--e_len", type=int, default=16000)
    p.add_argument("--num_lengths", type=int, default=10)
    p.add_argument("--num_depths", type=int, default=10)
    p.add_argument("--prompt_format", choices=["chat", "raw"], default="chat")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--out_dir", default="results/niah")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--allow_cpu", action="store_true", help="do not abort when CUDA is unavailable")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rc.setup_logging("ablate", os.path.basename(args.head_score).split(".json")[0])
    log.info(f"args: {vars(args)}")

    tok, model, args.device = rc.load_model(args.model_path, args.device, args.dtype, allow_cpu=args.allow_cpu)
    space = rc.HeadSpace(model)
    log.info(space.describe())
    scores = rc.load_head_scores(args.head_score)
    for key in space.keys:  # heads never scored (no successful sample) get 0
        scores.setdefault(key, 0.0)

    lengths, depths = rc.grid(args.s_len, args.e_len, args.num_lengths, args.num_depths)
    haystack = rc.read_haystack(os.path.join(args.haystack_root, "PaulGrahamEssays"), max(lengths))

    runs = []
    if not args.no_baseline:
        runs.append(("baseline", []))
    kinds = [k for k in args.kinds.split(",") if k]
    ks = [int(k) for k in args.ks.split(",") if k]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    for kind in kinds:
        n_kind = len(space.heads_of_kind(kind))
        ranked = rc.rank_heads(scores, space.kind_of, kind)
        for K in ks:
            if K > n_kind:
                log.info(f"skip {kind} K={K}: only {n_kind} heads")
                continue
            if "top" in args.modes:
                runs.append((f"{kind}_top{K}", ranked[:K]))
            if "random" in args.modes:
                for s in seeds:
                    runs.append((f"{kind}_random{K}_s{s}",
                                 rc.random_heads(scores, space.kind_of, kind, K, s, exclude_thresh=args.exclude_thresh)))

    todo = []
    for run, heads in runs:
        path = os.path.join(args.out_dir, f"{run}.jsonl")
        if os.path.exists(path) and not args.overwrite:
            log.info(f"skip {run}: {path} exists")
        else:
            todo.append((run, heads, path))
    n_per_run = len(lengths) * len(depths)
    log.info(f"{len(runs)} runs planned, {len(todo)} to do ({len(runs) - len(todo)} already done); "
             f"{n_per_run} samples per run, {len(todo) * n_per_run} samples total")
    prog = rc.Progress(len(todo) * n_per_run, len(todo) * len(depths) * sum(lengths), name="ablate")
    results = {}
    for i, (run, heads, path) in enumerate(todo):
        meta = dict(run=run, heads=heads, head_scores=[round(scores[h], 4) for h in heads],
                    kinds=[space.kind_of[h] for h in heads], scope=args.scope, model=args.model_path)
        log.info(f"=== run {i + 1}/{len(todo)}: {run} | {len(heads)} heads {heads[:16]}{'...' if len(heads) > 16 else ''}"
                 + (f" | their scores {meta['head_scores'][:16]}" if heads else ""))
        masker = rc.HeadMasker(space, heads) if heads else None
        try:
            results[run] = run_niah(model, tok, args, lengths, depths, haystack, masker, path, meta, prog)
        finally:
            if masker is not None:
                masker.remove()
    if results:
        log.info("summary of this session (mean NIAH score): "
                 + ", ".join(f"{r}={v:.1f}" for r, v in results.items()))


if __name__ == "__main__":
    main()
