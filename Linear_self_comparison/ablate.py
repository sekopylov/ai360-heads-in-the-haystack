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
    """Writes to <path>.part and renames to <path> when complete; an interrupted .part is continued."""
    tmp = path + ".part"
    scores, done = [], set()
    if os.path.exists(tmp):
        for line in open(tmp, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "meta" not in d:
                done.add((d["context_length"], d["depth_percent"]))
                scores.append(d["score"])
        if done:
            log.info(f"RESUME {run_meta['run']}: {len(done)} samples already in {tmp}")
            for L in lengths:  # keep the global progress honest
                for depth in depths:
                    if (L, depth) in done:
                        prog.total_work -= L
                        prog.total_items -= 1
    t_run = time.time()
    f = open_append(tmp)
    if not os.path.getsize(tmp):
        f.write(json.dumps({"meta": run_meta}) + "\n")
    with f:
        for L in lengths:
            for depth in depths:
                if (L, depth) in done:
                    continue
                t0 = time.time()
                context = rc.insert_needle(tok, haystack, SF_NEEDLE, depth, L)
                ids = rc.build_prompt_ids(tok, context, SF_QUESTION, args.prompt_format).to(args.device)
                gen = rc.greedy_generate(model, tok, ids, args.max_new_tokens, masker=masker, scope=args.scope,
                                         prefill_chunk=args.prefill_chunk)
                resp = tok.decode(gen, skip_special_tokens=True).strip()
                score = rc.rouge_recall(SF_ANSWER, resp)
                scores.append(score)
                f.write(json.dumps(dict(context_length=L, depth_percent=depth, score=score, response=resp),
                                   ensure_ascii=False) + "\n")
                f.flush()
                prog.update(L, f"{run_meta['run']}: len={L} depth={depth}% score={score:.0f} "
                               f"({time.time() - t0:.1f}s: {rc.gen_timing_str()}; run mean {np.mean(scores):.1f}) "
                               f"{resp[:40]!r}")
    os.replace(tmp, path)
    log.info(f"=== finished {run_meta['run']}: mean NIAH score {np.mean(scores):.2f} "
             f"over {len(scores)} samples in {rc.fmt_time(time.time() - t_run)} -> {path}")
    return float(np.mean(scores))


def read_part(tmp):
    """{(L, depth): score} already written to a run's .part file."""
    done = {}
    if os.path.exists(tmp):
        for line in open(tmp, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "meta" not in d:
                done[(d["context_length"], d["depth_percent"])] = d["score"]
    return done


def run_grid_batched(model, tok, args, space, lengths, depths, haystack, todo, metas):
    """scope=decode: the prefill is identical for every run at a grid point (masks act only while decoding),
    so each grid point is prefilled ONCE and all runs that still need it are decoded together as one batch,
    each batch row with its own head mask."""
    files, done = {}, {}
    for run, heads, path in todo:
        tmp = path + ".part"
        done[run] = read_part(tmp)
        f = open_append(tmp)
        if os.path.getsize(tmp) == 0:
            f.write(json.dumps({"meta": metas[run]}) + "\n")
            f.flush()
        files[run] = f
        if done[run]:
            log.info(f"RESUME {run}: {len(done[run])} samples already in {tmp}")
    grid = [(L, d) for L in lengths for d in depths]
    pending = {(L, d): [r for r, _, _ in todo if (L, d) not in done[r]] for L, d in grid}
    n_items = sum(len(v) for v in pending.values())
    prog = rc.Progress(n_items, sum(L * len(pending[(L, d)]) for L, d in grid), name="ablate")
    heads_of = {r: h for r, h, _ in todo}
    run_scores = {r: list(done[r].values()) for r, _, _ in todo}
    try:
        for L, depth in grid:
            rows = pending[(L, depth)]
            if not rows:
                continue
            t0 = time.time()
            context = rc.insert_needle(tok, haystack, SF_NEEDLE, depth, L)
            ids = rc.build_prompt_ids(tok, context, SF_QUESTION, args.prompt_format).to(args.device)
            tp = rc._sync_time()
            past = rc.prefill_for_decode(model, ids, args.prefill_chunk)
            t_prefill = rc._sync_time() - tp
            t_dec, steps = 0.0, 0
            for g in range(0, len(rows), args.batch_runs):
                group = rows[g:g + args.batch_runs]
                masker = rc.BatchHeadMasker(space, [heads_of[r] for r in group], args.device)
                try:
                    gens = rc.batched_greedy_decode(model, tok, past, ids[0, -1], len(group),
                                                    args.max_new_tokens, masker)
                finally:
                    masker.remove()
                t_dec += rc.LAST_GEN["batch_decode_s"]
                steps += rc.LAST_GEN["batch_steps"]
                for r, gen in zip(group, gens):
                    resp = tok.decode(gen, skip_special_tokens=True).strip()
                    score = rc.rouge_recall(SF_ANSWER, resp)
                    run_scores[r].append(score)
                    files[r].write(json.dumps(dict(context_length=L, depth_percent=depth, score=score, response=resp),
                                              ensure_ascii=False) + "\n")
                    files[r].flush()
            del past
            line = ", ".join(f"{r}={run_scores[r][-1]:.0f}" for r in rows[:6]) + (" ..." if len(rows) > 6 else "")
            prog.update(L * len(rows), f"len={L} depth={depth}%: {len(rows)} runs in {time.time() - t0:.1f}s "
                                       f"(prefill {t_prefill:.1f}s once + {steps} batched steps {t_dec:.1f}s) | {line}",
                        items=len(rows))
    finally:
        for f in files.values():
            f.close()
    results = {}
    for run, heads, path in todo:
        tmp = path + ".part"
        if len(read_part(tmp)) == len(grid):
            os.replace(tmp, path)
            results[run] = float(np.mean(run_scores[run]))
            log.info(f"=== finished {run}: mean NIAH score {results[run]:.2f} -> {path}")
    return results


def open_append(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            last = fh.read(1)
        if last != b"\n":
            with open(path, "a") as fh:
                fh.write("\n")
    return open(path, "a", encoding="utf-8")


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
    p.add_argument("--prefill_chunk", type=int, default=2048, help="prefill in chunks (bounds memory; 0 = off)")
    p.add_argument("--batch_runs", type=int, default=16,
                   help="runs decoded together from one shared prefill (scope=decode). Lower it if you hit OOM: "
                        "the KV cache of the 6 full-attention layers is replicated per row (~12 MB per 1k tokens)")
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
    metas = {run: dict(run=run, heads=heads, head_scores=[round(scores[h], 4) for h in heads],
                       kinds=[space.kind_of[h] for h in heads], scope=args.scope, model=args.model_path)
             for run, heads, _ in todo}
    for run, heads, _ in todo:
        log.info(f"   {run}: {len(heads)} heads {heads[:16]}{'...' if len(heads) > 16 else ''}"
                 + (f" | scores {metas[run]['head_scores'][:16]}" if heads else ""))
    results = {}
    if args.scope == "decode" and todo:
        results = run_grid_batched(model, tok, args, space, lengths, depths, haystack, todo, metas)
    else:  # masks also act during prefill -> every run needs its own prefill
        prog = rc.Progress(len(todo) * n_per_run, len(todo) * len(depths) * sum(lengths), name="ablate")
        for i, (run, heads, path) in enumerate(todo):
            log.info(f"=== run {i + 1}/{len(todo)}: {run}")
            masker = rc.HeadMasker(space, heads) if heads else None
            try:
                results[run] = run_niah(model, tok, args, lengths, depths, haystack, masker, path, metas[run], prog)
            finally:
                if masker is not None:
                    masker.remove()
    if results:
        log.info("summary of this session (mean NIAH score): "
                 + ", ".join(f"{r}={v:.1f}" for r, v in results.items()))


if __name__ == "__main__":
    main()