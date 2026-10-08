"""
Retrieval-head detection for Mamba-2 (linear attention) models,
following Wu et al. 2024 (arXiv:2404.15574), generalised via implicit attention.

For every (needle, context length, depth):
  1. build the NIAH prompt, greedy-decode the answer (stop at newline);
  2. if ROUGE-1 recall(real_needle, answer) > 50  (as in the paper),
     run one teacher-forced pass on prompt+answer and, for every generated token,
     check for every head whether its top-1 implicit attention lands on the needle
     position holding that very token (copy-paste event);
  3. per-run head score = #hits / |needle tokens|; final score = mean over runs.

Output (same format as the authors' head_score/*.json):
  {out_dir}/{name}.json        {"layer-head": [score_run1, score_run2, ...], ...}   (hard, paper's score)
  {out_dir}/{name}_soft.json   same with the soft (attention-mass) score
  {out_dir}/{name}_runs.jsonl  per-run log
The json files are rewritten after every used run, so partial results are always on disk.

Progress:  tail -f logs/detect_retrieval_heads_latest.log

Example:
  git clone https://github.com/nightdessert/Retrieval_Head
  python detect_retrieval_heads.py --model AntonV/mamba2-2.7b-hf \
      --haystack_dir Retrieval_Head/haystack_for_detect \
      --lengths 500 1000 1500 2000 --depth_intervals 10
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from log_utils import Progress, add_logging_args, gpu_mem, load_model_logged, setup_logging
from mamba2_implicit_attention import Mamba2RetrievalScorer, greedy_answer, head_geometry
from niah_utils import NeedleHaystack, load_needles, make_grid, rouge1_recall

BINS = [(0.0, 0.1), (0.1, 0.4), (0.4, 10.0)]


def summary(counter):
    means = np.array([np.mean(v) for v in counter.values()])
    return ", ".join(f"[{lo},{hi if hi < 10 else '1'}): {int(((means >= lo) & (means < hi)).sum())}"
                     for lo, hi in BINS)


def save(counter, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(counter, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="AntonV/mamba2-2.7b-hf")
    ap.add_argument("--name", default=None, help="output name (default: last part of --model)")
    ap.add_argument("--haystack_dir", default="Retrieval_Head/haystack_for_detect")
    ap.add_argument("--needles_file", default=None)
    ap.add_argument("--lengths", type=int, nargs="+", default=[500, 1000, 1500, 2000])
    ap.add_argument("--depth_intervals", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=50)
    ap.add_argument("--success_threshold", type=float, default=50.0,
                    help="accumulate head scores only for runs with ROUGE-1 recall > this (paper: 50)")
    ap.add_argument("--accumulate_all", action="store_true",
                    help="accumulate scores for all runs (useful if the model rarely succeeds)")
    ap.add_argument("--weighting", default="contrib", choices=["contrib", "alpha"],
                    help="contrib: argmax |alpha|*||x||  (default);  alpha: argmax |alpha|")
    ap.add_argument("--pos_tolerance", type=int, default=0,
                    help="also count a hit if the copied token is up to k positions before the "
                         "attended one (Mamba conv1d smears tokens over 4 positions). Paper: 0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out_dir", default="head_score")
    ap.add_argument("--top_print", type=int, default=10, help="how many top heads to log after each run")
    add_logging_args(ap)
    args = ap.parse_args()
    log = setup_logging("detect_retrieval_heads", args)

    name = args.name or args.model.rstrip("/").split("/")[-1]
    os.makedirs(args.out_dir, exist_ok=True)
    hard_path = os.path.join(args.out_dir, f"{name}.json")
    soft_path = os.path.join(args.out_dir, f"{name}_soft.json")
    runs_path = os.path.join(args.out_dir, f"{name}_runs.jsonl")

    tok, model = load_model_logged(args.model, args.dtype)
    L, H, P = head_geometry(model)
    log.info(f"{name}: {L} layers x {H} heads = {L * H} heads (head_dim={P})")

    scorer = Mamba2RetrievalScorer(model, weighting=args.weighting, pos_tolerance=args.pos_tolerance)
    nh = NeedleHaystack(tok)
    needles = load_needles(args.haystack_dir, args.needles_file)
    grid = make_grid(args.lengths, depth_intervals=args.depth_intervals)
    n_total = len(needles) * len(grid)
    log.info(f"{len(needles)} needles x {len(grid)} (length, depth) points = {n_total} NIAH runs")
    for i, it in enumerate(needles):
        log.info(f"  needle {i}: {it['needle'][:90]!r} | haystack {it['haystack_dir']}")
    log.info(f"outputs: {hard_path}, {soft_path}, {runs_path}")

    hard_counter, soft_counter = defaultdict(list), defaultdict(list)
    runs_f = open(runs_path, "w")
    n_ok, n_fail, n_nospan = 0, 0, 0
    prog = Progress(n_total, "detect")
    t_start = time.time()

    for ni, item in enumerate(needles):
        log.info("-" * 90)
        log.info(f"needle {ni + 1}/{len(needles)}: Q: {item['question']!r} | A: {item['real_needle']!r}")
        for (ctx_len, depth) in grid:
            t0 = time.time()
            prompt = nh.build_prompt(item, ctx_len, depth)
            prompt_ids = torch.tensor(tok(prompt, add_special_tokens=True).input_ids)
            ns, ne = nh.find_needle_span(prompt_ids.tolist(), item["real_needle"])
            if ns < 0:
                log.warning(f"needle span not found in prompt (len={ctx_len}, depth={depth})")
            gen = greedy_answer(model, tok, prompt_ids, args.max_new_tokens)
            t_gen = time.time() - t0
            response = tok.decode(gen, skip_special_tokens=True).strip()
            score = rouge1_recall(item["real_needle"], response)

            used, t_score = False, 0.0
            if ns >= 0 and len(gen) > 0 and (score > args.success_threshold or args.accumulate_all):
                t1 = time.time()
                full = torch.cat([prompt_ids, torch.tensor(gen, dtype=torch.long)])
                q_idx = torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(gen))
                hard, soft = scorer.score(full, q_idx, torch.tensor(gen), ns, ne)
                for l in range(L):
                    for h in range(H):
                        hard_counter[f"{l}-{h}"].append(float(hard[l, h]))
                        soft_counter[f"{l}-{h}"].append(float(soft[l, h]))
                save(hard_counter, hard_path)
                save(soft_counter, soft_path)
                t_score = time.time() - t1
                used = True
                n_ok += 1
            elif ns < 0:
                n_nospan += 1
            else:
                n_fail += 1

            runs_f.write(json.dumps(dict(needle=item["needle"], context_length=ctx_len, depth=depth,
                                         prompt_tokens=len(prompt_ids), needle_span=[ns, ne],
                                         response=response, rouge1_recall=score, used=used,
                                         t_generate=t_gen, t_score=t_score)) + "\n")
            runs_f.flush()

            status = "USED " if used else ("NOSPAN" if ns < 0 else "FAIL ")
            log.info(f"{prog.step()} {status} len={ctx_len:6d} ({len(prompt_ids)} tok) depth={depth:3d}% "
                     f"rouge={score:5.1f} | gen {t_gen:.1f}s score {t_score:.1f}s | {gpu_mem()} | "
                     f"answer: {response[:70]!r}")
            if used:
                this_run = sorted(((f"{l}-{h}", float(hard[l, h])) for l in range(L) for h in range(H)),
                                  key=lambda x: -x[1])[:args.top_print]
                log.info(f"    this run top heads: {[(k, round(v, 3)) for k, v in this_run]}")
                mean = sorted(((k, np.mean(v)) for k, v in hard_counter.items()), key=lambda x: -x[1])
                log.info(f"    running mean top heads ({n_ok} runs): "
                         f"{[(k, round(v, 3)) for k, v in mean[:args.top_print]]}")
                log.info(f"    running bins: {summary(hard_counter)}")
        log.info(f"needle {ni + 1} done | used {n_ok}, failed {n_fail}, no-span {n_nospan} so far")

    runs_f.close()
    save(hard_counter, hard_path)
    save(soft_counter, soft_path)
    log.info("=" * 90)
    log.info(f"finished in {time.time() - t_start:.0f}s | used {n_ok}/{n_total} runs, "
             f"failed (rouge<={args.success_threshold}) {n_fail}, needle not found {n_nospan}")
    if n_ok:
        log.info(f"retrieval score bins (hard): {summary(hard_counter)}")
        log.info(f"retrieval score bins (soft): {summary(soft_counter)}")
        log.info(f"saved {hard_path} and {soft_path}")
    else:
        log.warning("No successful runs! Try shorter --lengths or --accumulate_all.")


if __name__ == "__main__":
    main()