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

from mamba2_implicit_attention import Mamba2RetrievalScorer, greedy_answer, head_geometry
from niah_utils import NeedleHaystack, load_needles, make_grid, rouge1_recall


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
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = args.name or args.model.rstrip("/").split("/")[-1]
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
                                                 device_map="auto").eval()
    L, H, P = head_geometry(model)
    print(f"{name}: {L} layers x {H} heads = {L * H} heads (head_dim={P})")

    scorer = Mamba2RetrievalScorer(model, weighting=args.weighting, pos_tolerance=args.pos_tolerance)
    nh = NeedleHaystack(tok)
    needles = load_needles(args.haystack_dir, args.needles_file)
    grid = make_grid(args.lengths, depth_intervals=args.depth_intervals)

    hard_counter, soft_counter = defaultdict(list), defaultdict(list)
    runs_f = open(os.path.join(args.out_dir, f"{name}_runs.jsonl"), "w")
    n_ok = 0
    n_total = len(needles) * len(grid)
    t0 = time.time()
    for item in needles:
        for (ctx_len, depth) in grid:
            prompt = nh.build_prompt(item, ctx_len, depth)
            prompt_ids = torch.tensor(tok(prompt, add_special_tokens=True).input_ids)
            ns, ne = nh.find_needle_span(prompt_ids.tolist(), item["real_needle"])
            gen = greedy_answer(model, tok, prompt_ids, args.max_new_tokens)
            response = tok.decode(gen, skip_special_tokens=True).strip()
            score = rouge1_recall(item["real_needle"], response)
            used = False
            if ns >= 0 and len(gen) > 0 and (score > args.success_threshold or args.accumulate_all):
                full = torch.cat([prompt_ids, torch.tensor(gen, dtype=torch.long)])
                q_idx = torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(gen))
                hard, soft = scorer.score(full, q_idx, torch.tensor(gen), ns, ne)
                for l in range(L):
                    for h in range(H):
                        hard_counter[f"{l}-{h}"].append(float(hard[l, h]))
                        soft_counter[f"{l}-{h}"].append(float(soft[l, h]))
                used = True
                n_ok += 1
            runs_f.write(json.dumps(dict(needle=item["needle"], context_length=ctx_len, depth=depth,
                                         prompt_tokens=len(prompt_ids), needle_span=[ns, ne],
                                         response=response, rouge1_recall=score, used=used)) + "\n")
            runs_f.flush()
            print(f"[{time.time() - t0:7.0f}s] len={ctx_len:6d} depth={depth:3d} score={score:5.1f} "
                  f"used={used} | {response[:80]!r}")
            if used:
                mean = sorted(((k, np.mean(v)) for k, v in hard_counter.items()), key=lambda x: -x[1])
                print("   top heads:", [(k, round(v, 3)) for k, v in mean[:10]], f"(runs used: {n_ok})")

    runs_f.close()
    with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
        json.dump(hard_counter, f)
    with open(os.path.join(args.out_dir, f"{name}_soft.json"), "w") as f:
        json.dump(soft_counter, f)
    print(f"\nused {n_ok}/{n_total} runs (successful retrievals). Saved to {args.out_dir}/{name}.json")
    if n_ok == 0:
        print("No successful runs! Try shorter --lengths or --accumulate_all.")


if __name__ == "__main__":
    main()
