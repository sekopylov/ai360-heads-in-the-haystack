"""
Retrieval-head detection for Qwen3.5 (linear + full attention heads).
Port of retrieval_head_detection.py from https://github.com/nightdessert/Retrieval_Head

Example:
  python detect.py --model_path Qwen/Qwen3.5-0.8B --haystack_root ../Retrieval_Head --s_len 1000 --e_len 16000

Writes (format of the original repo, {"layer-head": [score per successful sample]}):
  head_score/<name>.json           variant "attn"    (argmax of the attention row)
  head_score/<name>.contrib.json   variant "contrib" (argmax of |A| * ||v||)
  head_score/<name>.meta.json      head kinds + run info
  head_score/<name>.samples.jsonl  per-sample responses / rouge
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

import rh_core as rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--name", default=None, help="output name (default: last part of model_path)")
    p.add_argument("--haystack_root", default="../Retrieval_Head", help="clone of the authors' repo (data only)")
    p.add_argument("-s", "--s_len", type=int, default=1000)
    p.add_argument("-e", "--e_len", type=int, default=16000)
    p.add_argument("--num_lengths", type=int, default=20)
    p.add_argument("--num_depths", type=int, default=10)
    p.add_argument("--prompt_format", choices=["chat", "raw"], default="chat")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--success_threshold", type=float, default=50.0, help="rouge-1 recall needed to count a sample")
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--out_dir", default="head_score")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    name = args.name or args.model_path.rstrip("/").split("/")[-1]
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, name)

    tok, model = rc.load_model(args.model_path, args.device, args.dtype)
    space = rc.HeadSpace(model)
    print(space.describe())
    cap = rc.AttnCapture(space, topk=args.topk, chunk=args.chunk)

    data_dir = os.path.join(args.haystack_root, "haystack_for_detect")
    needles = [json.loads(l) for l in open(os.path.join(data_dir, "needles.jsonl"))]
    lengths, depths = rc.grid(args.s_len, args.e_len, args.num_lengths, args.num_depths)

    counters = {v: defaultdict(list) for v in rc.VARIANTS}
    n_total = n_ok = 0
    samples_f = open(base + ".samples.jsonl", "w")
    for ni, nd in enumerate(needles):
        hay_dir = os.path.join(data_dir, f"part{ni + 1}")
        haystack = rc.read_haystack(hay_dir, max(lengths))
        for L in lengths:
            for depth in depths:
                t0 = time.time()
                context = rc.insert_needle(tok, haystack, nd["needle"], depth, L)
                input_ids = rc.build_prompt_ids(tok, context, nd["question"], args.prompt_format)
                prompt = input_ids[0]
                ns, ne = rc.find_needle_idx(tok, prompt, nd["real_needle"])
                input_ids = input_ids.to(args.device)
                gen = rc.greedy_generate(model, tok, input_ids, args.max_new_tokens)
                response = tok.decode(gen, skip_special_tokens=True).strip()
                score = rc.rouge_recall(nd["real_needle"], response)
                n_total += 1
                ok = score > args.success_threshold and ns >= 0
                if ok:
                    n_ok += 1
                    # sequence the model actually read while decoding: prompt + generated[:-1]
                    seq = torch.cat([input_ids[0], torch.tensor(gen[:-1], device=args.device, dtype=input_ids.dtype)])
                    q_start = input_ids.shape[1] - 1  # position of the token fed at decoding step 0
                    res = cap.run(model, seq[None], q_start)
                    for v in rc.VARIANTS:
                        sc = rc.retrieval_scores(res, space, seq, gen, (ns, ne), v)
                        for key, val in sc.items():
                            counters[v][key].append(val)
                rec = dict(needle=ni, context_length=L, depth_percent=depth, tokens=int(prompt.shape[0]),
                           needle_span=[ns, ne], score=score, response=response, used=ok,
                           seconds=round(time.time() - t0, 2))
                samples_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                samples_f.flush()
                print(f"[needle {ni}] len={L} depth={depth}% rouge={score:.0f} ok={ok} ({rec['seconds']}s) -> {response[:80]!r}")
                if ok:
                    mean = {k: np.mean(v) for k, v in counters["attn"].items()}
                    top = sorted(mean, key=mean.get, reverse=True)[:10]
                    print("   top heads:", ", ".join(f"{k}({space.kind_of[k][0]}):{mean[k]:.2f}" for k in top))
    samples_f.close()

    # same format as the original repo: one json line
    with open(base + ".json", "w") as f:
        json.dump(counters["attn"], f)
    with open(base + ".contrib.json", "w") as f:
        json.dump(counters["contrib"], f)
    meta = dict(model=args.model_path, kinds=space.kind_of, layer_kinds=space.kinds, n_heads=space.n_heads,
                n_samples=n_total, n_used=n_ok, args=vars(args))
    with open(base + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"done: {n_ok}/{n_total} samples used for scoring -> {base}.json")


if __name__ == "__main__":
    main()
