"""
Retrieval head detection with the new code. Writes one .npz per sample in the format of the dump of the
old code, so source/check_dump.py and rh.compare_dumps work on the result.

Replay the inputs of an old dump (checks the model side only, the inputs are taken from the dump):
python -m rh.detect --model_path Qwen/Qwen1.5-14B-Chat --legacy \
    --replay source/results/dump/Qwen1.5-14B-Chat/detect --needle 0 --out results/new/replay

Build the contexts (legacy = as the authors' code with the default provider):
python -m rh.detect --model_path Qwen/Qwen1.5-14B-Chat --legacy \
    --haystack_dir source/haystack_for_detect --s_len 1000 --e_len 30000 --out results/new/detect
"""
import argparse
import glob
import json
import os
import time

import numpy as np
from rouge_score import rouge_scorer

from . import haystack

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


def samples_from_dump(args):
    files = sorted(glob.glob(f"{args.replay}/*.npz"))
    for file in files:
        d = np.load(file)
        if args.needle is not None and int(d["needle_idx"]) != args.needle:
            continue
        yield dict(name=os.path.basename(file), input_ids=d["input_ids"].tolist(), needle_idx=int(d["needle_idx"]),
                   context_length=int(d["context_length"]), depth_percent=float(d["depth_percent"]),
                   needle_start=int(d["needle_start"]), needle_end=int(d["needle_end"]),
                   full_steps=sum(k.startswith("attn_step") for k in d.files))


def samples_from_grid(args, enc):
    context_lengths, depths = haystack.grid(args.s_len, args.e_len, args.context_intervals, args.depths)
    periods = haystack.period_tokens(enc, "llama" if args.legacy else "model")
    chat = enc.chat_template is not None and not args.raw_prompt
    pick = lambda l: {l[0], l[len(l) // 2], l[-1]}
    for ni, needle in enumerate(haystack.load_needles(args.haystack_dir)):
        if args.needle is not None and ni != args.needle:
            continue
        text = haystack.read_haystack(needle["haystack_dir"], max(context_lengths))
        tokens = enc.encode(text)
        for context_length in context_lengths:
            for depth in depths:
                context = haystack.build_context(enc, tokens, text, needle["needle"], context_length, depth, periods)
                if args.legacy:
                    input_ids = haystack.build_prompt(enc, context, needle["question"], chat)
                    start, end = haystack.find_needle_fuzzy(enc, input_ids, needle["real_needle"])
                else:
                    input_ids, start, end = haystack.build_prompt_exact(enc, context, needle["question"], chat, needle["real_needle"])
                full = args.full_steps if ni == 0 and context_length in pick(context_lengths) and depth in pick(depths) else 0
                yield dict(name=f"needle{ni}_len{context_length}_depth{depth}.npz", input_ids=input_ids, needle_idx=ni,
                           context_length=context_length, depth_percent=float(depth), needle_start=start, needle_end=end,
                           full_steps=full, real_needle=needle["real_needle"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--out', type=str, default=None, help='folder for the .npz files, not needed with --inputs_only')
    parser.add_argument('--legacy', action='store_true', help="repeat the authors' code: Llama period ids for the needle insertion and their stop condition")
    parser.add_argument('--replay', type=str, default=None, help='folder with the dump of the old code: take the inputs from it')
    parser.add_argument('--haystack_dir', type=str, default="source/haystack_for_detect")
    parser.add_argument('--s_len', type=int, default=1000)
    parser.add_argument('--e_len', type=int, default=30000)
    parser.add_argument('--context_intervals', type=int, default=20)
    parser.add_argument('--depths', type=lambda s: [int(x) for x in s.split(',')], default=None, help='comma separated, default 10 depths from 0 to 100')
    parser.add_argument('--needle', type=int, default=None, help='only this needle')
    parser.add_argument('--limit', type=int, default=None, help='stop after this number of samples')
    parser.add_argument('--every', type=int, default=1, help='take every N-th sample, to check a part of the grid')
    parser.add_argument('--full_steps', type=int, default=3, help='decode steps with full attention rows on a few samples, grid mode')
    parser.add_argument('--raw_prompt', action='store_true', help='do not use the chat template')
    parser.add_argument('--slow_tokenizer', action='store_true')
    parser.add_argument('--inputs_only', action='store_true', help='build the inputs and compare them with --ref_dump, the model is not loaded')
    parser.add_argument('--ref_dump', type=str, default=None)
    args = parser.parse_args()
    if args.inputs_only and not args.ref_dump:
        parser.error('--inputs_only needs --ref_dump')
    if not args.inputs_only and not args.out:
        parser.error('--out is required')

    if args.inputs_only:
        from transformers import AutoTokenizer
        enc = AutoTokenizer.from_pretrained(args.model_path, use_fast=not args.slow_tokenizer)
        same, total = 0, 0
        for sample in samples_from_grid(args, enc):
            ref = np.load(f"{args.ref_dump}/{sample['name']}")
            ok = ref["input_ids"].tolist() == sample["input_ids"] and int(ref["needle_start"]) == sample["needle_start"]
            same, total = same + ok, total + 1
            if not ok:
                print("differs:", sample["name"], "length", len(sample["input_ids"]), "vs", len(ref["input_ids"]),
                      "needle", sample["needle_start"], "vs", int(ref["needle_start"]))
        print(f"inputs identical to the old code: {same}/{total}")
        raise SystemExit

    from . import runner
    enc, model = runner.load(args.model_path, slow_tokenizer=args.slow_tokenizer)
    stop = runner.stop_tokens(enc, model, args.legacy)
    real_needles = [n["real_needle"] for n in haystack.load_needles(args.haystack_dir)]
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/run_meta.json", "w") as f:
        import torch, transformers
        json.dump({"args": vars(args), "torch": torch.__version__, "transformers": transformers.__version__,
                   "stop_tokens": sorted(stop)}, f, indent=1)

    samples = samples_from_dump(args) if args.replay else samples_from_grid(args, enc)
    done = 0
    for n, sample in enumerate(samples):
        if n % args.every:
            continue
        if args.limit is not None and done >= args.limit:
            break
        done += 1
        start_time = time.time()
        result = runner.run_sample(model, sample["input_ids"], sample["needle_start"], sample["needle_end"], stop,
                                   full_steps=sample["full_steps"])
        response = enc.decode(result["output_ids"], skip_special_tokens=True).strip()
        score = scorer.score(real_needles[sample["needle_idx"]], response)['rouge1'].recall * 100
        np.savez_compressed(
            f"{args.out}/{sample['name']}", input_ids=np.array(sample["input_ids"], dtype=np.int32),
            needle_start=sample["needle_start"], needle_end=sample["needle_end"], needle_idx=sample["needle_idx"],
            context_length=sample["context_length"], depth_percent=sample["depth_percent"],
            rouge=score, response=response, **result)
        print(f"{sample['name']}: {time.time() - start_time:.1f}s, rouge {score:.1f}, {response[:80]!r}", flush=True)
