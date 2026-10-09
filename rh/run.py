"""
Runs a model over the samples of a task and writes generations and attention into a spool folder (see rh.spool).
Knows nothing about the metrics; --with_metrics only starts rh.metrics as a separate process on the same folder.

Detection:
python -m rh.run --model_path Qwen/Qwen3-8B --task niah --needles data/needles_detect.jsonl \
    --out results/new/qwen3_detect --with_metrics

Masking the 30 best heads of that run on the evaluation needle, attention is not saved:
python -m rh.run --model_path Qwen/Qwen3-8B --task niah --needles data/needles_eval.jsonl \
    --mask_file results/new/qwen3_detect/head_score_copy_count.json --mask_top 30 --save none \
    --out results/new/qwen3_mask_top30 --with_metrics
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

from . import spool
from .tasks import TASKS


def from_masked_run(path):
    """True when the head_score file comes from a run with masked heads: by its name or by the run.json next to it."""
    if os.path.basename(path).startswith("head_score_masked_"):
        return True
    run_json = os.path.join(os.path.dirname(os.path.abspath(path)), "spool", "run.json")
    return os.path.exists(run_json) and bool(spool.load_json(run_json).get("block_list"))


def ranked_heads(path):
    """Heads of a head_score file, best first, as [layer, head]."""
    scores = spool.load_json(path)
    ranked = sorted(scores, key=lambda k: -float(np.mean(scores[k])) if len(scores[k]) else 0.0)
    return [[int(i) for i in k.split("-")] for k in ranked]


def choose_block_list(args, attn_layers, n_heads):
    if args.mask_top:
        return ranked_heads(args.mask_file)[:args.mask_top]
    if args.mask_random:
        # random heads outside the best ones, the same for every sample of the run
        best = {tuple(h) for h in ranked_heads(args.mask_file)[:max(100, args.mask_random)]} if args.mask_file else set()
        heads = [[l, h] for l in attn_layers for h in range(n_heads) if (l, h) not in best]
        return random.Random(args.seed).sample(heads, args.mask_random)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--task', type=str, default="niah", choices=sorted(TASKS))
    parser.add_argument('--out', type=str, required=True, help='run folder; the spool is <out>/spool unless --spool is given')
    parser.add_argument('--spool', type=str, default=None)
    parser.add_argument('--save', type=str, default="compact", choices=["compact", "rows", "none"],
                        help='compact: top-k positions and span masses per head; rows: also whole attention rows; none: answers only')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--max_new_tokens', type=int, default=256, help='cap of the answer length; the generation stops earlier at the end of the answer')
    parser.add_argument('--limits', type=lambda s: [int(x) for x in s.split(',')], default=[50, 64, 128, 256],
                        help='token limits to evaluate from this one run: the answer and the head metrics are also computed '
                             'as if the generation had been cut at each of them (see rh.limits)')
    parser.add_argument('--buffer_gb', type=float, default=2.0, help='wait while the unread spool is larger than this; 0: no limit')
    parser.add_argument('--chunk_mb', type=int, default=256)
    parser.add_argument('--mask_file', type=str, default=None, help='head_score json that ranks the heads')
    parser.add_argument('--mask_top', type=int, default=0, help='mask this number of the best heads of --mask_file')
    parser.add_argument('--mask_random', type=int, default=0, help='mask this number of random heads outside the best 100 of --mask_file')
    parser.add_argument('--seed', type=int, default=0, help='seed of the random heads')
    parser.add_argument('--limit', type=int, default=None, help='stop after this number of samples')
    parser.add_argument('--every', type=int, default=1, help='take every N-th sample')
    parser.add_argument('--dtype', type=str, default="auto")
    parser.add_argument('--with_metrics', action='store_true', help='start rh.metrics --follow --consume on the spool as a separate process')
    task_class = TASKS[parser.parse_known_args()[0].task]
    task_class.add_arguments(parser)
    args = parser.parse_args()
    if args.mask_top and args.mask_random:
        parser.error('--mask_top and --mask_random are exclusive')
    if args.mask_top and not args.mask_file:
        parser.error('--mask_top needs --mask_file')
    if args.mask_file and from_masked_run(args.mask_file):
        parser.error(f'{args.mask_file} comes from a run with masked heads; the heads must be ranked by a run without a mask')
    spool_dir = args.spool or f"{args.out}/spool"

    from . import model as rh_model
    enc, model, attn_layers = rh_model.load(args.model_path, args.dtype)
    stop = rh_model.stop_tokens(enc, model)
    block_list = choose_block_list(args, attn_layers, model.config.num_attention_heads)
    task = task_class(enc, args)

    done_ids = set()
    if os.path.exists(f"{args.out}/samples.jsonl"):
        with open(f"{args.out}/samples.jsonl", encoding="utf-8") as f:
            done_ids = {json.loads(l)["id"] for l in f if l.strip()}
        print(f"{len(done_ids)} samples are already in {args.out}/samples.jsonl and will be skipped", flush=True)

    import torch, transformers
    writer = spool.SpoolWriter(spool_dir, args.save, args.topk, args.buffer_gb, args.chunk_mb)
    if writer.pending_ids:
        print(f"{len(writer.pending_ids)} complete samples wait in the spool for the metrics and will be skipped", flush=True)
    done_ids |= writer.pending_ids
    limits = sorted({l for l in args.limits if l < args.max_new_tokens} | {args.max_new_tokens})
    writer.start_run({"args": vars(args), "attn_layers": attn_layers, "block_list": block_list, "stop_tokens": sorted(stop),
                      "limits": limits,
                      "torch": torch.__version__, "transformers": transformers.__version__})
    metrics = None
    if args.with_metrics:
        metrics = subprocess.Popen([sys.executable, "-m", "rh.metrics", spool_dir, "--out", args.out, "--follow", "--consume"])

    taken, completed = 0, False
    try:
        for n, sample in enumerate(task.samples()):
            if n % args.every or sample.id in done_ids:
                continue
            if args.limit is not None and taken >= args.limit:
                break
            taken += 1
            start_time = time.time()
            writer.start_sample(sample)
            output = []
            for token, rows in rh_model.generate(model, sample.input_ids, stop, args.max_new_tokens, block_list, capture=args.save != "none"):
                output.append(token)
                writer.add_step(token, rows)
            response = enc.decode(output, skip_special_tokens=True).strip()
            # the text the answer would have had under every shorter limit; decoded here, the metrics have no tokenizer
            responses_at = {str(l): enc.decode(output[:l], skip_special_tokens=True).strip() for l in limits if l < len(output)}
            writer.end_sample(output, response, round(time.time() - start_time, 2),
                              stopped=bool(output) and output[-1] in stop, responses_at=responses_at)
            if metrics is None:
                print(f"{sample.id}: {time.time() - start_time:.1f}s, {len(output)} tokens, {response[:70]!r}", flush=True)
        completed = True
    finally:
        # also after an error, so that the metrics do not wait for data that will never come
        writer.end_run(completed)
        if metrics is not None:
            metrics.wait()
