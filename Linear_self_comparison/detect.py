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

log = rc.log


def log_top(counter, space, n=10):
    if not counter:
        return
    mean = {k: np.mean(v) for k, v in counter.items()}
    for kind in ("full", "linear"):
        ks = [k for k in mean if space.kind_of[k] == kind]
        top = sorted(ks, key=mean.get, reverse=True)[:n]
        n01 = sum(mean[k] >= 0.1 for k in ks)
        log.info(f"   top {kind:6s} heads ({n01}/{len(ks)} with score>=0.1): "
                 + ", ".join(f"{k}:{mean[k]:.2f}" for k in top))


def save(base, counters, space, args, n_total, n_ok, partial):
    """Write head scores (same json format as the original repo). Partial checkpoints go to <name>.partial.*"""
    b = base + (".partial" if partial else "")
    with open(b + ".json", "w") as f:
        json.dump(counters["attn"], f)
    with open(b + ".contrib.json", "w") as f:
        json.dump(counters["contrib"], f)
    meta = dict(model=args.model_path, kinds=space.kind_of, layer_kinds=space.kinds, n_heads=space.n_heads,
                n_samples=n_total, n_used=n_ok, partial=partial, args=vars(args))
    with open(b + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    if partial:
        log.info(f"   checkpoint saved: {b}.json ({n_ok} used samples)")


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
    p.add_argument("--chunk", type=int, default=256, help="block size of the implicit-attention scan (exact for any value)")
    p.add_argument("--prefill_chunk", type=int, default=2048,
                   help="prefill / capture in chunks of this many tokens (bounds attention memory; 0 = no chunking)")
    p.add_argument("--log_top_every", type=int, default=10, help="log current top heads every N used samples")
    p.add_argument("--out_dir", default="head_score")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--allow_cpu", action="store_true", help="do not abort when CUDA is unavailable")
    p.add_argument("--fresh", action="store_true", help="ignore previous results of this name and start over")
    args = p.parse_args()

    name = args.name or args.model_path.rstrip("/").split("/")[-1]
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, name)
    rc.setup_logging("detect", name)
    log.info(f"args: {vars(args)}")

    tok, model, args.device = rc.load_model(args.model_path, args.device, args.dtype, allow_cpu=args.allow_cpu)
    space = rc.HeadSpace(model)
    log.info(space.describe())
    cap = rc.AttnCapture(space, topk=args.topk, chunk=args.chunk, prefill_chunk=args.prefill_chunk)

    data_dir = os.path.join(args.haystack_root, "haystack_for_detect")
    needles = [json.loads(l) for l in open(os.path.join(data_dir, "needles.jsonl"))]
    lengths, depths = rc.grid(args.s_len, args.e_len, args.num_lengths, args.num_depths)

    counters = {v: defaultdict(list) for v in rc.VARIANTS}
    n_total = n_ok = 0
    grid_keys = {(ni, L, d) for ni in range(len(needles)) for L in lengths for d in depths}
    samples_path, persample_path = base + ".samples.jsonl", base + ".persample.jsonl"

    # ---- resume: samples already in <name>.samples.jsonl are skipped, their head scores re-loaded
    done = {}
    if args.fresh:
        for pth in (samples_path, persample_path):
            if os.path.exists(pth):
                os.replace(pth, pth + f".old_{time.strftime('%Y%m%d_%H%M%S')}")
    else:
        done = {k: r for k, r in read_jsonl(samples_path, lambda r: (r["needle"], r["context_length"], r["depth_percent"]))
                if k in grid_keys and not r.get("error")}
        per = dict(read_jsonl(persample_path, lambda r: (r["needle"], r["context_length"], r["depth_percent"])))
        if done and not per:
            per = migrate_from_partial(base, samples_path, persample_path)
        for k, r in done.items():
            n_total += 1
            if r["used"]:
                if k not in per:  # scores lost (crash between the two writes): redo this sample
                    done[k] = None
                    n_total -= 1
                    continue
                n_ok += 1
                for v in rc.VARIANTS:
                    for key, val in per[k][v].items():
                        counters[v][key].append(val)
        done = {k: r for k, r in done.items() if r is not None}
        if done:
            log.info(f"RESUME: {len(done)} samples already done ({n_ok} used) -> skipping them "
                     f"(use --fresh to start over)")

    todo = [(ni, L, d) for ni in range(len(needles)) for L in lengths for d in depths if (ni, L, d) not in done]
    log.info(f"grid: {len(needles)} needles x {len(lengths)} lengths {lengths} x {len(depths)} depths = "
             f"{len(grid_keys)} samples, {len(todo)} to do")
    prog = rc.Progress(len(todo), sum(L for _, L, _ in todo), name="detect")
    samples_f = open_append(samples_path)
    persample_f = open_append(persample_path)
    haystacks = {}

    def process(ni, L, depth, chunk):
        """One sample. Returns (rec, rec_scores or None). Raises torch.OutOfMemoryError."""
        nd = needles[ni]
        t0 = time.time()
        context = rc.insert_needle(tok, haystacks[ni], nd["needle"], depth, L)
        input_ids = rc.build_prompt_ids(tok, context, nd["question"], args.prompt_format)
        prompt = input_ids[0]
        ns, ne = rc.find_needle_idx(tok, prompt, nd["real_needle"])
        input_ids = input_ids.to(args.device)
        rc.STAGE = "generate"
        # the capture hooks store each layer's inputs during this generation -> no second pass over the prompt
        gen = rc.greedy_generate(model, tok, input_ids, args.max_new_tokens, prefill_chunk=chunk, capture=cap)
        timing = rc.gen_timing_str()
        response = tok.decode(gen, skip_special_tokens=True).strip()
        score = rc.rouge_recall(nd["real_needle"], response)
        ok = score > args.success_threshold and ns >= 0
        if ns < 0:
            log.warning(f"needle not found in prompt tokens (needle {ni}, len {L}, depth {depth}) -> sample not used")
        rec_scores = None
        if ok:
            rc.STAGE = "capture"
            rc.peak_reset()
            t_cap = rc._sync_time()
            # sequence the model actually read while decoding: prompt + generated[:-1]
            seq = torch.cat([input_ids[0], torch.tensor(gen[:-1], device=args.device, dtype=input_ids.dtype)])
            q_start = input_ids.shape[1] - 1  # position of the token fed at decoding step 0
            res = cap.finalize(q_start)
            rec_scores = {}
            for v in rc.VARIANTS:
                sc = rc.retrieval_scores(res, space, seq, gen, (ns, ne), v)
                rec_scores[v] = {k: round(x, 5) for k, x in sc.items()}
            timing += f", capture {rc._sync_time() - t_cap:.1f}s (peak {rc.peak_gb():.1f}G)"
        cap.reset()  # drop collected hidden states (also for unused samples)
        rc.STAGE = None
        rec = dict(needle=ni, context_length=L, depth_percent=depth, tokens=int(prompt.shape[0]),
                   needle_span=[ns, ne], score=score, response=response, used=ok,
                   seconds=round(time.time() - t0, 2), timing=timing, prefill_chunk=chunk)
        return rec, rec_scores

    for ni, L, depth in todo:
        if ni not in haystacks:
            haystacks[ni] = rc.read_haystack(os.path.join(data_dir, f"part{ni + 1}"), max(lengths))
        chunk, rec, rec_scores = args.prefill_chunk, None, None
        for attempt in range(3):
            try:
                rec, rec_scores = process(ni, L, depth, chunk)
                break
            except torch.OutOfMemoryError:
                import traceback

                stage = rc.STAGE
                log.error(f"CUDA OOM at needle {ni} len={L} depth={depth}% in stage '{stage}' with prefill_chunk={chunk} "
                          f"| {rc.gpu_mem()}\n" + traceback.format_exc(limit=-6).rstrip())
                cap.reset()
                rc.free_gpu()
                log.error(f"   after cleanup: {rc.gpu_mem()}")
                chunk = max(256, (chunk or 4096) // 4)
        if rec is None:
            log.error(f"   giving up on needle {ni} len={L} depth={depth}% (recorded as error, retried on next resume)")
            samples_f.write(json.dumps(dict(needle=ni, context_length=L, depth_percent=depth, used=False,
                                            error="oom")) + "\n")
            samples_f.flush()
            prog.update(L, f"needle {ni} len={L} depth={depth}% SKIPPED (OOM)")
            continue
        n_total += 1
        ok = rec["used"]
        if ok:
            n_ok += 1
            for v in rc.VARIANTS:
                for key, val in rec_scores[v].items():
                    counters[v][key].append(val)
            persample_f.write(json.dumps(dict(needle=ni, context_length=L, depth_percent=depth, **rec_scores)) + "\n")
            persample_f.flush()
        samples_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        samples_f.flush()
        prog.update(L, f"needle {ni} len={L} depth={depth}% rouge={rec['score']:.0f} used={ok} "
                       f"({rec['seconds']}s: {rec['timing']}; used {n_ok}/{n_total}) -> {rec['response'][:50]!r}")
        if L >= 8000:
            rc.free_gpu()
        if ok and (n_ok % args.log_top_every == 1 or args.log_top_every == 1):
            log_top(counters["attn"], space)
            save(base, counters, space, args, n_total, n_ok, partial=True)
    log_top(counters["attn"], space)
    samples_f.close()
    persample_f.close()
    save(base, counters, space, args, n_total, n_ok, partial=False)
    log.info(f"done: {n_ok}/{n_total} samples used for scoring -> {base}.json")


def migrate_from_partial(base, samples_path, persample_path):
    """Runs made before per-sample files existed: the partial checkpoint keeps, for every head, the list of
    scores in the order of used samples -> map the i-th used sample in samples.jsonl to the i-th entry."""
    part = {v: base + ".partial" + ("" if v == "attn" else "." + v) + ".json" for v in rc.VARIANTS}
    if not all(os.path.exists(p) for p in part.values()):
        return {}
    lists = {v: json.loads(open(p).readline()) for v, p in part.items()}
    n = min(len(next(iter(lists[v].values()), [])) for v in rc.VARIANTS)
    used = [k for k, r in read_jsonl(samples_path, lambda r: (r["needle"], r["context_length"], r["depth_percent"]))
            if r["used"]]
    per = {}
    with open(persample_path, "a") as f:
        for i, k in enumerate(used[:n]):
            rec = {v: {h: s[i] for h, s in lists[v].items()} for v in rc.VARIANTS}
            per[k] = rec
            f.write(json.dumps(dict(needle=k[0], context_length=k[1], depth_percent=k[2], **rec)) + "\n")
    log.info(f"RESUME: recovered head scores of {len(per)}/{len(used)} used samples from {part['attn']}; "
             f"the remaining {len(used) - len(per)} used samples will be recomputed")
    return per


def open_append(path):
    """Append mode; if a previous crash left a half-written last line, start on a fresh line."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                with open(path, "a") as g:
                    g.write("\n")
    return open(path, "a", encoding="utf-8")


def read_jsonl(path, key):
    """(key, record) pairs from a jsonl file; a truncated last line (crash mid-write) is ignored."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                out.append((key(r), r))
            except (json.JSONDecodeError, KeyError):
                log.warning(f"{path}: skipping unreadable line {i + 1}")
    return out


if __name__ == "__main__":
    main()