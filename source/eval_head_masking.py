#!/usr/bin/env python
"""
Этап 3: Needle-in-a-Haystack с выкинутыми (замаскированными) головами.

Условия:
  baseline   -- без маскирования
  top        -- K голов с наибольшим retrieval score
  bottom     -- K голов с наименьшим score (ничьи среди нулей -- случайно, фиксированный сид)
  random_sN  -- K случайных голов, сид N (по умолчанию из пула без top-K, как в статье;
                --random_pool all -- из всех голов)

Маскирование: выход головы обнуляется перед o_proj (см. HeadMasker в rh_common.py).
Точность пробы = ROUGE-1 recall ответа относительно иглы (0..100), как в оригинале;
дополнительно считается accuracy = доля проб с recall > порога.

Выход: results/masking/<model>/<condition>.jsonl и summary.json
"""
import argparse
import json
import os
import random
import time

import numpy as np

from rh_common import (DEFAULT_MODEL, SF_NEEDLE, build_context, build_prompt_text, load_detect_needles,
                       load_head_scores, model_tag, period_ids_of, read_haystack_ids, rouge_recall,
                       setup_logger, stage)


def select_heads(scores, k, seed=0, random_seeds=(0,), random_pool="non_top"):
    """Возвращает {condition: [(layer, head), ...]} -- без torch, чтобы легко тестировать."""
    heads = list(scores)
    rng = random.Random(seed)
    tie = {h: rng.random() for h in heads}
    desc = sorted(heads, key=lambda x: (-scores[x], tie[x]))
    asc = sorted(heads, key=lambda x: (scores[x], tie[x]))
    top = desc[:k]
    conds = {"baseline": [], "top": top, "bottom": asc[:k]}
    pool = [h for h in heads if h not in set(top)] if random_pool == "non_top" else heads
    for s in random_seeds:
        conds[f"random_s{s}"] = sorted(random.Random(1000 + s).sample(pool, k))
    return conds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--scores", default=None, help="по умолчанию head_score/<model>.json")
    p.add_argument("--k", type=int, default=20, help="сколько голов выкидывать")
    p.add_argument("--random_seeds", type=int, default=3, help="число случайных наборов")
    p.add_argument("--random_pool", default="non_top", choices=["non_top", "all"])
    p.add_argument("--conditions", default="baseline,top,bottom,random")
    p.add_argument("--eval_needles", default="sf", choices=["sf", "detect"],
                   help="sf: игла про San Francisco в PaulGrahamEssays (как в статье); "
                        "detect: иглы из haystack_for_detect")
    p.add_argument("--eval_haystack_dir", default="./PaulGrahamEssays")
    p.add_argument("--haystack_dir", default="./haystack_for_detect")
    p.add_argument("-s", "--s_len", type=int, default=1000)
    p.add_argument("-e", "--e_len", type=int, default=16000)
    p.add_argument("--num_lengths", type=int, default=8)
    p.add_argument("--num_depths", type=int, default=10)
    p.add_argument("--buffer", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--score_threshold", type=float, default=50.0)
    p.add_argument("--no_chat", action="store_true")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    a = p.parse_args()

    import torch
    from rh_common import HeadMasker, eos_ids_of, find_needle_span, greedy_generate, load_model

    a.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log = setup_logger("03_mask")
    log.info(f"Аргументы: {vars(a)}")
    tag = model_tag(a.model_path)
    out_dir = f"results/masking/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    with stage(log, "Выбор голов для маскирования"):
        path = a.scores or f"head_score/{tag}.json"
        scores = load_head_scores(path)
        n_zero = sum(1 for s in scores.values() if s == 0)
        log.info(f"{path}: {len(scores)} голов, из них с нулевым score: {n_zero}")
        all_conds = select_heads(scores, a.k, random_seeds=range(a.random_seeds), random_pool=a.random_pool)
        wanted = set(a.conditions.split(","))
        conds = {c: hs for c, hs in all_conds.items()
                 if c in wanted or (c.startswith("random_s") and "random" in wanted)}
        for c, hs in conds.items():
            ms = np.mean([scores[h] for h in hs]) if hs else 0.0
            log.info(f"  {c:<10} {len(hs):3d} голов, средний score={ms:.3f}: "
                     f"{[f'{l}-{h}' for l, h in hs]}")
        if a.random_pool == "non_top":
            log.info(f"  случайные головы выбираются из {len(scores) - a.k} голов вне top-{a.k}")

    with stage(log, "Загрузка модели"):
        model, tok, info = load_model(a.model_path, a.dtype, a.device, log)
        masker = HeadMasker(model, info["n_heads"], info["head_dim"])
        eos = eos_ids_of(model, tok)
        period_ids = period_ids_of(tok)

    with stage(log, "Подготовка промптов (общих для всех условий)"):
        lengths = np.round(np.linspace(a.s_len, a.e_len, a.num_lengths)).astype(int)
        depths = np.round(np.linspace(0, 100, a.num_depths)).astype(int)
        if a.eval_needles == "sf":
            hay_dir = a.eval_haystack_dir if os.path.isdir(a.eval_haystack_dir) else None
            needles = [dict(SF_NEEDLE, hay_dir=hay_dir)]
        else:
            needles = load_detect_needles(a.haystack_dir, log)
        samples = []
        for ni, nd in enumerate(needles):
            hay_ids = read_haystack_ids(tok, nd.get("hay_dir"), int(lengths.max()), log)
            needle_ids = tok(nd["needle"], add_special_tokens=False)["input_ids"]
            for length in lengths:
                for depth in depths:
                    ctx = build_context(tok, hay_ids, needle_ids, length, depth, a.buffer, period_ids)
                    text = build_prompt_text(tok, ctx, nd["question"], not a.no_chat)
                    ids, ns, _ = find_needle_span(tok, text, nd["real_needle"])
                    if ids is None or ns < 0:
                        log.warning(f"[skip] needle={ni} len={length} depth={depth:.0f}: игла не найдена")
                        continue
                    samples.append(dict(needle=ni, length=int(length), depth=float(depth),
                                        ids=ids, real_needle=nd["real_needle"]))
        log.info(f"проб на условие: {len(samples)}; условий: {len(conds)}; "
                 f"всего прогонов: {len(samples) * len(conds)}")

    summary = {}
    for ci, (cond, heads) in enumerate(conds.items()):
        with stage(log, f"Условие {ci + 1}/{len(conds)}: {cond} (выкинуто {len(heads)} голов)"):
            masker.set_heads(heads)
            recs, t_start = [], time.time()
            with open(f"{out_dir}/{cond}.jsonl", "w", encoding="utf-8") as f:
                for si, smp in enumerate(samples):
                    t0 = time.time()
                    input_ids = torch.tensor([smp["ids"]], device=a.device)
                    try:
                        gen = greedy_generate(model, tok, input_ids, a.max_new_tokens, eos)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        log.error(f"[OOM] len={smp['length']} -- пропускаю")
                        continue
                    resp = tok.decode(gen, skip_special_tokens=True).strip()
                    r = rouge_recall(smp["real_needle"], resp)
                    rec = dict(condition=cond, needle=smp["needle"], context_length=smp["length"],
                               depth_percent=smp["depth"], score=r, response=resp,
                               seconds=time.time() - t0)
                    recs.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    if (si + 1) % 10 == 0 or si + 1 == len(samples):
                        eta = (time.time() - t_start) / (si + 1) * (len(samples) - si - 1)
                        log.info(f"  [{cond}] {si + 1}/{len(samples)} средний score="
                                 f"{np.mean([x['score'] for x in recs]):.1f} ETA {eta / 60:.1f}m "
                                 f"| последний: len={smp['length']} depth={smp['depth']:.0f}% "
                                 f"score={r:.0f} {resp[:60]!r}")
            sc = np.array([x["score"] for x in recs]) if recs else np.array([np.nan])
            summary[cond] = dict(heads=[f"{l}-{h}" for l, h in heads], n=len(recs),
                                 mean_score=float(np.mean(sc)),
                                 accuracy=float(np.mean(sc > a.score_threshold) * 100))
            log.info(f"  ИТОГ {cond}: средний ROUGE-1 recall={summary[cond]['mean_score']:.1f}, "
                     f"accuracy={summary[cond]['accuracy']:.1f}%")
    masker.set_heads([])

    with stage(log, "Сводка"):
        rnd = [v for k, v in summary.items() if k.startswith("random_s")]
        if rnd:
            summary["random"] = dict(
                n_seeds=len(rnd),
                mean_score=float(np.mean([v["mean_score"] for v in rnd])),
                std_score=float(np.std([v["mean_score"] for v in rnd])),
                accuracy=float(np.mean([v["accuracy"] for v in rnd])),
                std_accuracy=float(np.std([v["accuracy"] for v in rnd])))
        meta = dict(model=a.model_path, k=a.k, random_pool=a.random_pool, eval_needles=a.eval_needles,
                    lengths=lengths.tolist(), depths=depths.tolist(), threshold=a.score_threshold)
        with open(f"{out_dir}/summary.json", "w", encoding="utf-8") as f:
            json.dump(dict(meta=meta, conditions=summary), f, ensure_ascii=False, indent=1)
        log.info(f"{'условие':<12}{'ROUGE-1 recall':>16}{'accuracy, %':>14}")
        for c in ("baseline", "top", "random", "bottom"):
            if c in summary:
                v = summary[c]
                extra = f" ± {v['std_score']:.1f}" if "std_score" in v else ""
                log.info(f"{c:<12}{v['mean_score']:>10.1f}{extra:<6}{v['accuracy']:>14.1f}")
        log.info(f"сохранено: {out_dir}/summary.json")


if __name__ == "__main__":
    main()