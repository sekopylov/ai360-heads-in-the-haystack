#!/usr/bin/env python
"""
Этап 1: детекция retrieval heads (Wu et al., 2024) для Qwen2.5-0.5B-Instruct.

Retrieval score головы: доля токенов иглы, которые голова "скопировала" -- на шаге
декодирования её top-1 внимание указывает на позицию внутри иглы, и токен на этой
позиции совпадает с только что сгенерированным. Скоры копятся только по пробам,
где модель правильно ответила (ROUGE-1 recall > порога).

Выход:
  head_score/<model>.json      {"layer-head": [score по каждой успешной пробе]}
  head_score/<model>_top.json  отсортированный список (layer-head, mean score)
  results/<model>/detect_samples.jsonl  лог каждой пробы
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from rh_common import (DEFAULT_MODEL, AttnGrabber, build_context, build_prompt_text, eos_ids_of,
                       find_needle_span, greedy_generate, load_detect_needles, load_model, model_tag,
                       period_ids_of, read_haystack_ids, rouge_recall, setup_logger, stage)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--haystack_dir", default="./haystack_for_detect")
    p.add_argument("-s", "--s_len", type=int, default=1000)
    p.add_argument("-e", "--e_len", type=int, default=16000)
    p.add_argument("--num_lengths", type=int, default=8)
    p.add_argument("--num_depths", type=int, default=10)
    p.add_argument("--buffer", type=int, default=200, help="запас токенов под вопрос/ответ")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--score_threshold", type=float, default=50.0, help="ROUGE-1 recall, %%")
    p.add_argument("--no_chat", action="store_true", help="не использовать chat template")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    log = setup_logger("01_detect")
    log.info(f"Аргументы: {vars(a)}")
    tag = model_tag(a.model_path)

    with stage(log, "Загрузка модели"):
        model, tok, info = load_model(a.model_path, a.dtype, a.device, log)
        L, H = info["n_layers"], info["n_heads"]
        grab = AttnGrabber(model)
        eos = eos_ids_of(model, tok)
        period_ids = period_ids_of(tok)
        log.info(f"stop-токены: {sorted(eos)}; токенов-точек: {len(period_ids)}")

    with stage(log, "Подготовка игл и сетки"):
        needles = load_detect_needles(a.haystack_dir, log)
        lengths = np.round(np.linspace(a.s_len, a.e_len, a.num_lengths)).astype(int)
        depths = np.linspace(0, 100, a.num_depths)
        total = len(needles) * len(lengths) * len(depths)
        log.info(f"длины: {lengths.tolist()}")
        log.info(f"глубины: {np.round(depths, 1).tolist()}")
        log.info(f"всего проб: {total}")

    head_counter = defaultdict(list)
    n_ok = n_done = 0
    os.makedirs(f"results/{tag}", exist_ok=True)
    os.makedirs("head_score", exist_ok=True)

    def save():
        with open(f"head_score/{tag}.json", "w", encoding="utf-8") as f:
            json.dump(head_counter, f)
        ranked = sorted(((k, float(np.mean(v))) for k, v in head_counter.items()),
                        key=lambda x: x[1], reverse=True)
        with open(f"head_score/{tag}_top.json", "w", encoding="utf-8") as f:
            json.dump([(k, round(s, 4)) for k, s in ranked], f, indent=1)
        return ranked

    with stage(log, "Детекция retrieval heads"), \
            open(f"results/{tag}/detect_samples.jsonl", "w", encoding="utf-8") as out_f:
        t_start = time.time()
        for ni, nd in enumerate(needles):
            log.info(f"--- игла {ni + 1}/{len(needles)}: {nd['real_needle']!r}")
            hay_ids = read_haystack_ids(tok, nd.get("hay_dir"), int(lengths.max()), log)
            needle_ids = tok(nd["needle"], add_special_tokens=False)["input_ids"]
            for length in lengths:
                for depth in depths:
                    n_done += 1
                    t0 = time.time()
                    ctx = build_context(tok, hay_ids, needle_ids, length, depth, a.buffer, period_ids)
                    text = build_prompt_text(tok, ctx, nd["question"], not a.no_chat)
                    ids, ns, ne = find_needle_span(tok, text, nd["real_needle"])
                    if ids is None or ns < 0:
                        log.warning(f"[skip] len={length} depth={depth:.0f}: игла не найдена в промпте")
                        continue
                    input_ids = torch.tensor([ids], device=a.device)
                    prompt_ids = input_ids[0]
                    score = torch.zeros(L, H, device=a.device)
                    needle_len = ne - ns

                    def on_step(weights, tok_id):
                        for li in range(L):
                            top = weights[li][0, :, -1, :].argmax(dim=-1)  # [H]
                            in_needle = (top >= ns) & (top < ne)
                            copied = prompt_ids[top.clamp(max=len(prompt_ids) - 1)] == tok_id
                            score[li] += (in_needle & copied).float() / needle_len

                    try:
                        gen = greedy_generate(model, tok, input_ids, a.max_new_tokens, eos,
                                              grabber=grab, on_step=on_step)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        log.error(f"[OOM] len={length} depth={depth:.0f} -- пропускаю")
                        continue
                    resp = tok.decode(gen, skip_special_tokens=True).strip()
                    r = rouge_recall(nd["real_needle"], resp)
                    ok = r > a.score_threshold
                    sc = score.cpu()
                    if ok:
                        n_ok += 1
                        for li in range(L):
                            for h in range(H):
                                head_counter[f"{li}-{h}"].append(float(sc[li, h]))
                    out_f.write(json.dumps(dict(needle=ni, context_length=int(length),
                                                depth_percent=float(depth), prompt_tokens=len(ids),
                                                needle_span=[ns, ne], rouge1_recall=r, success=ok,
                                                response=resp, seconds=time.time() - t0),
                                           ensure_ascii=False) + "\n")
                    out_f.flush()
                    eta = (time.time() - t_start) / n_done * (total - n_done)
                    log.info(f"[{n_done}/{total}] len={length} depth={depth:5.1f}% tok={len(ids)} "
                             f"rouge={r:5.1f} {'OK  ' if ok else 'FAIL'} {time.time() - t0:5.1f}s "
                             f"ETA {eta / 60:5.1f}m | {resp[:70]!r}")
                    if ok and n_ok % 10 == 0:
                        ranked = save()
                        log.info(f"промежуточный top-5: {[(k, round(s, 3)) for k, s in ranked[:5]]}")

    with stage(log, "Сохранение скоров"):
        if n_ok == 0:
            log.error("Ни одной успешной пробы -- скоры не посчитаны. Попробуй меньшие длины "
                      "(--e_len) или --score_threshold ниже.")
            raise SystemExit(2)
        ranked = save()
        log.info(f"успешных проб: {n_ok}/{n_done} ({100 * n_ok / max(n_done, 1):.1f}%)")
        log.info(f"сохранено: head_score/{tag}.json, head_score/{tag}_top.json")
        log.info("Top-20 retrieval heads (layer-head: score):")
        for k, s in ranked[:20]:
            log.info(f"   {k:>6}: {s:.4f}")


if __name__ == "__main__":
    main()