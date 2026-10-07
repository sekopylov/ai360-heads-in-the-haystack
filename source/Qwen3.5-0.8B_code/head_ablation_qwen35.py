#!/usr/bin/env python
"""
Блокировка голов Qwen3.5 и замер точности needle-in-a-haystack.

Условия для каждого K из --ks:
  top    -- блокируем K голов с наибольшим retrieval rate (из head_score/<model>.json)
  random -- блокируем K случайных голов (из остальных кандидатов, --n_seeds раз)
  K=0    -- baseline без блокировки

Блокировка головы = обнуление её выхода перед o_proj у full-attention слоя
(forward_pre_hook на self_attn.o_proj, срез [h*head_dim:(h+1)*head_dim]).
Голова не влияет ни на остаток, ни на другие головы. DeltaNet-слои не трогаются.

Метрики: accuracy (доля проб с ROUGE-1 recall > порога) и средний ROUGE-1 recall.

  python -u head_ablation_qwen35.py --model_path Qwen/Qwen3.5-0.8B-Base ^
      --scores head_score/Qwen3.5-0.8B-Base.json --s_len 1000 --e_len 8000 ^
      --num_lengths 4 --num_depths 5 --ks 0 2 4 8 16 --n_seeds 3
"""
import argparse
import csv
import glob
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)


# ----------------------------------------------------------------------------
# блокировка голов
# ----------------------------------------------------------------------------
def get_decoder_layers(model):
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Не нашёл decoder-слои, сделай print(model) и поправь get_decoder_layers")


class HeadBlocker:
    def __init__(self, model, full_layers, num_heads, dtype, device):
        self.num_heads = num_heads
        self.masks, self.handles = {}, []
        layers = get_decoder_layers(model)
        for li in full_layers:
            o_proj = layers[li].self_attn.o_proj
            self.head_dim = o_proj.in_features // num_heads
            self.masks[li] = torch.ones(o_proj.in_features, dtype=dtype, device=device)
            self.handles.append(o_proj.register_forward_pre_hook(self._hook(li)))

    def _hook(self, li):
        def hook(module, args):
            return (args[0] * self.masks[li],) + tuple(args[1:])
        return hook

    def set_blocked(self, heads):
        """heads: список (layer, head); пустой список = ничего не блокируем."""
        for m in self.masks.values():
            m.fill_(1)
        for li, h in heads:
            self.masks[li][h * self.head_dim:(h + 1) * self.head_dim] = 0


# ----------------------------------------------------------------------------
# данные
# ----------------------------------------------------------------------------
def load_haystack_ids(tok, hay_dir, need):
    files = sorted(glob.glob(f"{hay_dir}/*.txt"))
    if not files:
        raise FileNotFoundError(f"В {hay_dir} нет *.txt")
    text = ""
    while len(text.split()) < need:
        for f in files:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                text += fh.read()
    return tok(text, add_special_tokens=False)["input_ids"]


def build_prompt(tok, hay_ids, needle_ids, question, length, depth, buffer, period_ids, chat):
    budget = max(length - buffer - len(needle_ids), 0)
    ctx = hay_ids[:budget]
    if depth >= 100:
        new = ctx + needle_ids
    else:
        ins = int(len(ctx) * depth / 100)
        while ins > 0 and ctx[ins - 1] not in period_ids:
            ins -= 1
        new = ctx[:ins] + needle_ids + ctx[ins:]
    context = tok.decode(new)
    q = f"Based on the content of the book, Question: {question}\nAnswer:"
    if chat:
        msgs = [{"role": "user", "content": f"<book>{context}</book>\n{q}"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    return context + q


# ----------------------------------------------------------------------------
# основной код
# ----------------------------------------------------------------------------
def load_ranking(path, full_layers, num_heads):
    with open(path, encoding="utf-8") as f:
        counter = json.load(f)
    rates = {k: (float(np.mean(v)) if len(v) else 0.0) for k, v in counter.items()}
    # гарантируем, что в ранжировании есть все кандидаты (даже с нулевым скором)
    for li in full_layers:
        for h in range(num_heads):
            rates.setdefault(f"{li}-{h}", 0.0)
    ranked = sorted(rates.items(), key=lambda x: -x[1])
    return [tuple(map(int, k.split("-"))) for k, _ in ranked], rates


@torch.no_grad()
def generate_answer(model, tok, input_ids, max_new, eos_ids):
    out = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                         max_new_tokens=max_new, do_sample=False, temperature=None,
                         top_p=None, top_k=None, eos_token_id=list(eos_ids),
                         pad_token_id=next(iter(eos_ids)))
    text = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    lines = [l for l in text.split("\n") if l.strip()]
    return (lines[0] if lines else "").strip()


def summarize(rows, out_dir, ks):
    # rows: dict с ключами mode, k, seed, rouge, ok
    by = defaultdict(list)
    for r in rows:
        by[(r["mode"], r["k"], r["seed"])].append(r)
    table = []
    for mode in ("top", "bottom", "random"):
        for k in ks:
            if k == 0:
                continue
            seeds = sorted({s for (m, kk, s) in by if m == mode and kk == k})
            accs = [np.mean([x["ok"] for x in by[(mode, k, s)]]) * 100 for s in seeds]
            rouges = [np.mean([x["rouge"] for x in by[(mode, k, s)]]) for s in seeds]
            if accs:
                table.append(dict(mode=mode, k=k, n_seeds=len(seeds), acc=np.mean(accs),
                                  acc_std=np.std(accs), rouge=np.mean(rouges),
                                  rouge_std=np.std(rouges)))
    base = [r for r in rows if r["mode"] == "baseline"]
    base_acc = np.mean([x["ok"] for x in base]) * 100 if base else float("nan")
    base_rouge = np.mean([x["rouge"] for x in base]) if base else float("nan")

    with open(f"{out_dir}/ablation_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "k", "n_seeds", "accuracy_%", "accuracy_std", "rouge1_recall", "rouge1_std"])
        w.writerow(["baseline", 0, 1, f"{base_acc:.2f}", 0, f"{base_rouge:.2f}", 0])
        for t in table:
            w.writerow([t["mode"], t["k"], t["n_seeds"], f"{t['acc']:.2f}", f"{t['acc_std']:.2f}",
                        f"{t['rouge']:.2f}", f"{t['rouge_std']:.2f}"])

    print(f"\n{'mode':<8}{'K':>4}{'acc %':>10}{'±':>7}{'rouge1':>10}{'±':>7}")
    print(f"{'baseline':<8}{0:>4}{base_acc:>10.1f}{'':>7}{base_rouge:>10.1f}")
    for t in table:
        print(f"{t['mode']:<8}{t['k']:>4}{t['acc']:>10.1f}{t['acc_std']:>7.1f}"
              f"{t['rouge']:>10.1f}{t['rouge_std']:>7.1f}")

    print("Сохранено:", f"{out_dir}/ablation_summary.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--scores", required=True, help="head_score/<model>.json")
    ap.add_argument("--haystack_dir", default="./haystack_for_detect")
    ap.add_argument("-s", "--s_len", type=int, default=1000)
    ap.add_argument("-e", "--e_len", type=int, default=8000)
    ap.add_argument("--num_lengths", type=int, default=4)
    ap.add_argument("--num_depths", type=int, default=5)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 2, 4, 8, 16])
    ap.add_argument("--n_seeds", type=int, default=3, help="сколько случайных наборов на каждое K")
    ap.add_argument("--random_from_all", action="store_true",
                    help="случайные головы брать из всех кандидатов (по умолчанию -- вне top-K)")
    ap.add_argument("--buffer", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--score_threshold", type=float, default=50.0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    version = a.model_path.rstrip("/").split("/")[-1]
    out_dir = f"results/{version}"
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model_path)
    cfg = AutoConfig.from_pretrained(a.model_path).get_text_config()
    full_layers = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    num_heads = cfg.num_attention_heads
    dtype = getattr(torch, a.dtype)
    model = AutoModelForCausalLM.from_pretrained(a.model_path, dtype=dtype, attn_implementation="sdpa")
    model = model.to(a.device).eval()
    blocker = HeadBlocker(model, full_layers, num_heads, dtype, a.device)

    eos = set()
    ge = getattr(model.generation_config, "eos_token_id", None)
    for e in ([ge] if isinstance(ge, int) else (ge or [])):
        eos.add(int(e))
    if tok.eos_token_id is not None:
        eos.add(int(tok.eos_token_id))
    period_ids = {i for t, i in tok.get_vocab().items() if t.rstrip("ĊĠ").endswith(".")}

    ranked, rates = load_ranking(a.scores, full_layers, num_heads)
    candidates = [h for h in ranked if h[0] in full_layers]
    print(f"Кандидатов: {len(candidates)}; top-5: "
          f"{[(f'{l}-{h}', round(rates[f'{l}-{h}'], 3)) for l, h in candidates[:5]]}")

    # проверка, что блокировка реально меняет выход
    ids = tok("The quick brown fox jumps over the lazy dog. " * 20, return_tensors="pt",
              add_special_tokens=False).input_ids.to(a.device)
    with torch.no_grad():
        blocker.set_blocked([])
        l0 = model(input_ids=ids, logits_to_keep=1).logits.float()
        blocker.set_blocked(candidates)
        l1 = model(input_ids=ids, logits_to_keep=1).logits.float()
        blocker.set_blocked([])
    diff = (l0 - l1).abs().max().item()
    print(f"Проверка блокировки: max|d_logits| при блокировке всех голов = {diff:.3f}")
    assert diff > 0, "Блокировка не влияет на выход -- хук не сработал"

    needles = [json.loads(l) for l in open(f"{a.haystack_dir}/needles.jsonl", encoding="utf-8")]
    lengths = np.round(np.linspace(a.s_len, a.e_len, a.num_lengths)).astype(int)
    depths = np.linspace(0, 100, a.num_depths)
    ks = sorted(set(a.ks) | {0})

    # наборы голов для каждого условия (фиксированы на весь прогон)
    conditions = [("baseline", 0, 0, [])]
    for k in ks:
        if k == 0:
            continue
        top = candidates[:k]
        conditions.append(("top", k, 0, top))
        # "обычные" головы: K голов с наименьшим retrieval rate (контроль)
        conditions.append(("bottom", k, 0, candidates[-k:]))
        pool = candidates if a.random_from_all else [h for h in candidates if h not in top]
        for s in range(a.n_seeds):
            rng = random.Random(a.seed * 1000 + k * 10 + s)
            conditions.append(("random", k, s, rng.sample(pool, min(k, len(pool)))))
    print(f"Условий: {len(conditions)}, проб на условие: {len(needles) * len(lengths) * len(depths)}")

    rows = []
    with open(f"{out_dir}/ablation_samples.jsonl", "w", encoding="utf-8") as out_f:
        for ni, nd in enumerate(needles):
            hay = load_haystack_ids(tok, f"{a.haystack_dir}/part{ni + 1}", int(lengths.max()))
            needle_ids = tok(nd["needle"], add_special_tokens=False)["input_ids"]
            for L in lengths:
                for d in depths:
                    text = build_prompt(tok, hay, needle_ids, nd["question"], L, d,
                                        a.buffer, period_ids, a.chat)
                    input_ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
                    input_ids = input_ids.to(a.device)
                    t0 = time.time()
                    msg = []
                    for mode, k, seed, heads in conditions:
                        blocker.set_blocked(heads)
                        try:
                            resp = generate_answer(model, tok, input_ids, a.max_new_tokens, eos)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            print(f"[OOM] len={L}")
                            continue
                        rouge = scorer.score(nd["real_needle"], resp)["rouge1"].recall * 100
                        row = dict(needle=ni, length=int(L), depth=float(d), mode=mode, k=k,
                                   seed=seed, rouge=rouge, ok=int(rouge > a.score_threshold),
                                   response=resp)
                        rows.append(row)
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        if mode in ("baseline", "top"):
                            msg.append(f"{mode}{k}:{rouge:.0f}")
                    out_f.flush()
                    blocker.set_blocked([])
                    print(f"needle={ni} len={L} depth={d:.0f}% tokens={input_ids.shape[1]} "
                          f"{time.time() - t0:.0f}s | {' '.join(msg)}")

    blocker.set_blocked([])
    summarize(rows, out_dir, ks)
    try:
        from plot_ablation import make_plots
        make_plots(out_dir)
    except Exception as e:  # графики можно построить отдельно: python plot_ablation.py --dir ...
        print('Графики не построены:', repr(e))


if __name__ == "__main__":
    main()