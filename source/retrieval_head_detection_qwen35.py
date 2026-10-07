#!/usr/bin/env python
"""
Retrieval-head detection (Wu et al., "Retrieval Head Mechanistically Explains
Long-Context Factuality") для Qwen3.5-0.8B и других Qwen3.5.

Отличия от оригинального retrieval_head_detection.py:
  * Не нужны кастомные faiss_attn/source/modeling_*.py -- используется родной
    transformers (>=5.x), где есть Qwen3_5ForCausalLM.
  * Qwen3.5 -- гибрид: 18 слоёв Gated DeltaNet (линейный attention, весов
    внимания нет) + 6 слоёв обычного gated full attention (слои 3,7,11,15,19,23).
    Retrieval heads ищутся только среди этих 6 слоёв x 8 голов = 48 голов.
  * Prefill идёт через sdpa (быстро, без матрицы n x n), а decode-шаги --
    через eager, чтобы получить веса внимания. Веса забираются forward-хуками.
  * logits_to_keep=1 на prefill: словарь 248k, логиты для всех позиций
    заняли бы десятки ГБ.
  * Правильные stop-токены / точки вставки иглы для токенизатора Qwen
    (в оригинале были захардкожены id от Llama/Mistral).
  * Формат выхода тот же: head_score/<model>.json, ключи "layer-head" -> список
    скоров по успешным пробам (layer -- реальный индекс слоя 0..23).

Пример:
  python -u retrieval_head_detection_qwen35.py \
      --model_path Qwen/Qwen3.5-0.8B-Base --s_len 1000 --e_len 16000 \
      --num_lengths 8 --num_depths 10 2>&1 | tee logs/qwen35_0.8b.log
"""
import argparse
import glob
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def get_decoder_layers(model):
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Не нашёл список decoder-слоёв, сделай print(model) и поправь get_decoder_layers")


class AttnGrabber:
    """Forward-хуки на self_attn full-attention слоёв. Сохраняют attn_weights (eager)."""

    def __init__(self, model, full_layer_ids):
        self.enabled = False
        self.weights = {}
        self.handles = []
        layers = get_decoder_layers(model)
        for li in full_layer_ids:
            layer = layers[li]
            if not hasattr(layer, "self_attn"):
                raise RuntimeError(
                    f"Слой {li} по config.layer_types full_attention, но у него нет .self_attn: "
                    f"{[n for n, _ in layer.named_children()]}"
                )
            self.handles.append(layer.self_attn.register_forward_hook(self._make_hook(li)))

    def _make_hook(self, li):
        def hook(module, args, output):
            if self.enabled and isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self.weights[li] = output[1].detach()

        return hook


def update_retrieval_scores(weights, full_layer_ids, prompt_ids, needle_start, needle_end, new_token, score):
    """
    weights: {layer_idx: [1, H, 1, kv_len]}; score: [n_full_layers, H] (на том же девайсе).
    Голова получает 1/needle_len, если её top-1 внимание попало в иглу И токен на этой
    позиции == только что сгенерированному токену (copy-paste, как в оригинале).
    """
    needle_len = needle_end - needle_start
    n_prompt = prompt_ids.shape[0]
    for row, li in enumerate(full_layer_ids):
        w = weights[li][0, :, -1, :]  # [H, kv_len]
        top = w.argmax(dim=-1)  # [H]
        in_needle = (top >= needle_start) & (top < needle_end)
        copied = prompt_ids[top.clamp(max=n_prompt - 1)] == new_token
        score[row] += (in_needle & copied).float() / needle_len


def find_needle_span(tokenizer, text, real_needle):
    """Точный поиск токенов иглы через offset_mapping. Возвращает (start, end) или (-1, -1)."""
    cs = text.find(real_needle)
    if cs < 0:
        return None, -1, -1
    ce = cs + len(real_needle)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    hit = [i for i, (a, b) in enumerate(offs) if b > cs and a < ce]
    if not hit:
        return ids, -1, -1
    return ids, hit[0], hit[-1] + 1


# --------------------------------------------------------------------------- #
# main class
# --------------------------------------------------------------------------- #
class RetrievalHeadDetector:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.model_version = args.model_path.rstrip("/").split("/")[-1]
        if args.model_name_suffix:
            self.model_version += "_" + args.model_name_suffix

        self.tok = AutoTokenizer.from_pretrained(args.model_path)
        cfg = AutoConfig.from_pretrained(args.model_path).get_text_config()
        self.layer_types = list(cfg.layer_types)
        self.full_layers = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
        self.head_num = cfg.num_attention_heads
        print(f"layers: {len(self.layer_types)}, full-attention layers: {self.full_layers}, "
              f"heads per layer: {self.head_num}")

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=getattr(torch, args.dtype), attn_implementation="sdpa"
        ).to(self.device).eval()
        self._impl = "sdpa"
        self.grab = AttnGrabber(self.model, self.full_layers)

        # stop tokens
        eos = set()
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        for e in ([gen_eos] if isinstance(gen_eos, int) else (gen_eos or [])):
            eos.add(int(e))
        if self.tok.eos_token_id is not None:
            eos.add(int(self.tok.eos_token_id))
        self.eos_ids = eos

        # токены, заканчивающиеся на '.', -- точки вставки иглы (byte-level BPE: 'Ċ' = \n, 'Ġ' = пробел)
        self.period_ids = {i for t, i in self.tok.get_vocab().items() if t.rstrip("ĊĠ").endswith(".")}

        self.head_counter = defaultdict(list)
        self.n_success = 0

    # ----- attention implementation switch -----
    def set_impl(self, name):
        if self._impl == name:
            return
        try:
            self.model.set_attn_implementation(name)
        except Exception:
            for m in self.model.modules():
                c = getattr(m, "config", None)
                if c is not None and hasattr(c, "_attn_implementation"):
                    c._attn_implementation = name
        self._impl = name

    # ----- data -----
    def load_haystack_ids(self, hay_dir, need_tokens):
        files = sorted(glob.glob(f"{hay_dir}/*.txt"))
        if not files:
            raise FileNotFoundError(f"В {hay_dir} нет *.txt")
        text = ""
        while len(text.split()) < need_tokens:  # слов >= токенов нужно, т.к. токенов обычно >= слов
            for f in files:
                with open(f, "r") as fh:
                    text += fh.read()
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def build_context(self, hay_ids, needle_ids, length, depth):
        budget = max(length - self.args.buffer - len(needle_ids), 0)
        ctx = hay_ids[:budget]
        if depth >= 100:
            new = ctx + needle_ids
        else:
            ins = int(len(ctx) * depth / 100)
            while ins > 0 and ctx[ins - 1] not in self.period_ids:
                ins -= 1
            new = ctx[:ins] + needle_ids + ctx[ins:]
        return self.tok.decode(new)

    def build_prompt_text(self, context, question):
        q = f"Based on the content of the book, Question: {question}\nAnswer:"
        if self.args.chat:
            msgs = [{"role": "user", "content": f"<book>{context}</book>\n{q}"}]
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        return context + q

    # ----- one sample -----
    @torch.no_grad()
    def run_sample(self, input_ids, prompt_ids, ns, ne, max_new):
        score = torch.zeros(len(self.full_layers), self.head_num, device=self.device)

        # prefill: sdpa, без весов внимания и без логитов по всем позициям
        self.set_impl("sdpa")
        self.grab.enabled = False
        out = self.model(input_ids=input_ids[:, :-1], use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        del out

        # decode: eager, хуки забирают веса
        self.set_impl("eager")
        self.grab.enabled = True
        inp = input_ids[:, -1:]
        generated = []
        for _ in range(max_new):
            self.grab.weights.clear()
            o = self.model(input_ids=inp, past_key_values=past, use_cache=True, logits_to_keep=1)
            past = o.past_key_values
            nxt = o.logits[0, -1].argmax()
            tid = int(nxt.item())
            if tid in self.eos_ids:
                break
            generated.append(tid)
            if not self.grab.weights:
                raise RuntimeError("Хуки не получили attn_weights: eager-режим не включился. "
                                   "Запусти sanity_check_qwen35.py")
            update_retrieval_scores(self.grab.weights, self.full_layers, prompt_ids, ns, ne, nxt, score)
            piece = self.tok.decode([tid])
            if "\n" in piece and self.tok.decode(generated).strip():
                break
            inp = nxt.view(1, 1)
        self.grab.enabled = False
        self.grab.weights.clear()
        return generated, score.cpu()

    def evaluate(self, needle_idx, hay_ids, needle_ids, nd, length, depth, out_f):
        a = self.args
        t0 = time.time()
        context = self.build_context(hay_ids, needle_ids, length, depth)
        text = self.build_prompt_text(context, nd["question"])
        ids, ns, ne = find_needle_span(self.tok, text, nd["real_needle"])
        if ids is None or ns < 0:
            print(f"[skip] needle {needle_idx} len={length} depth={depth}: не нашёл иглу в промпте")
            return
        input_ids = torch.tensor([ids], device=self.device)
        prompt_ids = input_ids[0]
        try:
            gen, score = self.run_sample(input_ids, prompt_ids, ns, ne, a.max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[OOM] needle {needle_idx} len={length} depth={depth}")
            return
        response = self.tok.decode(gen, skip_special_tokens=True).strip()
        rouge = scorer.score(nd["real_needle"], response)["rouge1"].recall * 100
        ok = rouge > a.score_threshold
        if ok:
            self.n_success += 1
            for row, li in enumerate(self.full_layers):
                for h in range(self.head_num):
                    self.head_counter[f"{li}-{h}"].append(float(score[row, h]))
        rec = dict(needle=needle_idx, context_length=int(length), depth_percent=float(depth),
                   prompt_tokens=len(ids), needle_span=[ns, ne], rouge1_recall=rouge,
                   success=ok, response=response, seconds=time.time() - t0)
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        print(f"needle={needle_idx} len={length} depth={depth:.0f}% tokens={len(ids)} "
              f"rouge={rouge:.0f} {'OK ' if ok else 'FAIL'} {time.time() - t0:.1f}s | {response[:80]!r}")
        if ok and self.n_success % 10 == 0:
            self.save_scores()
            print("top-10:", self.top_heads(10))

    # ----- output -----
    def top_heads(self, k=20):
        ranked = sorted(((h, float(np.mean(v))) for h, v in self.head_counter.items() if v),
                        key=lambda x: x[1], reverse=True)
        return [(h, round(s, 4)) for h, s in ranked[:k]]

    def save_scores(self):
        os.makedirs("head_score", exist_ok=True)
        with open(f"head_score/{self.model_version}.json", "w") as f:
            json.dump(self.head_counter, f)
        with open(f"head_score/{self.model_version}_top.json", "w") as f:
            json.dump(self.top_heads(len(self.head_counter)), f, indent=1)

    def run(self):
        a = self.args
        needles = [json.loads(l) for l in open(f"{a.haystack_dir}/needles.jsonl")]
        lengths = np.round(np.linspace(a.s_len, a.e_len, a.num_lengths)).astype(int)
        depths = np.linspace(0, 100, a.num_depths)
        print(f"needles={len(needles)} lengths={lengths.tolist()} depths={depths.tolist()}")
        os.makedirs(f"results/{self.model_version}", exist_ok=True)
        with open(f"results/{self.model_version}/samples.jsonl", "w") as out_f:
            for ni, nd in enumerate(needles):
                hay_ids = self.load_haystack_ids(f"{a.haystack_dir}/part{ni + 1}", int(lengths.max()))
                needle_ids = self.tok(nd["needle"], add_special_tokens=False)["input_ids"]
                for L in lengths:
                    for d in depths:
                        self.evaluate(ni, hay_ids, needle_ids, nd, L, d, out_f)
        self.save_scores()
        print(f"\nУспешных проб: {self.n_success}")
        print("Top-20 retrieval heads (layer-head, score):")
        for h, s in self.top_heads(20):
            print(f"  {h:>6}  {s:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_name_suffix", default=None)
    p.add_argument("--haystack_dir", default="./haystack_for_detect")
    p.add_argument("-s", "--s_len", type=int, default=1000)
    p.add_argument("-e", "--e_len", type=int, default=16000)
    p.add_argument("--num_lengths", type=int, default=8)
    p.add_argument("--num_depths", type=int, default=10)
    p.add_argument("--buffer", type=int, default=200, help="запас токенов под вопрос/ответ")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--score_threshold", type=float, default=50.0, help="rouge1 recall, %%")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--chat", action="store_true",
                   help="использовать chat template (для post-trained Qwen3.5-0.8B, не Base)")
    RetrievalHeadDetector(p.parse_args()).run()


if __name__ == "__main__":
    main()