"""
Needle-in-a-Haystack с маскированием голов, адаптировано под Qwen3.5-0.8B.

    --mask_topk K > 0  : замаскировать top-K retrieval heads (из head_score/<model>.json)
    --mask_topk K < 0  : замаскировать |K| случайных голов (не из top-|K|)
    --mask_topk 0      : без маскирования

У Qwen3.5-0.8B всего 48 голов с полным вниманием (6 слоёв x 8), поэтому разумные K: 5–20.

Пример:
    python needle_in_haystack_with_mask_qwen35.py --model_path ./models/Qwen3.5-0.8B --mask_topk 10 --s_len 1000 --e_len 100000
Результаты: ./results/graph/<model>_block_top10/ (или _block_random10, или <model>/)
"""
import argparse
import json
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
from rouge_score import rouge_scorer

import qwen35_common as qc

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


class MaskedNeedleTester:
    def __init__(self, model_path, mask_topk=0,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 real_needle="eat a sandwich and sit in Dolores Park on a sunny day",
                 haystack_dir="PaulGrahamEssays",
                 retrieval_question="What is the best thing to do in San Francisco?",
                 context_lengths_min=1000, context_lengths_max=128000, context_lengths_num_intervals=40,
                 depth_intervals=10, use_chat="auto", model_name_suffix=None, head_score_name=None, seed=42):
        self.needle, self.real_needle = needle, real_needle
        self.haystack_dir, self.retrieval_question = haystack_dir, retrieval_question
        self.mask_topk = mask_topk
        self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max,
                                                    num=context_lengths_num_intervals, endpoint=True)).astype(int)
        self.depths = np.round(np.linspace(0, 100, num=depth_intervals, endpoint=True)).astype(int)
        self.model_path = model_path
        base_name = os.path.basename(os.path.normpath(model_path))
        self.model_version = base_name + (f"_{model_name_suffix}" if model_name_suffix else "")
        random.seed(seed)

        self.model, self.enc, self.full_layers, self.head_num = qc.load_model_and_tokenizer(model_path)
        self.use_chat = ("Base" not in base_name) if use_chat == "auto" else (use_chat == "on")
        print(f"chat template: {self.use_chat}")
        self.period_ids = qc.period_token_ids(self.enc)

        self.all_heads = [(l, h) for l in self.full_layers for h in range(self.head_num)]
        if mask_topk != 0:
            name = head_score_name or base_name
            with open(f"head_score/{name}.json") as f:
                scores = json.loads(f.readline())
            ranked = sorted(((k, np.mean(v)) for k, v in scores.items()), key=lambda x: x[1], reverse=True)
            self.ranked_heads = [tuple(int(x) for x in k.split("-")) for k, _ in ranked]
            k = abs(mask_topk)
            if k > len(self.all_heads):
                raise ValueError(f"mask_topk={mask_topk}, но у модели только {len(self.all_heads)} голов с полным вниманием")
            if mask_topk > 0:
                print(f"masking out top {k} retrieval heads: {self.ranked_heads[:k]}")
            else:
                if len(self.all_heads) - k < k:
                    raise ValueError(f"Нельзя выбрать {k} случайных голов вне top-{k}: всего {len(self.all_heads)} голов")
                print(f"masking out {k} random non-top heads (new sample per test)")

        self.context_tokens = None

    def block_dict(self):
        if self.mask_topk > 0:
            heads = self.ranked_heads[: self.mask_topk]
        elif self.mask_topk < 0:
            k = -self.mask_topk
            pool = [x for x in self.all_heads if x not in set(self.ranked_heads[:k])]
            heads = random.sample(pool, k)
        else:
            return {}
        d = {}
        for l, h in heads:
            d.setdefault(l, []).append(h)
        return d

    @property
    def save_name(self):
        if self.mask_topk > 0:
            return f"{self.model_version}_block_top{self.mask_topk}"
        if self.mask_topk < 0:
            return f"{self.model_version}_block_random{-self.mask_topk}"
        return self.model_version

    def generate_context(self, context_length, depth_percent):
        if self.context_tokens is None:
            text = qc.read_haystack(self.haystack_dir, self.enc, max(self.context_lengths))
            self.context_tokens = self.enc.encode(text)
        context = self.enc.decode(self.context_tokens[:context_length])
        return qc.insert_needle(self.enc, context, self.needle, depth_percent, context_length, self.period_ids)

    def evaluate_and_log(self, context_length, depth_percent):
        context = self.generate_context(context_length, depth_percent)
        input_ids = qc.build_input_ids(self.enc, context, self.retrieval_question, self.use_chat)
        block = self.block_dict()

        t0 = time.time()
        qc.STATE.reset()
        qc.STATE.block = block
        response = qc.prefill_and_decode(self.model, self.enc, input_ids, 50)
        qc.STATE.reset()
        elapsed = time.time() - t0

        score = scorer.score(self.real_needle, response)["rouge1"].recall * 100
        print("-- Test Summary --")
        print(f"Duration: {elapsed:.1f} s | Context: {context_length} | Depth: {depth_percent}% | Score: {score:.1f}")
        print(f"Response: {response}\n")

        results = {
            "model": self.model_path, "context_length": int(context_length), "depth_percent": float(depth_percent),
            "version": 1, "needle": self.needle, "model_response": response, "score": score,
            "masked_heads": {str(k): v for k, v in block.items()},
            "test_duration_seconds": elapsed,
            "test_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
        }
        name = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent * 100)}'
        os.makedirs(f"results/graph/{self.save_name}", exist_ok=True)
        p = f"results/graph/{self.save_name}/{name}_results.json"
        with open(p, "w") as f:
            json.dump(results, f)
        print("Writing at %s" % p)

    def start_test(self, s_len, e_len):
        print(f"Needle: {self.needle.strip()}")
        for L in self.context_lengths:
            if L < s_len or L > e_len:
                continue
            for d in self.depths:
                self.evaluate_and_log(int(L), d)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--s_len", type=int, default=1000)
    p.add_argument("-e", "--e_len", type=int, default=100000)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--model_name_suffix", type=str, default=None)
    p.add_argument("--mask_topk", type=int, default=0,
                   help="K>0: top-K retrieval heads, K<0: |K| случайных голов, 0: без маски")
    p.add_argument("--head_score_name", type=str, default=None,
                   help="имя файла в head_score/ (по умолчанию = имя папки модели)")
    p.add_argument("--haystack_dir", type=str, default="PaulGrahamEssays")
    p.add_argument("--num_lengths", type=int, default=40)
    p.add_argument("--chat", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    # Пути к данным/результатам — относительно папки скрипта (source/), модель — относительно места запуска
    a.model_path = os.path.abspath(a.model_path) if os.path.exists(a.model_path) else a.model_path
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    t =MaskedNeedleTester(a.model_path, mask_topk=a.mask_topk, haystack_dir=a.haystack_dir,
                           context_lengths_min=max(a.s_len, 1000), context_lengths_max=a.e_len,
                           context_lengths_num_intervals=a.num_lengths, use_chat=a.chat,
                           model_name_suffix=a.model_name_suffix, head_score_name=a.head_score_name, seed=a.seed)
    t.start_test(a.s_len, a.e_len)