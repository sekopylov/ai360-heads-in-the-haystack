"""
Retrieval head detection (Wu et al., 2024), адаптировано под Qwen3.5-0.8B.

Пример:
    python retrieval_head_detection_qwen35.py --model_path ./models/Qwen3.5-0.8B --s_len 0 --e_len 50000

Результат: ./head_score/<model_name>.json в том же формате, что и в оригинале:
    {"layer-head": [scores...]}, где layer — глобальный индекс слоя (3, 7, 11, ...).
"""
import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from rouge_score import rouge_scorer

import qwen35_common as qc

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


class RetrievalHeadDetector:
    def __init__(self, model_path, haystack_dir="./haystack_for_detect",
                 context_lengths_min=1000, context_lengths_max=50000, context_lengths_num_intervals=20,
                 depth_intervals=10, use_chat="auto", model_name_suffix=None, save_contexts=False):
        with open(f"{haystack_dir}/needles.jsonl") as f:
            needles = [json.loads(l) for l in f if l.strip()]
        self.needle_list = [l["needle"] for l in needles]
        self.question_list = [l["question"] for l in needles]
        self.real_needle_list = [l["real_needle"] for l in needles]
        self.haystack_dir_list = [f"{haystack_dir}/part{i}" for i in range(1, len(needles) + 1)]

        self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max,
                                                    num=context_lengths_num_intervals, endpoint=True)).astype(int)
        self.depths = np.round(np.linspace(0, 100, num=depth_intervals, endpoint=True)).astype(int)

        self.model_version = os.path.basename(os.path.normpath(model_path))
        if model_name_suffix:
            self.model_version += "_" + model_name_suffix
        self.model_path = model_path
        self.save_contexts = save_contexts

        self.model, self.enc, self.full_layers, self.head_num = qc.load_model_and_tokenizer(model_path)
        self.use_chat = ("Base" not in self.model_version) if use_chat == "auto" else (use_chat == "on")
        print(f"chat template: {self.use_chat}")
        self.period_ids = qc.period_token_ids(self.enc)
        self.head_counter = defaultdict(list)
        self._haystack_cache = {}

    # ---------- retrieval score ----------
    def retrieval_calculate(self, retrieval_score, tok):
        tok_id = tok.item()
        span = self.needle_end - self.needle_start
        for layer in self.full_layers:
            top1 = qc.STATE.top1.get(layer)
            if top1 is None:
                continue
            for h in range(self.head_num):
                i = top1[h].item()
                if self.needle_start <= i < self.needle_end and tok_id == self.prompt_ids[i].item():
                    retrieval_score[layer][h] += 1 / span

    def retrieval_head_accumulate(self, retrieval_score):
        for layer in self.full_layers:
            for h in range(self.head_num):
                self.head_counter[f"{layer}-{h}"].append(retrieval_score[layer][h])

    # ---------- один прогон ----------
    def generate_context(self, context_length, depth_percent):
        key = self.haystack_dir
        if key not in self._haystack_cache:
            self._haystack_cache[key] = qc.read_haystack(self.haystack_dir, self.enc, max(self.context_lengths))
        context = self._haystack_cache[key]
        tokens = self.enc.encode(context)
        if len(tokens) > context_length:
            context = self.enc.decode(tokens[:context_length])
        return qc.insert_needle(self.enc, context, self.needle, depth_percent, context_length, self.period_ids)

    def evaluate_and_log(self, context_length, depth_percent):
        context = self.generate_context(context_length, depth_percent)
        input_ids = qc.build_input_ids(self.enc, context, self.retrieval_question, self.use_chat)
        self.prompt_ids = input_ids[0]
        self.needle_start, self.needle_end = qc.find_needle_idx(self.enc, self.prompt_ids, self.real_needle)
        if self.needle_start < 0:
            print(f"!! needle not found in prompt (len={context_length}, depth={depth_percent}), skip")
            return

        retrieval_score = {l: [0.0] * self.head_num for l in self.full_layers}
        t0 = time.time()
        qc.STATE.reset()
        qc.STATE.record = True
        response = qc.prefill_and_decode(self.model, self.enc, input_ids, 50,
                                         on_step=lambda tok: self.retrieval_calculate(retrieval_score, tok))
        qc.STATE.reset()
        elapsed = time.time() - t0

        score = scorer.score(self.real_needle, response)["rouge1"].recall * 100
        if score > 50:  # как в оригинале: учитываем только успешные извлечения
            self.retrieval_head_accumulate(retrieval_score)
            head_score = sorted(((k, np.mean(v)) for k, v in self.head_counter.items()), key=lambda x: x[1], reverse=True)
            print([[k, round(float(v), 3)] for k, v in head_score[:20]])

        print("-- Test Summary --")
        print(f"Duration: {elapsed:.1f} s | Context: {context_length} | Depth: {depth_percent}% | "
              f"Prompt tokens: {input_ids.shape[1]} | Score: {score:.1f}")
        print(f"Response: {response}\n")

        results = {
            "model": self.model_path, "context_length": int(context_length), "depth_percent": float(depth_percent),
            "needle": self.needle, "model_response": response, "score": score,
            "test_duration_seconds": elapsed,
            "test_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
        }
        name = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent * 100)}'
        os.makedirs(f"results/graph/{self.model_version}", exist_ok=True)
        with open(f"results/graph/{self.model_version}/{name}_results.json", "w") as f:
            json.dump(results, f)
        if self.save_contexts:
            os.makedirs(f"contexts/{self.model_version}", exist_ok=True)
            with open(f"contexts/{self.model_version}/{name}_context.txt", "w") as f:
                f.write(context)

    def start_test(self, s_len, e_len):
        for ni in range(len(self.needle_list)):
            self.needle = self.needle_list[ni]
            self.haystack_dir = self.haystack_dir_list[ni]
            self.real_needle = self.real_needle_list[ni]
            self.retrieval_question = self.question_list[ni]
            print(f"\n=== Needle {ni + 1}/{len(self.needle_list)}: {self.needle.strip()}\n")
            for L in self.context_lengths:
                if L < s_len or L > e_len:
                    continue
                for d in self.depths:
                    self.evaluate_and_log(int(L), d)

        os.makedirs("head_score", exist_ok=True)
        path = f"head_score/{self.model_version}.json"
        if os.path.exists(path):
            with open(path) as f:
                for k, v in json.loads(f.readline()).items():
                    self.head_counter[k] += v
        with open(path, "w") as f:
            json.dump(self.head_counter, f)
        print(f"Saved head scores to {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--s_len", type=int, default=0)
    p.add_argument("-e", "--e_len", type=int, default=50000)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--model_name_suffix", type=str, default=None)
    p.add_argument("--haystack_dir", type=str, default="./haystack_for_detect")
    p.add_argument("--num_lengths", type=int, default=20, help="число длин контекста между s и e")
    p.add_argument("--chat", choices=["auto", "on", "off"], default="auto",
                   help="использовать chat template (auto: да для Instruct, нет для *-Base)")
    p.add_argument("--save_contexts", action="store_true")
    a = p.parse_args()

    # Пути к данным/результатам — относительно папки скрипта (source/), модель — относительно места запуска
    a.model_path = os.path.abspath(a.model_path) if os.path.exists(a.model_path) else a.model_path
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    det =RetrievalHeadDetector(a.model_path, haystack_dir=a.haystack_dir,
                                context_lengths_min=max(a.s_len, 1000), context_lengths_max=a.e_len,
                                context_lengths_num_intervals=a.num_lengths, use_chat=a.chat,
                                model_name_suffix=a.model_name_suffix, save_contexts=a.save_contexts)
    det.start_test(a.s_len, a.e_len)