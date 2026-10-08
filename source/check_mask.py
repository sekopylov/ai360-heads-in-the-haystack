"""
Checks that head masking (block_list) reaches the attention of Qwen2: compares the logits of one decode step
with and without masked heads, in one process and on the same input.

python check_mask.py --model_path Qwen/Qwen1.5-14B-Chat
"""
import argparse
import glob
import json
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.append("./faiss_attn/")
from source.modeling_qwen2 import Qwen2ForCausalLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--head_score', type=str, default=None, help='head score json, default head_score/<model>.json')
    parser.add_argument('--tokens', type=int, default=2000, help='length of the test context')
    args = parser.parse_args()

    name = args.model_path.split("/")[-1]
    with open(args.head_score or f"head_score/{name}.json") as file:
        scores = json.loads(file.readline())
    ranked = sorted(scores.items(), key=lambda x: np.mean(x[1]), reverse=True)
    ranked = [[int(i) for i in k.split("-")] for k, _ in ranked]

    enc = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = Qwen2ForCausalLM.from_pretrained(
        args.model_path, torch_dtype="auto", device_map='auto', use_flash_attention_2="flash_attention_2").eval()
    text = open(sorted(glob.glob("PaulGrahamEssays/*.txt"))[0]).read()
    ids = enc(text, return_tensors="pt")["input_ids"][:, :args.tokens].to(model.device)

    with torch.no_grad():
        # the tuple cache is not modified by a decode step, so it can be reused
        past = model.model(input_ids=ids[:, :-1], use_cache=True, return_dict=True).past_key_values

        def step(block_list):
            out = model(input_ids=ids[:, -1:], past_key_values=past, use_cache=True, block_list=block_list)
            return out.logits[0, -1].float()

        base = step(None)
        noise = (step(None) - base).abs().max().item()
        print(f"context {ids.size(1)} tokens, logit range {base.min().item():.2f}..{base.max().item():.2f}")
        print(f"no mask twice:    max |diff| {noise:.6f}")
        ok = True
        for k in (1, 30, 100):
            masked = step(ranked[:k])
            diff = (masked - base).abs().max().item()
            ok = ok and diff > 10 * noise and diff > 0
            print(f"top-{k:<3d} masked:   max |diff| {diff:.6f}, next token changed: {bool(masked.argmax() != base.argmax())}")
    print("MASKING WORKS" if ok else "MASKING HAS NO EFFECT")
