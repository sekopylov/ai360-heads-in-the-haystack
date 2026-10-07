#!/usr/bin/env python
"""
Быстрая проверка перед основным запуском (~1 минута):
  1) transformers загружает Qwen3.5 и видно layer_types;
  2) хуки получают веса внимания только у full-attention слоёв;
  3) переключение sdpa -> eager на лету не ломает результат
     (argmax следующего токена совпадает с чистым sdpa);
  4) веса внимания нормированы (сумма по ключам ~ 1).

  python sanity_check_qwen35.py --model_path Qwen/Qwen3.5-0.8B-Base
"""
import argparse

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from retrieval_head_detection_qwen35 import AttnGrabber


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    print("transformers", transformers.__version__)

    cfg = AutoConfig.from_pretrained(a.model_path).get_text_config()
    types = list(cfg.layer_types)
    full = [i for i, t in enumerate(types) if t == "full_attention"]
    print(f"layers={len(types)} full_attention={full} heads={cfg.num_attention_heads} "
          f"kv_heads={cfg.num_key_value_heads} head_dim={cfg.head_dim}")

    tok = AutoTokenizer.from_pretrained(a.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to(a.device).eval()
    print("model class:", type(model).__name__)
    grab = AttnGrabber(model, full)

    text = "The quick brown fox jumps over the lazy dog. " * 40 + "The secret code is 7421. What is the secret code? The secret code is"
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(a.device)
    print("prompt tokens:", ids.shape[1])

    with torch.no_grad():
        # эталон: всё на sdpa
        ref = model(input_ids=ids, logits_to_keep=1).logits[0, -1].argmax().item()

        # prefill sdpa -> decode eager
        out = model(input_ids=ids[:, :-1], use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        try:
            model.set_attn_implementation("eager")
        except Exception as e:  # noqa
            print("set_attn_implementation не сработал:", repr(e))
            raise
        grab.enabled = True
        o = model(input_ids=ids[:, -1:], past_key_values=past, use_cache=True, logits_to_keep=1)
        got = o.logits[0, -1].argmax().item()

    print("captured layers:", sorted(grab.weights))
    assert sorted(grab.weights) == full, "хуки должны сработать ровно на full-attention слоях"
    for li, w in grab.weights.items():
        s = w[0, :, -1, :].float().sum(-1)
        print(f"  layer {li}: attn shape {tuple(w.shape)}, row-sum min/max = {s.min():.3f}/{s.max():.3f}")
    print(f"next token sdpa={tok.decode([ref])!r} vs sdpa->eager={tok.decode([got])!r}")
    print("OK" if ref == got else "ВНИМАНИЕ: токены разные -- проверь переключение attention-реализации")


if __name__ == "__main__":
    main()