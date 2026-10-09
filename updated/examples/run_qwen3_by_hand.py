"""Local checkpoint tutorial: inspect, HF generate, or explicit greedy forward loop.

This example does NOT use the project's adapter or masking. No implicit download.
See MODEL_EXECUTION_GUIDE.md for setup, commands, and the project call chain.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import inspect
from pathlib import Path
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM


def greedy_forward(model, input_ids, max_new_tokens, eos_token_ids):
    """Full-prompt prefill, then cached one-token forwards; no head masking."""
    generated = []
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        print("prefill logits:", tuple(output.logits.shape))
        cache = output.past_key_values
        print("cache type:", type(cache).__name__, "length:", cache.get_seq_length())
        for step in range(max_new_tokens):
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated.append(token_id)
            if step < 5:
                print(f"step={step} predicted_id={token_id} cache_length={cache.get_seq_length()}")
            if token_id in eos_token_ids or step + 1 == max_new_tokens:
                break
            output = model(input_ids=next_token, past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="downloaded local Qwen3 checkpoint")
    parser.add_argument("--mode", choices=("config", "inspect", "generate", "manual"), default="manual")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--attention", choices=("eager", "sdpa", "torch-flash", "flash_attention_2"), default="sdpa")
    parser.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--thinking", action="store_true", help="default OFF here, unlike the experiment")
    parser.add_argument("--trace", action="store_true", help="print first-layer Q/K/V shapes; adds overhead")
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error("--model must be an existing local checkpoint directory")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    print("versions:", "torch", torch.__version__, "transformers", transformers.__version__)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    print("config class:", type(config).__name__)
    for name in ("model_type", "num_hidden_layers", "hidden_size", "num_attention_heads",
                 "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings"):
        print(name, getattr(config, name, None))
    if args.mode == "config":
        return  # Neither GPU nor weights are loaded.
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: check the driver and PyTorch build")
        print("GPU:", torch.cuda.get_device_name(device))
        if args.attention == "torch-flash" and torch.cuda.get_device_capability(device)[0] < 8:
            raise RuntimeError("torch-flash example requires Ampere or newer")
    elif args.attention == "torch-flash":
        raise ValueError("torch-flash requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=False)
    model = Qwen3ForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=getattr(torch, args.dtype),
        device_map=str(device),
        attn_implementation="sdpa" if args.attention == "torch-flash" else args.attention,
    ).eval()
    print("model Python class:", type(model).__name__)
    print("model source:", inspect.getsourcefile(type(model)))
    print("parameters:", sum(p.numel() for p in model.parameters()))
    for name in ("model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
                 "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight",
                 "lm_head.weight"):
        p = model.get_parameter(name)
        print(name, "shape=", tuple(p.shape), "dtype=", p.dtype, "device=", p.device)
    if args.mode == "inspect":
        print(model.model.layers[0])
        return
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=args.thinking,
    )
    print("formatted prompt:", repr(text))
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    print("input IDs before transfer:", encoded.input_ids.device, tuple(encoded.input_ids.shape), encoded.input_ids.dtype)
    input_ids = encoded.input_ids.to(device)
    print("input IDs after transfer:", input_ids.device)
    handles = []
    if args.trace:
        def hook(name):
            def show(module, inputs, output):
                print(name, "input=", tuple(inputs[0].shape), "output=", tuple(output.shape),
                      "device=", output.device)
            return show
        attn = model.model.layers[0].self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            handles.append(getattr(attn, name).register_forward_hook(hook(name)))
    kernel = sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]) if args.attention == "torch-flash" else nullcontext()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode(), kernel:
            if args.mode == "manual":
                eos = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
                token_ids = greedy_forward(model, input_ids, args.max_new_tokens, eos)
            else:
                output_ids = model.generate(
                    input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    do_sample=False, max_new_tokens=args.max_new_tokens,
                    use_cache=True, pad_token_id=tokenizer.eos_token_id,
                )
                token_ids = output_ids[0, input_ids.shape[1]:].tolist()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
    duration = time.perf_counter() - started
    print("generated IDs:", token_ids)
    print("raw output:", tokenizer.decode(token_ids, skip_special_tokens=False))
    print("readable output:", tokenizer.decode(token_ids, skip_special_tokens=True))
    print(f"generation_seconds={duration:.3f} generated_tokens={len(token_ids)}")
    if device.type == "cuda":
        print("peak_allocated_GiB:", torch.cuda.max_memory_allocated(device) / 1024**3)


if __name__ == "__main__":
    main()
