"""
Model loading with the capturing attention and greedy generation that yields the attention of every step.
"""
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import attention
from .attention import STATE


def find_layers(model):
    """The decoder layers: the longest ModuleList named 'layers'."""
    found = [m for name, m in model.named_modules() if name.split(".")[-1] == "layers" and isinstance(m, nn.ModuleList)]
    return max(found, key=len)


def load(model_path, dtype="auto"):
    """Returns the tokenizer, the model and the indices of the layers that have attention."""
    enc = AutoTokenizer.from_pretrained(model_path)
    dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, attn_implementation=attention.NAME, torch_dtype=dtype, device_map="auto").eval()
    attn_layers = attention.bind(find_layers(model))
    print(f"loaded {model_path}: {len(find_layers(model))} layers, attention in {len(attn_layers)} of them", flush=True)
    return enc, model, attn_layers


def stop_tokens(enc, model):
    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    return {i for i in eos + [enc.eos_token_id] if i is not None}


def generate(model, input_ids, stop, max_new_tokens, block_list=None, capture=True):
    """
    Greedy decoding of one prompt. The prompt is run on the fast path and fills the cache; then one token per step,
    with the heads of block_list masked. Yields (token, rows) per step: rows is the attention of the generating
    position, [layer, head, kv_len] float32 on the model device, over the layers that have attention; None
    with capture=False (the computation is the same, the rows are just not returned). The step that produces a stop token is the last one yielded.
    """
    device = model.device
    prompt = torch.tensor(input_ids, device=device)
    attention.set_block(None)
    STATE.capture = False
    try:
        with torch.no_grad():
            cache = model.base_model(input_ids=prompt[None, :-1], use_cache=True).past_key_values
        attention.set_block(block_list)
        inp = prompt[None, -1:]
        for _ in range(max_new_tokens):
            # the decode step always takes the explicit attention path, also when the rows are not needed:
            # the fast path differs from it in the last digits and that is enough to change a generated token
            STATE.capture, STATE.rows = True, {}
            with torch.no_grad():
                outputs = model(input_ids=inp, past_key_values=cache, use_cache=True)
            STATE.capture = False
            cache = outputs.past_key_values
            token = outputs.logits[0, -1].argmax()
            rows = torch.stack([STATE.rows[l].to(device) for l in sorted(STATE.rows)]) if capture else None
            STATE.rows = {}
            yield token.item(), rows
            if token.item() in stop:
                break
            inp = token.view(1, 1)
    finally:
        STATE.capture, STATE.rows = False, {}
        attention.set_block(None)
