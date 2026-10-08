"""
Model loading and one needle-in-a-haystack sample: the context is run without attention capture,
then the answer is decoded greedily token by token with the attention of every head captured.
"""
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import attention
from ..attention import STATE


def find_layers(model):
    """The decoder layers: the longest ModuleList named 'layers'."""
    found = [m for name, m in model.named_modules() if name.split(".")[-1] == "layers" and isinstance(m, nn.ModuleList)]
    return max(found, key=len)


def load(model_path, slow_tokenizer=False, **kwargs):
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=not slow_tokenizer)
    kwargs.setdefault("torch_dtype", "auto")
    kwargs.setdefault("device_map", "auto")
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attention.NAME, **kwargs).eval()
    layers = attention.bind(find_layers(model))
    print(f"loaded {model_path}: {len(find_layers(model))} layers, attention in {len(layers)} of them")
    return enc, model


def stop_tokens(enc, model, legacy):
    """legacy: the authors' condition, written for Llama; for other tokenizers it almost never fires."""
    if legacy:
        return {144} | {i for i in [enc.convert_tokens_to_ids('<0x0A>')] if isinstance(i, int) and i >= 0 and i != enc.unk_token_id}
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, (list, tuple)) else [eos]
    return {i for i in eos + [enc.eos_token_id] if i is not None}


@torch.no_grad()
def run_sample(model, input_ids, needle_start, needle_end, stop, decode_len=50, block_list=None, full_steps=0):
    """
    input_ids: list of prompt ids. Returns a dict of numpy arrays in the format of the dump of the old code:
    top1_idx / top1_val / needle_mass are [step, layer, head] over the layers that have attention,
    retrieval_score is the authors' score of this sample, [layer, head].
    """
    device = model.device
    prompt_ids = torch.tensor(input_ids, device=device)
    attention.set_block(None)
    STATE.capture = False
    cache = model.base_model(input_ids=prompt_ids[None, :-1], use_cache=True).past_key_values

    # as in the authors' code, heads are masked only while the answer is generated
    attention.set_block(block_list)
    STATE.capture = True
    inp = prompt_ids[None, -1:]
    output, top1_idx, top1_val, needle_mass, full = [], [], [], [], {}
    score = None
    for step in range(decode_len):
        STATE.rows = {}
        outputs = model(input_ids=inp, past_key_values=cache, use_cache=True)
        cache = outputs.past_key_values
        token = outputs.logits[0, -1].argmax()
        output.append(token.item())

        rows = torch.stack([STATE.rows[l].to(device) for l in sorted(STATE.rows)])  # [layer, head, kv_len]
        val, idx = rows.max(dim=-1)
        top1_idx.append(idx.cpu().numpy().astype(np.int32))
        top1_val.append(val.cpu().numpy().astype(np.float32))
        if needle_start >= 0:
            needle_mass.append(rows[:, :, needle_start:needle_end].sum(-1).cpu().numpy().astype(np.float32))
            # the authors' rule: the top-1 attention is on a needle token and this token is the generated one
            hit = (idx >= needle_start) & (idx < needle_end) & (prompt_ids[idx.clamp(max=len(prompt_ids) - 1)] == token)
            hit = hit.double() / (needle_end - needle_start)
        else:
            needle_mass.append(np.zeros(idx.shape, dtype=np.float32))
            hit = torch.zeros(idx.shape, dtype=torch.double, device=device)
        score = hit if score is None else score + hit
        if step < full_steps:
            full[f"attn_step{step}"] = rows.half().cpu().numpy()

        if token.item() in stop:
            break
        inp = token.view(1, 1)

    layers = np.array(sorted(STATE.rows), dtype=np.int32)
    STATE.capture = False
    STATE.rows = {}
    attention.set_block(None)
    return dict(output_ids=np.array(output, dtype=np.int32), top1_idx=np.stack(top1_idx), top1_val=np.stack(top1_val),
                needle_mass=np.stack(needle_mass), retrieval_score=score.cpu().numpy(), attn_layers=layers, **full)
