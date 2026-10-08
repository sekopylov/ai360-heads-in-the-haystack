"""
Attention function for the transformers AttentionInterface: fast by default, and on request it
returns the attention of the last query token and masks heads.

The model is loaded with attn_implementation=NAME, every attention layer of the model then calls
attention_forward. Only single unpadded sequences are supported: the attention mask of the model
is ignored and a causal mask is built here.
"""
import torch
from torch import nn
from transformers import AttentionInterface

NAME = "rh_capture"


class State:
    def __init__(self):
        self.layer_of = {}   # id(attention module) -> layer index
        self.capture = False
        self.block = {}      # layer index -> list of masked heads
        self.rows = {}       # layer index -> [head, kv_len], attention of the last query token, float32


STATE = State()


def bind(layers):
    """Registers the attention modules of the decoder layers; layers without self_attn are skipped."""
    STATE.layer_of = {id(layer.self_attn): i for i, layer in enumerate(layers) if hasattr(layer, "self_attn")}
    return sorted(STATE.layer_of.values())


def set_block(block_list):
    """block_list: [[layer, head], ...] as in the authors' code, or None."""
    STATE.block = {}
    for layer, head in block_list or []:
        STATE.block.setdefault(int(layer), []).append(int(head))


def causal_mask(q_len, k_len, device):
    # the query tokens are the last q_len positions of the k_len keys
    return torch.ones(q_len, k_len, dtype=torch.bool, device=device).tril(diagonal=k_len - q_len)


def attention_forward(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None, **kwargs):
    # query [batch, head, q_len, dim], key/value [batch, kv_head, k_len, dim]
    n_rep = query.shape[1] // key.shape[1]
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    q_len, k_len = query.shape[2], key.shape[2]
    if scaling is None:
        scaling = query.shape[-1] ** -0.5

    layer = STATE.layer_of.get(id(module))
    heads = STATE.block.get(layer)
    if layer is None or not (STATE.capture or heads):
        if q_len == k_len:
            out = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=q_len > 1, scale=scaling)
        else:
            mask = causal_mask(q_len, k_len, query.device) if q_len > 1 else None
            out = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if heads:
        # the authors' masking: zero logits, i.e. uniform attention of the head
        scores[:, heads] = 0
    if q_len > 1:
        scores = scores.masked_fill(~causal_mask(q_len, k_len, scores.device), torch.finfo(scores.dtype).min)
    weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
    if STATE.capture:
        STATE.rows[layer] = weights[0, :, -1]
    weights = weights.to(query.dtype)
    out = torch.matmul(weights, value)
    return out.transpose(1, 2).contiguous(), weights


AttentionInterface.register(NAME, attention_forward)
