"""
Retrieval-head detection and ablation for hybrid Qwen3.5 models
(Gated DeltaNet "linear attention" layers + gated softmax "full attention" layers).

Port of https://github.com/nightdessert/Retrieval_Head (Wu et al., 2024, arXiv:2404.15574).

Key differences from the original code
--------------------------------------
* No patched modeling files. Everything is done with forward hooks on the stock
  `transformers` Qwen3.5 modules, so prefill / decoding run with the normal fast kernels.
* Retrieval scores are computed in ONE extra forward pass (no KV cache) over
  `prompt + generated tokens`, instead of eager attention at every decoding step.
  The query positions are exactly the ones the original code looked at
  (the token fed at each decoding step), so the score definition is unchanged.
* Linear-attention (Gated DeltaNet) heads have no explicit attention matrix. We use the
  EXACT implicit attention matrix of the gated delta rule:

        S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T ,   o_t = S_t^T q_t
    =>  o_t = sum_{i<=t} A[t,i] v_i ,
        A[t,i] = b_i * k_i^T  N_{i+1} N_{i+2} ... N_t  q_t ,   N_j = a_j (I - b_j k_j k_j^T)

  (q, k are L2-normalised, q is scaled by d^-1/2, exactly as in the kernel).
  A row of A is computed with a backward scan (chunked + a triangular solve, exact).
  A[t,i] can be negative; the retrieval criterion uses argmax_i A[t,i] (variant "attn"),
  and additionally argmax_i |A[t,i]| * ||v_i|| (variant "contrib" = size of token i's
  contribution to the head output). For softmax heads "attn" is the usual attention
  probability, "contrib" is p * ||v||.
* Head ablation = zeroing the head's output (before out_proj / o_proj), the same operation
  for both head types. As in the original code it is applied while decoding
  (the last prompt token + generated tokens); `scope="all"` also masks the prefill.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

VARIANTS = ("attn", "contrib")


# ----------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------
def load_model(model_path, device="cuda", dtype="bfloat16", attn_implementation="sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    torch_dtype = getattr(torch, dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch_dtype, attn_implementation=attn_implementation
        )
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, attn_implementation=attn_implementation
        )
    model.to(device).eval()
    return tok, model


def get_decoder_layers(model):
    """Find the text decoder ModuleList (works for *ForCausalLM and *ForConditionalGeneration)."""
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
            first = mod[0]
            if hasattr(first, "mlp") and (hasattr(first, "linear_attn") or hasattr(first, "self_attn")):
                return mod
    raise RuntimeError("Could not find decoder layers with `linear_attn` / `self_attn`.")


class HeadSpace:
    """All heads of the model: key 'layer-head', kind 'linear' (Gated DeltaNet) or 'full'."""

    def __init__(self, model):
        self.layers = get_decoder_layers(model)
        self.kinds, self.n_heads = [], []
        for layer in self.layers:
            if hasattr(layer, "linear_attn"):
                self.kinds.append("linear")
                self.n_heads.append(int(layer.linear_attn.num_v_heads))
            else:
                at = layer.self_attn
                self.kinds.append("full")
                self.n_heads.append(int(at.config.num_attention_heads))
        self.keys = [f"{l}-{h}" for l in range(len(self.layers)) for h in range(self.n_heads[l])]
        self.kind_of = {f"{l}-{h}": self.kinds[l] for l in range(len(self.layers)) for h in range(self.n_heads[l])}

    def heads_of_kind(self, kind):
        return [k for k in self.keys if kind == "all" or self.kind_of[k] == kind]

    def describe(self):
        n_lin = sum(n for k, n in zip(self.kinds, self.n_heads) if k == "linear")
        n_full = sum(n for k, n in zip(self.kinds, self.n_heads) if k == "full")
        pattern = "".join("L" if k == "linear" else "F" for k in self.kinds)
        return f"{len(self.layers)} layers [{pattern}], linear heads={n_lin}, full heads={n_full}"


# ----------------------------------------------------------------------------------------
# Ablation: zero a head's output
# ----------------------------------------------------------------------------------------
class HeadMasker:
    """Zeroes the outputs of the given heads while `enabled` is True.

    linear layer: pre-hook on `linear_attn.norm` (input = core_attn_out reshaped to [B*T*H, d_v]);
                  RMSNormGated(0) = 0, so the head contributes nothing to out_proj.
    full layer:   pre-hook on `self_attn.o_proj` (input = gated head outputs [B, T, H*d]).
    """

    def __init__(self, space: HeadSpace, heads):
        self.enabled = False
        self.handles = []
        self.heads = [tuple(int(x) for x in str(h).split("-")) if not isinstance(h, tuple) else h for h in heads]
        by_layer = defaultdict(list)
        for l, h in self.heads:
            by_layer[l].append(h)
        for l, hs in by_layer.items():
            layer, H = space.layers[l], space.n_heads[l]
            idx = torch.tensor(sorted(set(hs)), dtype=torch.long)
            if space.kinds[l] == "linear":
                self.handles.append(layer.linear_attn.norm.register_forward_pre_hook(self._make_linear_hook(idx, H)))
            else:
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._make_full_hook(idx, H)))

    def _make_linear_hook(self, idx, H):
        def hook(module, args):
            if not self.enabled:
                return None
            x = args[0]
            xv = x.reshape(-1, H, x.shape[-1]).clone()
            xv[:, idx.to(x.device)] = 0
            return (xv.reshape(x.shape),) + tuple(args[1:])

        return hook

    def _make_full_hook(self, idx, H):
        def hook(module, args):
            if not self.enabled:
                return None
            x = args[0]
            xv = x.reshape(*x.shape[:-1], H, x.shape[-1] // H).clone()
            xv[..., idx.to(x.device), :] = 0
            return (xv.reshape(x.shape),) + tuple(args[1:])

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ----------------------------------------------------------------------------------------
# Implicit attention of a Gated DeltaNet head (exact)
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def gdn_implicit_attention(q, k, beta, g, q_start, chunk=64):
    """Rows q_start..T-1 of the implicit attention matrix of the gated delta rule.

    q, k : [H, T, d]  float32, already L2-normalised; q already multiplied by d^-0.5
    beta : [H, T]     write strength (sigmoid)
    g    : [H, T]     log decay (<= 0), state is multiplied by exp(g_t) at step t
    returns A : [H, n, T] with n = T - q_start and  o_t = sum_i A[:, t-q_start, i] * v_i
    """
    H, T, d = q.shape
    n = T - q_start
    A = q.new_zeros(H, n, T)
    G = q.new_zeros(H, n, d)  # G[:, s] = N_{i+1} ... N_{t_s} q_{t_s} at the current position i
    # 1) tail: positions that are themselves query positions -> exact step-by-step scan
    for i in range(T - 1, q_start - 1, -1):
        G[:, i - q_start] = q[:, i]
        ki = k[:, i]  # [H, d]
        kG = torch.einsum("hnd,hd->hn", G, ki)
        A[:, :, i] = beta[:, i, None] * kG
        G = torch.exp(g[:, i])[:, None, None] * (G - (beta[:, i, None] * kG)[..., None] * ki[:, None, :])
    # 2) prefix: chunked backward scan. Inside a chunk [s, e):
    #    N_{j+1}..N_{e-1} = exp(c_j) * P_j,  P_j = I - sum_{m>j} k_m r_m,
    #    r_j = b_j k_j^T P_j  <=>  (I + U) R = diag(b) K,  U_jm = b_j <k_j, k_m> (m > j)
    e = q_start
    eye = None
    while e > 0:
        s = max(0, e - chunk)
        C = e - s
        Kc, bc, gc = k[:, s:e], beta[:, s:e], g[:, s:e]
        incl = torch.flip(torch.cumsum(torch.flip(gc, [1]), 1), [1])  # sum_{m>=j} g_m
        excl = incl - gc  # sum_{m>j} g_m
        if eye is None or eye.shape[0] != C:
            eye = torch.eye(C, device=q.device, dtype=q.dtype)
        M = (bc[..., None] * (Kc @ Kc.transpose(1, 2))).triu(1) + eye
        R = torch.linalg.solve_triangular(M, bc[..., None] * Kc, upper=True, unitriangular=True)  # [H, C, d]
        RG = torch.einsum("hcd,hnd->hnc", R, G)  # [H, n, C]
        A[:, :, s:e] = RG * torch.exp(excl)[:, None, :]
        G = torch.exp(incl[:, :1])[..., None] * (G - torch.einsum("hcd,hnc->hnd", Kc, RG))
        e = s
    return A


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def gdn_recompute_qkv(module, x):
    """Recompute (q, k, v, beta, g) of a Qwen3_5GatedDeltaNet module from its input hidden states,
    in exactly the form the kernel consumes them. x: [1, T, D]. Returns [H,T,d] tensors (fp32)."""
    from transformers.activations import ACT2FN

    B, T, _ = x.shape
    assert B == 1, "batch size 1 only"
    mixed = module.in_proj_qkv(x).transpose(1, 2)
    w = module.conv1d.weight
    mixed = F.conv1d(mixed.to(w.dtype), w, module.conv1d.bias, padding=w.shape[-1] - 1, groups=mixed.shape[1])[..., :T]
    mixed = ACT2FN[module.activation](mixed).transpose(1, 2)
    q, k, v = torch.split(mixed, [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    q = q.reshape(T, -1, module.head_k_dim)
    k = k.reshape(T, -1, module.head_k_dim)
    v = v.reshape(T, -1, module.head_v_dim)
    beta = module.in_proj_b(x).sigmoid()[0].float()  # [T, Hv]
    g = (-module.A_log.float().exp() * F.softplus(module.in_proj_a(x).float() + module.dt_bias.float()))[0]
    rep = module.num_v_heads // module.num_k_heads
    if rep > 1:
        q = q.repeat_interleave(rep, dim=1)
        k = k.repeat_interleave(rep, dim=1)
    q, k, v = (t.transpose(0, 1).float().contiguous() for t in (q, k, v))  # [H, T, d]
    q = _l2norm(q) * (q.shape[-1] ** -0.5)
    k = _l2norm(k)
    return q, k, v, beta.t().contiguous(), g.t().contiguous()


@torch.no_grad()
def full_attention_rows(module, x, cos, sin, q_start):
    """Attention probabilities of a Qwen3_5Attention module for query rows q_start..T-1.
    Returns P [H, n, T] (fp32), v [H, T, d] (repeated to query heads) and the output gate [n, H*d]."""
    rope = sys.modules[type(module).__module__].apply_rotary_pos_emb
    input_shape = x.shape[:-1]
    hd = module.head_dim
    q, gate = torch.chunk(module.q_proj(x).view(*input_shape, -1, hd * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    q = module.q_norm(q.reshape(*input_shape, -1, hd)).transpose(1, 2)
    k = module.k_norm(module.k_proj(x).view(*input_shape, -1, hd)).transpose(1, 2)
    v = module.v_proj(x).view(*input_shape, -1, hd).transpose(1, 2)
    q, k = rope(q, k, cos, sin)
    T = k.shape[2]
    rep = module.num_key_value_groups
    qs = q[0, :, q_start:].float()  # [H, n, d]
    kk = k[0].float().repeat_interleave(rep, dim=0)
    vv = v[0].float().repeat_interleave(rep, dim=0)
    logits = torch.einsum("hnd,htd->hnt", qs, kk) * module.scaling
    pos = torch.arange(T, device=x.device)
    future = pos[None, :] > torch.arange(q_start, T, device=x.device)[:, None]
    logits.masked_fill_(future[None], float("-inf"))
    P = torch.softmax(logits, dim=-1)
    return P, vv, gate[0, q_start:].float()


# ----------------------------------------------------------------------------------------
# Capture: one forward over the whole sequence -> top-k attended positions per head/step
# ----------------------------------------------------------------------------------------
class AttnCapture:
    def __init__(self, space: HeadSpace, topk=1, chunk=64, verify=False):
        self.space, self.topk, self.chunk, self.verify = space, topk, chunk, verify
        self.active = False
        self.q_start = None
        self.results = {}  # layer -> {variant: (vals [H,n,k], idx [H,n,k])}
        self.recon, self.actual = {}, {}
        self.handles = []
        for l, layer in enumerate(space.layers):
            if space.kinds[l] == "linear":
                self.handles.append(layer.linear_attn.register_forward_pre_hook(self._linear_hook(l), with_kwargs=True))
                if verify:
                    self.handles.append(layer.linear_attn.norm.register_forward_pre_hook(self._actual_linear_hook(l)))
            else:
                self.handles.append(layer.self_attn.register_forward_pre_hook(self._full_hook(l), with_kwargs=True))
                if verify:
                    self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._actual_full_hook(l)))

    def _store(self, l, A, vnorm):
        out = {}
        for var in VARIANTS:
            S = A if var == "attn" else A.abs() * vnorm[:, None, :]
            vals, idx = S.topk(self.topk, dim=-1)
            out[var] = (vals.cpu(), idx.cpu())
        self.results[l] = out

    def _linear_hook(self, l):
        def hook(module, args, kwargs):
            if not self.active:
                return None
            x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            q, k, v, beta, g = gdn_recompute_qkv(module, x)
            A = gdn_implicit_attention(q, k, beta, g, self.q_start, self.chunk)
            self._store(l, A, v.norm(dim=-1))
            if self.verify:
                self.recon[l] = torch.einsum("hnt,htd->nhd", A, v).cpu()
            return None

        return hook

    def _full_hook(self, l):
        def hook(module, args, kwargs):
            if not self.active:
                return None
            x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            cos, sin = kwargs["position_embeddings"]
            P, vv, gate = full_attention_rows(module, x, cos, sin, self.q_start)
            self._store(l, P, vv.norm(dim=-1))
            if self.verify:
                o = torch.einsum("hnt,htd->nhd", P, vv).reshape(P.shape[1], -1)
                self.recon[l] = (o * torch.sigmoid(gate)).reshape(P.shape[1], P.shape[0], -1).cpu()
            return None

        return hook

    def _actual_linear_hook(self, l):
        H = self.space.n_heads[l]

        def hook(module, args):
            if self.active:
                x = args[0]
                self.actual[l] = x.reshape(-1, H, x.shape[-1])[self.q_start:].float().cpu()

        return hook

    def _actual_full_hook(self, l):
        H = self.space.n_heads[l]

        def hook(module, args):
            if self.active:
                x = args[0][0]  # [T, H*d]
                self.actual[l] = x[self.q_start:].reshape(-1, H, x.shape[-1] // H).float().cpu()

        return hook

    @torch.no_grad()
    def run(self, model, seq_ids, q_start):
        """seq_ids: [1, T] on device. Captures rows q_start..T-1."""
        self.results, self.recon, self.actual = {}, {}, {}
        self.q_start, self.active = q_start, True
        try:
            model(input_ids=seq_ids, use_cache=False, logits_to_keep=1)
        finally:
            self.active = False
        return self.results

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def retrieval_scores(results, space: HeadSpace, seq_ids, gen_ids, needle_span, variant="attn"):
    """Retrieval score of every head for one sample (Wu et al. 2024, Eq. 1):
    fraction of needle tokens copied by the head = #{steps where the top attended token is inside the needle
    and equals the generated token} / |needle|."""
    ns, ne = needle_span
    seq = seq_ids.view(-1).cpu()
    gen = torch.as_tensor(gen_ids)
    scores = {}
    for l, res in results.items():
        _, idx = res[variant]  # [H, n, k]
        n = idx.shape[1]
        tgt = gen[:n][None, :, None]
        hit = (idx >= ns) & (idx < ne) & (seq[idx] == tgt)
        hit = hit.any(-1)  # a step counts once (original code: break after first hit among top-k)
        sc = hit.sum(1).float() / max(ne - ns, 1)
        for h in range(idx.shape[0]):
            scores[f"{l}-{h}"] = float(sc[h])
    return scores


# ----------------------------------------------------------------------------------------
# Needle-in-a-haystack (port of the original tester)
# ----------------------------------------------------------------------------------------
_HAYSTACK_CACHE = {}


def read_haystack(haystack_dir, max_words):
    key = (haystack_dir, max_words)
    if key not in _HAYSTACK_CACHE:
        files = sorted(glob.glob(os.path.join(haystack_dir, "*.txt")))
        if not files:
            raise FileNotFoundError(f"No .txt files in {haystack_dir}")
        context = ""
        while len(context.split()) < max_words:
            for f in files:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    context += fh.read()
        _HAYSTACK_CACHE[key] = context
    return _HAYSTACK_CACHE[key]


def insert_needle(tok, context, needle, depth_percent, context_length, buffer=200):
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    tokens_needle, tokens_context = enc(needle), enc(context)
    context_length -= buffer
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[: context_length - len(tokens_needle)]
    if depth_percent == 100:
        new = tokens_context + tokens_needle
    else:
        point = int(len(tokens_context) * (depth_percent / 100))
        period = set(enc("."))
        while point > 0 and tokens_context[point - 1] not in period:
            point -= 1
        new = tokens_context[:point] + tokens_needle + tokens_context[point:]
    return tok.decode(new)


def build_prompt_ids(tok, context, question, prompt_format="chat"):
    body = f"<book>{context}</book>\nBased on the content of the book, Question: {question}\nAnswer:"
    if prompt_format == "chat":
        msgs = [{"role": "user", "content": body}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = context + f"\nBased on the content of the book, Question: {question}\nAnswer:"
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids


def find_needle_idx(tok, prompt_ids, needle):
    """Original fuzzy search: first window whose unique-token overlap with the needle is > 0.9."""
    needle_ids = tok.encode(needle, add_special_tokens=False)
    span, nset = len(needle_ids), set(needle_ids)
    ids = prompt_ids.tolist()
    for i in range(len(ids)):
        if len(set(ids[i : i + span]) & nset) / len(nset) > 0.9:
            return i, i + span
    return -1, -1


def stop_token_ids(tok):
    ids = set()
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            ids.add(tid)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return ids


@torch.no_grad()
def greedy_generate(model, tok, input_ids, max_new_tokens=50, masker: HeadMasker | None = None, scope="decode"):
    """Original decoding protocol: prefill prompt[:-1], then feed prompt[-1] and decode greedily,
    stopping at the first newline after some text (or EOS). Masking is active for the decoding steps
    (and for the prefill too if scope == 'all')."""
    stops = stop_token_ids(tok)
    try:
        if masker is not None:
            masker.enabled = scope == "all"
        out = model(input_ids=input_ids[:, :-1], use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        if masker is not None:
            masker.enabled = True
        inp = input_ids[:, -1:]
        gen = []
        for _ in range(max_new_tokens):
            out = model(input_ids=inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            gen.append(nxt)
            if nxt in stops:
                break
            if "\n" in tok.decode([nxt]) and tok.decode(gen).strip():
                break
            inp = torch.tensor([[nxt]], device=input_ids.device)
    finally:
        if masker is not None:
            masker.enabled = False
    return gen


_SCORER = None


def rouge_recall(reference, response):
    global _SCORER
    if _SCORER is None:
        from rouge_score import rouge_scorer

        _SCORER = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    return _SCORER.score(reference, response)["rouge1"].recall * 100


def grid(s_len, e_len, num_lengths, num_depths):
    lengths = np.round(np.linspace(s_len, e_len, num=num_lengths, endpoint=True)).astype(int).tolist()
    depths = np.round(np.linspace(0, 100, num=num_depths, endpoint=True)).astype(int).tolist()
    return lengths, depths


# ----------------------------------------------------------------------------------------
# Head-score files (same json format as the original repo: {"layer-head": [scores...]})
# ----------------------------------------------------------------------------------------
def load_head_scores(path):
    with open(path) as f:
        d = json.loads(f.readline())
    return {k: float(np.mean(v)) if len(v) else 0.0 for k, v in d.items()}


def rank_heads(mean_scores, kind_of, kind):
    keys = [k for k in mean_scores if kind == "all" or kind_of[k] == kind]
    return sorted(keys, key=lambda k: mean_scores[k], reverse=True)


def random_heads(mean_scores, kind_of, kind, k, seed, exclude_top=None, exclude_thresh=0.1):
    """K random heads of a kind, excluding retrieval heads (score >= exclude_thresh) and the top-`exclude_top`."""
    ranked = rank_heads(mean_scores, kind_of, kind)
    excl = set(ranked[: (exclude_top if exclude_top is not None else k)])
    excl |= {h for h in ranked if mean_scores[h] >= exclude_thresh}
    pool = [h for h in ranked if h not in excl]
    if len(pool) < k:  # not enough non-retrieval heads: only exclude the top-k
        pool = ranked[k:]
    rng = random.Random(seed)
    return sorted(rng.sample(pool, min(k, len(pool))), key=lambda s: tuple(map(int, s.split("-"))))
