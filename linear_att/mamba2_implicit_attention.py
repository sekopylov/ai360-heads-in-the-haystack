"""
Implicit ("hidden") attention of Mamba-2 heads + retrieval-score computation
and head masking.

Mamba-2 (SSD) is linear attention with a data-dependent scalar decay per head
(Dao & Gu, 2024, "Transformers are SSMs"). For head h:

    S_t = exp(dt_t * A_h) * S_{t-1} + dt_t * x_t B_t^T          (state, P x N)
    y_t = S_t C_t + D_h x_t

Unrolling the recurrence gives an attention form

    y_t = sum_{s<=t} alpha^h_{t,s} x_s + D_h x_t,
    alpha^h_{t,s} = (C_t . B_s) * exp( sum_{j=s+1..t} dt_j A_h ) * dt_s

i.e.   query  q_t = C_t,   key  k_s = B_s,   value  v_s = x_s (per head),
       mask   M_{t,s} = prod_{j=s+1..t} a_j,  a_j = exp(dt_j A_h)  in (0, 1].

alpha is NOT normalised and can be negative, and |x_s| differs a lot between
tokens, so instead of argmax(alpha) we use the norm of the per-token
contribution  ||alpha_{t,s} x_s|| = |alpha_{t,s}| * ||x_s||
(the "norm-based" attention analysis of Kobayashi et al., 2020).
`weighting="alpha"` reproduces a plain argmax |alpha|.

Works with HF `transformers` Mamba2ForCausalLM (e.g. AntonV/mamba2-2.7b-hf,
mistralai/Mamba-Codestral-7B-v0.1). B, C, x, dt are recomputed from the
mixer input inside a forward-pre-hook, so it is independent of whether the
model internally runs the torch path or the CUDA kernels.
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def get_mixers(model):
    base = getattr(model, "backbone", None) or getattr(model, "model", None) or model
    return [layer.mixer for layer in base.layers]


def get_backbone(model):
    return getattr(model, "backbone", None) or getattr(model, "model", None)


def head_geometry(model):
    m = get_mixers(model)[0]
    return len(get_mixers(model)), m.num_heads, m.head_dim


# ----------------------------------------------------------------------------
#  Recompute SSM parameters of one mixer from its input hidden states
# ----------------------------------------------------------------------------
@torch.no_grad()
def mamba2_ssm_inputs(mixer, hidden_states):
    """hidden_states: [1, T, d_model] (input of the mixer, i.e. after the block norm).
    Returns float32 tensors:
        x  [T, H, P]  values,   B [T, G, N] keys,   C [T, G, N] queries,
        dt [T, H]     step sizes (after softplus / clamp),   A [H] (negative)."""
    H, P = mixer.num_heads, mixer.head_dim
    G, N = mixer.n_groups, mixer.ssm_state_size
    d_in = mixer.intermediate_size
    conv_dim = d_in + 2 * G * N
    T = hidden_states.shape[1]

    proj = mixer.in_proj(hidden_states)[0]                      # [T, proj]
    # layout of in_proj output: [ (d_mlp, d_mlp,) gate z, xBC, dt ] -> take from the end
    dt_raw = proj[:, -H:].float()
    xBC = proj[:, -H - conv_dim: -H].float()                    # [T, conv_dim]

    w = mixer.conv1d.weight.float()                             # [conv_dim, 1, k]
    b = mixer.conv1d.bias.float() if mixer.conv1d.bias is not None else None
    k = w.shape[-1]
    xBC = F.conv1d(xBC.t().unsqueeze(0), w, b, padding=k - 1, groups=conv_dim)[0, :, :T].t()
    if getattr(mixer, "activation", "silu") in ("silu", "swish"):
        xBC = F.silu(xBC)
    else:
        xBC = mixer.act(xBC)

    x = xBC[:, :d_in].reshape(T, H, P)
    B = xBC[:, d_in: d_in + G * N].reshape(T, G, N)
    C = xBC[:, d_in + G * N:].reshape(T, G, N)

    dt = F.softplus(dt_raw + mixer.dt_bias.float())
    lim = getattr(mixer, "time_step_limit", (0.0, float("inf")))
    dt = dt.clamp(min=float(lim[0]), max=float(lim[1]))
    A = -torch.exp(mixer.A_log.float())
    return x, B, C, dt, A


@torch.no_grad()
def mamba2_implicit_attention(x, B, C, dt, A, query_idx, weighting="contrib"):
    """Rows of the implicit attention matrix for the given query positions.
    query_idx: LongTensor [Q] (positions t).
    Returns W [H, Q, T] (non-negative "attention" used for argmax / mass) and
    alpha [H, Q, T] (signed implicit attention weights)."""
    T, H, P = x.shape
    G = B.shape[1]
    dev = x.device
    query_idx = query_idx.to(dev)

    cum = torch.cumsum((dt * A[None, :]).double(), dim=0)      # [T, H]  log of decay prefix
    CB = torch.einsum("qgn,sgn->gqs", C[query_idx], B)          # [G, Q, T]
    CB = CB.repeat_interleave(H // G, dim=0)                     # [H, Q, T]
    seg = cum[query_idx].t()[:, :, None] - cum.t()[:, None, :]   # [H, Q, T] = sum_{j=s+1..t} dt_j A
    causal = (torch.arange(T, device=dev)[None, :] <= query_idx[:, None])  # [Q, T]
    decay = torch.exp(seg.clamp(max=0.0)).float() * causal[None]
    alpha = CB * decay * dt.t()[:, None, :]                      # [H, Q, T]

    if weighting == "contrib":
        vnorm = x.norm(dim=-1).t()                                # [H, T]
        W = alpha.abs() * vnorm[:, None, :]
    elif weighting == "alpha":
        W = alpha.abs()
    else:
        raise ValueError(weighting)
    return W, alpha


# ----------------------------------------------------------------------------
#  Retrieval-score collector
# ----------------------------------------------------------------------------
class Mamba2RetrievalScorer:
    """
    Usage:
        scorer = Mamba2RetrievalScorer(model)
        hard, soft = scorer.score(full_ids, query_idx, gen_tokens, needle_start, needle_end)

    full_ids    : LongTensor [T]  prompt + generated answer (teacher forcing)
    query_idx   : LongTensor [m]  position whose output predicts generated token j
                                  (= len(prompt) - 1 + j)
    gen_tokens  : LongTensor [m]  generated tokens
    Returns two tensors [L, H]:
        hard  - paper's retrieval score: fraction of needle tokens for which the
                head's top-1 (implicit) attention lands on the needle position
                holding exactly the token being generated (copy-paste event);
        soft  - same but with the share of the head's attention mass that lands
                on such positions instead of a 0/1 top-1 indicator.
    """

    def __init__(self, model, weighting="contrib", pos_tolerance=0, query_chunk=16):
        self.model = model
        self.mixers = get_mixers(model)
        self.L, self.H, self.P = head_geometry(model)
        self.weighting = weighting
        self.pos_tolerance = pos_tolerance
        self.query_chunk = query_chunk
        self._ctx = None
        self._hooks = []
        for li, mixer in enumerate(self.mixers):
            self._hooks.append(
                mixer.register_forward_pre_hook(self._make_hook(li), with_kwargs=True))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _make_hook(self, li):
        def hook(module, args, kwargs):
            if self._ctx is None:
                return None
            hs = args[0] if len(args) > 0 else kwargs["hidden_states"]
            self._layer_score(li, module, hs)
            return None
        return hook

    @torch.no_grad()
    def _layer_score(self, li, mixer, hs):
        c = self._ctx
        dev = hs.device
        x, B, C, dt, A = mamba2_ssm_inputs(mixer, hs)
        ids = c["full_ids"].to(dev)
        q_all = c["query_idx"].to(dev)
        g_all = c["gen_tokens"].to(dev)
        ns, ne = c["needle_start"], c["needle_end"]
        T = ids.shape[0]
        pos = torch.arange(T, device=dev)
        in_needle = (pos >= ns) & (pos < ne)

        hard = torch.zeros(self.H, device=dev)
        soft = torch.zeros(self.H, device=dev)
        for s in range(0, q_all.numel(), self.query_chunk):
            q = q_all[s: s + self.query_chunk]
            g = g_all[s: s + self.query_chunk]
            W, _ = mamba2_implicit_attention(x, B, C, dt, A, q, self.weighting)  # [H,Q,T]

            # copy-source positions for each step: inside needle and token == generated token
            src = in_needle[None, :] & (ids[None, :] == g[:, None])            # [Q, T]
            if self.pos_tolerance > 0:  # allow the source token to be up to k positions
                acc = src.clone()       # before the attended position (conv1d smears tokens)
                for d in range(1, self.pos_tolerance + 1):
                    acc[:, d:] |= src[:, :-d]
                src_hard = acc
            else:
                src_hard = src

            top = W.argmax(dim=-1)                                              # [H, Q]
            hit = torch.gather(src_hard[None].expand(self.H, -1, -1), 2, top[..., None])[..., 0]
            hard += hit.float().sum(dim=1)

            mass = (W * src_hard[None].float()).sum(-1) / W.sum(-1).clamp_min(1e-30)  # [H, Q]
            soft += mass.sum(dim=1)

        n = float(ne - ns)
        c["hard"][li] = (hard / n).cpu()
        c["soft"][li] = (soft / n).cpu()

    @torch.no_grad()
    def score(self, full_ids, query_idx, gen_tokens, needle_start, needle_end):
        self._ctx = dict(full_ids=full_ids, query_idx=query_idx, gen_tokens=gen_tokens,
                         needle_start=needle_start, needle_end=needle_end,
                         hard=torch.zeros(self.L, self.H), soft=torch.zeros(self.L, self.H))
        try:
            backbone = get_backbone(self.model)
            first_dev = next(self.model.parameters()).device
            backbone(input_ids=full_ids[None].to(first_dev), use_cache=False)
            return self._ctx["hard"], self._ctx["soft"]
        finally:
            self._ctx = None


# ----------------------------------------------------------------------------
#  Head masking
# ----------------------------------------------------------------------------
@contextmanager
def mask_heads(model, heads, mode="dt"):
    """
    heads: iterable of (layer, head).
    mode="dt"  : dt_h := 0 for the head  ->  the head never writes into its state,
                 its token-mixing (implicit attention) part is exactly zero, only the
                 local skip D_h * x_t remains. This is the linear-attention analogue of
                 the authors' masking (they zero the query so the head cannot attend
                 selectively). Implemented as dt_bias_h = -1e4 and zero dt row in in_proj,
                 so it works in torch and CUDA-kernel paths and during generation.
    mode="out" : zero the head's columns of out_proj (removes the whole head output).
    """
    mixers = get_mixers(model)
    saved = []
    try:
        with torch.no_grad():
            for (l, h) in heads:
                m = mixers[l]
                if mode == "dt":
                    H = m.num_heads
                    row = m.in_proj.weight.shape[0] - H + h
                    saved.append(("w", m.in_proj.weight, row, m.in_proj.weight[row].clone()))
                    saved.append(("b1", m.dt_bias, h, m.dt_bias[h].clone()))
                    m.in_proj.weight[row].zero_()
                    m.dt_bias[h] = -1e4
                    if m.in_proj.bias is not None:
                        saved.append(("b1", m.in_proj.bias, row, m.in_proj.bias[row].clone()))
                        m.in_proj.bias[row] = 0
                elif mode == "out":
                    P = m.head_dim
                    sl = slice(h * P, (h + 1) * P)
                    saved.append(("col", m.out_proj.weight, sl, m.out_proj.weight[:, sl].clone()))
                    m.out_proj.weight[:, sl] = 0
                else:
                    raise ValueError(mode)
        yield
    finally:
        with torch.no_grad():
            for kind, p, idx, val in reversed(saved):
                if kind == "col":
                    p[:, idx] = val
                else:
                    p[idx] = val


# ----------------------------------------------------------------------------
#  Generation helper
# ----------------------------------------------------------------------------
@torch.no_grad()
def greedy_answer(model, tokenizer, input_ids, max_new_tokens=50):
    """Greedy decoding, stops at the first token containing a newline (as the authors).
    Returns list[int] of generated tokens up to (not including) the first newline that
    follows non-empty content. Leading newline tokens are kept so that teacher forcing on
    prompt + answer reproduces the generation exactly."""
    dev = next(model.parameters()).device
    ids = input_ids[None].to(dev)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         do_sample=False, use_cache=True,
                         pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                         else tokenizer.eos_token_id)
    gen = out[0, ids.shape[1]:].tolist()
    res, has_content = [], False
    for t in gen:
        if t == tokenizer.eos_token_id:
            break
        piece = tokenizer.decode([t])
        if "\n" in piece and has_content:
            break
        res.append(t)
        has_content = has_content or bool(piece.strip())
    return res
