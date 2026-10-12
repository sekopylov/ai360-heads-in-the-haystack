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
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation at long T
import torch
import torch.nn.functional as F

VARIANTS = ("attn", "contrib")
log = logging.getLogger("rh")


# ----------------------------------------------------------------------------------------
# Logging & progress
# ----------------------------------------------------------------------------------------
def setup_logging(script, name="", log_dir="logs"):
    """Log to stdout and to logs/<script>_<name>_<timestamp>.log (line-buffered, survives crashes)."""
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"{script}{'_' + name if name else ''}_{stamp}.log")
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, mode="a", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.info(f"log file: {os.path.abspath(path)}")
    log.info("command: " + " ".join(sys.argv) + f" | pid {os.getpid()}")
    install_crash_logging(path)
    return path


_CRASH_FH = None


def install_crash_logging(path):
    """Make every way the process can die visible in the log file:
    uncaught exceptions (incl. CUDA OOM) -> full traceback; SIGTERM/SIGHUP/SIGINT -> a line naming the signal;
    segfaults / fatal CUDA errors -> faulthandler dump. Only SIGKILL (OOM-killer, `kill -9`, notebook/container
    limits) cannot be caught: then the log simply stops - see the RAM numbers in the progress lines."""
    global _CRASH_FH
    import atexit
    import faulthandler
    import signal
    import traceback

    def excepthook(t, v, tb):
        log.error("UNCAUGHT EXCEPTION -> process stops:\n" + "".join(traceback.format_exception(t, v, tb)).rstrip())
        log.error(f"memory at crash: {gpu_mem()}")

    sys.excepthook = excepthook
    _CRASH_FH = open(path, "a", buffering=1)
    faulthandler.enable(file=_CRASH_FH, all_threads=True)

    def on_signal(signum, frame):
        name = signal.Signals(signum).name
        hint = {"SIGHUP": "terminal/SSH session closed - run with nohup, tmux or screen",
                "SIGTERM": "killed by job scheduler / notebook / `kill`",
                "SIGINT": "Ctrl+C"}.get(name, "")
        log.error(f"received {name} ({hint}) -> stopping. Re-run the same command to resume.")
        raise SystemExit(128 + signum)

    for name in ("SIGTERM", "SIGHUP", "SIGINT"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), on_signal)
    atexit.register(lambda: log.info(f"process exit (pid {os.getpid()})"))


def fmt_time(sec):
    if sec is None or not math.isfinite(sec):
        return "--:--:--"
    sec = int(sec)
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return (f"{d}d " if d else "") + f"{h:02d}:{m:02d}:{s:02d}"


def _proc_kb(path, field):
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def gpu_mem():
    """GPU memory + process RAM + free system RAM (to spot host OOM kills)."""
    parts = []
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 2**30
        p = torch.cuda.max_memory_allocated() / 2**30
        r = torch.cuda.memory_reserved() / 2**30
        parts.append(f"gpu {a:.1f}G (peak {p:.1f}G, reserved {r:.1f}G)")
    else:
        parts.append("cpu")
    rss = _proc_kb("/proc/self/status", "VmRSS")
    avail = _proc_kb("/proc/meminfo", "MemAvailable")
    if rss is not None:
        parts.append(f"ram {rss / 2**20:.1f}G" + (f" (free {avail / 2**20:.1f}G)" if avail is not None else ""))
    return " | ".join(parts)


class Progress:
    """Progress with %, elapsed and ETA. Every item has a work weight (here: context length in tokens),
    so the ETA accounts for long contexts being slower than short ones."""

    def __init__(self, total_items, total_work=None, name=""):
        self.total_items = max(int(total_items), 1)
        self.total_work = float(total_work if total_work is not None else total_items)
        self.name = name
        self.items = 0
        self.work = 0.0
        self.t0 = time.time()

    def update(self, work=1.0, msg="", items=1):
        self.items += items
        self.work += work
        el = time.time() - self.t0
        frac = min(self.work / self.total_work, 1.0) if self.total_work > 0 else 1.0
        eta = el / frac * (1 - frac) if frac > 0 else float("nan")
        finish = time.strftime("%H:%M", time.localtime(time.time() + eta)) if math.isfinite(eta) else "--:--"
        log.info(f"[{self.name} {self.items}/{self.total_items} | {100 * frac:5.1f}%] "
                 f"elapsed {fmt_time(el)} | ETA {fmt_time(eta)} (~{finish}) | {gpu_mem()} | {msg}")


# ----------------------------------------------------------------------------------------
# Device checks
# ----------------------------------------------------------------------------------------
def check_device(device, allow_cpu=False):
    """Fail early (and loudly) if a GPU was requested but is not usable."""
    import transformers

    log.info(f"python {sys.version.split()[0]} | torch {torch.__version__} (CUDA build {torch.version.cuda}) | "
             f"transformers {transformers.__version__}")
    import importlib.util

    fast = {m: importlib.util.find_spec(m) is not None for m in ("fla", "causal_conv1d", "kernels")}
    log.info("fast linear-attention kernels: " + ", ".join(f"{m}={'yes' if v else 'no'}" for m, v in fast.items())
             + ("" if fast["fla"] and fast["causal_conv1d"] else
                "  -> Gated DeltaNet probably runs on the slow torch fallback; "
                "try `pip install flash-linear-attention causal-conv1d kernels` (then re-run selftest.py)"))
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            msg = ("CUDA is NOT available: torch.cuda.is_available() == False. "
                   "Check `nvidia-smi`, CUDA_VISIBLE_DEVICES and that torch is a CUDA build "
                   f"(torch.version.cuda = {torch.version.cuda}).")
            if allow_cpu:
                log.warning(msg + " --allow_cpu given -> continuing on CPU (VERY slow).")
                return "cpu"
            log.error(msg + " Use --allow_cpu to run on CPU anyway.")
            raise SystemExit(msg + " Use --allow_cpu to run on CPU anyway.")
        idx = torch.device(device).index or 0
        prop = torch.cuda.get_device_properties(idx)
        log.info(f"GPU OK: cuda:{idx} = {prop.name}, {prop.total_memory / 2**30:.1f} GiB, "
                 f"capability {prop.major}.{prop.minor}, visible GPUs = {torch.cuda.device_count()}, "
                 f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
                 f"bf16 supported = {torch.cuda.is_bf16_supported()}")
        # tiny kernel to make sure the device really executes work
        x = torch.randn(256, 256, device=device)
        torch.cuda.synchronize()
        assert torch.isfinite(x @ x).all()
    else:
        log.warning(f"Running on {device} (no GPU requested).")
    return device


def check_model_device(model, device):
    devs = {str(p.device) for p in model.parameters()} | {str(b.device) for b in model.buffers()}
    log.info(f"model parameter devices: {sorted(devs)} | dtype: {next(model.parameters()).dtype} | "
             f"attn_implementation: {getattr(model.config, '_attn_implementation', '?')} | "
             f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M | {gpu_mem()}")
    if str(device).startswith("cuda") and any(not d.startswith("cuda") for d in devs):
        log.error(f"Some model tensors are not on the GPU: {sorted(devs)}")
        raise SystemExit(f"Some model tensors are not on the GPU: {sorted(devs)}")


# ----------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------
def load_model(model_path, device="cuda", dtype="bfloat16", attn_implementation="sdpa", allow_cpu=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = check_device(device, allow_cpu)
    if device == "cpu" and dtype == "bfloat16":
        dtype = "float32"
    if str(device).startswith("cuda") and torch.cuda.get_device_capability()[0] < 8 and dtype == "bfloat16":
        log.warning("GPU is pre-Ampere: no flash / memory-efficient SDPA kernels for bf16 -> full-attention layers use "
                    "the O(T^2) math backend. Prefill is chunked (--prefill_chunk) to keep memory bounded. "
                    "--dtype float16 may be faster here, but check it with selftest.py first.")
    t0 = time.time()
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
    model.requires_grad_(False)
    torch.set_grad_enabled(False)  # this project never needs gradients
    log.info(f"loaded {model_path} in {time.time() - t0:.1f}s")
    check_model_device(model, device)
    return tok, model, device


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
    hd = module.head_dim
    xq = x[:, q_start:]  # only the query rows need q / gate
    n = xq.shape[1]
    q, gate = torch.chunk(module.q_proj(xq).view(1, n, -1, hd * 2), 2, dim=-1)
    gate = gate.reshape(1, n, -1)
    q = module.q_norm(q.reshape(1, n, -1, hd)).transpose(1, 2)
    k = module.k_norm(module.k_proj(x).view(*x.shape[:-1], -1, hd)).transpose(1, 2)
    v = module.v_proj(x).view(*x.shape[:-1], -1, hd).transpose(1, 2)
    q = rope(q, q, cos[..., q_start:, :], sin[..., q_start:, :])[0]
    k = rope(k, k, cos, sin)[0]
    T = k.shape[2]
    rep = module.num_key_value_groups
    qs = q[0].float()  # [H, n, d]
    kk = k[0].float().repeat_interleave(rep, dim=0)
    vv = v[0].float().repeat_interleave(rep, dim=0)
    logits = torch.einsum("hnd,htd->hnt", qs, kk) * module.scaling
    pos = torch.arange(T, device=x.device)
    future = pos[None, :] > torch.arange(q_start, T, device=x.device)[:, None]
    logits.masked_fill_(future[None], float("-inf"))
    P = torch.softmax(logits, dim=-1)
    return P, vv, gate[0].float()


# ----------------------------------------------------------------------------------------
# Capture: forward over prompt + generated tokens -> top-k attended positions per head/step.
# The prefix [0, q_start) is run in chunks with a KV/recurrent cache ("collect": the hooks only store
# each layer's input hidden states); the last chunk [q_start, T) is run with the hooks in "compute" mode,
# which concatenate the stored inputs and compute the attention rows of the query positions.
# Memory is O(T) instead of the O(T^2) of a single full-length SDPA call on GPUs without flash kernels.
# ----------------------------------------------------------------------------------------
class AttnCapture:
    """Two ways to use it:
    * detection (fast): pass `capture=cap` to greedy_generate -> the hooks only store each layer's input hidden
      states during the prefill and decoding forwards that generation does anyway; then `cap.finalize(q_start)`
      computes the attention rows. No extra forward pass over the long prompt.
    * verification (selftest): `cap.run(model, seq, q_start)` does its own chunked forward and, with verify=True,
      also records the real head outputs to compare against.
    """

    def __init__(self, space: HeadSpace, topk=1, chunk=256, verify=False, prefill_chunk=2048):
        self.space, self.topk, self.chunk, self.verify = space, topk, chunk, verify
        self.prefill_chunk = prefill_chunk
        self.mode = None  # None | "collect" | "compute"
        self.q_start = None
        self.buf = defaultdict(list)  # layer -> list of hidden-state chunks [1, t, D]
        self.pos = defaultdict(list)  # full layers -> list of (cos, sin) chunks
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

    def _compute_linear(self, l, module, X):
        q, k, v, beta, g = gdn_recompute_qkv(module, X)
        A = gdn_implicit_attention(q, k, beta, g, self.q_start, self.chunk)
        self._store(l, A, v.norm(dim=-1))
        if self.verify:
            self.recon[l] = torch.einsum("hnt,htd->nhd", A, v).cpu()

    def _compute_full(self, l, module, X, C, S):
        P, vv, gate = full_attention_rows(module, X, C, S, self.q_start)
        self._store(l, P, vv.norm(dim=-1))
        if self.verify:
            o = torch.einsum("hnt,htd->nhd", P, vv).reshape(P.shape[1], -1)
            self.recon[l] = (o * torch.sigmoid(gate)).reshape(P.shape[1], P.shape[0], -1).cpu()

    def _linear_hook(self, l):
        def hook(module, args, kwargs):
            if self.mode is None:
                return None
            x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            self.buf[l].append(x.detach())
            if self.mode == "compute":
                X = torch.cat(self.buf[l], dim=1)
                self.buf[l] = []
                self._compute_linear(l, module, X)
            return None

        return hook

    def _full_hook(self, l):
        def hook(module, args, kwargs):
            if self.mode is None:
                return None
            x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            self.buf[l].append(x.detach())
            self.pos[l].append(kwargs["position_embeddings"])
            if self.mode == "compute":
                X, C, S = self._take_full(l)
                self._compute_full(l, module, X, C, S)
            return None

        return hook

    def _take_full(self, l):
        X = torch.cat(self.buf[l], dim=1)
        C = torch.cat([c for c, _ in self.pos[l]], dim=-2)
        S = torch.cat([s_ for _, s_ in self.pos[l]], dim=-2)
        self.buf[l], self.pos[l] = [], []
        return X, C, S

    # verification hooks: in "compute" mode the current chunk is exactly positions q_start..T-1
    def _actual_linear_hook(self, l):
        H = self.space.n_heads[l]

        def hook(module, args):
            if self.mode == "compute":
                x = args[0]
                self.actual[l] = x.reshape(-1, H, x.shape[-1]).float().cpu()

        return hook

    def _actual_full_hook(self, l):
        H = self.space.n_heads[l]

        def hook(module, args):
            if self.mode == "compute":
                x = args[0][0]  # [n, H*d]
                self.actual[l] = x.reshape(-1, H, x.shape[-1] // H).float().cpu()

        return hook

    def start_collect(self):
        self.reset()
        self.mode = "collect"

    def stop_collect(self):
        self.mode = None

    def reset(self):
        self.mode = None
        self.buf.clear()
        self.pos.clear()
        self.results, self.recon, self.actual = {}, {}, {}

    @torch.no_grad()
    def finalize(self, q_start):
        """Compute attention rows q_start..T-1 from the hidden states collected during generation."""
        self.q_start = q_start
        self.results = {}
        try:
            for l, layer in enumerate(self.space.layers):
                if self.space.kinds[l] == "linear":
                    X = torch.cat(self.buf[l], dim=1)
                    self.buf[l] = []
                    self._compute_linear(l, layer.linear_attn, X)
                else:
                    X, C, S = self._take_full(l)
                    self._compute_full(l, layer.self_attn, X, C, S)
                del X
        finally:
            self.buf.clear()
            self.pos.clear()
        return self.results

    @torch.no_grad()
    def run(self, model, seq_ids, q_start):
        """Own chunked forward over seq_ids [1, T]; captures rows q_start..T-1 (used by selftest)."""
        self.reset()
        self.q_start = q_start
        past = None
        try:
            self.mode = "collect"
            cs = self.prefill_chunk or q_start
            for s in range(0, q_start, max(cs, 1)):
                e = min(s + cs, q_start)
                out = model(input_ids=seq_ids[:, s:e], past_key_values=past, use_cache=True, logits_to_keep=1)
                past = out.past_key_values
            self.mode = "compute"
            model(input_ids=seq_ids[:, q_start:], past_key_values=past, use_cache=True, logits_to_keep=1)
        finally:
            self.mode = None
            self.buf.clear()
            self.pos.clear()
            del past
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


class BatchHeadMasker:
    """Row b of the batch gets heads head_lists[b] zeroed (same hook points as HeadMasker).
    Lets several ablation runs decode together from one shared prefill."""

    def __init__(self, space: HeadSpace, head_lists, device):
        self.enabled = False
        self.handles = []
        B = len(head_lists)
        masks = {}
        for b, heads in enumerate(head_lists):
            for h in heads:
                l, hh = (int(x) for x in str(h).split("-"))
                if l not in masks:
                    masks[l] = torch.ones(B, space.n_heads[l])
                masks[l][b, hh] = 0.0
        for l, m in masks.items():
            layer = space.layers[l]
            if space.kinds[l] == "linear":
                self.handles.append(layer.linear_attn.norm.register_forward_pre_hook(self._hook(m.to(device), "linear")))
            else:
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._hook(m.to(device), "full")))

    def _hook(self, m, kind):
        B, H = m.shape

        def hook(module, args):
            if not self.enabled:
                return None
            x = args[0]
            if kind == "linear":  # [B*T*H, d]
                xv = x.reshape(B, -1, H, x.shape[-1]) * m[:, None, :, None].to(x.dtype)
            else:  # [B, T, H*d]
                xv = x.reshape(B, x.shape[1], H, -1) * m[:, None, :, None].to(x.dtype)
            return (xv.reshape(x.shape),) + tuple(args[1:])

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def prefill_for_decode(model, input_ids, prefill_chunk=2048):
    """Prefill prompt[:-1] (unmasked, as in the paper's protocol). Returns the cache."""
    return chunked_prefill(model, input_ids[:, :-1], prefill_chunk).past_key_values


@torch.no_grad()
def batched_greedy_decode(model, tok, past, last_token, n_rows, max_new_tokens=50, masker=None):
    """Greedy decoding of n_rows sequences that share the same prefilled cache `past` (batch 1; it is
    replicated, the original is left untouched). Same stopping rule per row as greedy_generate.
    Row b uses the head mask of masker row b. Returns a list of token lists."""
    import copy

    past = copy.deepcopy(past)
    past.reorder_cache(torch.zeros(n_rows, dtype=torch.long, device=last_token.device))
    stops = stop_token_ids(tok)
    inp = last_token.reshape(1, 1).expand(n_rows, 1).contiguous()
    gens = [[] for _ in range(n_rows)]
    done = [False] * n_rows
    t0 = _sync_time()
    steps = 0
    try:
        if masker is not None:
            masker.enabled = True
        for _ in range(max_new_tokens):
            out = model(input_ids=inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            steps += 1
            logits = out.logits[:, -1]
            nxt = logits.argmax(-1)
            nxt_l = nxt.tolist()
            for b in range(n_rows):
                if done[b]:
                    continue
                if not torch.isfinite(logits[b]).all():
                    raise RuntimeError("non-finite logits (fp16 overflow?) -> use --dtype bfloat16 or float32")
                t = nxt_l[b]
                gens[b].append(t)
                if t in stops or ("\n" in tok.decode([t]) and tok.decode(gens[b]).strip()):
                    done[b] = True
            if all(done):
                break
            inp = nxt[:, None]
    finally:
        if masker is not None:
            masker.enabled = False
        del past
    LAST_GEN.update(batch_decode_s=_sync_time() - t0, batch_steps=steps, batch_rows=n_rows)
    return gens


@torch.no_grad()
def chunked_prefill(model, ids, chunk=2048):
    """Prefill with a cache in chunks of `chunk` tokens (identical result, O(T) attention memory)."""
    past, out = None, None
    T = ids.shape[1]
    cs = chunk if chunk and chunk > 0 else T
    for s in range(0, T, cs):
        out = model(input_ids=ids[:, s:s + cs], past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
    return out


@torch.no_grad()
def greedy_generate(model, tok, input_ids, max_new_tokens=50, masker: HeadMasker | None = None, scope="decode",
                    prefill_chunk=2048, capture: "AttnCapture | None" = None):
    """Original decoding protocol: prefill prompt[:-1], then feed prompt[-1] and decode greedily,
    stopping at the first newline after some text (or EOS). Masking is active for the decoding steps
    (and for the prefill too if scope == 'all')."""
    global STAGE
    stops = stop_token_ids(tok)
    peak_reset()
    STAGE = "prefill"
    t0 = _sync_time()
    if capture is not None:
        # every token fed below (prompt[:-1], prompt[-1], gen[:-1]) is exactly the sequence scored afterwards
        capture.start_collect()
    try:
        if masker is not None:
            masker.enabled = scope == "all"
        out = chunked_prefill(model, input_ids[:, :-1], prefill_chunk)
        past = out.past_key_values
        t1 = _sync_time()
        p_prefill = peak_gb()
        peak_reset()
        STAGE = "decode"
        if masker is not None:
            masker.enabled = True
        inp = input_ids[:, -1:]
        gen = []
        for _ in range(max_new_tokens):
            out = model(input_ids=inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            if not torch.isfinite(out.logits[0, -1]).all():
                raise RuntimeError("non-finite logits (fp16 overflow?) -> use --dtype bfloat16 or float32")
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
        if capture is not None:
            capture.stop_collect()
    t2 = _sync_time()
    LAST_GEN.update(prefill_s=t1 - t0, decode_s=t2 - t1, steps=len(gen), prompt_tokens=int(input_ids.shape[1]),
                    prefill_peak=p_prefill, decode_peak=peak_gb())
    return gen


LAST_GEN = {}
STAGE = None  # which stage of a sample is running (for OOM reports)


def peak_reset():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def free_gpu():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def gen_timing_str():
    g = LAST_GEN
    if not g:
        return ""
    per = g["decode_s"] / max(g["steps"], 1)
    return (f"prefill {g['prefill_s']:.1f}s (peak {g['prefill_peak']:.1f}G) + {g['steps']} tok x {per:.2f}s "
            f"(peak {g['decode_peak']:.1f}G)")


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