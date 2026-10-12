"""
Sanity checks. Run once on your GPU before the long experiments:
  python selftest.py --model_path Qwen/Qwen3.5-0.8B --haystack_root ../Retrieval_Head

1) synthetic: implicit attention of the gated delta rule reproduces the recurrent kernel output exactly
2) model: implicit/explicit attention rows reconstruct the REAL head outputs of every layer
   (linear: input of linear_attn.norm; full: input of o_proj), i.e. hooks match this transformers version
3) model: HeadMasker really zeroes the selected heads
4) model: a short needle-in-a-haystack answer + top retrieval heads on that sample
"""
import argparse
import os
import time

import torch

import rh_core as rc

log = rc.log


def reference_gated_delta_rule(q, k, v, beta, g):
    """Plain recurrence (same as transformers' torch_recurrent_gated_delta_rule), inputs as in gdn_implicit_attention."""
    H, T, dk = q.shape
    S = torch.zeros(H, dk, v.shape[-1], dtype=q.dtype, device=q.device)
    out = torch.zeros_like(v)
    for t in range(T):
        S = S * torch.exp(g[:, t])[:, None, None]
        kv_mem = (S * k[:, t, :, None]).sum(-2)
        delta = (v[:, t] - kv_mem) * beta[:, t, None]
        S = S + k[:, t, :, None] * delta[:, None, :]
        out[:, t] = (S * q[:, t, :, None]).sum(-2)
    return out


def synthetic_test(device):
    torch.manual_seed(0)
    H, T, dk, dv = 4, 300, 32, 24
    q = rc._l2norm(torch.randn(H, T, dk, device=device, dtype=torch.float64)) * dk ** -0.5
    k = rc._l2norm(torch.randn(H, T, dk, device=device, dtype=torch.float64))
    v = torch.randn(H, T, dv, device=device, dtype=torch.float64)
    beta = torch.rand(H, T, device=device, dtype=torch.float64)
    g = -torch.rand(H, T, device=device, dtype=torch.float64) * 0.2
    ref = reference_gated_delta_rule(q, k, v, beta, g)
    for q_start, chunk in [(T - 1, 64), (T - 20, 64), (250, 16), (5, 7)]:
        A = rc.gdn_implicit_attention(q, k, beta, g, q_start, chunk)
        rec = torch.einsum("hnt,htd->hnd", A, v)
        err = (rec - ref[:, q_start:]).abs().max().item()
        fut = max(A[:, s, q_start + s + 1:].abs().max().item() if q_start + s + 1 < T else 0.0 for s in range(T - q_start))
        log.info(f"  synthetic q_start={q_start} chunk={chunk}: max|o_rec - o_ref| = {err:.2e}, max|A_future| = {fut:.1e}")
        assert err < 1e-8 and fut == 0.0, "implicit attention mismatch"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--haystack_root", default="../Retrieval_Head")
    p.add_argument("--length", type=int, default=3000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--prompt_format", default="chat")
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--prefill_chunk", type=int, default=512, help="small on purpose: exercises chunking")
    p.add_argument("--long", type=int, default=16000, help="length for the memory test (0 = skip)")
    args = p.parse_args()
    rc.setup_logging("selftest")

    log.info("[0] device check")
    args.device = rc.check_device(args.device, args.allow_cpu)
    log.info("[1] synthetic gated-delta-rule check")
    synthetic_test(args.device)

    tok, model, args.device = rc.load_model(args.model_path, args.device, args.dtype, allow_cpu=args.allow_cpu)
    space = rc.HeadSpace(model)
    log.info("model: " + space.describe())

    hay = rc.read_haystack(os.path.join(args.haystack_root, "PaulGrahamEssays"), args.length)
    needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    answer = "eat a sandwich and sit in Dolores Park on a sunny day"
    ctx = rc.insert_needle(tok, hay, needle, 50, args.length)
    ids = rc.build_prompt_ids(tok, ctx, "What is the best thing to do in San Francisco?", args.prompt_format).to(args.device)
    gen = rc.greedy_generate(model, tok, ids, 50, prefill_chunk=args.prefill_chunk)
    resp = tok.decode(gen, skip_special_tokens=True).strip()
    log.info(f"[4] answer: {resp!r}  rouge-1 recall = {rc.rouge_recall(answer, resp):.0f}")

    log.info(f"[5] chunked prefill (chunk {args.prefill_chunk}) vs one-shot prefill, {ids.shape[1]} tokens")
    with torch.no_grad():
        l1 = rc.chunked_prefill(model, ids, args.prefill_chunk).logits[0, -1].float()
        l0 = model(input_ids=ids, use_cache=True, logits_to_keep=1).logits[0, -1].float()
    rel = ((l1 - l0).norm() / l0.norm()).item()
    log.info(f"  last-token logits rel.diff = {rel:.2e}, same argmax = {bool(l1.argmax() == l0.argmax())} "
             f"-> {'OK' if rel < 0.05 else 'CHECK'}")

    log.info("[2] reconstruction of real head outputs (relative error per layer; bf16 model => ~1e-2 is fine)")
    cap = rc.AttnCapture(space, verify=True, prefill_chunk=args.prefill_chunk)
    seq = torch.cat([ids[0], torch.tensor(gen[:-1], device=args.device, dtype=ids.dtype)])
    q_start = ids.shape[1] - 1
    res = cap.run(model, seq[None], q_start)
    worst = 0.0
    for l in range(len(space.layers)):
        a, r = cap.actual[l], cap.recon[l]
        rel = ((a - r).norm() / a.norm().clamp_min(1e-6)).item()
        worst = max(worst, rel)
        log.info(f"  layer {l:2d} {space.kinds[l]:6s} rel.err = {rel:.3e}")
    cap.remove()
    log.info(f"  worst = {worst:.3e} -> {'OK' if worst < 0.05 else 'CHECK! hooks may not match this transformers version'}")

    ns, ne = rc.find_needle_idx(tok, ids[0], answer)
    sc = rc.retrieval_scores(res, space, seq, gen, (ns, ne), "attn")
    top = sorted(sc, key=sc.get, reverse=True)[:10]
    log.info("  top heads on this sample: " + ", ".join(f"{h}({space.kind_of[h]}):{sc[h]:.2f}" for h in top))

    log.info("[2b] fast path used by detect.py (hidden states collected during generation) vs separate pass [2]")
    cap2 = rc.AttnCapture(space)
    gen2 = rc.greedy_generate(model, tok, ids, 50, prefill_chunk=args.prefill_chunk, capture=cap2)
    res2 = cap2.finalize(q_start)
    cap2.remove()
    if gen2 != gen:
        log.info("  generation differs from [4] -> skipping comparison")
    else:
        for v in rc.VARIANTS:
            a = rc.retrieval_scores(res, space, seq, gen, (ns, ne), v)
            b = rc.retrieval_scores(res2, space, seq, gen, (ns, ne), v)
            d = max(abs(a[k] - b[k]) for k in a)
            same_idx = sum(bool((res[l][v][1] == res2[l][v][1]).all()) for l in res)
            log.info(f"  {v:7s}: max |score diff| = {d:.3f}, layers with identical argmax positions {same_idx}/{len(res)} "
                     f"-> {'OK' if d <= 0.05 else 'CHECK'}")

    log.info("[3] masker check")
    lin = next(l for l, k in enumerate(space.kinds) if k == "linear")
    full = next(l for l, k in enumerate(space.kinds) if k == "full")
    heads = [f"{lin}-{h}" for h in range(space.n_heads[lin])] + [f"{full}-{h}" for h in range(space.n_heads[full])]
    outs = {}
    hooks = [space.layers[lin].linear_attn.register_forward_hook(lambda m, a, o: outs.__setitem__("lin", o)),
             space.layers[full].self_attn.register_forward_hook(lambda m, a, o: outs.__setitem__("full", o[0]))]
    masker = rc.HeadMasker(space, heads)
    masker.enabled = True
    with torch.no_grad():
        model(input_ids=ids[:, :64], use_cache=False, logits_to_keep=1)
    masker.enabled = False
    masker.remove()
    for h in hooks:
        h.remove()
    log.info(f"  all heads of layer {lin} (linear) masked: max|out| = {outs['lin'].abs().max().item():.2e}")
    log.info(f"  all heads of layer {full} (full)   masked: max|out| = {outs['full'].abs().max().item():.2e}")
    log.info("[7] batched ablation decoding (one shared prefill, per-row head masks) vs one run at a time")
    lin_layers = [l for l, k in enumerate(space.kinds) if k == "linear"]
    full_layers = [l for l, k in enumerate(space.kinds) if k == "full"]
    head_sets = [[], [f"{full_layers[2]}-{h}" for h in range(4)], [f"{lin_layers[0]}-{h}" for h in range(8)],
                 [f"{l}-{h}" for l in full_layers for h in range(space.n_heads[l])]]  # last: all full heads
    past = rc.prefill_for_decode(model, ids, args.prefill_chunk)
    bm = rc.BatchHeadMasker(space, head_sets, args.device)
    gens_b = rc.batched_greedy_decode(model, tok, past, ids[0, -1], len(head_sets), 50, bm)
    bm.remove()
    for hs, gb in zip(head_sets, gens_b):
        m = rc.HeadMasker(space, hs) if hs else None
        g1 = rc.greedy_generate(model, tok, ids, 50, masker=m, prefill_chunk=args.prefill_chunk)
        if m is not None:
            m.remove()
        same = sum(a == b for a, b in zip(g1, gb))
        log.info(f"  {len(hs):2d} heads masked: identical tokens {same}/{max(len(g1), len(gb))} "
                 f"-> {'OK' if g1 == gb else 'DIFFERS'} | {tok.decode(gb, skip_special_tokens=True).strip()[:50]!r}")

    if args.long:
        log.info(f"[6] memory test at {args.long} tokens (prefill chunk 2048 + capture)")
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        hay = rc.read_haystack(os.path.join(args.haystack_root, "PaulGrahamEssays"), args.long)
        ctx = rc.insert_needle(tok, hay, needle, 50, args.long)
        ids = rc.build_prompt_ids(tok, ctx, "What is the best thing to do in San Francisco?", args.prompt_format)
        ids = ids.to(args.device)
        t0 = time.time()
        cap = rc.AttnCapture(space)
        gen = rc.greedy_generate(model, tok, ids, 50, prefill_chunk=2048, capture=cap)
        t1 = time.time()
        cap.finalize(ids.shape[1] - 1)
        cap.remove()
        rc._sync_time()
        log.info(f"  {ids.shape[1]} tokens: generate {t1 - t0:.1f}s ({rc.gen_timing_str()}), capture {time.time() - t1:.1f}s | "
                 f"{rc.gpu_mem()} | answer {tok.decode(gen, skip_special_tokens=True).strip()[:60]!r}")
    log.info("selftest finished")


if __name__ == "__main__":
    main()