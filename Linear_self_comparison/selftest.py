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

import torch

import rh_core as rc


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
        print(f"  synthetic q_start={q_start} chunk={chunk}: max|o_rec - o_ref| = {err:.2e}, max|A_future| = {fut:.1e}")
        assert err < 1e-8 and fut == 0.0, "implicit attention mismatch"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--haystack_root", default="../Retrieval_Head")
    p.add_argument("--length", type=int, default=3000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--prompt_format", default="chat")
    args = p.parse_args()

    print("[1] synthetic gated-delta-rule check")
    synthetic_test(args.device)

    tok, model = rc.load_model(args.model_path, args.device)
    space = rc.HeadSpace(model)
    print("model:", space.describe())

    hay = rc.read_haystack(os.path.join(args.haystack_root, "PaulGrahamEssays"), args.length)
    needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    answer = "eat a sandwich and sit in Dolores Park on a sunny day"
    ctx = rc.insert_needle(tok, hay, needle, 50, args.length)
    ids = rc.build_prompt_ids(tok, ctx, "What is the best thing to do in San Francisco?", args.prompt_format).to(args.device)
    gen = rc.greedy_generate(model, tok, ids, 50)
    resp = tok.decode(gen, skip_special_tokens=True).strip()
    print(f"[4] answer: {resp!r}  rouge-1 recall = {rc.rouge_recall(answer, resp):.0f}")

    print("[2] reconstruction of real head outputs (relative error per layer; bf16 model => ~1e-2 is fine)")
    cap = rc.AttnCapture(space, verify=True)
    seq = torch.cat([ids[0], torch.tensor(gen[:-1], device=args.device, dtype=ids.dtype)])
    q_start = ids.shape[1] - 1
    res = cap.run(model, seq[None], q_start)
    worst = 0.0
    for l in range(len(space.layers)):
        a, r = cap.actual[l], cap.recon[l]
        rel = ((a - r).norm() / a.norm().clamp_min(1e-6)).item()
        worst = max(worst, rel)
        print(f"  layer {l:2d} {space.kinds[l]:6s} rel.err = {rel:.3e}")
    cap.remove()
    print(f"  worst = {worst:.3e} -> {'OK' if worst < 0.05 else 'CHECK! hooks may not match this transformers version'}")

    ns, ne = rc.find_needle_idx(tok, ids[0], answer)
    sc = rc.retrieval_scores(res, space, seq, gen, (ns, ne), "attn")
    top = sorted(sc, key=sc.get, reverse=True)[:10]
    print("  top heads on this sample:", ", ".join(f"{h}({space.kind_of[h]}):{sc[h]:.2f}" for h in top))

    print("[3] masker check")
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
    print(f"  all heads of layer {lin} (linear) masked: max|out| = {outs['lin'].abs().max().item():.2e}")
    print(f"  all heads of layer {full} (full)   masked: max|out| = {outs['full'].abs().max().item():.2e}")
    print("selftest finished")


if __name__ == "__main__":
    main()
