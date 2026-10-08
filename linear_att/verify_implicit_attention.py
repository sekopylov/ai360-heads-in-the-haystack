"""
Sanity check: the implicit attention reconstructs the real Mamba-2 SSM output.

For every layer we capture
  * the mixer input (to recompute x, B, C, dt), and
  * the true SSM output y (input of mixer.norm, i.e. before gating / out_proj),
then compare y with   sum_s alpha_{t,s} x_s + D x_t.
Also checks that `mask_heads(mode="dt")` really kills the token-mixing part.

    python verify_implicit_attention.py --model AntonV/mamba2-2.7b-hf
"""
import argparse

import torch

from mamba2_implicit_attention import (get_backbone, get_mixers, mamba2_implicit_attention,
                                       mamba2_ssm_inputs, mask_heads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="AntonV/mamba2-2.7b-hf")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--seq_len", type=int, default=300)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
                                                 device_map="auto").eval()
    mixers = get_mixers(model)
    layers = args.layers if args.layers else sorted({0, len(mixers) // 2, len(mixers) - 1})

    text = ("The quick brown fox jumps over the lazy dog. " * 20 +
            "The secret number is 4823. Remember it. " + "Paul Graham writes essays. " * 20)
    ids = tok(text, return_tensors="pt").input_ids[0, : args.seq_len]
    T = ids.numel()

    store = {}
    hooks = []
    for li in layers:
        m = mixers[li]
        hooks.append(m.register_forward_pre_hook(
            lambda mod, a, kw, li=li: store.__setitem__(("in", li), (a[0] if a else kw["hidden_states"]).detach()),
            with_kwargs=True))
        hooks.append(m.norm.register_forward_pre_hook(
            lambda mod, a, li=li: store.__setitem__(("y", li), a[0].detach())))

    dev = next(model.parameters()).device
    with torch.no_grad():
        get_backbone(model)(input_ids=ids[None].to(dev), use_cache=False)
    for h in hooks:
        h.remove()

    ok = True
    for li in layers:
        if ("y", li) not in store:
            print(f"layer {li}: mixer.norm hook did not fire (fused kernel path?) - cannot compare")
            ok = False
            continue
        m = mixers[li]
        x, B, C, dt, A = mamba2_ssm_inputs(m, store[("in", li)])
        _, alpha = mamba2_implicit_attention(x, B, C, dt, A, torch.arange(T))       # [H,T,T]
        y_rec = torch.einsum("hts,shp->thp", alpha, x) + m.D.float()[None, :, None] * x
        y_true = store[("y", li)][0].float().reshape(T, m.num_heads, m.head_dim)
        rel = (y_rec - y_true).norm() / y_true.norm()
        per_head = ((y_rec - y_true).norm(dim=(0, 2)) / y_true.norm(dim=(0, 2)).clamp_min(1e-9))
        print(f"layer {li:3d}: relative error {rel.item():.2e}   (worst head {per_head.max().item():.2e})")
        ok &= rel.item() < (1e-3 if args.dtype == "float32" else 5e-2)

    # masking check
    li, hh = layers[-1], 0
    m = mixers[li]
    store.clear()
    hk = m.norm.register_forward_pre_hook(lambda mod, a: store.__setitem__("y", a[0].detach()))
    with torch.no_grad(), mask_heads(model, [(li, hh)], mode="dt"):
        hk_in = m.register_forward_pre_hook(
            lambda mod, a, kw: store.__setitem__("in", (a[0] if a else kw["hidden_states"]).detach()),
            with_kwargs=True)
        get_backbone(model)(input_ids=ids[None].to(dev), use_cache=False)
        x, B, C, dt, A = mamba2_ssm_inputs(m, store["in"])
        hk_in.remove()
    hk.remove()
    y = store["y"][0].float().reshape(T, m.num_heads, m.head_dim)[:, hh]
    skip = m.D.float()[hh] * x[:, hh]
    print(f"masked head ({li},{hh}): max dt = {dt[:, hh].max().item():.2e}, "
          f"|y - D x| / |y| = {((y - skip).norm() / y.norm()).item():.2e}  (should be ~0)")
    print("OK" if ok else "CHECK FAILED")


if __name__ == "__main__":
    main()
