#!/usr/bin/env python
"""Case study: watch a retrieval head copy the needle, step by step.

Produces the paper's Fig. 1 intuition (`retrieval_attention_dist.pdf`) from a
real run: the attention row of a strong retrieval head at the moment it pastes a
needle token, next to the same row for a head that never retrieves anything.

    .venv/bin/python scripts/case_study.py --model qwen3.5-0.8b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retrieval_heads.cli import _load  # noqa: E402
from retrieval_heads.detection import DEFAULT_NEEDLES  # noqa: E402
from retrieval_heads.haystack import HaystackBuilder, build_needle_sample  # noqa: E402
from retrieval_heads.plotting import plot_attention_distribution, save_fig  # noqa: E402
from retrieval_heads.scoring import (  # noqa: E402
    credits_from_trace,
    decode_with_attention,
)
from retrieval_heads.provenance import add_provenance  # noqa: E402
from retrieval_heads.utils import HeadRef, save_json  # noqa: E402


def top_heads(credits, info, k=5):
    ranked = sorted(info.scoreable_heads, key=lambda h: -len(credits[h]))
    return ranked[:k]


def find_copy_step(trace, sample, head: HeadRef, pairing: str):
    """First decoding step at which ``head`` pastes a needle token.

    Steps are filtered by ``applies_to`` exactly as the scorer does: the captured
    prefill row belongs to ``next_step`` only, so using it for ``same_step`` would
    draw a panel the scorer does not credit.
    """
    # Same set the scorer credits: the needle *text* tokenization.
    needle_set = set(sample.needle_text_ids)
    start, end = sample.needle_span
    prompt = sample.input_ids[0]
    for step in trace.steps:
        if step.applies_to is not None and pairing not in step.applies_to:
            continue
        token = step.fed_token if pairing == "same_step" else step.predicted_token
        if token not in needle_set:
            continue
        row = step.attn.get(head.layer)
        if row is None or head.head >= row.shape[0]:
            continue
        j = int(row[head.head].argmax())
        if start <= j < end and int(prompt[j]) == token:
            return step, token, j
    return None, None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen3.5-0.8b")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--depth", type=float, default=0.5)
    parser.add_argument("--pairing", default="next_step", choices=["next_step", "same_step"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--needle-index", type=int, default=0)
    parser.add_argument("--out", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()

    model, tokenizer, info = _load(args.model)
    needle, question = DEFAULT_NEEDLES[args.needle_index]
    sample = build_needle_sample(
        tokenizer, needle=needle, question=question, target_tokens=args.length,
        depth=args.depth, builder=HaystackBuilder(seed=0),
    )
    print(f"prompt={sample.length} tokens  needle={sample.needle_span} "
          f"({sample.n_unique_needle_tokens} unique)\n")

    trace, generated = decode_with_attention(
        model, info, sample.input_ids, max_new_tokens=args.max_new_tokens, tokenizer=tokenizer,
        # This script needs the real attention distribution for its figure, so it
        # opts into storing the rows; everything else keeps argmax only.  The
        # chunked prefill avoids the one-shot fp32 path that OOM'd at 16K.
        prefill_chunk=4096, store_rows=True,
    )
    credits, _, considered = credits_from_trace(trace, sample, info, pairing=args.pairing)
    print("generated:", repr(tokenizer.decode(generated, skip_special_tokens=True)[:220]), "\n")

    denom = max(sample.n_unique_needle_tokens, 1)
    ranked = top_heads(credits, info, k=6)
    print(f"top heads ({args.pairing}):")
    for head in ranked:
        print(f"  {head}: score={len(credits[head]) / denom:.2f} "
              f"tokens={sorted(credits[head])}")

    strong = ranked[0]
    step_s, token_s, pos_s = find_copy_step(trace, sample, strong, args.pairing)
    if step_s is None:
        print("no copy step found -- try a longer context or more new tokens")
        return 1

    # Fig. 1 compares two heads *at the same step*.  Pick a non-retrieval head that
    # actually has a row in this step; never fall back to the strong head's row
    # under the weak head's label (that drew the same panel twice).
    weak = next(
        (h for h in reversed(info.scoreable_heads)
         if not credits[h] and h.layer in step_s.attn
         and h.head < step_s.attn[h.layer].shape[0]),
        None,
    )
    distributions = {
        f"{strong} copying token {token_s!r} (input position {pos_s})":
            (step_s.attn[strong.layer][strong.head].numpy(), sample.needle_span),
    }
    if weak is None:
        print("no non-retrieval head has an attention row at this step; "
              "plotting the strong head only")
    else:
        distributions[f"{weak} (score {len(credits[weak]) / denom:.2f}) at the same step"] = (
            step_s.attn[weak.layer][weak.head].numpy(), sample.needle_span,
        )
    fig_dir = Path(args.out) / "figures"
    save_fig(plot_attention_distribution(distributions), fig_dir / "retrieval_attention_dist.pdf")

    save_json(
        add_provenance({
            "model": info.name,
            "pairing": args.pairing,
            "prompt_tokens": sample.length,
            "needle_span": list(sample.needle_span),
            "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
            "top_heads": [
                {"head": str(h), "score": len(credits[h]) / denom,
                 "considered_steps": considered[h],
                 "copied_tokens": sorted(credits[h])}
                for h in ranked
            ],
            "case": {
                "head": str(strong), "token": token_s, "input_position": pos_s,
                "step": step_s.step,
                "argmax_attention": float(step_s.attn[strong.layer][strong.head].max()),
            },
        }),
        Path(args.out) / "case_study.json",
    )
    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sys.exit(main())
