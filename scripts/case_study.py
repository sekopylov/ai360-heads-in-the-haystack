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


def find_copy_step(trace, sample, head: HeadRef, pairing: str, domain: str = "prompt"):
    """First decoding step at which ``head`` pastes a needle token.

    Steps are filtered by ``applies_to`` exactly as the scorer does: the captured
    prefill row belongs to ``next_step`` only, so using it for ``same_step`` would
    draw a panel the scorer does not credit.

    ``domain`` must match the run being illustrated: the scorer's criterion (2)
    takes the argmax over the prompt by default, over the whole row with
    ``--argmax-domain full``, and over the context span alone with
    ``--argmax-domain haystack``.  Hard-coding the prompt here would silently look
    at a different position than the scores being explained.
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
        # `full` lets already-generated positions compete; the span test below keeps
        # only prompt positions anyway, so `prompt[j]` stays in range.
        if domain == "full":
            lo, hi = 0, row.shape[1]
        elif domain == "prompt":
            lo, hi = 0, sample.length
        elif domain == "haystack":
            if sample.haystack_span is None:
                raise ValueError(
                    "argmax_domain='haystack' but this sample has no haystack span"
                )
            lo, hi = sample.haystack_span
        else:
            raise ValueError(f"unknown argmax domain {domain!r}")
        j = int(row[head.head][lo:hi].argmax()) + lo
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
    parser.add_argument("--argmax-domain", default="haystack",
                        choices=["prompt", "full", "haystack"],
                        help="must match the detect run whose scores this illustrates "
                             "(default: the paper's haystack domain)")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--needle-index", type=int, default=0)
    parser.add_argument("--scores", default=None,
                        help="a detect run directory (scores_<pairing>.json); its recorded "
                             "conditions are reused so the figure explains that run")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--capture-method", default="patch", choices=["patch", "output_attentions"])
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--out", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()

    model, tokenizer, info = _load(args.model, dtype=args.dtype)
    if not 0 <= args.needle_index < len(DEFAULT_NEEDLES):
        parser.error(f"--needle-index must be in [0, {len(DEFAULT_NEEDLES) - 1}], "
                     f"got {args.needle_index}")
    needle, question = DEFAULT_NEEDLES[args.needle_index]
    # Conditions: reuse what the illustrated run recorded.  Without this the figure
    # could show a different prompt mode than the heads it is supposed to explain
    # (the same drift `resolve_detection_settings` fixes for the ablations).
    if args.scores:
        sidecar = Path(args.scores) / f"scores_{args.pairing}.json"
        if not sidecar.exists():
            parser.error(f"{sidecar} does not exist; --scores needs a detect run directory")
        recorded = (json.loads(sidecar.read_text(encoding="utf-8")).get("meta") or {})
        config = recorded.get("config") or {}
        # (label, recorded value, apply, what the command line implied).  The flags
        # are not named like the fields (`--no-chat-template`, `--thinking`), so the
        # mapping is explicit.
        for label, value, apply, current in (
            ("chat_template", config.get("chat_template"),
             lambda v: setattr(args, "no_chat_template", not v), not args.no_chat_template),
            ("system_prompt", config.get("system_prompt"),
             lambda v: setattr(args, "system_prompt", v), args.system_prompt),
            ("enable_thinking", config.get("enable_thinking"),
             lambda v: setattr(args, "thinking", bool(v)), args.thinking),
            ("argmax_domain", config.get("argmax_domain"),
             lambda v: setattr(args, "argmax_domain", v), args.argmax_domain),
            ("capture_method", config.get("capture_method"),
             lambda v: setattr(args, "capture_method", v), args.capture_method),
        ):
            if value is None and label != "system_prompt":
                continue
            if current != value:
                print(f"note: {label}={value!r} taken from {sidecar.name} (the command "
                      f"line implied {current!r})")
            apply(value)
    sample = build_needle_sample(
        tokenizer, needle=needle, question=question, target_tokens=args.length,
        depth=args.depth, builder=HaystackBuilder(seed=0),
        chat_template=not args.no_chat_template,
        system_prompt=args.system_prompt,
        enable_thinking=True if args.thinking else False,
    )
    print(f"prompt={sample.length} tokens  needle={sample.needle_span} "
          f"({sample.n_unique_needle_text_tokens} unique)\n")

    trace, generated = decode_with_attention(
        model, info, sample.input_ids, max_new_tokens=args.max_new_tokens, tokenizer=tokenizer,
        # This script needs the real attention distribution for its figure, so it
        # opts into storing the rows; everything else keeps argmax only.  The
        # chunked prefill avoids the one-shot fp32 path that OOM'd at 16K.
        prefill_chunk=4096, store_rows=True,
        # Must reach the capture, not just `find_copy_step`: the credits (and hence
        # `top_heads` in the artifact) come from `credits_from_trace`, i.e. from the
        # argmax this call records.  Passing it only to the display path made the
        # figure and the JSON describe two different domains.
        argmax_domain=args.argmax_domain,
        # The capture needs the span, not just the domain name: `credits_from_trace`
        # reads the argmax this call records, so a `haystack` run without it would
        # either raise or (worse) be scored in another domain.
        argmax_span=(sample.haystack_span if args.argmax_domain == "haystack" else None),
        capture_method=args.capture_method,
    )
    credits, _, considered = credits_from_trace(trace, sample, info, pairing=args.pairing)
    print("generated:", repr(tokenizer.decode(generated, skip_special_tokens=True)[:220]), "\n")

    denom = max(sample.n_unique_needle_text_tokens, 1)
    ranked = top_heads(credits, info, k=6)
    print(f"top heads ({args.pairing}):")
    for head in ranked:
        print(f"  {head}: score={len(credits[head]) / denom:.2f} "
              f"tokens={sorted(credits[head])}")

    strong = ranked[0]
    step_s, token_s, pos_s = find_copy_step(trace, sample, strong, args.pairing,
                                            args.argmax_domain)
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
            (step_s.attn[strong.layer][strong.head].cpu().numpy(), sample.needle_span),
    }
    if weak is None:
        print("no non-retrieval head has an attention row at this step; "
              "plotting the strong head only")
    else:
        distributions[f"{weak} (score {len(credits[weak]) / denom:.2f}) at the same step"] = (
            step_s.attn[weak.layer][weak.head].cpu().numpy(), sample.needle_span,
        )
    fig_dir = Path(args.out) / "figures"
    save_fig(plot_attention_distribution(distributions), fig_dir / "retrieval_attention_dist.pdf")

    save_json(
        add_provenance(dtype=args.dtype, payload={
            "model": info.name,
            "pairing": args.pairing,
            "argmax_domain": args.argmax_domain,
            "capture_method": args.capture_method,
            "chat_template": not args.no_chat_template,
            "system_prompt": args.system_prompt,
            "enable_thinking": True if args.thinking else False,
            "prompt_tokens": sample.length,
            "needle_span": list(sample.needle_span),
            "haystack_span": (list(sample.haystack_span)
                              if sample.haystack_span is not None else None),
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
