"""Command line interface.

    python -m retrieval_heads.cli describe  --model qwen3.5-0.8b
    python -m retrieval_heads.cli detect    --model qwen3.5-0.8b --profile smoke
    python -m retrieval_heads.cli mask      --model qwen3.5-0.8b --k 1 2 4 8
    python -m retrieval_heads.cli qa        --model qwen3.5-0.8b
    python -m retrieval_heads.cli cot       --model qwen3.5-0.8b
    python -m retrieval_heads.cli compare   --runs results/qwen3.5-0.8b results/qwen3-0.6b
    python -m retrieval_heads.cli figures   --runs results/qwen3.5-0.8b results/qwen3-0.6b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from retrieval_heads.utils import ensure_dir, get_logger, load_json, save_json, set_seed

log = get_logger("cli")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "models.json"

#: Named grids, so a laptop run and a GPU run differ by one flag.
PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {"lengths": [1024], "depths": 1, "needles": 1, "max_new_tokens": 16},
    "laptop": {"lengths": [1024, 2048, 4096], "depths": 3, "needles": 2, "max_new_tokens": 32},
    "paper": {"lengths": [1024, 2048, 4096, 8192, 16384, 32768, 49152], "depths": 10,
              "needles": 3, "max_new_tokens": 48},
}


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    return load_json(path or DEFAULT_REGISTRY)["models"]


def resolve_model(name: str | Path, registry: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """Accept a registry key or a raw path; return ``(path, settings)``."""
    registry = registry if registry is not None else load_registry()
    if name in registry:
        return registry[name]["path"], registry[name]
    candidate = Path(name)
    if candidate.exists():
        return str(candidate), {"path": str(candidate), "dtype": "float32"}
    raise SystemExit(f"unknown model {name!r}; known: {sorted(registry)} (or pass an existing path)")


def resolve_k(args: argparse.Namespace, info) -> list[int]:
    """Turn ``--k`` (absolute) and ``--k-frac`` (fraction of heads) into one list.

    Absolute K is meaningless across models: K=8 is 17% of Qwen3.5-0.8B's 48
    scoreable heads but only 1.8% of Qwen3-0.6B's 448.  Fractions make the two
    directly comparable, which is what the paper's "about 5% of heads" framing
    actually requires.
    """
    values: set[int] = set()
    for k in getattr(args, "k", None) or []:
        if k > 0:
            values.add(int(k))
    for frac in getattr(args, "k_frac", None) or []:
        if frac > 0:
            values.add(max(1, round(frac * info.n_scoreable_heads)))
    if not values:
        values = {max(1, round(0.05 * info.n_scoreable_heads))}
    return sorted(values)


def _load(name: str, *, attn_implementation: str = "eager"):
    import torch

    from retrieval_heads.models import describe_model, load_model

    path, settings = resolve_model(name)
    model, tokenizer, info = load_model(
        path, dtype=settings.get("dtype", "float32"),
        attn_implementation=attn_implementation,
    )
    if torch.cuda.is_available():
        model.to("cuda")  # pragma: no cover - GPU path
        log.info("moved model to CUDA")
    print(describe_model(info))
    print()
    return model, tokenizer, info


# --------------------------------------------------------------------------- commands
def cmd_describe(args: argparse.Namespace) -> int:
    _, _, info = _load(args.model)
    if args.out:
        save_json(info.as_dict(), Path(args.out) / "model_info.json")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    from retrieval_heads.detection import DEFAULT_NEEDLES, DetectionConfig, run_detection

    profile = PROFILES.get(args.profile, {}) if args.profile else {}
    lengths = args.lengths or profile.get("lengths", [1024, 2048, 4096])
    depths = args.depths or profile.get("depths", 3)
    needles = args.needles or profile.get("needles", len(DEFAULT_NEEDLES))
    out_dir = Path(args.out or (REPO_ROOT / "results" / args.model))

    model, tokenizer, info = _load(args.model)
    cfg = DetectionConfig(
        lengths=list(lengths),
        depths_per_length=int(depths),
        needles=list(DEFAULT_NEEDLES[: int(needles)]),
        max_new_tokens=args.max_new_tokens or profile.get("max_new_tokens", 32),
        threshold=args.threshold,
        pairing=args.pairing,
        chat_template=not args.no_chat_template,
        enable_thinking=None if args.thinking else False,
        capture_method=args.capture_method,
        prefill_impl=args.prefill_impl,
        prefill_chunk=args.prefill_chunk,
        seed=args.seed,
        limit=args.limit,
    )
    if args.corpus:
        from retrieval_heads.haystack import load_corpus

        corpus = load_corpus(args.corpus)
    else:
        corpus = None
    run_detection(model, tokenizer, info, cfg, corpus=corpus, out_dir=out_dir)
    return 0


def cmd_mask(args: argparse.Namespace) -> int:
    import torch

    from retrieval_heads.detection import DEFAULT_NEEDLES
    from retrieval_heads.masking import (
        make_eval_samples, masking_curve, token_mixer_ablation,
    )
    from retrieval_heads.scoring import RetrievalScores

    out_dir = Path(args.out or (REPO_ROOT / "results" / args.model))
    scores = RetrievalScores.load(out_dir / f"scores_{args.pairing}")
    model, tokenizer, info = _load(args.model)

    needle, question = DEFAULT_NEEDLES[0]
    # Ablations re-run the *prefill* for every masking configuration, so context
    # length dominates their cost.  One length x two depths keeps the laptop run
    # honest without turning it into an overnight job.
    samples = make_eval_samples(
        tokenizer, lengths=args.lengths or (1024,), depths=(0.2, 0.5, 0.8),
        needle=needle, question=question, seed=args.seed + 7,
        chat_template=not args.no_chat_template,
        enable_thinking=None if args.thinking else False,
    )
    k_values = resolve_k(args, info)
    log.info("masking K values %s (%.1f%%-%.1f%% of %d scoreable heads)",
             k_values, 100 * k_values[0] / info.n_scoreable_heads,
             100 * k_values[-1] / info.n_scoreable_heads, info.n_scoreable_heads)
    curve = masking_curve(
        model, tokenizer, info, scores, samples,
        k_values=k_values, n_random_trials=args.random_trials,
        max_new_tokens=args.max_new_tokens, seed=args.seed,
        prefill_chunk=args.prefill_chunk or None,
    )
    curve.save(out_dir / "masking_curve.json")

    if info.linear_layers:
        ablation = token_mixer_ablation(model, tokenizer, info, samples,
                                        k_values=tuple(k for k in k_values if k <= 4) or (1, 2),
                                        max_new_tokens=args.max_new_tokens,
                                        prefill_chunk=args.prefill_chunk or None)
        save_json(ablation.as_dict(), out_dir / "mixer_ablation.json")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def cmd_qa(args: argparse.Namespace) -> int:
    from retrieval_heads.downstream import builtin_qa_samples, load_qa_jsonl, qa_ablation
    from retrieval_heads.scoring import RetrievalScores

    out_dir = Path(args.out or (REPO_ROOT / "results" / args.model))
    scores = RetrievalScores.load(out_dir / f"scores_{args.pairing}")
    model, tokenizer, info = _load(args.model)

    samples = load_qa_jsonl(args.data) if args.data else builtin_qa_samples()
    result = qa_ablation(model, tokenizer, info, scores, samples,
                         k_values=resolve_k(args, info), n_random_trials=args.random_trials,
                         seed=args.seed, max_new_tokens=args.max_new_tokens)
    save_json(result, out_dir / "task_qa.json")
    return 0


def cmd_cot(args: argparse.Namespace) -> int:
    from retrieval_heads.downstream import (
        builtin_reasoning_samples, cot_ablation, load_reasoning_jsonl,
    )
    from retrieval_heads.scoring import RetrievalScores

    out_dir = Path(args.out or (REPO_ROOT / "results" / args.model))
    scores = RetrievalScores.load(out_dir / f"scores_{args.pairing}")
    model, tokenizer, info = _load(args.model)

    samples = load_reasoning_jsonl(args.data) if args.data else builtin_reasoning_samples()
    result = cot_ablation(model, tokenizer, info, scores, samples,
                          k=resolve_k(args, info)[0], n_random_trials=args.random_trials,
                          seed=args.seed, max_new_tokens=args.max_new_tokens)
    save_json(result, out_dir / "task_cot.json")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from retrieval_heads.properties import correlation_matrix, head_overlap
    from retrieval_heads.scoring import RetrievalScores

    runs = {}
    for run in args.runs:
        path = Path(run)
        scores = RetrievalScores.load(path / f"scores_{args.pairing}")
        runs[scores.info.name] = scores
    names = list(runs)
    mode = args.mode or ("sorted" if _different_shapes(runs) else "grid")
    corr = correlation_matrix([runs[n] for n in names], mode=mode, labels=names)
    save_json(corr.as_dict(), Path(args.out or REPO_ROOT / "results") / "correlation.json")
    print(json.dumps(corr.as_dict(), indent=2))
    if len(names) == 2:
        overlap = head_overlap(runs[names[0]], runs[names[1]], threshold=args.threshold, mode=mode)
        save_json(overlap.as_dict(), Path(args.out or REPO_ROOT / "results") / "overlap.json")
        print(json.dumps(overlap.as_dict(), indent=2))
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from retrieval_heads.plotting import (
        plot_corr_map, plot_heat_map, plot_layer_profile, plot_masking_curve,
        plot_mixer_ablation, plot_score_distribution, plot_score_pie, plot_task_cot,
        plot_task_qa, save_fig,
    )
    from retrieval_heads.properties import correlation_matrix
    from retrieval_heads.scoring import RetrievalScores

    fig_dir = ensure_dir(args.out or REPO_ROOT / "results" / "figures")
    runs, curves, qa, cot, mixers = {}, {}, {}, {}, {}
    for run in args.runs:
        path = Path(run)
        scores = RetrievalScores.load(path / f"scores_{args.pairing}")
        runs[scores.info.name] = scores
        for name, store in (("masking_curve.json", curves), ("task_qa.json", qa),
                            ("task_cot.json", cot), ("mixer_ablation.json", mixers)):
            candidate = path / name
            if candidate.exists():
                store[scores.info.name] = json.loads(candidate.read_text(encoding="utf-8"))

    if runs:
        save_fig(plot_score_pie(runs), fig_dir / "ring_graph.pdf")
        save_fig(plot_score_distribution(runs), fig_dir / "score_distribution.pdf")
        save_fig(plot_heat_map(runs), fig_dir / "heat_map.pdf")
        save_fig(plot_layer_profile(runs), fig_dir / "layer_profile.pdf")
        mode = "sorted" if _different_shapes(runs) else "grid"
        corr = correlation_matrix(list(runs.values()), mode=mode, labels=list(runs))
        save_fig(plot_corr_map(corr), fig_dir / "corr_map.pdf")
    if curves:
        save_fig(plot_masking_curve(curves), fig_dir / "masking_heads.pdf")
    if qa:
        first = next(iter(qa))
        save_fig(plot_task_qa(qa[first]), fig_dir / "task_qa.pdf")
    if cot:
        first = next(iter(cot))
        save_fig(plot_task_cot(cot[first]), fig_dir / "task_cot.pdf")
    if mixers:
        save_fig(plot_mixer_ablation(mixers), fig_dir / "mixer_ablation.pdf")
    return 0


def _different_shapes(runs: dict[str, Any]) -> bool:
    shapes = {run.score.shape for run in runs.values()}
    return len(shapes) > 1


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval_heads", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", required=True, help="registry key (see configs/models.json) or path")
        p.add_argument("--out", default=None, help="results directory")
        p.add_argument("--pairing", default="next_step", choices=["next_step", "same_step"])
        p.add_argument("--no-chat-template", action="store_true")
        p.add_argument("--thinking", action="store_true", help="leave the model's thinking mode on")
        # Also accepted per-subcommand (not just before it): job drivers build
        # argv as `detect --model ... --seed 0 ...`, and argparse only allows a
        # parent-parser option *before* the subcommand.
        p.add_argument("--seed", type=int, default=argparse.SUPPRESS)

    p = sub.add_parser("describe", help="print the architecture and scoreable-head census")
    add_common(p)
    p.set_defaults(func=cmd_describe)

    p = sub.add_parser("detect", help="run the retrieval-head detection grid")
    add_common(p)
    p.add_argument("--profile", default="laptop", choices=sorted(PROFILES))
    p.add_argument("--lengths", type=int, nargs="*")
    p.add_argument("--depths", type=int)
    p.add_argument("--needles", type=int)
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--capture-method", default="output_attentions",
                   choices=["output_attentions", "patch"])
    p.add_argument("--prefill-impl", default="sdpa")
    p.add_argument("--prefill-chunk", type=int, default=4096,
                   help="feed the prefill in chunks of this size (0 = one shot); bounds "
                        "memory when float32 SDPA falls back to the matmul kernel")
    p.add_argument("--corpus", default=None, help="text file of filler sentences")
    p.add_argument("--limit", type=int, default=None, help="cap the number of instances")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("mask", help="mask top-K retrieval heads vs K random heads")
    add_common(p)
    p.add_argument("--k", type=int, nargs="+", default=None,
                   help="absolute numbers of heads to mask")
    p.add_argument("--k-frac", type=float, nargs="+", default=[0.02, 0.04, 0.08, 0.17, 0.33],
                   help="K as a fraction of scoreable heads -- comparable across models")
    p.add_argument("--lengths", type=int, nargs="*", default=None)
    p.add_argument("--random-trials", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--prefill-chunk", type=int, default=4096)
    p.set_defaults(func=cmd_mask)

    p = sub.add_parser("qa", help="extractive-QA ablation")
    add_common(p)
    p.add_argument("--data", default=None, help="JSONL with context/question/answer")
    p.add_argument("--k", type=int, nargs="+", default=None)
    p.add_argument("--k-frac", type=float, nargs="+", default=[0.04, 0.08, 0.17])
    p.add_argument("--random-trials", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.set_defaults(func=cmd_qa)

    p = sub.add_parser("cot", help="chain-of-thought ablation")
    add_common(p)
    p.add_argument("--data", default=None, help="JSONL with question/answer")
    p.add_argument("--k", type=int, nargs="+", default=None)
    p.add_argument("--k-frac", type=float, nargs="+", default=[0.08])
    p.add_argument("--random-trials", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.set_defaults(func=cmd_cot)

    p = sub.add_parser("compare", help="correlate / overlap models")
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--pairing", default="next_step", choices=["next_step", "same_step"])
    p.add_argument("--mode", default=None, choices=["grid", "sorted"])
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("figures", help="regenerate every figure from saved results")
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--pairing", default="next_step", choices=["next_step", "same_step"])
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_figures)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_seed(getattr(args, "seed", 0))
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
