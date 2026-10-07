from __future__ import annotations

import argparse
from pathlib import Path

from retrieval_heads.attention import (
    AttentionRequest,
    CompositeCollector,
    FullTraceCollector,
)
from retrieval_heads.experiment.data import (
    ContextBuilder,
    depth_grid,
    linear_grid,
    load_detection_cases,
    parse_depths,
    parse_lengths,
)
from retrieval_heads.experiment.locator import LegacyOverlapLocator
from retrieval_heads.experiment.runner import ExperimentRunner
from retrieval_heads.experiment.scoring import (
    RetrievalScoreCollector,
    merge_scores,
    rank_heads,
)
from retrieval_heads.experiment.storage import (
    result_payload,
    result_stem,
    write_json,
)
from retrieval_heads.models import available_models, create_model

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect retrieval heads")
    parser.add_argument(
        "--adapter",
        default="qwen35",
        choices=available_models(),
    )
    parser.add_argument("--model", "--model_path", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--s",
        "--min-context",
        dest="min_context",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--e",
        "--max-context",
        dest="max_context",
        type=int,
        default=50000,
    )
    parser.add_argument("--context-intervals", type=int, default=20)
    parser.add_argument(
        "--lengths",
        help="explicit comma-separated context lengths; overrides --s/--e",
    )
    parser.add_argument("--depths")
    parser.add_argument("--context-seed", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--success-threshold", type=float, default=50.0)
    parser.add_argument(
        "--capture",
        choices=["top1", "full"],
        default="top1",
        help="top1 matches the source metric; full exposes every attention value",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prefill-attention", default="sdpa")
    parser.add_argument(
        "--haystack-root",
        type=Path,
        default=HERE / "data" / "haystack_for_detect",
    )
    parser.add_argument("--output-root", type=Path, default=HERE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lengths = (
        parse_lengths(args.lengths)
        if args.lengths
        else linear_grid(args.min_context, args.max_context, args.context_intervals)
    )
    depths = parse_depths(args.depths) if args.depths else depth_grid(10)

    model = create_model(
        args.adapter,
        model_id=args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        prefill_attention=args.prefill_attention,
    )
    context_builder = ContextBuilder(
        model.tokenizer,
        max_context_length=max(lengths),
        period_tokens=model.period_tokens,
        context_seed=args.context_seed,
    )
    runner = ExperimentRunner(
        model,
        context_builder,
        LegacyOverlapLocator(model.tokenizer, threshold=0.9),
        max_new_tokens=args.max_new_tokens,
    )

    detection_dir = args.output_root / "detection"
    results_dir = detection_dir / "results"
    contexts_dir = detection_dir / "contexts"
    attention_dir = detection_dir / "attention"
    score_path = detection_dir / "head_scores.json"
    config_path = detection_dir / "run.json"
    history: dict[str, list[float]] = {}
    cases = load_detection_cases(args.haystack_root)

    total = len(cases) * len(lengths) * len(depths)
    run_config = {
        "kind": "retrieval_head_detection",
        "complete": False,
        "model": model.model_id,
        "model_version": model.model_version,
        "lengths": lengths,
        "depths": depths,
        "capture": args.capture,
        "context_seed": args.context_seed,
        "success_threshold": args.success_threshold,
        "max_new_tokens": args.max_new_tokens,
        "total_cases": total,
    }
    write_json(config_path, run_config)
    write_json(score_path, history)

    completed = 0
    for case in cases:
        for context_length in lengths:
            for depth in depths:
                completed += 1
                print(
                    f"[{completed}/{total}] case={case.case_id} "
                    f"context={context_length} depth={depth:g}%"
                )
                prepared = runner.prepare(
                    case,
                    context_length=context_length,
                    depth_percent=depth,
                )
                collector = RetrievalScoreCollector(
                    eligible_heads=model.eligible_heads,
                    prompt_token_ids=prepared.prompt.token_ids,
                    needle_span=prepared.needle_span,
                )
                full_trace = (
                    FullTraceCollector(prepared.prompt.token_ids)
                    if args.capture == "full"
                    else None
                )
                observer = (
                    CompositeCollector(collector, full_trace)
                    if full_trace is not None
                    else collector
                )
                result = runner.run(
                    prepared,
                    attention=AttentionRequest(capture=args.capture),
                    observer=observer,
                )

                stem = result_stem(
                    model.model_version,
                    context_length,
                    depth,
                    case_id=case.case_id,
                )
                write_json(
                    results_dir / f"{stem}_results.json",
                    result_payload(
                        result,
                        model.model_id,
                        experiment={
                            "kind": "retrieval_head_detection",
                            "capture": args.capture,
                            "success_threshold": args.success_threshold,
                        },
                    ),
                )
                context_path = contexts_dir / f"{stem}_context.txt"
                context_path.parent.mkdir(parents=True, exist_ok=True)
                context_path.write_text(prepared.context, encoding="utf-8")
                if full_trace is not None:
                    full_trace.save(attention_dir / f"{stem}.pt")

                if result.score > args.success_threshold:
                    merge_scores(history, collector.scores)
                    write_json(score_path, history)
                print(
                    f"  score={result.score:.1f} "
                    f"response={result.generation.text!r}"
                )
                leaders = rank_heads(history)[:10]
                if leaders:
                    print(f"  top heads: {leaders}")

    write_json(score_path, history)
    run_config["complete"] = True
    run_config["completed_cases"] = completed
    run_config["successful_cases"] = max(
        (len(values) for values in history.values()),
        default=0,
    )
    write_json(config_path, run_config)


if __name__ == "__main__":
    main()
