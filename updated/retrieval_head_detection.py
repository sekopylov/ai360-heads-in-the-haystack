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
    available_retrieval_metrics,
    create_retrieval_collector,
    select_retrieval_metrics,
    retrieval_capture_requirements,
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
    parser.add_argument("--model-search-dir", action="append",
                        help="explicit root containing MODEL_ID; repeat for ordered search; no default search directory")
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
    metrics = parser.add_mutually_exclusive_group()
    metrics.add_argument("--retrieval-metric", choices=available_retrieval_metrics(),
                        default=None,
                        help="legacy: uncapped repeated hits; needle_token_multiset_v1: token-frequency quotas")
    metrics.add_argument("--retrieval-metrics", help="comma-separated independent metrics computed in the same generation")
    parser.add_argument(
        "--capture",
        choices=["top1", "full"],
        default="top1",
        help="top1 is sufficient for retrieval scoring; full saves analyzed decode probabilities",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prefill-attention", default="sdpa")
    parser.add_argument("--attention-scope", choices=("answer_only", "all_decode_tokens"),
                        help="attention analysis scope; defaults to adapter policy; ROUGE remains final-answer-only")
    parser.add_argument(
        "--haystack-root",
        type=Path,
        default=HERE / "data" / "haystack_for_detect",
    )
    parser.add_argument("--output-root", type=Path, default=HERE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    single_metric, metric_names = select_retrieval_metrics(args.retrieval_metric, args.retrieval_metrics)
    needs_top1, needs_mass = retrieval_capture_requirements(metric_names)
    effective_capture = args.capture if needs_top1 or args.capture == "full" else "none"
    lengths = (
        parse_lengths(args.lengths)
        if args.lengths
        else linear_grid(args.min_context, args.max_context, args.context_intervals)
    )
    depths = parse_depths(args.depths) if args.depths else depth_grid(10)

    model = create_model(
        args.adapter,
        model_id=args.model,
        model_search_dirs=args.model_search_dir,
        device_map=args.device_map,
        dtype=args.dtype,
        prefill_attention=args.prefill_attention,
        attention_scope=args.attention_scope,
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
    histories = {name: {} for name in metric_names}
    metric_files = {name: f"head_scores_{name}.json" for name in metric_names}

    def save_scores():
        if single_metric is not None:
            write_json(score_path, histories[single_metric])
        for name in metric_names:
            write_json(detection_dir / metric_files[name], histories[name])

    cases = load_detection_cases(args.haystack_root)

    total = len(cases) * len(lengths) * len(depths)
    run_config = {
        "kind": "retrieval_head_detection",
        "complete": False,
        "model": model.model_id,
        "model_version": model.model_version,
        "lengths": lengths,
        "depths": depths,
        "capture": effective_capture,
        "retrieval_metric": single_metric,
        "retrieval_metrics": metric_names,
        "head_score_files": metric_files,
        "needle_mass_capture": needs_mass,
        "attention_scope": model.attention_scope,
        "context_seed": args.context_seed,
        "success_threshold": args.success_threshold,
        "max_new_tokens": args.max_new_tokens,
        "total_cases": total,
    }
    write_json(config_path, run_config)
    save_scores()

    completed = 0
    successful = 0
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
                collectors = {name: create_retrieval_collector(
                    name,
                    eligible_heads=model.eligible_heads,
                    prompt_token_ids=prepared.prompt.token_ids,
                    needle_span=prepared.needle_span,
                ) for name in metric_names}
                full_trace = (
                    FullTraceCollector(prepared.prompt.token_ids)
                    if args.capture == "full"
                    else None
                )
                observer = CompositeCollector(*collectors.values(), *([full_trace] if full_trace is not None else []))
                result = runner.run(
                    prepared,
                    attention=AttentionRequest(capture=effective_capture,
                        needle_span=(prepared.needle_span.start, prepared.needle_span.end) if needs_mass else None),
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
                            "capture": effective_capture,
                            "retrieval_metric": single_metric,
                            "retrieval_metrics": metric_names,
                            "retrieval_scores": {name: c.scores for name, c in collectors.items()},
                            "retrieval_qualifying_steps": {name: c.qualifying_steps for name, c in collectors.items()
                                                           if hasattr(c, "qualifying_steps")},
                            "attention_scope": model.attention_scope,
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
                    successful += 1
                    for name, collector in collectors.items():
                        merge_scores(histories[name], collector.scores)
                    save_scores()
                print(
                    f"  score={result.score:.1f} "
                    f"response={result.generation.text!r}"
                )
                for name in metric_names:
                    leaders = rank_heads(histories[name])[:10]
                    if leaders:
                        print(f"  top heads [{name}]: {leaders}")

    save_scores()
    run_config["complete"] = True
    run_config["completed_cases"] = completed
    run_config["successful_cases"] = successful
    write_json(config_path, run_config)


if __name__ == "__main__":
    main()
