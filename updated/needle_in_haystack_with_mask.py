from __future__ import annotations

import argparse
import random
from pathlib import Path

from retrieval_heads.attention import AttentionRequest, Head
from retrieval_heads.experiment.data import (
    ContextBuilder,
    default_mask_case,
    depth_grid,
    linear_grid,
    load_validation_cases,
    parse_depths,
    parse_lengths,
)
from retrieval_heads.experiment.runner import ExperimentRunner
from retrieval_heads.experiment.scoring import rank_heads
from retrieval_heads.experiment.storage import (
    read_json,
    result_payload,
    result_stem,
    write_json,
)
from retrieval_heads.models import available_models, create_model

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Needle test with head masking")
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
        default=128000,
    )
    parser.add_argument("--context-intervals", type=int, default=40)
    parser.add_argument(
        "--lengths",
        help="explicit comma-separated context lengths; overrides --s/--e",
    )
    parser.add_argument("--depths")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument(
        "--mask-topk",
        "--mask_topk",
        type=int,
        default=0,
        help="positive: top retrieval heads; negative: random non-retrieval heads",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--head-selection", choices=("top", "bottom"), default="top",
                        help="ranked selection for positive --mask-topk; negative still selects random")
    parser.add_argument("--context-seed", type=int,
                        help="select the haystack order/window reproducibly; defaults to --seed")
    parser.add_argument(
        "--random-exclusion-top",
        type=int,
        help="exclude this many ranked heads; defaults to abs(mask-topk)",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["zero_output", "legacy_uniform"],
        default="legacy_uniform",
        help="legacy_uniform reproduces the source; zero_output is an optional ablation",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prefill-attention", default="sdpa")
    parser.add_argument("--decode-attention", choices=("eager", "sdpa_flash"), default="eager",
                        help="sdpa_flash: native Flash GQA masking on A100, without capture")
    parser.add_argument("--attention-scope", choices=("answer_only", "all_decode_tokens"),
                        help="attention analysis scope; defaults to adapter policy; does not change masking or ROUGE")
    data = parser.add_mutually_exclusive_group()
    data.add_argument(
        "--haystack-dir",
        type=Path,
        help="legacy single-case haystack directory (uses the San Francisco needle)",
    )
    data.add_argument(
        "--validation-root",
        type=Path,
        help="directory of cases, each containing corpus.txt and needle.json",
    )
    parser.add_argument("--case-id", action="append",
                        help="select a validation case; repeat to select multiple cases")
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--head-scores", type=Path)
    return parser.parse_args()


def ranking_path(args: argparse.Namespace) -> Path:
    if args.head_scores is not None:
        return args.head_scores
    manifest = args.output_root / "detection" / "aggregation" / "run.json"
    run = read_json(manifest)
    source_path = args.output_root / 'detection' / 'run.json'
    source = read_json(source_path) if source_path.exists() else {}
    if source.get('run_id') and run.get('source_run_id') != source['run_id']:
        raise ValueError('Aggregation belongs to another detection run; rerun aggregation')
    files = run.get("head_score_files", {})
    if len(files) != 1:
        raise ValueError("Explicitly select a ranking JSON with --head-scores; no unique named metric file")
    if not run.get("complete"):
        raise RuntimeError(f"Detection is incomplete: {manifest}. Wait for it to finish.")
    return manifest.parent / next(iter(files.values()))


def load_ranked_heads(args: argparse.Namespace, model) -> list[Head]:
    score_path = ranking_path(args)
    if not score_path.exists():
        raise FileNotFoundError(
            f"Head scores not found: {score_path}. Run detection first."
        )
    eligible = set(model.eligible_heads)
    return [
        head
        for head, _ in rank_heads(read_json(score_path))
        if head in eligible
    ]


def choose_blocked_heads(
    args: argparse.Namespace,
    model,
    ranked: list[Head],
) -> frozenset[Head]:
    count = abs(args.mask_topk)
    if count == 0:
        return frozenset()
    eligible_count = len(model.eligible_heads)
    if count > eligible_count:
        raise ValueError(
            f"Cannot mask {count} heads; model exposes only "
            f"{eligible_count} full-attention heads"
        )
    if args.mask_topk > 0:
        if len(ranked) < count:
            raise ValueError(f"Only {len(ranked)} eligible heads are ranked")
        if getattr(args, "head_selection", "top") == "bottom":
            return frozenset(ranked[-count:])
        return frozenset(ranked[:count])

    exclusion_requested = (
        count
        if args.random_exclusion_top is None
        else args.random_exclusion_top
    )
    if exclusion_requested < 0:
        raise ValueError("random-exclusion-top must not be negative")
    exclusion_count = min(exclusion_requested, eligible_count - count)
    excluded = set(ranked[:exclusion_count])
    candidates = tuple(sorted(set(model.eligible_heads).difference(excluded)))
    if len(candidates) < count:
        raise ValueError(
            f"Top-{exclusion_requested} exclusion leaves "
            f"{len(candidates)} candidates for {count} random heads"
        )
    return frozenset(random.sample(candidates, count))


def main() -> None:
    args = parse_args()
    if args.validation_root is None and args.haystack_dir is None:
        raise ValueError("Use --validation-root or --haystack-dir")
    if args.mask_topk < 0 and args.head_selection == "bottom":
        raise ValueError("bottom selection requires positive --mask-topk; negative means random")
    context_seed = args.context_seed if args.context_seed is not None else args.seed
    if args.validation_root is not None and context_seed is None:
        raise ValueError("Validation corpora require --context-seed or --seed")
    if args.seed is not None:
        random.seed(args.seed)
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
        decode_attention=args.decode_attention,
        attention_scope=args.attention_scope,
    )
    context_builder = ContextBuilder(
        model.tokenizer,
        max_context_length=max(lengths),
        period_tokens=model.period_tokens,
        context_seed=context_seed,
        random_start=args.validation_root is not None,
    )
    runner = ExperimentRunner(
        model,
        context_builder,
        # Masking evaluates only the generated final answer. Locating answer
        # tokens in the prompt is a detection concern and can be brittle at
        # tokenizer boundaries, so it is intentionally skipped here.
        None,
        max_new_tokens=args.max_new_tokens,
    )
    ranked = load_ranked_heads(args, model) if args.mask_topk else []
    if args.validation_root is not None:
        cases = load_validation_cases(args.validation_root)
        if args.case_id is not None:
            missing = set(args.case_id).difference(case.case_id for case in cases)
            if missing:
                raise ValueError(f"Unknown validation cases: {sorted(missing)}")
            cases = [case for case in cases if case.case_id in args.case_id]
        data_source = str(args.validation_root.resolve())
        context_window = "seeded_contiguous_offset"
    else:
        if args.case_id is not None:
            raise ValueError("--case-id requires --validation-root")
        cases = [default_mask_case(args.haystack_dir)]
        data_source = str(args.haystack_dir.resolve())
        context_window = "prefix"

    if args.mask_topk > 0:
        condition = f"{args.head_selection}{args.mask_topk}"
    elif args.mask_topk < 0:
        condition = f"random{-args.mask_topk}"
    else:
        condition = "baseline"

    stable_blocked = (
        choose_blocked_heads(args, model, ranked)
        if args.mask_topk >= 0
        else frozenset()
    )
    condition_dir = args.output_root / "evaluation" / condition
    results_dir = condition_dir / "results"
    contexts_dir = args.output_root / "evaluation" / "contexts"
    run_path = condition_dir / "run.json"
    run_config = {
        "kind": "head_masking",
        "complete": False,
        "condition": condition,
        "model": model.model_id,
        "model_version": model.model_version,
        "adapter": args.adapter,
        "lengths": lengths,
        "depths": depths,
        "mask_mode": args.mask_mode,
        "head_selection": "random" if args.mask_topk < 0 else args.head_selection if args.mask_topk else "baseline",
        "head_scores_file": str(ranking_path(args).resolve())
                            if args.mask_topk else None,
        "blocked_heads": (
            [list(head) for head in sorted(stable_blocked)]
            if args.mask_topk >= 0
            else None
        ),
        "randomized_per_case": args.mask_topk < 0,
        "seed": args.seed,
        "context_seed": context_seed,
        "context_window": context_window,
        "data_source": data_source,
        "case_ids": [case.case_id for case in cases],
        "random_exclusion_top": (
            abs(args.mask_topk)
            if args.mask_topk < 0 and args.random_exclusion_top is None
            else args.random_exclusion_top
        ),
        "max_new_tokens": args.max_new_tokens,
        "prefill_attention": args.prefill_attention,
        "decode_attention": args.decode_attention,
    }
    write_json(run_path, run_config)
    if args.mask_topk >= 0:
        print(
            f"condition={condition} mask_mode={args.mask_mode} "
            f"blocked_heads={sorted(stable_blocked)}"
        )
    else:
        print(
            f"condition={condition} mask_mode={args.mask_mode} "
            "blocked_heads=randomized per case"
        )

    total = len(cases) * len(lengths) * len(depths)
    completed = 0
    for case in cases:
        for context_length in lengths:
            for depth in depths:
                completed += 1
                # Match the source experiment: draw a fresh random control for
                # every case/context/depth tuple. Exact heads are saved in JSON.
                blocked = (
                    choose_blocked_heads(args, model, ranked)
                    if args.mask_topk < 0
                    else stable_blocked
                )
                print(
                    f"[{completed}/{total}] case={case.case_id} "
                    f"context={context_length} depth={depth:g}% "
                    f"blocked={len(blocked)}"
                )
                prepared = runner.prepare(
                    case,
                    context_length=context_length,
                    depth_percent=depth,
                )
                result = runner.run(
                    prepared,
                    attention=AttentionRequest(
                        blocked_heads=blocked,
                        mask_mode=args.mask_mode,
                    ),
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
                            "kind": "head_masking",
                            "prefill_attention": args.prefill_attention,
                            "decode_attention": args.decode_attention,
                            "condition": condition,
                            "mask_mode": args.mask_mode,
                            "blocked_heads": [
                                list(head) for head in sorted(blocked)
                            ],
                            "seed": args.seed,
                            "context_seed": context_seed,
                            "context_window": context_window,
                        },
                    ),
                )
                context_path = contexts_dir / f"{stem}_context.txt"
                context_path.parent.mkdir(parents=True, exist_ok=True)
                context_path.write_text(prepared.context, encoding="utf-8")
                print(
                    f"  score={result.score:.1f} "
                    f"response={result.generation.text!r}"
                )

    run_config["complete"] = True
    run_config["completed_cases"] = completed
    write_json(run_path, run_config)


if __name__ == "__main__":
    main()
