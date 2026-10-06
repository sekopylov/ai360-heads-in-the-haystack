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
    parse_depths,
)
from retrieval_heads.experiment.locator import LegacyOverlapLocator
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
REPO_ROOT = HERE.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Needle test with head masking")
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
        default=128000,
    )
    parser.add_argument("--context-intervals", type=int, default=40)
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
    parser.add_argument("--random-exclusion-top", type=int, default=100)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prefill-attention", default="flash_attention_2")
    parser.add_argument(
        "--haystack-dir",
        type=Path,
        default=REPO_ROOT / "source" / "PaulGrahamEssays",
    )
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--head-scores", type=Path)
    return parser.parse_args()


def load_ranked_heads(args: argparse.Namespace, model) -> list[Head]:
    score_path = args.head_scores or (
        args.output_root / "head_score" / f"{model.model_version}.json"
    )
    if not score_path.exists():
        raise FileNotFoundError(
            f"Head scores not found: {score_path}. Run detection first."
        )
    eligible = set(model.eligible_heads)
    return [
        head
        for head, _ in rank_heads(read_json(score_path))
        if head in eligible
    ][:100]


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
        return frozenset(ranked[:count])

    # Qwen3.5-0.8B has only 48 ordinary attention heads. Keep the source
    # top-100 exclusion when possible, but never exclude the entire pool.
    if args.random_exclusion_top < 0:
        raise ValueError("random-exclusion-top must not be negative")
    exclusion_count = min(args.random_exclusion_top, eligible_count - count)
    excluded = set(ranked[:exclusion_count])
    candidates = tuple(sorted(set(model.eligible_heads).difference(excluded)))
    if len(candidates) < count:
        raise ValueError(
            f"Legacy top-{args.random_exclusion_top} exclusion leaves "
            f"{len(candidates)} candidates for {count} random heads"
        )
    return frozenset(random.sample(candidates, count))


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    lengths = linear_grid(args.min_context, args.max_context, args.context_intervals)
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
    )
    runner = ExperimentRunner(
        model,
        context_builder,
        LegacyOverlapLocator(model.tokenizer, threshold=0.9),
        max_new_tokens=args.max_new_tokens,
    )
    ranked = load_ranked_heads(args, model) if args.mask_topk else []
    case = default_mask_case(args.haystack_dir)

    if args.mask_topk > 0:
        run_name = f"{model.model_version}_block_top{args.mask_topk}"
    elif args.mask_topk < 0:
        run_name = f"{model.model_version}_block_random{-args.mask_topk}"
    else:
        run_name = model.model_version
    results_dir = args.output_root / "results" / "graph" / run_name

    total = len(lengths) * len(depths)
    completed = 0
    for context_length in lengths:
        for depth in depths:
            completed += 1
            blocked = choose_blocked_heads(args, model, ranked)
            print(
                f"[{completed}/{total}] context={context_length} "
                f"depth={depth:g}% blocked={len(blocked)}"
            )
            prepared = runner.prepare(
                case,
                context_length=context_length,
                depth_percent=depth,
            )
            result = runner.run(
                prepared,
                attention=AttentionRequest(blocked_heads=blocked),
            )
            stem = result_stem(
                model.model_version,
                context_length,
                depth,
                case_id=case.case_id,
            )
            write_json(
                results_dir / f"{stem}_results.json",
                result_payload(result, model.model_id),
            )
            print(
                f"  score={result.score:.1f} "
                f"response={result.generation.text!r}"
            )


if __name__ == "__main__":
    main()
