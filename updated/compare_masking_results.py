from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare top retrieval-head masking with random-head masking"
    )
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--model-version", default="Qwen3.5-0.8B")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--success-threshold", type=float, default=50.0)
    return parser.parse_args()


def load_results(directory: Path) -> dict[tuple[int, float], dict[str, Any]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Results directory not found: {directory}")

    results: dict[tuple[int, float], dict[str, Any]] = {}
    for path in sorted(directory.glob("*_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("case_id") != "san-francisco":
            continue
        key = int(payload["context_length"]), float(payload["depth_percent"])
        if key in results:
            raise ValueError(f"Duplicate result for {key}: {path}")
        results[key] = payload
    return results


def main() -> None:
    args = parse_args()
    if args.topk < 1:
        raise ValueError("--topk must be positive")

    graph_root = args.output_root / "results" / "graph"
    top = load_results(
        graph_root / f"{args.model_version}_block_top{args.topk}"
    )
    random = load_results(
        graph_root / f"{args.model_version}_block_random{args.topk}"
    )

    if set(top) != set(random):
        raise ValueError(
            "Top and random runs do not contain the same context/depth pairs"
        )
    if not top:
        raise ValueError("No paired masking results found")

    keys = sorted(top)
    top_scores = [float(top[key]["score"]) for key in keys]
    random_scores = [float(random[key]["score"]) for key in keys]
    deltas = [top_score - random_score for top_score, random_score in zip(top_scores, random_scores)]

    top_successes = sum(score > args.success_threshold for score in top_scores)
    random_successes = sum(score > args.success_threshold for score in random_scores)

    print(f"paired cases:       {len(keys)}")
    print(f"random-{args.topk} mean:      {mean(random_scores):.2f}")
    print(f"retrieval-top-{args.topk} mean: {mean(top_scores):.2f}")
    print(f"mean delta:         {mean(deltas):+.2f}  (top - random)")
    print(
        f"successes > {args.success_threshold:g}: "
        f"random {random_successes}, top {top_successes}"
    )

    by_context: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for key in keys:
        context_length, _ = key
        by_context[context_length].append(
            (float(random[key]["score"]), float(top[key]["score"]))
        )

    print("\ncontext   random      top   top-random")
    for context_length, pairs in sorted(by_context.items()):
        random_mean = mean(pair[0] for pair in pairs)
        top_mean = mean(pair[1] for pair in pairs)
        print(
            f"{context_length:>7}   {random_mean:>6.2f}   "
            f"{top_mean:>6.2f}   {top_mean-random_mean:>+10.2f}"
        )


if __name__ == "__main__":
    main()
