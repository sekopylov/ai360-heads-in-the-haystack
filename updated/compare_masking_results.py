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
        description="Compare baseline, retrieval-head and random-head masking"
    )
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--success-threshold", type=float, default=50.0)
    return parser.parse_args()


def load_condition(
    evaluation_root: Path,
    condition: str,
) -> tuple[dict[str, Any], dict[tuple[int, float], dict[str, Any]]]:
    condition_dir = evaluation_root / condition
    run_path = condition_dir / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"Run metadata not found: {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if not run.get("complete"):
        raise RuntimeError(f"Run is incomplete: {run_path}")

    results: dict[tuple[int, float], dict[str, Any]] = {}
    for path in sorted((condition_dir / "results").glob("*_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = int(payload["context_length"]), float(payload["depth_percent"])
        if key in results:
            raise ValueError(f"Duplicate result for {key}: {path}")
        results[key] = payload
    if not results:
        raise ValueError(f"No results found for condition {condition!r}")
    return run, results


def main() -> None:
    args = parse_args()
    if args.topk < 1:
        raise ValueError("--topk must be positive")

    evaluation_root = args.output_root / "evaluation"
    names = ("baseline", f"top{args.topk}", f"random{args.topk}")
    loaded = {name: load_condition(evaluation_root, name) for name in names}
    baseline_run, baseline = loaded["baseline"]
    top_run, top = loaded[f"top{args.topk}"]
    random_run, random = loaded[f"random{args.topk}"]

    if top_run["mask_mode"] != random_run["mask_mode"]:
        raise ValueError("Top and random runs use different mask modes")
    if not (
        baseline_run["lengths"]
        == top_run["lengths"]
        == random_run["lengths"]
    ):
        raise ValueError("Conditions use different context lengths")
    if not (
        baseline_run["depths"]
        == top_run["depths"]
        == random_run["depths"]
    ):
        raise ValueError("Conditions use different depths")
    if not (set(baseline) == set(top) == set(random)):
        raise ValueError("Conditions do not contain the same context/depth pairs")

    keys = sorted(baseline)
    scores = {
        "baseline": [float(baseline[key]["score"]) for key in keys],
        "top": [float(top[key]["score"]) for key in keys],
        "random": [float(random[key]["score"]) for key in keys],
    }
    averages = {name: mean(values) for name, values in scores.items()}
    successes = {
        name: sum(value > args.success_threshold for value in values)
        for name, values in scores.items()
    }

    print(f"paired cases:          {len(keys)}")
    print(f"mask mode:             {top_run['mask_mode']}")
    print(f"baseline mean:         {averages['baseline']:.2f}")
    print(f"random-{args.topk} mean:         {averages['random']:.2f}")
    print(f"retrieval-top-{args.topk} mean:  {averages['top']:.2f}")
    print(
        f"random - baseline:     "
        f"{averages['random']-averages['baseline']:+.2f}"
    )
    print(
        f"top - baseline:        "
        f"{averages['top']-averages['baseline']:+.2f}"
    )
    print(f"top - random:          {averages['top']-averages['random']:+.2f}")
    print(
        f"successes > {args.success_threshold:g}: "
        f"baseline {successes['baseline']}, "
        f"random {successes['random']}, top {successes['top']}"
    )
    print(f"top heads:             {top_run['blocked_heads']}")
    if random_run.get("randomized_per_case"):
        print("random heads:          randomized per case (stored in result JSON)")
    else:
        print(f"random heads:          {random_run['blocked_heads']}")

    by_context: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for key in keys:
        context_length, _ = key
        by_context[context_length].append(
            (
                float(baseline[key]["score"]),
                float(random[key]["score"]),
                float(top[key]["score"]),
            )
        )

    print("\ncontext   baseline   random      top   top-random")
    for context_length, triples in sorted(by_context.items()):
        baseline_mean = mean(item[0] for item in triples)
        random_mean = mean(item[1] for item in triples)
        top_mean = mean(item[2] for item in triples)
        print(
            f"{context_length:>7}   {baseline_mean:>8.2f}   "
            f"{random_mean:>6.2f}   {top_mean:>6.2f}   "
            f"{top_mean-random_mean:>+10.2f}"
        )


if __name__ == "__main__":
    main()
