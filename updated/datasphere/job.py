#!/usr/bin/env python3
"""Run the current Qwen3.5 retrieval-head experiment in one DataSphere job."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROFILES = {
    "smoke": {
        "lengths": "1000",
        "depths": "50",
        "topk": 1,
    },
    "full": {
        "lengths": "6000",
        "depths": "25,50,75",
        "topk": 8,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--detection-data", type=Path, required=True)
    parser.add_argument("--mask-data", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--prefill-attention", default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lengths")
    parser.add_argument("--depths")
    parser.add_argument("--context-seed", type=int,
                        help="haystack seed; defaults to --seed")
    parser.add_argument("--context-count", type=int, default=3)
    parser.add_argument("--random-repeats", type=int, default=3)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--topk", type=int)
    selection.add_argument("--topks", help="comma-separated head counts, e.g. 1,2,4,8,16")
    return parser.parse_args()


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, script, *arguments]
    print(f"\n[job] $ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    profile = PROFILES[args.profile]
    lengths = args.lengths or str(profile["lengths"])
    depths = args.depths or str(profile["depths"])
    topks = (
        [int(value.strip()) for value in args.topks.split(",")]
        if args.topks else [args.topk if args.topk is not None else int(profile["topk"])]
    )
    if any(k < 1 for k in topks) or len(topks) != len(set(topks)):
        raise SystemExit("--topks must contain distinct positive integers")
    if args.context_count < 1 or args.random_repeats < 1:
        raise SystemExit("Context count and random repeats must be positive")
    if args.profile == "full" and args.mask_data is None:
        raise SystemExit("--mask-data is required for the full profile")
    context_seed = args.context_seed if args.context_seed is not None else args.seed
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    common = (
        "--model",
        args.model,
        "--device-map",
        args.device_map,
        "--dtype",
        args.dtype,
        "--prefill-attention",
        args.prefill_attention,
    )
    grid = ("--lengths", lengths, "--depths", depths)

    run(
        "retrieval_head_detection.py",
        *common,
        *grid,
        "--output-root", str(output_root),
        "--context-seed", str(context_seed),
        "--haystack-root",
        str(args.detection_data.resolve()),
    )

    # Smoke proves that the checkpoint, data and observable attention backend
    # work. The full profile additionally performs the paired masking test.
    if args.profile == "smoke":
        print(f"\n[job] smoke artifacts: {output_root}")
        return 0

    if args.mask_data is None:
        raise SystemExit("--mask-data is required for the full profile")

    masking_common = (
        *common,
        *grid,
        "--haystack-dir",
        str(args.mask_data.resolve()),
        "--mask-mode",
        "legacy_uniform",
        "--head-scores", str(output_root / "detection" / "head_scores.json"),
    )
    for index in range(args.context_count):
        seed = context_seed + index
        context_root = output_root / "evaluation" / f"context-{seed}"
        context_arguments = (*masking_common, "--context-seed", str(seed))
        run("needle_in_haystack_with_mask.py", *context_arguments,
            "--output-root", str(context_root), "--mask-topk", "0")
        for topk in topks:
            run("needle_in_haystack_with_mask.py", *context_arguments,
                "--output-root", str(context_root), "--mask-topk", str(topk))
            for repeat in range(args.random_repeats):
                repeat_root = context_root / "random" / f"repeat-{repeat}"
                run("needle_in_haystack_with_mask.py", *context_arguments,
                    "--output-root", str(repeat_root), "--mask-topk", str(-topk),
                    "--seed", str(args.seed + repeat))
    print(f"\n[job] full experiment artifacts: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
