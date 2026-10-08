#!/usr/bin/env python3
"""Run detection, paired masking, or both in one DataSphere job."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
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
    "mask": {
        "lengths": "6000",
        "depths": "25,50,75",
        "topk": 8,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--detection-data", type=Path)
    parser.add_argument("--mask-data", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--adapter", default="qwen35",
                        help="model architecture adapter: qwen35 or qwen3")
    parser.add_argument("--model-search-dir", action="append",
                        help="explicit root containing MODEL_ID; repeat for ordered search; no default search directory")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    metrics = parser.add_mutually_exclusive_group()
    metrics.add_argument("--retrieval-metric", choices=("legacy", "needle_token_multiset_v1", "needle_attention_mass_v1"),
                        default=None)
    metrics.add_argument("--retrieval-metrics", help="comma-separated retrieval metrics to compute together")
    parser.add_argument("--head-scores", type=Path,
                        help="ranking JSON; required for mask and full multi-metric runs")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--prefill-attention", default="sdpa")
    parser.add_argument("--attention-scope", choices=("answer_only", "all_decode_tokens"),
                        help="include thinking in retrieval analysis, or analyze only final answer")
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


@dataclass
class Progress:
    total: int
    completed: int = 0

    def report(self) -> None:
        print(
            f"[progress] {self.completed}/{self.total} generations "
            f"({100 * self.completed / self.total:.1f}%)",
            flush=True,
        )

    def advance(self) -> None:
        self.completed += 1
        self.report()


def run_command(script: str, *arguments: str, progress: Progress) -> None:
    # Stream output immediately, including when DataSphere redirects it to logs.
    command = [sys.executable, "-u", script, *arguments]
    print(f"\n[job] $ {' '.join(command)}", flush=True)
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            # Both experiment CLIs emit exactly one score line per finished case.
            if line.lstrip().startswith("score="):
                progress.advance()
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


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
    if args.profile != "mask" and args.detection_data is None:
        raise SystemExit("--detection-data is required for smoke/full profiles")
    if args.profile in ("full", "mask") and args.mask_data is None:
        raise SystemExit("--mask-data is required for full/mask profiles")
    if args.profile == "mask":
        if args.head_scores is None:
            raise SystemExit("--head-scores is required for the mask profile")
        if not args.head_scores.is_file():
            raise SystemExit(f"Ranking file does not exist: {args.head_scores}")
        if args.retrieval_metric is not None or args.retrieval_metrics is not None:
            raise SystemExit("Retrieval metrics are computed by detection, not the mask profile; use --head-scores")
    if (args.profile == "full" and args.retrieval_metrics is not None
            and len(args.retrieval_metrics.split(",")) > 1 and args.head_scores is None):
        raise SystemExit("--head-scores must explicitly select a ranking file for full multi-metric runs")
    context_seed = args.context_seed if args.context_seed is not None else args.seed
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    import torch
    runtime = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
               "cuda_available": torch.cuda.is_available()}
    if runtime["cuda_available"]:
        runtime["gpu"] = torch.cuda.get_device_name(0)
        runtime["capability"] = list(torch.cuda.get_device_capability(0))
        runtime["compiled_architectures"] = torch.cuda.get_arch_list()
    print(f"[job] runtime: {json.dumps(runtime)}", flush=True)
    (output_root / "runtime.json").write_text(json.dumps(runtime, indent=2))
    if args.device_map.startswith("cuda"):
        if not runtime["cuda_available"]:
            raise RuntimeError("GPU requested but PyTorch CUDA is unavailable; check build and NVIDIA driver")
        # Exercise a real kernel before loading weights or processing long inputs.
        probe = torch.ones((32, 32), device=args.device_map)
        probe @ probe
        torch.cuda.synchronize()

    common = (
        "--adapter",
        args.adapter,
        "--model",
        args.model,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--device-map",
        args.device_map,
        "--dtype",
        args.dtype,
        "--prefill-attention",
        args.prefill_attention,
    )
    for directory in args.model_search_dir or []:
        common += ("--model-search-dir", directory)
    if args.attention_scope is not None:
        common += ("--attention-scope", args.attention_scope)
    grid = ("--lengths", lengths, "--depths", depths)

    grid_size = len([v for v in lengths.split(",") if v.strip()]) * len(
        [v for v in depths.split(",") if v.strip()]
    )
    detection_cases = 0
    if args.profile != "mask":
        detection_cases = sum(
            bool(line)
            for line in (args.detection_data / "needles.jsonl")
            .read_text(encoding="utf-8").splitlines()
        )
    total = detection_cases * grid_size
    if args.profile in ("full", "mask"):
        total += args.context_count * grid_size * (
            1 + len(topks) * (1 + args.random_repeats)
        )
    if total < 1:
        raise SystemExit("Experiment must contain at least one case")
    progress = Progress(total)
    progress.report()
    run = partial(run_command, progress=progress)

    if args.profile != "mask":
        run(
            "retrieval_head_detection.py",
            *(("--retrieval-metric", args.retrieval_metric) if args.retrieval_metric else ()),
            *(("--retrieval-metrics", args.retrieval_metrics) if args.retrieval_metrics is not None else ()),
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

    masking_common = (
        *common,
        *grid,
        "--haystack-dir",
        str(args.mask_data.resolve()),
        "--mask-mode",
        "legacy_uniform",
        "--head-scores", str(args.head_scores.resolve() if args.head_scores is not None
                             else output_root / "detection" / "head_scores.json"),
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
    print(f"\n[job] {args.profile} experiment artifacts: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
