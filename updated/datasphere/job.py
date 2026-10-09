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
    parser.add_argument("--mask-case-id", action="append",
                        help="select a validation case; repeat to select multiple cases")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--adapter", default="qwen35",
                        help="model adapter registered in retrieval_heads.models")
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
    parser.add_argument("--decode-attention", choices=("eager", "sdpa_flash"), default="eager",
                        help="masking decode backend; detection always uses observed eager")
    parser.add_argument("--attention-scope", choices=("answer_only", "all_decode_tokens"),
                        help="include thinking in retrieval analysis, or analyze only final answer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lengths")
    parser.add_argument("--depths")
    parser.add_argument("--context-seed", type=int,
                        help="haystack seed; defaults to --seed")
    parser.add_argument("--context-count", type=int, default=3)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--mask-selections", default="top,random",
                        help="comma-separated masking strategies: top,bottom,random; baseline always runs")
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


def validation_case_count(root: Path, selected_ids: list[str] | None = None) -> int:
    """Return the number of corpus.txt + needle.json validation directories."""
    count = 0
    found_ids = set()
    if selected_ids and len(selected_ids) != len(set(selected_ids)):
        raise SystemExit("--mask-case-id must not contain duplicates")
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        corpus = case_dir / "corpus.txt"
        needle = case_dir / "needle.json"
        if not corpus.exists() and not needle.exists():
            continue
        if not corpus.is_file() or not needle.is_file():
            raise SystemExit(
                f"Validation case {case_dir} must contain corpus.txt and needle.json"
            )
        if selected_ids:
            case_id = json.loads(needle.read_text(encoding="utf-8"))["case_id"]
            found_ids.add(case_id)
            count += case_id in selected_ids
        else:
            count += 1
    if selected_ids and set(selected_ids).difference(found_ids):
        raise SystemExit(f"Unknown validation cases: {sorted(set(selected_ids).difference(found_ids))}")
    return count


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
    mask_selections = [v.strip() for v in args.mask_selections.split(",")]
    if (not mask_selections or any(v not in {"top", "bottom", "random"} for v in mask_selections)
            or len(mask_selections) != len(set(mask_selections))):
        raise SystemExit("--mask-selections must contain distinct strategies: top,bottom,random")
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
    if args.mask_data is not None and not args.mask_data.is_dir():
        raise SystemExit(f"Mask data directory does not exist: {args.mask_data}")
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
    mask_case_count = (
        validation_case_count(args.mask_data, args.mask_case_id)
        if args.profile in ("full", "mask") else 0
    )
    # A root containing case subdirectories uses the held-out validation
    # format. A directory with plain .txt files remains the legacy one-case
    # San Francisco format.
    if args.profile in ("full", "mask") and mask_case_count == 0:
        mask_case_count = 1
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
        total += mask_case_count * args.context_count * grid_size * (
            1 + len(topks) * (sum(v != "random" for v in mask_selections)
                              + (args.random_repeats if "random" in mask_selections else 0))
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

    if args.profile == "full":
        run("aggregate_head_scores.py", "--run", str(output_root))

    if args.head_scores is not None:
        head_scores = args.head_scores.resolve()
    else:
        detection_dir = output_root / "detection" / "aggregation"
        files = json.loads((detection_dir / "run.json").read_text())["head_score_files"]
        if len(files) != 1:
            raise SystemExit("--head-scores must explicitly select a ranking file")
        head_scores = detection_dir / next(iter(files.values()))
    mask_data_arguments = (
        ("--validation-root", str(args.mask_data.resolve()))
        if validation_case_count(args.mask_data)
        else ("--haystack-dir", str(args.mask_data.resolve()))
    )
    masking_common = (
        *common,
        "--decode-attention", args.decode_attention,
        *grid,
        *mask_data_arguments,
        "--mask-mode",
        "legacy_uniform",
        "--head-scores", str(head_scores),
    )
    for case_id in args.mask_case_id or []:
        masking_common += ("--case-id", case_id)
    for index in range(args.context_count):
        seed = context_seed + index
        context_root = output_root / "evaluation" / f"context-{seed}"
        context_arguments = (*masking_common, "--context-seed", str(seed))
        run("needle_in_haystack_with_mask.py", *context_arguments,
            "--output-root", str(context_root), "--mask-topk", "0")
        for topk in topks:
            for selection in mask_selections:
                if selection != "random":
                    run("needle_in_haystack_with_mask.py", *context_arguments,
                        "--output-root", str(context_root), "--mask-topk", str(topk),
                        "--head-selection", selection)
            for repeat in range(args.random_repeats if "random" in mask_selections else 0):
                repeat_root = context_root / "random" / f"repeat-{repeat}"
                run("needle_in_haystack_with_mask.py", *context_arguments,
                    "--output-root", str(repeat_root), "--mask-topk", str(-topk),
                    "--seed", str(args.seed + repeat))
    print(f"\n[job] {args.profile} experiment artifacts: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
