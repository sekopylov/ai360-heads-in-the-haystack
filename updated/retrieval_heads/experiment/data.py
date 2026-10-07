from __future__ import annotations

import json
import random
from pathlib import Path

from .types import ExperimentCase


def linear_grid(start: int, end: int, intervals: int) -> list[int]:
    if intervals < 1:
        raise ValueError("intervals must be positive")
    if start > end:
        raise ValueError("start must not exceed end")
    if intervals == 1:
        return [start]
    return [
        round(start + (end - start) * index / (intervals - 1))
        for index in range(intervals)
    ]


def depth_grid(intervals: int = 10) -> list[int]:
    if intervals < 1:
        raise ValueError("intervals must be positive")
    if intervals == 1:
        return [0]
    return [round(100.0 * index / (intervals - 1)) for index in range(intervals)]


def parse_depths(value: str) -> list[float]:
    depths = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not depths or any(depth < 0 or depth > 100 for depth in depths):
        raise ValueError("depths must contain values from 0 to 100")
    return depths


def parse_lengths(value: str) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("lengths must contain positive integers")
    if len(lengths) != len(set(lengths)):
        raise ValueError("lengths must not contain duplicates")
    return lengths


def load_detection_cases(root: Path) -> list[ExperimentCase]:
    rows = [
        json.loads(line)
        for line in (root / "needles.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    cases: list[ExperimentCase] = []
    for index, row in enumerate(rows, start=1):
        haystack_dir = root / f"part{index}"
        if not haystack_dir.is_dir():
            raise FileNotFoundError(f"Missing haystack directory: {haystack_dir}")
        cases.append(
            ExperimentCase(
                case_id=f"detect-{index}",
                needle=row["needle"],
                question=row["question"],
                expected_answer=row["real_needle"],
                haystack_dir=haystack_dir,
            )
        )
    return cases


def default_mask_case(haystack_dir: Path) -> ExperimentCase:
    return ExperimentCase(
        case_id="san-francisco",
        needle=(
            "\nThe best thing to do in San Francisco is eat a sandwich and sit in "
            "Dolores Park on a sunny day.\n"
        ),
        question="What is the best thing to do in San Francisco?",
        expected_answer="eat a sandwich and sit in Dolores Park on a sunny day",
        haystack_dir=haystack_dir,
    )


class ContextBuilder:
    """Create the author-style haystack context for one tokenizer."""

    def __init__(
        self,
        tokenizer,
        *,
        max_context_length: int,
        period_tokens: list[int],
        final_context_length_buffer: int = 200,
        context_seed: int | None = None,
    ) -> None:
        if max_context_length < 1:
            raise ValueError("max_context_length must be positive")
        if final_context_length_buffer < 0:
            raise ValueError("final_context_length_buffer must not be negative")
        if not period_tokens:
            raise ValueError("period_tokens must not be empty")
        self.tokenizer = tokenizer
        self.context_seed = context_seed
        self.max_context_length = max_context_length
        self.period_tokens = period_tokens
        self.final_context_length_buffer = final_context_length_buffer
        self._tokens: dict[Path, list[int]] = {}

    def _read_repeated_tokens(self, directory: Path) -> list[int]:
        if directory in self._tokens:
            return self._tokens[directory]
        files = sorted(directory.glob("*.txt"))
        if self.context_seed is not None:
            # Local RNG: head sampling and condition execution order cannot
            # change the haystack. Seed the canonical file list anew per corpus.
            random.Random(self.context_seed).shuffle(files)
        if not files:
            raise FileNotFoundError(f"No .txt files found in {directory}")
        base = "".join(path.read_text(encoding="utf-8") for path in files)
        if not base:
            raise ValueError(f"Haystack files in {directory} are empty")

        repetitions = 1
        tokens = self.tokenizer.encode(base, add_special_tokens=False)
        while len(tokens) < self.max_context_length:
            repetitions *= 2
            tokens = self.tokenizer.encode(
                base * repetitions,
                add_special_tokens=False,
            )
        self._tokens[directory] = tokens
        return tokens

    def build(
        self,
        case: ExperimentCase,
        *,
        context_length: int,
        depth_percent: float,
    ) -> str:
        if context_length > self.max_context_length:
            raise ValueError(
                f"context_length exceeds configured maximum "
                f"{self.max_context_length}"
            )
        if not 0 <= depth_percent <= 100:
            raise ValueError("depth_percent must be from 0 to 100")
        context_tokens = self._read_repeated_tokens(case.haystack_dir)
        needle_tokens = self.tokenizer.encode(
            case.needle,
            add_special_tokens=False,
        )
        available = context_length - self.final_context_length_buffer
        if available <= len(needle_tokens):
            raise ValueError("Context is too short for the buffer and needle")

        context_tokens = context_tokens[:context_length]
        if len(context_tokens) + len(needle_tokens) > available:
            context_tokens = context_tokens[: available - len(needle_tokens)]

        if depth_percent == 100:
            result = context_tokens + needle_tokens
        else:
            insertion = int(len(context_tokens) * depth_percent / 100)
            prefix = context_tokens[:insertion]
            while prefix and prefix[-1] not in self.period_tokens:
                insertion -= 1
                prefix = context_tokens[:insertion]
            result = prefix + needle_tokens + context_tokens[insertion:]
        return self.tokenizer.decode(result)
