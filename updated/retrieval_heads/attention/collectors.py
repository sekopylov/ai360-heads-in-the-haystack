from __future__ import annotations

from pathlib import Path

import torch

from .types import AttentionObserver, AttentionStep


class NullCollector:
    def on_step(self, step: AttentionStep) -> None:
        pass


class FullTraceCollector:
    """Keep every received decode-attention step in CPU memory."""

    def __init__(self, prompt_token_ids: list[int]) -> None:
        self.prompt_token_ids = list(prompt_token_ids)
        self.steps: list[AttentionStep] = []

    def on_step(self, step: AttentionStep) -> None:
        self.steps.append(step)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "prompt_token_ids": self.prompt_token_ids,
            "steps": [
                {
                    "index": step.index,
                    "token_id": step.token_id,
                    "layers": step.layers,
                }
                for step in self.steps
            ],
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)


class CompositeCollector:
    """Send each attention step to several independent consumers."""

    def __init__(self, *collectors: AttentionObserver) -> None:
        self.collectors = collectors

    def on_step(self, step: AttentionStep) -> None:
        for collector in self.collectors:
            collector.on_step(step)
