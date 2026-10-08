from __future__ import annotations

from typing import Any

import torch

from .collectors import NullCollector
from .types import AttentionObserver, AttentionRequest, AttentionStep, Head


class AttentionController:
    """Mutable per-run attention state owned by a model adapter.

    The adapter calls ``begin_step`` before a decode forward. Each full
    attention layer then calls ``record``. Once Qwen predicts the next token,
    ``end_step`` publishes one complete snapshot to the observer.
    """

    def __init__(self, expected_layers: tuple[int, ...] = ()) -> None:
        self.expected_layers = frozenset(expected_layers)
        self.request = AttentionRequest()
        self._observer: AttentionObserver = NullCollector()
        self._layers: dict[int, Any] = {}
        self._needle_mass: dict[int, Any] = {}
        self._step_index = 0
        self._active = False
        self._step_open = False

    def start(
        self,
        request: AttentionRequest,
        observer: AttentionObserver | None = None,
    ) -> None:
        if self._active:
            raise RuntimeError("AttentionController is already active")
        if request.needle_span is not None:
            start, end = request.needle_span
            if not 0 <= start < end:
                raise ValueError("Needle span must be nonempty and nonnegative")
        self.request = request
        self._observer = observer or NullCollector()
        self._layers = {}
        self._needle_mass = {}
        self._step_index = 0
        self._active = True
        self._step_open = False

    def finish(self) -> None:
        self.request = AttentionRequest()
        self._observer = NullCollector()
        self._layers = {}
        self._needle_mass = {}
        self._active = False
        self._step_open = False

    def begin_step(self) -> None:
        if not self._active:
            raise RuntimeError("AttentionController has not been started")
        if self._step_open:
            raise RuntimeError("Previous attention step has not been ended")
        self._layers = {}
        self._needle_mass = {}
        self._step_open = True

    def blocked_heads(self, layer_idx: int) -> list[int]:
        return [
            head
            for layer, head in self.request.blocked_heads
            if layer == layer_idx
        ]

    def record(self, layer_idx: int, probabilities: torch.Tensor) -> None:
        if not self._step_open:
            raise RuntimeError("No attention step is active")
        if self.request.capture == "none" and self.request.needle_span is None:
            return

        last_query = probabilities[0, :, -1, :].detach()
        if self.request.needle_span is not None:
            start, end = self.request.needle_span
            if end > last_query.shape[-1]:
                raise ValueError("Needle span exceeds attention key length")
            self._needle_mass[layer_idx] = last_query[:, start:end].sum(
                dim=-1, dtype=torch.float32).to(device="cpu")
        if self.request.capture == "none":
            return
        if self.request.capture == "top1":
            value = last_query.argmax(dim=-1).to(device="cpu")
        elif self.request.capture == "full":
            # Keep the model's original probability dtype; bf16/fp16 both use
            # two bytes, and avoiding a conversion preserves the exact values.
            value = last_query.to(device="cpu").clone()
        else:
            raise ValueError(f"Unknown capture mode: {self.request.capture!r}")
        self._layers[layer_idx] = value

    def end_step(self, token_id: int, *, publish: bool = True) -> None:
        if not self._step_open:
            raise RuntimeError("No attention step is active")
        if self.request.capture != "none" or self.request.needle_span is not None:
            missing = self.expected_layers.difference(
                self._layers if self.request.capture != "none" else self._needle_mass)
            if self.request.needle_span is not None:
                missing |= self.expected_layers.difference(self._needle_mass)
            if missing:
                raise RuntimeError(
                    f"Attention backend did not report layers: {sorted(missing)}"
                )
            if publish:
                self._observer.on_step(
                    AttentionStep(
                        index=self._step_index,
                        token_id=token_id,
                        layers=dict(self._layers),
                        needle_attention_mass=dict(self._needle_mass),
                    )
                )
        self._step_index += 1
        self._step_open = False
        self._layers = {}
        self._needle_mass = {}


def validate_blocked_heads(
    requested: frozenset[Head],
    eligible: tuple[Head, ...],
) -> None:
    invalid = requested.difference(eligible)
    if invalid:
        raise ValueError(f"Heads are not available for masking: {sorted(invalid)}")
