"""Attention-map capture and head/token-mixer ablation.

Two independent capabilities live here:

* :class:`AttentionRecorder` -- run a forward pass and get back the softmax
  attention probabilities of every *scoreable* layer, keyed by absolute layer
  index.  Only the decoding steps need this: during a cached decode ``q_len`` is
  1, so each map is ``(batch, heads, 1, kv_len)`` and costs almost nothing even
  at very long context.  The prefill can therefore run on a cheap kernel.

* :class:`HeadMasker` / :class:`TokenMixerMasker` -- "mask out" a head or a whole
  linear-attention layer.  Masking a head is implemented by zeroing that head's
  slice of the tensor entering ``o_proj``.  Because attention output for head
  ``h`` only ever touches ``o_proj``'s input slice ``[h*d : (h+1)*d]``, zeroing
  the input is exactly equivalent to zeroing the head's attention row, and it
  needs no surgery on the attention kernel itself.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from torch import nn

from retrieval_heads.models import ModelInfo, _head_dim
from retrieval_heads.utils import HeadRef, get_logger

log = get_logger("attention")


# --------------------------------------------------------------------------- impl switching
def _collect_configs(model: nn.Module) -> list[Any]:
    configs, seen = [], set()
    candidates = [getattr(model, "config", None)]
    candidates += [getattr(m, "config", None) for m in model.modules()]
    for cfg in candidates:
        if cfg is not None and id(cfg) not in seen:
            seen.add(id(cfg))
            configs.append(cfg)
    return configs


def set_attn_implementation(model: nn.Module, impl: str) -> str | None:
    """Force every config shared by the model's modules onto ``impl``.

    Returns the implementation that was in place before the call, so callers can
    restore it.  Attention modules read ``self.config._attn_implementation`` at
    call time, so flipping this between the prefill and the decode is safe.
    """
    previous = None
    for cfg in _collect_configs(model):
        if hasattr(cfg, "_attn_implementation"):
            if previous is None:
                previous = cfg._attn_implementation
            cfg._attn_implementation = impl
    return previous


# --------------------------------------------------------------------------- capture
@dataclass
class AttentionRecorder:
    """Captures per-layer attention probabilities for scoreable layers.

    ``method="output_attentions"`` uses the public Transformers API; the
    recorded tuple is mapped onto scoreable layers in ascending layer order.
    ``method="patch"`` wraps the model's own ``eager_attention_forward`` and keys
    results by ``layer_idx`` directly.  The patch variant is the safety net for
    architectures whose ``_can_record_outputs`` does not cover the module.
    """

    model: nn.Module
    info: ModelInfo
    method: str = "output_attentions"

    # -- public API ---------------------------------------------------------
    @contextmanager
    def _active(self) -> Iterator[dict[int, torch.Tensor]]:
        store: dict[int, torch.Tensor] = {}
        if self.method == "output_attentions":
            yield store
        elif self.method == "patch":
            with self._patched_eager(store):
                yield store
        else:
            raise ValueError(f"unknown capture method: {self.method!r}")

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = True,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[Any, dict[int, torch.Tensor]]:
        """One decode-step forward; returns ``(outputs, {layer: attn_probs})``.

        This **advances** ``past_key_values`` by one position, as any decode step
        does.  Pass a freshly prefilled cache per call: handing the same cache to
        two captures compares rows of different lengths (a subtle way to get a
        shape mismatch that looks like a bug in the model).
        """
        with self._active() as store:
            out = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                attention_mask=attention_mask,
                output_attentions=self.method == "output_attentions",
            )
            if self.method == "output_attentions":
                attn = getattr(out, "attentions", None)
                if attn is None:
                    raise RuntimeError(
                        "model returned no attentions; use method='patch' or "
                        "attn_implementation='eager'"
                    )
                layers = self.info.scoreable_layers
                if len(attn) != len(layers):
                    raise RuntimeError(
                        f"expected {len(layers)} attention maps for scoreable layers "
                        f"{layers}, got {len(attn)}"
                    )
                for layer, tensor in zip(layers, attn):
                    store[layer] = tensor
            # method="patch" fills `store` from inside the wrapper instead.
        return out, store

    # -- patch fallback -----------------------------------------------------
    @contextmanager
    def _patched_eager(self, store: dict[int, torch.Tensor]) -> Iterator[None]:
        """Wrap the modeling module's ``eager_attention_forward`` for the duration.

        The attention blocks resolve ``eager_attention_forward`` as a module
        global at call time, so patching that global is enough -- and it is
        restored exactly, once per modelling module, even on an exception.
        """
        patched: dict[int, tuple[Any, Any]] = {}  # id(module) -> (module, original)
        for module in self.info.attention_modules.values():
            modeling = sys.modules.get(type(module).__module__)
            if modeling is None or id(modeling) in patched:
                continue
            original = getattr(modeling, "eager_attention_forward", None)
            if original is None:
                continue
            patched[id(modeling)] = (modeling, original)

            def wrapper(module_, query, key, value, attention_mask, scaling,
                        dropout=0.0, _orig=original, **kwargs):
                out, weights = _orig(module_, query, key, value, attention_mask,
                                     scaling, dropout, **kwargs)
                if weights is not None:
                    store[int(getattr(module_, "layer_idx", -1))] = weights.detach()
                return out, weights

            modeling.eager_attention_forward = wrapper

        if not patched:
            raise RuntimeError(
                "could not locate a module-level eager_attention_forward to patch; "
                "use method='output_attentions'"
            )
        try:
            yield
        finally:
            for modeling, original in patched.values():
                modeling.eager_attention_forward = original


# --------------------------------------------------------------------------- masking
class _HookGroup:
    def __init__(self) -> None:
        self.handles: list[Any] = []

    def add(self, handle: Any) -> None:
        self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "_HookGroup":
        return self

    def __exit__(self, *exc: object) -> None:
        self.remove()


class HeadMasker(_HookGroup):
    """Mask out individual attention heads by zeroing their ``o_proj`` input slice.

    This reproduces the paper's "completely pruning retrieval heads": the head's
    attention row is replaced by zeros, so the head contributes nothing to the
    residual stream.  Non-retrieval heads are untouched.
    """

    def __init__(self, model: nn.Module, info: ModelInfo, heads: Sequence[HeadRef]) -> None:
        super().__init__()
        self.model = model
        self.info = info
        self.heads = list(heads)
        self._install()

    def _install(self) -> None:
        by_layer: dict[int, list[int]] = {}
        for head in self.heads:
            module = self.info.attention_modules.get(head.layer)
            if module is None:
                raise KeyError(
                    f"layer {head.layer} has no scoreable attention module "
                    f"(scoreable layers: {self.info.scoreable_layers})"
                )
            by_layer.setdefault(head.layer, []).append(head.head)

        for layer, head_ids in by_layer.items():
            module = self.info.attention_modules[layer]
            head_dim = _head_dim(module)
            index = torch.tensor(sorted(set(head_ids)), dtype=torch.long)
            handle = module.o_proj.register_forward_pre_hook(self._make_hook(index, head_dim))
            self.add(handle)

    @staticmethod
    def _make_hook(head_index: torch.Tensor, head_dim: int):
        def pre_hook(module: nn.Module, args: tuple[Any, ...]):
            hidden = args[0]
            if hidden.shape[-1] % head_dim != 0:
                raise RuntimeError(
                    f"o_proj input width {hidden.shape[-1]} is not a multiple of head_dim {head_dim}"
                )
            masked = hidden.clone()
            view = masked.view(*masked.shape[:-1], -1, head_dim)
            view[..., head_index, :] = 0
            return (masked,) + args[1:]

        return pre_hook


class TokenMixerMasker(_HookGroup):
    """Zero the output of whole token-mixer layers (linear/recurrent, or attention).

    Used for the section-5 style control on hybrid models: if full attention is
    what makes retrieval possible, silencing a Gated DeltaNet layer should hurt
    far less than silencing a Gated Attention layer.
    """

    def __init__(self, model: nn.Module, info: ModelInfo, layers: Sequence[int]) -> None:
        super().__init__()
        self.info = info
        self.layers = list(layers)
        for layer in self.layers:
            module = info.attention_modules.get(layer) or info.linear_modules.get(layer)
            if module is None:
                raise KeyError(f"no token mixer found for layer {layer}")
            self.add(module.register_forward_hook(self._zero()))

    @staticmethod
    def _zero():
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any):
            if isinstance(output, tuple):
                first = output[0]
                return (torch.zeros_like(first),) + tuple(output[1:])
            return torch.zeros_like(output)

        return hook


@contextmanager
def masked_heads(model: nn.Module, info: ModelInfo, heads: Sequence[HeadRef]) -> Iterator[None]:
    """Convenience wrapper: ``with masked_heads(model, info, heads): ...``"""
    masker = HeadMasker(model, info, heads)
    try:
        yield
    finally:
        masker.remove()


@contextmanager
def masked_token_mixers(model: nn.Module, info: ModelInfo, layers: Sequence[int]) -> Iterator[None]:
    masker = TokenMixerMasker(model, info, layers)
    try:
        yield
    finally:
        masker.remove()
