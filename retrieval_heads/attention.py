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


def set_attn_implementation(model: nn.Module, impl: str) -> dict[int, str]:
    """Force every config shared by the model's modules onto ``impl``.

    Returns ``{id(config): previous_impl}`` so :func:`restore_attn_implementation`
    can put each config back to *its own* value.  (Returning a single value for all
    of them pinned every config to the first one's implementation on restore.)
    """
    previous: dict[int, str] = {}
    for cfg in _collect_configs(model):
        if hasattr(cfg, "_attn_implementation"):
            previous[id(cfg)] = cfg._attn_implementation
            cfg._attn_implementation = impl
    return previous


def restore_attn_implementation(model: nn.Module, previous: dict[int, str]) -> None:
    """Put each config back to the implementation it had before the set call."""
    for cfg in _collect_configs(model):
        # `previous` may legitimately hold None ("not set"); `get(...) is not None`
        # skipped those and left the forced implementation in place for good.
        if id(cfg) in previous:
            cfg._attn_implementation = previous[id(cfg)]


# --------------------------------------------------------------------------- capture
@dataclass(eq=False, repr=False)
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
                    expected = self.info.num_heads.get(layer)
                    if expected is not None and tensor.shape[1] != expected:
                        raise RuntimeError(
                            f"attention map #{layer} has {tensor.shape[1]} heads but the "
                            f"model reports {expected} for that layer; the returned maps "
                            f"are probably ordered differently from scoreable_layers. Use "
                            f"capture_method='patch' (keyed by layer_idx) to be sure."
                        )
                    store[layer] = tensor
            # method="patch" fills `store` from inside the wrapper instead.
        if self.method == "patch":
            missing = [layer for layer in self.info.scoreable_layers if layer not in store]
            if missing:
                raise RuntimeError(
                    f"the eager_attention_forward patch captured nothing for layers "
                    f"{missing}; the model is probably running a non-eager kernel, so "
                    f"every retrieval score would silently be 0. Set capture_impl='eager' "
                    f"or use method='output_attentions'."
                )
        return out, store

    # -- patch fallback -----------------------------------------------------
    @contextmanager
    def _patched_eager(self, store: dict[int, torch.Tensor]) -> Iterator[None]:
        """Wrap the modeling module's ``eager_attention_forward`` for the duration.

        The attention blocks resolve ``eager_attention_forward`` as a module
        global at call time, so patching that global is enough -- and it is
        restored exactly, once per modelling module, even on an exception.
        """
        # Only the modules this model actually scores may write into `store`: a
        # foreign attention block (vision tower, second attention class) can share
        # a `layer_idx` and would otherwise overwrite a scored layer's row.
        known_modules = {id(module) for module in self.info.attention_modules.values()}
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
                        dropout=0.0, _orig=original, _known=known_modules, **kwargs):
                out, weights = _orig(module_, query, key, value, attention_mask,
                                     scaling, dropout, **kwargs)
                if weights is not None and id(module_) in _known:
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
        # `model` is accepted for symmetry with TokenMixerMasker but not needed: the
        # hooks go on the modules held by `info`.
        super().__init__()
        self.info = info
        self.heads = list(heads)
        # `info` can come from `ModelInfo.from_dict`, i.e. from a different model
        # instance; masking the wrong modules would silently do nothing useful.
        self._model_module_ids = ({id(m) for m in model.modules()}
                                  if isinstance(model, nn.Module) else None)
        self._install()

    def _install(self) -> None:
        by_layer: dict[int, list[int]] = {}
        for head in self.heads:
            n_heads = self.info.num_heads.get(head.layer)
            if head.head < 0 or (n_heads is not None and head.head >= n_heads):
                # A negative index would silently zero the *last* head of the layer
                # (`view[..., -1, :]`), i.e. mask a head nobody asked for.
                raise KeyError(
                    f"head {head} is out of range: layer {head.layer} has {n_heads} heads"
                )
            module = self.info.attention_modules.get(head.layer)
            if module is None:
                raise KeyError(
                    f"layer {head.layer} has no scoreable attention module "
                    f"(scoreable layers: {self.info.scoreable_layers})"
                )
            if self._model_module_ids is not None and id(module) not in self._model_module_ids:
                raise KeyError(
                    f"layer {head.layer}'s attention module does not belong to the model "
                    f"passed to HeadMasker; the ModelInfo probably came from another "
                    f"checkpoint"
                )
            by_layer.setdefault(head.layer, []).append(head.head)

        # Resolve head_dim for every layer *before* installing anything: `_head_dim`
        # can raise on a gated/MLA block, and it used to run inside the registration
        # loop, leaving the earlier layers' hooks live with no masker object left to
        # remove them.
        resolved: list[tuple[nn.Module, int, torch.Tensor, int | None]] = []
        for layer, head_ids in by_layer.items():
            module = self.info.attention_modules[layer]
            head_dim = _head_dim(module)
            index = torch.tensor(sorted(set(head_ids)), dtype=torch.long)
            resolved.append((module, head_dim, index, self.info.num_heads.get(layer)))

        for module, head_dim, index, num_heads in resolved:
            handle = module.o_proj.register_forward_pre_hook(
                self._make_hook(index, head_dim, num_heads), with_kwargs=True
            )
            self.add(handle)

    @staticmethod
    def _make_hook(head_index: torch.Tensor, head_dim: int, num_heads: int | None = None):
        def pre_hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            # `o_proj` is normally called positionally, but a keyword call must not
            # blow up with an IndexError on `args[0]`.
            hidden = args[0] if args else kwargs.get("input", kwargs.get("hidden_states"))
            if hidden is None:
                raise RuntimeError(
                    "HeadMasker could not find the o_proj input tensor; pass it "
                    "positionally or as kwargs['input']"
                )
            if hidden.shape[-1] % head_dim != 0:
                raise RuntimeError(
                    f"o_proj input width {hidden.shape[-1]} is not a multiple of head_dim {head_dim}"
                )
            if num_heads is not None and hidden.shape[-1] != num_heads * head_dim:
                raise RuntimeError(
                    f"o_proj input width {hidden.shape[-1]} != num_heads {num_heads} x "
                    f"head_dim {head_dim}; _head_dim's fallback is wrong for this "
                    f"architecture (gated/MLA attention?), so masking would cut the "
                    f"wrong slice"
                )
            masked = hidden.clone()
            # `clone()` keeps strides, and `reshape` on a non-contiguous tensor
            # returns a *copy*: the in-place zeroing below would then be discarded
            # and the mask would silently do nothing.  Force contiguity and prove
            # the view shares storage.
            if masked.stride(-1) != 1:
                masked = masked.contiguous()
            view = masked.reshape(*masked.shape[:-1], -1, head_dim)
            if view.data_ptr() != masked.data_ptr():
                raise RuntimeError(
                    "o_proj input could not be viewed as (..., heads, head_dim) without "
                    "copying; masking would silently do nothing"
                )
            view[..., head_index, :] = 0
            if args:
                return (masked,) + args[1:], kwargs
            new_kwargs = dict(kwargs)
            new_kwargs["input" if "input" in kwargs else "hidden_states"] = masked
            return args, new_kwargs

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
        # Resolve *every* module before registering anything: the previous version
        # registered as it went, so a missing layer left the earlier hooks live and
        # every later forward pass ran with those layers silently zeroed.
        model_module_ids = ({id(m) for m in model.modules()}
                            if isinstance(model, nn.Module) else None)
        resolved = []
        for layer in self.layers:
            # `or` on an nn.Module would be wrong for a module whose __len__ is 0.
            module = (info.attention_modules[layer] if layer in info.attention_modules
                      else info.linear_modules.get(layer))
            if module is None:
                raise KeyError(f"no token mixer found for layer {layer}")
            if model_module_ids is not None and id(module) not in model_module_ids:
                raise KeyError(
                    f"layer {layer}'s token-mixer module does not belong to the model "
                    f"passed to TokenMixerMasker; the ModelInfo probably came from "
                    f"another checkpoint"
                )
            resolved.append(module)
        for module in resolved:
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
