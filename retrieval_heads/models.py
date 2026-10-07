"""Model loading and *architecture-aware* head discovery.

Why this module exists
----------------------
The paper measures, for every attention head, the probability mass its softmax
attention map puts on the needle.  That quantity simply does not exist for a
linear/recurrent token mixer: a Gated DeltaNet layer keeps a running state and
never builds a ``[query x context]`` matrix.

Qwen3.5-0.8B is exactly such a hybrid (24 layers = 18 Gated DeltaNet +
6 Gated Attention), so naively iterating "all layers x all heads" would either
crash or silently invent numbers.  Instead we *discover* which modules are
scoreable by looking for a softmax attention signature (``q_proj``/``k_proj``/
``v_proj``/``o_proj`` plus a head count), and record everything else as a
non-scoreable token mixer.  Dense transformers such as Qwen3-0.6B simply yield
all their layers as scoreable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from retrieval_heads.utils import HeadRef, get_logger

log = get_logger("models")

#: Module attributes that together identify a softmax multi-head attention block.
_ATTENTION_SIGNATURE = ("q_proj", "k_proj", "v_proj", "o_proj")

#: Recognised recurrent / linear token mixers (kept for reporting + ablation).
_LINEAR_MARKERS = ("DeltaNet", "LinearAttention", "Mamba", "Recurrent", "SSM", "RWKV")


def text_config(config: Any) -> Any:
    """Return the language-model config, descending through VL wrappers."""
    seen = set()
    while hasattr(config, "text_config") and id(config) not in seen:
        seen.add(id(config))
        config = config.text_config
    return config


def _is_attention_module(module: nn.Module) -> bool:
    return all(hasattr(module, attr) for attr in _ATTENTION_SIGNATURE)


def _looks_linear(module: nn.Module) -> bool:
    return any(marker in type(module).__name__ for marker in _LINEAR_MARKERS)


def _head_dim(module: nn.Module) -> int:
    if hasattr(module, "head_dim"):
        return int(module.head_dim)
    return int(module.q_proj.out_features // module.num_heads)


@dataclass
class ModelInfo:
    """Static description of where retrieval can possibly live inside a model."""

    name: str
    path: str
    model_type: str
    num_layers: int
    layer_types: list[str]
    num_heads: dict[int, int]
    num_kv_heads: dict[int, int]
    head_dim: int
    hidden_size: int
    max_position_embeddings: int | None
    attention_modules: dict[int, nn.Module] = field(repr=False, default_factory=dict)
    linear_modules: dict[int, nn.Module] = field(repr=False, default_factory=dict)
    module_names: dict[int, str] = field(default_factory=dict)
    #: Explicit layer lists, so a ModelInfo can be rebuilt from JSON without
    #: loading any weights (needed by the plotting / comparison entry points).
    scoreable_layers_: list[int] | None = None
    linear_layers_: list[int] | None = None

    # ------------------------------------------------------------------ queries
    @property
    def scoreable_layers(self) -> list[int]:
        """Absolute layer indices whose token mixer is a softmax attention block."""
        if self.scoreable_layers_ is not None:
            return list(self.scoreable_layers_)
        return sorted(self.attention_modules)

    @property
    def linear_layers(self) -> list[int]:
        """Absolute layer indices whose token mixer is linear/recurrent."""
        if self.linear_layers_ is not None:
            return list(self.linear_layers_)
        return sorted(self.linear_modules)

    @property
    def is_hybrid(self) -> bool:
        return bool(self.linear_layers) and bool(self.scoreable_layers)

    @property
    def max_heads(self) -> int:
        return max(self.num_heads.values()) if self.num_heads else 0

    @property
    def scoreable_heads(self) -> list[HeadRef]:
        return [HeadRef(l, h) for l in self.scoreable_layers for h in range(self.num_heads[l])]

    @property
    def n_scoreable_heads(self) -> int:
        return sum(self.num_heads[l] for l in self.scoreable_layers)

    @property
    def n_all_heads(self) -> int:
        """Head count including the heads of non-scoreable linear mixers (if any)."""
        return self.n_scoreable_heads

    def layer_type(self, layer: int) -> str:
        if layer < len(self.layer_types):
            return self.layer_types[layer]
        return "full_attention" if layer in self.attention_modules else "unknown"

    # ------------------------------------------------------------------ matrices
    def empty_matrix(self, fill: float = float("nan")) -> torch.Tensor:
        """A ``[num_layers, max_heads]`` matrix, NaN where no score is defined.

        Dense matrices (rather than flat per-head lists) keep the paper's
        layer x head heatmaps honest on hybrid models.
        """
        return torch.full((self.num_layers, self.max_heads), fill, dtype=torch.float32)

    def scoreable_mask(self) -> torch.Tensor:
        mask = torch.zeros((self.num_layers, self.max_heads), dtype=torch.bool)
        for layer in self.scoreable_layers:
            mask[layer, : self.num_heads[layer]] = True
        return mask

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "model_type": self.model_type,
            "num_layers": self.num_layers,
            "num_scoreable_layers": len(self.scoreable_layers),
            "num_linear_layers": len(self.linear_layers),
            "is_hybrid": self.is_hybrid,
            "layer_types": self.layer_types,
            "scoreable_layers": self.scoreable_layers,
            "linear_layers": self.linear_layers,
            "num_heads": {str(k): v for k, v in sorted(self.num_heads.items())},
            "num_kv_heads": {str(k): v for k, v in sorted(self.num_kv_heads.items())},
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "max_position_embeddings": self.max_position_embeddings,
            "n_scoreable_heads": self.n_scoreable_heads,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelInfo":
        """Rebuild from :meth:`as_dict` output -- no weights, no modules.

        Sufficient for scoring bookkeeping, statistics and every figure; only
        :class:`~retrieval_heads.attention.HeadMasker` needs real modules.
        """
        return cls(
            name=data["name"],
            path=data.get("path", ""),
            model_type=data.get("model_type", "unknown"),
            num_layers=int(data["num_layers"]),
            layer_types=list(data.get("layer_types", [])),
            num_heads={int(k): int(v) for k, v in data.get("num_heads", {}).items()},
            num_kv_heads={int(k): int(v) for k, v in data.get("num_kv_heads", {}).items()},
            head_dim=int(data.get("head_dim", 0)),
            hidden_size=int(data.get("hidden_size", 0)),
            max_position_embeddings=data.get("max_position_embeddings"),
            scoreable_layers_=list(data.get("scoreable_layers", [])),
            linear_layers_=list(data.get("linear_layers", [])),
        )


def model_device(model: nn.Module) -> torch.device:
    """Where the model's weights live.

    Every generation path must place its input tensors here: the CPU->CUDA move
    happens once, at load time, while the batches are built later from tokenizer
    output that is always on the CPU.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - a model with no parameters
        return torch.device("cpu")


def describe_model(info: ModelInfo) -> str:
    """Human-readable summary, printed by every entry-point script."""
    lines = [
        f"model            : {info.name}  ({info.model_type})",
        f"layers           : {info.num_layers}",
        f"scoreable layers : {len(info.scoreable_layers)} {info.scoreable_layers}",
        f"linear layers    : {len(info.linear_layers)}"
        + (f" {info.linear_layers}" if info.linear_layers else ""),
        f"hybrid           : {info.is_hybrid}",
        f"heads/layer      : {sorted(set(info.num_heads.values()))}"
        f"  (kv: {sorted(set(info.num_kv_heads.values()))})",
        f"head_dim         : {info.head_dim}",
        f"scoreable heads  : {info.n_scoreable_heads}",
        f"max context      : {info.max_position_embeddings}",
    ]
    if info.is_hybrid:
        lines.append(
            "NOTE             : hybrid architecture -- the paper's retrieval score is only "
            f"defined for the {len(info.scoreable_layers)} full-attention layers "
            f"({info.n_scoreable_heads} heads). The remaining "
            f"{len(info.linear_layers)} layers are linear/recurrent token mixers with no "
            "attention map over the context; they can only be ablated, not scored."
        )
    return "\n".join(lines)


def discover_modules(model: nn.Module) -> tuple[dict[int, nn.Module], dict[int, nn.Module], dict[int, str]]:
    """Split every token mixer in ``model`` into scoreable vs linear."""
    attention: dict[int, nn.Module] = {}
    linear: dict[int, nn.Module] = {}
    names: dict[int, str] = {}
    for name, module in model.named_modules():
        if _is_attention_module(module):
            layer = getattr(module, "layer_idx", None)
            if layer is None:
                continue
            if layer in attention:  # keep the first match (outermost) deterministically
                continue
            attention[int(layer)] = module
            names[int(layer)] = name
        elif _looks_linear(module):
            layer = getattr(module, "layer_idx", None)
            if layer is None or layer in linear:
                continue
            linear[int(layer)] = module
            names.setdefault(int(layer), name)
    return attention, linear, names


def load_model(
    path: str,
    *,
    dtype: torch.dtype | str = torch.float32,
    attn_implementation: str = "eager",
    device: str = "cpu",
    name: str | None = None,
) -> tuple[nn.Module, Any, ModelInfo]:
    """Load a causal LM plus its tokenizer and an architecture description.

    ``attn_implementation`` defaults to ``"eager"`` because the retrieval score
    needs the actual attention probabilities; ``sdpa``/``flash_attention_2``
    never materialise them.  :class:`~retrieval_heads.attention.AttentionRecorder`
    can temporarily switch to a cheaper kernel for the prefill pass.
    """
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    config = AutoConfig.from_pretrained(path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
    except Exception:  # pragma: no cover - VL-only repos
        from transformers import AutoProcessor

        tokenizer = AutoProcessor.from_pretrained(path).tokenizer

    model_kwargs: dict[str, Any] = {"dtype": dtype, "attn_implementation": attn_implementation}
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
    except Exception as exc:  # pragma: no cover - e.g. conditional-generation-only repos
        log.warning("AutoModelForCausalLM failed (%s); trying AutoModelForImageTextToText", exc)
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(path, **model_kwargs)

    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    info = build_model_info(model, config, path=path, name=name)
    log.info("loaded %s: %d scoreable layers, %d scoreable heads",
             info.name, len(info.scoreable_layers), info.n_scoreable_heads)
    return model, tokenizer, info


def build_model_info(model: nn.Module, config: Any, *, path: str, name: str | None = None) -> ModelInfo:
    tcfg = text_config(config)
    attention, linear, names = discover_modules(model)

    num_heads = {
        layer: int(getattr(module, "config", tcfg).num_attention_heads)
        for layer, module in attention.items()
    }
    num_kv_heads = {
        layer: int(
            getattr(getattr(module, "config", tcfg), "num_key_value_heads",
                    getattr(module, "config", tcfg).num_attention_heads)
        )
        for layer, module in attention.items()
    }
    hidden = {
        int(getattr(module, "config", tcfg).hidden_size) for module in attention.values()
    }
    head_dims = {_head_dim(module) for module in attention.values()}

    num_layers = int(getattr(tcfg, "num_hidden_layers", 0))
    if not num_layers:
        num_layers = max(list(attention) + list(linear) + [-1]) + 1

    layer_types = list(getattr(tcfg, "layer_types", None) or [])
    if not layer_types:
        layer_types = [
            "full_attention" if i in attention else ("linear_attention" if i in linear else "unknown")
            for i in range(num_layers)
        ]

    return ModelInfo(
        name=name or str(path).rstrip("/").split("/")[-1],
        path=str(path),
        model_type=str(getattr(config, "model_type", getattr(tcfg, "model_type", "unknown"))),
        num_layers=num_layers,
        layer_types=layer_types,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=(head_dims.pop() if len(head_dims) == 1 else 0),
        hidden_size=(hidden.pop() if len(hidden) == 1 else 0),
        max_position_embeddings=getattr(tcfg, "max_position_embeddings", None),
        attention_modules=attention,
        linear_modules=linear,
        module_names=names,
    )
