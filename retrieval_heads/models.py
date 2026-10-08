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

"Architecture-agnostic" therefore means: a HF-style model whose attention blocks
expose **separate** ``q_proj``/``k_proj``/``v_proj``/``o_proj`` projections and a
``layer_idx``.  Fused-QKV blocks, or blocks without ``layer_idx``, are not
supported by design -- discovery finds nothing and :func:`require_scoreable`
raises rather than inventing numbers.
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
    """Return the language-model config, descending through VL wrappers.

    A VL config can carry ``text_config = None``; falling through to it would
    return ``None`` and silently produce an empty census, so the descent stops
    there and the outer config is kept.
    """
    if config is None:
        return None
    seen = set()
    while hasattr(config, "text_config") and id(config) not in seen:
        seen.add(id(config))
        nested = config.text_config
        if nested is None:
            break
        config = nested
    return config


def _is_attention_module(module: nn.Module) -> bool:
    # A recurrent/linear mixer that happens to expose q/k/v/o projections must not
    # be scored as softmax attention: its "attention maps" would be invented, which
    # is the failure mode this module exists to prevent.  Known mixer class names
    # win over the projection signature.
    if any(marker in type(module).__name__ for marker in _LINEAR_MARKERS):
        return False
    return all(hasattr(module, attr) for attr in _ATTENTION_SIGNATURE)


def _looks_linear(module: nn.Module) -> bool:
    return any(marker in type(module).__name__ for marker in _LINEAR_MARKERS)


#: Substrings that mark a vision/audio tower rather than the language model.
_UNSCOREABLE_TOWER_MARKERS = ("visual", "vision", "audio", "speech")


def _is_vision_module(qualified_name: str) -> bool:
    lowered = qualified_name.lower()
    return any(marker in lowered for marker in _UNSCOREABLE_TOWER_MARKERS)


def _head_dim(module: nn.Module) -> int:
    if hasattr(module, "head_dim"):
        return int(module.head_dim)
    # `o_proj` is not gated: it maps heads*head_dim -> hidden, so it is the reliable
    # fallback.  `q_proj.out_features` is twice the width for gated attention.
    heads = getattr(module, "num_heads", None)
    o_width = getattr(getattr(module, "o_proj", None), "in_features", None)
    if heads and o_width:
        return int(o_width // heads)
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
    #: Precision the model was loaded in ("float32", "bfloat16"); recorded so an
    #: artifact cannot be mistaken for one produced at another dtype.
    dtype: str | None = None
    #: The concrete class that was instantiated (e.g. ``Qwen3_5ForCausalLM``).  On a
    #: VL checkpoint the auto-class descends into ``text_config``, so this is what
    #: actually ran -- and the fallback below is only taken when it fails.
    model_class: str | None = None
    #: Value-head count of each linear/recurrent token mixer (0 when unknown).
    #: These heads have no attention map and cannot be scored, but they are
    #: token-mixer channels an ablation can silence, so they are counted
    #: separately from the scoreable ones.
    num_linear_heads: dict[int, int] = field(default_factory=dict)
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
        """Scoreable heads **plus** the value heads of linear/recurrent mixers.

        Only the scoreable ones have an attention map and can carry a retrieval
        score; the linear heads are still token-mixer channels that masking can
        remove.  Keeping both is what lets a "8 of 336" style statement be
        checked against the model rather than typed into prose.
        """
        return self.n_scoreable_heads + sum(self.num_linear_heads.values())

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
            "num_linear_heads": {str(k): v for k, v in sorted(self.num_linear_heads.items())},
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "max_position_embeddings": self.max_position_embeddings,
            "dtype": self.dtype,
            "model_class": self.model_class,
            "n_scoreable_heads": self.n_scoreable_heads,
            "n_all_heads": self.n_all_heads,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelInfo":
        """Rebuild from :meth:`as_dict` output -- no weights, no modules.

        Sufficient for scoring bookkeeping, statistics and every figure; only
        :class:`~retrieval_heads.attention.HeadMasker` needs real modules.
        """
        info = cls(
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
            dtype=data.get("dtype"),
            model_class=data.get("model_class"),
            num_linear_heads={int(k): int(v) for k, v in data.get("num_linear_heads", {}).items()},
            scoreable_layers_=_layer_list(data, "scoreable_layers", "num_heads"),
            linear_layers_=_layer_list(data, "linear_layers", "num_linear_heads"),
        )
        if not info.scoreable_layers:
            # Without this, a metadata dict lacking `scoreable_layers` produced an
            # empty model: 0 heads, all-zero matrices, silently useless figures.
            raise ValueError(
                f"{info.name!r}: no scoreable_layers in the metadata (and no num_heads "
                f"to derive them from); the artifact is incomplete"
            )
        missing_heads = sorted(set(info.scoreable_layers) - set(info.num_heads))
        if missing_heads:
            raise ValueError(
                f"{info.name!r}: scoreable_layers {missing_heads} have no entry in "
                f"num_heads; the metadata is inconsistent"
            )
        return info


def _layer_list(data: Mapping[str, Any], key: str, fallback_key: str) -> list[int] | None:
    """An explicit layer list, or the keys of a per-layer map, or ``None``.

    ``None`` means "not specified", which :meth:`ModelInfo.from_dict` rejects when
    no scoreable layer can be derived -- better a loud error than a model with no
    heads that quietly produces empty figures.
    """
    value = data.get(key)
    if value is not None:
        return [int(v) for v in value]
    fallback = data.get(fallback_key)
    return sorted(int(k) for k in fallback) if fallback else None


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


def _inhomogeneous(field: str, values: set[int]) -> int:
    """0 for a model whose scoreable layers disagree on ``field`` -- loudly."""
    log.warning("scoreable layers disagree on %s (%s); recording 0", field, sorted(values))
    return 0


def require_scoreable(info: ModelInfo) -> ModelInfo:
    """Fail fast when a model exposes no softmax attention module.

    Without this check a model whose projections are named differently (or whose
    attention blocks lack ``layer_idx``) would print an all-zero census, run the
    entire detection grid for nothing, and then die with an ``IndexError`` deep
    in the aggregation.  The failure belongs here, before any compute.
    """
    if not info.scoreable_layers:
        raise RuntimeError(
            f"{info.name}: found no scoreable softmax-attention layers. The retrieval "
            f"score needs a block with q_proj/k_proj/v_proj/o_proj and a layer_idx "
            f"attribute; discovered {len(info.linear_layers)} linear/recurrent token "
            f"mixer(s) and no attention. If this model does have attention, its module "
            f"naming has changed and retrieval_heads/models.py needs updating."
        )
    return info


def _heads_per_layer(info: ModelInfo) -> str:
    """`8` when uniform, else the explicit layer->heads mapping."""
    values = set(info.num_heads.values())
    if len(values) == 1:
        return str(values.pop())
    return str(dict(sorted(info.num_heads.items())))


def describe_model(info: ModelInfo) -> str:
    """Human-readable summary, printed by every entry-point script."""
    lines = [
        f"model            : {info.name}  ({info.model_type})",
        f"layers           : {info.num_layers}",
        f"scoreable layers : {len(info.scoreable_layers)} {info.scoreable_layers}",
        f"linear layers    : {len(info.linear_layers)}"
        + (f" {info.linear_layers}" if info.linear_layers else ""),
        f"hybrid           : {info.is_hybrid}",
        f"heads/layer      : {_heads_per_layer(info)}"
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
    """Split every token mixer in ``model`` into scoreable vs linear.

    A module that looks like a token mixer but carries no ``layer_idx`` cannot be
    addressed, and dropping it silently undercounts ``n_all_heads`` (the "8 of N"
    number), so it is reported.
    """
    attention: dict[int, nn.Module] = {}
    linear: dict[int, nn.Module] = {}
    names: dict[int, str] = {}
    unindexed: list[str] = []
    collisions: list[str] = []
    for name, module in model.named_modules():
        if _is_vision_module(name):
            # A VL checkpoint's vision tower can expose the same q/k/v/o_proj
            # signature with its own layer_idx range; it is not part of the
            # language model and must never win the "first match" race.
            continue
        if _is_attention_module(module):
            layer = getattr(module, "layer_idx", None)
            if layer is None:
                unindexed.append(name)
                continue
            if layer in attention:
                # Silently keeping the first match would collapse a whole model onto
                # one layer and still print a plausible census.
                collisions.append(f"{name} collides with {names[int(layer)]} at layer_idx={layer}")
                continue
            attention[int(layer)] = module
            names[int(layer)] = name
        elif _looks_linear(module):
            layer = getattr(module, "layer_idx", None)
            if layer is None:
                unindexed.append(name)
                continue
            if layer in linear:
                collisions.append(f"{name} collides with {names[int(layer)]} at layer_idx={layer}")
                continue
            linear[int(layer)] = module
            names.setdefault(int(layer), name)
    # A layer that is *both* a scoreable attention layer and a token mixer would be
    # silently double-counted (scoreable_layers and linear_layers both contain it).
    cross = sorted(set(attention) & set(linear))
    if cross:
        collisions.extend(
            f"layer {layer} is both {names.get(layer)} (attention) and "
            f"{type(linear[layer]).__name__} (token mixer)" for layer in cross
        )
    if collisions:
        raise RuntimeError(
            "attention/token-mixer modules share a layer_idx, so they cannot be told "
            f"apart: {collisions[:5]}. Scoring would silently collapse them onto one "
            "layer. Fix the ModelInfo or the architecture discovery."
        )
    if unindexed:
        log.warning("token-mixer modules without a layer_idx were skipped (they cannot "
                    "be addressed for scoring or ablation): %s", unindexed[:5])
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
    dtype_name = dtype if isinstance(dtype, str) else str(dtype).replace("torch.", "")
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    config = AutoConfig.from_pretrained(path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
    except (ValueError, OSError, KeyError) as exc:  # pragma: no cover - VL-only repos
        from transformers import AutoProcessor

        try:
            tokenizer = AutoProcessor.from_pretrained(path).tokenizer
        except Exception as second:
            # Keep the original cause: a broken shard, a 401 or an OOM must not be
            # relabelled as "this repo needs a processor".
            raise RuntimeError(
                f"could not load a tokenizer for {path} (neither AutoTokenizer nor "
                f"AutoProcessor worked): {second}"
            ) from exc

    model_kwargs: dict[str, Any] = {"dtype": dtype, "attn_implementation": attn_implementation}
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
    except (ValueError, KeyError) as exc:  # pragma: no cover - conditional-generation-only repos
        # Architecture resolution raises ValueError or (UnrecognizedConfigError, a
        # KeyError subclass); OOM, a truncated shard or an auth failure must
        # propagate unchanged.
        log.warning("AutoModelForCausalLM could not resolve the architecture (%s); "
                    "trying AutoModelForImageTextToText", exc)
        from transformers import AutoModelForImageTextToText

        try:
            model = AutoModelForImageTextToText.from_pretrained(path, **model_kwargs)
        except Exception as second:
            raise RuntimeError(
                f"could not load {path} as either a causal LM or an image-text-to-text "
                f"model: {second}"
            ) from exc

    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    info = build_model_info(model, config, path=path, name=name)
    info.dtype = dtype_name
    info.model_class = type(model).__name__
    require_scoreable(info)
    log.info("loaded %s as %s (checkpoint architectures: %s)",
             info.name, info.model_class,
             getattr(config, "architectures", None) or getattr(text_config(config), "architectures", None))
    log.info("loaded %s: %d scoreable layers, %d scoreable heads",
             info.name, len(info.scoreable_layers), info.n_scoreable_heads)
    return model, tokenizer, info


def build_model_info(model: nn.Module, config: Any, *, path: str, name: str | None = None) -> ModelInfo:
    tcfg = text_config(config)
    attention, linear, names = discover_modules(model)

    # Coverage invariant: every transformer layer must be recognised as either a
    # scoreable attention block or a token mixer.  A renamed/unknown mixer class
    # otherwise disappears silently -- the model would look dense, `linear_layers`
    # would be empty and `n_all_heads` would quietly change the published numbers.
    num_layers = int(getattr(tcfg, "num_hidden_layers", 0) or 0)
    if num_layers:
        missing = sorted(set(range(num_layers)) - (set(attention) | set(linear)))
        if missing:
            raise RuntimeError(
                f"layers {missing} have neither a scoreable attention module nor a "
                f"recognised token mixer (unknown class name?); refusing to describe "
                f"the model as if those layers did not exist"
            )

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

    for layer, module in attention.items():
        heads, dim = num_heads.get(layer), _head_dim(module)
        o_width = getattr(getattr(module, "o_proj", None), "in_features", None)
        if heads and o_width and o_width != heads * dim:
            # Not a "looks gated" note: with head_dim from the module and o_proj the
            # un-gated projection, a disagreement means the geometry is wrong and
            # every masked slice would land on the wrong positions.
            raise RuntimeError(
                f"layer {layer}: o_proj takes {o_width} features but num_heads {heads} "
                f"x head_dim {dim} = {heads * dim}; the head geometry is inconsistent"
            )
        q_width = getattr(getattr(module, "q_proj", None), "out_features", None)
        if q_width is not None and heads and dim and q_width != heads * dim:
            log.debug("layer %d: q_proj outputs %d = 2 x %d; the block is gated",
                      layer, q_width, heads * dim)

    num_linear_heads = {
        layer: int(
            getattr(module, "num_v_heads", None)
            or getattr(getattr(module, "config", tcfg), "linear_num_value_heads", 0)
            or 0
        )
        for layer, module in linear.items()
    }

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
        head_dim=(head_dims.pop() if len(head_dims) == 1 else _inhomogeneous("head_dim", head_dims)),
        hidden_size=(hidden.pop() if len(hidden) == 1 else _inhomogeneous("hidden_size", hidden)),
        max_position_embeddings=getattr(tcfg, "max_position_embeddings", None),
        num_linear_heads=num_linear_heads,
        attention_modules=attention,
        linear_modules=linear,
        module_names=names,
    )
