from __future__ import annotations

from collections.abc import Callable

from .base import ModelAdapter
from .qwen35 import Qwen35Adapter
from .qwen3 import Qwen3Adapter

ModelFactory = Callable[..., ModelAdapter]
_MODELS: dict[str, ModelFactory] = {
    "qwen35": Qwen35Adapter,
    "qwen3": Qwen3Adapter,
}


def create_model(name: str, **kwargs: object) -> ModelAdapter:
    try:
        factory = _MODELS[name]
    except KeyError as error:
        available = ", ".join(sorted(_MODELS))
        raise ValueError(
            f"Unknown model adapter {name!r}. Available: {available}"
        ) from error
    return factory(**kwargs)


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_MODELS))
