"""Small shared helpers: seeding, logging, IO, and the ``HeadRef`` type."""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER_NAME = "retrieval_heads"


def get_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")
    if not logging.getLogger().handlers:
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            level=os.environ.get("RH_LOGLEVEL", "INFO"),
        )
    return logger


log = get_logger()


@dataclass(frozen=True, order=True)
class HeadRef:
    """A single attention head, addressed by absolute layer index and head index.

    Layer indices are absolute and may be non-contiguous: on a hybrid model only
    the full-attention layers are addressable at all.
    """

    layer: int
    head: int

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"L{self.layer}H{self.head}"

    def as_dict(self) -> dict[str, int]:
        return {"layer": self.layer, "head": self.head}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)
    log.info("wrote %s", path)


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def human_int(n: int) -> str:
    """Format a token count the way long-context papers do (1K, 32K, ...)."""
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}K"
    if n >= 1000:
        return f"{n / 1000:.1f}K"
    return str(n)
