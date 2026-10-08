"""Small shared helpers: seeding, logging, IO, and the ``HeadRef`` type."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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


def squad_f1(predicted: Sequence[Any], gold: Sequence[Any]) -> float:
    """SQuAD-style F1 between two sequences of hashable units (token ids or words).

    One implementation for `masking.token_f1` (ids) and `downstream.word_f1`
    (words), which had drifted into two near-identical copies.
    """
    from collections import Counter

    if not predicted or not gold:
        return 0.0
    pred, target = Counter(predicted), Counter(gold)
    overlap = sum((pred & target).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def eos_ids(model: Any, tokenizer: Any = None) -> set[int]:
    """Every EOS id the model might emit: config, generation config, tokenizer.

    One shared implementation on purpose.  The three generation loops used to
    collect these separately, and ``masking`` had already drifted: it ignored
    ``generation_config.eos_token_id``, so on a model that defines EOS only there
    the NIAH ablations never stopped and kept generating past the answer.
    """
    ids: set[int] = set()
    for source in (getattr(model, "config", None), getattr(model, "generation_config", None)):
        value = getattr(source, "eos_token_id", None) if source is not None else None
        if isinstance(value, int):
            ids.add(int(value))
        elif isinstance(value, (list, tuple, set)):
            ids.update(int(v) for v in value)
    if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(int(tokenizer.eos_token_id))
    return ids


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    # Write-then-rename with a *unique* temp name: two writers of the same artifact
    # must not have `os.replace` pick up each other's half-written file.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            # `allow_nan=False`: a bare `NaN` token is invalid strict JSON (jq, JS
            # and pandas reject it).  Non-finite floats are written as `null`.
            json.dump(finite_json(obj), fh, indent=2, ensure_ascii=False,
                      default=_json_default, allow_nan=False)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    os.replace(tmp_name, path)
    log.info("wrote %s", path)


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def finite_json(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` (JSON ``null``)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: finite_json(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(value) for value in obj]
    return obj


def json_default(obj: Any) -> Any:
    """Strict JSON encoder shared by every writer (no `default=str` fallback)."""
    return _json_default(obj)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return finite_json(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")
