"""Shared fixtures.

Heavy tests load real checkpoints; they are skipped automatically when the model
directories are absent, so the fast unit suite always runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_DIRS = {
    "qwen3.5-0.8b": REPO_ROOT / "models" / "Qwen3.5-0.8B",
    "qwen3-0.6b": REPO_ROOT / "models" / "Qwen3-0.6B",
}


def _weights_present(path: Path) -> bool:
    if not path.exists():
        return False
    return any(p.stat().st_size > 10_000_000 for p in path.glob("*.safetensors" ))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def tokenizer():
    """The Qwen3-0.6B tokenizer: local, fast, and no weights needed."""
    path = MODEL_DIRS["qwen3-0.6b"]
    if not (path / "tokenizer.json").exists():
        pytest.skip("Qwen3-0.6B tokenizer not downloaded")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


def _load(name: str, dtype: str = "float32"):
    import torch

    from retrieval_heads.models import load_model

    path = MODEL_DIRS[name]
    if not _weights_present(path):
        pytest.skip(f"{name} weights not downloaded")
    torch.set_grad_enabled(False)
    return load_model(str(path), dtype=getattr(torch, dtype), attn_implementation="eager")


@pytest.fixture(scope="session")
def qwen35():
    return _load("qwen3.5-0.8b")


@pytest.fixture(scope="session")
def qwen3():
    return _load("qwen3-0.6b")
