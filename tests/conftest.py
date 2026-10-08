"""Shared fixtures.

Heavy tests load real checkpoints; they are skipped automatically when the model
directories are absent, so the fast unit suite always runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_DIRS = {
    "qwen3.5-0.8b": REPO_ROOT / "models" / "Qwen3.5-0.8B",
    "qwen3-0.6b": REPO_ROOT / "models" / "Qwen3-0.6B",
}

REGISTRY_ENV = "RETRIEVAL_HEADS_MODELS_JSON"


@pytest.fixture(autouse=True)
def _no_registry_env_leak():
    """The driver sets this env var inside a test; never let it reach the next one.

    ``monkeypatch.delenv`` on an already-absent variable records no undo, so the
    driver's own ``os.environ[...] = ...`` leaked into unrelated tests.
    """
    saved = os.environ.pop(REGISTRY_ENV, None)
    yield
    os.environ.pop(REGISTRY_ENV, None)
    if saved is not None:
        os.environ[REGISTRY_ENV] = saved


def _weights_present(path: Path) -> bool:
    """True only when every weight shard the index names is present and non-trivial.

    Checking for *any* large ``*.safetensors`` let a half-downloaded sharded
    checkpoint through, and the failure then surfaced as a load error instead of a
    skip.
    """
    if not path.exists():
        return False
    index = path / "model.safetensors.index.json"
    if index.exists():
        import json

        try:
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        except Exception:  # noqa: BLE001 - a corrupt index means "not usable"
            return False
        if not shards:
            return False
    else:
        shards = {p.name for p in path.glob("*.safetensors")}
        if not shards:
            return False
    return all((path / name).exists() and (path / name).stat().st_size > 10_000_000
               for name in shards)


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
    # No global torch.set_grad_enabled(False) here: load_model already clears
    # requires_grad on every parameter, and flipping the global flag from a
    # session fixture leaked into whatever test ran next.
    return load_model(str(path), dtype=getattr(torch, dtype), attn_implementation="eager")


@pytest.fixture(scope="session")
def tiny_hybrid():
    """A ~50k-parameter Qwen3.5-style hybrid built locally, no download.

    Four layers: three Gated DeltaNet + one Gated Attention.  That is enough to
    exercise architecture discovery, both capture paths, o_proj masking and
    chunked-prefill equivalence in the *fast* suite, which otherwise only sees
    these things through the integration tests.
    """
    transformers = pytest.importorskip("transformers")
    for attr in ("Qwen3_5TextConfig", "Qwen3_5ForCausalLM"):
        if not hasattr(transformers, attr):
            pytest.skip(f"transformers {getattr(transformers, '__version__', '?')} has no {attr}")
    from retrieval_heads.models import build_model_info, require_scoreable

    config = transformers.Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256,
        layer_types=["linear_attention", "linear_attention", "full_attention",
                     "linear_attention"],
        linear_num_key_heads=4, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )
    model = transformers.Qwen3_5ForCausalLM(config)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    info = build_model_info(model, config, path="tiny-hybrid", name="tiny-hybrid")
    info.dtype = "float32"
    info.model_class = type(model).__name__
    require_scoreable(info)
    return model, info


@pytest.fixture(scope="session")
def qwen35():
    return _load("qwen3.5-0.8b")


@pytest.fixture(scope="session")
def qwen3():
    return _load("qwen3-0.6b")
