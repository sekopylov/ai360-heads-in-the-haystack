"""Resolve checkpoint files without importing Torch or Transformers."""
from __future__ import annotations

import json
from pathlib import Path


def validate_checkpoint(directory: Path) -> None:
    if not (directory / "config.json").is_file():
        raise ValueError(f"Incomplete model directory: {directory}: config.json missing")
    weights = [directory / "model.safetensors", directory / "pytorch_model.bin"]
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / name
        if index.is_file():
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            if shards and all((directory / shard).is_file() for shard in shards):
                return
            raise ValueError(f"Incomplete model directory: {directory}: weight shards missing")
    if not any(path.is_file() for path in weights):
        raise ValueError(f"Incomplete model directory: {directory}: weights missing")


def resolve_model_source(model_id: str, search_dirs: list[str] | None = None) -> tuple[str, bool]:
    """Return (source, local_only); search only explicitly supplied roots.

    Each root contains the full Hub ID, e.g. ROOT/Qwen/Qwen3.5-0.8B.
    Existing but incomplete checkpoints fail instead of silently downloading.
    """
    direct = Path(model_id).expanduser()
    if direct.is_dir():
        validate_checkpoint(direct)
        source = str(direct.resolve())
        print(f"[model] source=local path={source}", flush=True)
        return source, True
    if direct.is_absolute() or model_id.startswith(("./", "../", "~")):
        raise FileNotFoundError(f"Model directory not found: {direct}")
    parts = model_id.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid model ID: {model_id!r}")

    for root in search_dirs or []:
        candidate = Path(root).expanduser() / model_id
        if candidate.exists():
            validate_checkpoint(candidate)
            source = str(candidate.resolve())
            print(f"[model] source=local path={source} model_id={model_id}", flush=True)
            return source, True
        print(f"[model] not found: {candidate}", flush=True)
    print(f"[model] source=HuggingFace model_id={model_id} (normal HF cache applies)", flush=True)
    return model_id, False
