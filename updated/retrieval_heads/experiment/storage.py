from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import RunResult


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def result_stem(
    model_version: str,
    context_length: int,
    depth_percent: float,
    *,
    case_id: str | None = None,
) -> str:
    stem = (
        f'{model_version.replace(".", "_")}_len_{context_length}_'
        f"depth_{int(depth_percent * 100)}"
    )
    return f"{case_id}_{stem}" if case_id else stem


def result_payload(result: RunResult, model_id: str) -> dict[str, Any]:
    prepared = result.prepared
    return {
        "case_id": prepared.case.case_id,
        "model": model_id,
        "context_length": int(prepared.context_length),
        "depth_percent": float(prepared.depth_percent),
        "version": 1,
        "needle": prepared.case.needle,
        "model_response": result.generation.text,
        "score": result.score,
        "test_duration_seconds": result.duration_seconds,
        "test_timestamp_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S%z"
        ),
    }
