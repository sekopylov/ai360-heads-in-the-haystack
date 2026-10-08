from __future__ import annotations

import json
import hashlib
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
    depth_slug = f"{depth_percent:g}".replace(".", "p")
    stem = (
        f'{model_version.replace(".", "_")}_len_{context_length}_'
        f"depth_{depth_slug}"
    )
    return f"{case_id}_{stem}" if case_id else stem


def result_payload(
    result: RunResult,
    model_id: str,
    *,
    experiment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prepared = result.prepared
    payload = {
        "case_id": prepared.case.case_id,
        "model": model_id,
        "context_length": int(prepared.context_length),
        "depth_percent": float(prepared.depth_percent),
        "version": 2,
        "needle": prepared.case.needle,
        "expected_answer": prepared.case.expected_answer,
        "prompt_sha256": hashlib.sha256(
            json.dumps(prepared.prompt.token_ids).encode("utf-8")
        ).hexdigest(),
        "model_response": result.generation.text,
        "score": result.score,
        "test_duration_seconds": result.duration_seconds,
        "test_timestamp_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S%z"
        ),
    }
    if result.generation.raw_text is not None:
        payload["raw_model_response"] = result.generation.raw_text
        payload["generated_token_ids"] = result.generation.token_ids
        payload["finish_reason"] = result.generation.finish_reason
    if experiment is not None:
        payload["experiment"] = experiment
    return payload
