"""Small structured-event writer shared by long-running project commands."""
from __future__ import annotations

import json
import hashlib
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_hashes(paths: list[Path]) -> dict[str, str | None]:
    return {str(path): file_sha256(path) if path.exists() else None for path in paths}


class EventWriter:
    """Append-only JSONL events; every call is flushed for live dashboards."""

    def __init__(self, path: Path, *, run_id: str, stage: int, lineage: dict | None = None):
        self.path = path
        self.run_id = run_id
        self.stage = stage
        self.lineage = lineage or {}
        self.git_sha = _git_sha()
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        event_type: str,
        *,
        phase: str | None = None,
        current: int | float | None = None,
        total: int | float | None = None,
        metrics: dict[str, Any] | None = None,
        message: str | None = None,
        **extra: Any,
    ) -> dict:
        event = {
            "type": event_type,
            "run_id": self.run_id,
            "stage": self.stage,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "phase": phase,
            "current": current,
            "total": total,
            "metrics": metrics or {},
            "message": message,
            "git_sha": self.git_sha,
            "host": socket.gethostname(),
            **self.lineage,
            **extra,
        }
        with self.path.open("a") as handle:
            handle.write(json.dumps(event, default=str) + "\n")
            handle.flush()
        return event
