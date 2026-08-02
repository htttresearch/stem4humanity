"""Prod/dev tracing: dev sessions capture solver intermediate states.

``prod`` mode (the default) never touches the filesystem beyond the
results the harness writes; ``dev`` mode opens a timestamped session
directory under ``private/sessions`` and lets individual solvers write
JSONL traces of their intermediate states as they run.

Tracing is intentionally ad hoc for now: solvers opt in by calling
:func:`open_trace` and :func:`write_record` directly. A generic
problem/solver abstraction can replace this once the schema is proven.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSIONS_ROOT = PROJECT_ROOT / "private" / "sessions"

_trace_dir: Path | None = None


def is_dev() -> bool:
    """True when a dev session is active (traces will be written)."""
    return _trace_dir is not None


def trace_dir() -> Path | None:
    """The active session directory, or None in prod mode."""
    return _trace_dir


def start_session(root: str | Path | None = None, label: str = "session") -> Path:
    """Create a timestamped dev session directory and make it active."""
    global _trace_dir
    root = Path(root) if root is not None else SESSIONS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    session = root / f"{label}-{time.strftime('%Y%m%d-%H%M%S')}"
    session.mkdir(parents=True, exist_ok=False)
    _trace_dir = session
    return session


def close_session() -> None:
    """Deactivate tracing (subsequent solves write nothing)."""
    global _trace_dir
    _trace_dir = None


def open_trace(instance_name: str, solver_id: str) -> Path | None:
    """Ad-hoc JSONL trace path for one (instance, solver), or None in prod.

    Files are named ``<instance>__<solver>.jsonl``; records are appended
    as solver runs and each line is a self-describing JSON object.
    """
    if _trace_dir is None:
        return None
    safe = instance_name.replace("/", "_").replace(":", "-")
    return _trace_dir / f"{safe}__{solver_id}.jsonl"


def write_record(path: Path | None, record: dict[str, Any]) -> None:
    """Append one JSON record to a trace file (no-op in prod mode)."""
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=float) + "\n")


__all__ = [
    "SESSIONS_ROOT",
    "close_session",
    "is_dev",
    "open_trace",
    "start_session",
    "trace_dir",
    "write_record",
]
