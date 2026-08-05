"""Session coordinator: parent-owned run lifecycle for dev mode.

The session reserves a run ID and writes ``invocation.json`` *before*
the worker starts, and writes the authoritative ``outcome.json`` and the
``index.jsonl`` row *after* the worker returns or is killed. The worker
only appends its own event stream; it never owns terminal state.

Layout (see docs/dev-mode-observability-design.md):

    private/sessions/session-<UTC>-<random>/
      session.json
      index.jsonl
      instances/<sha256>.json
      problem-states/<sha256>.json
      artifacts/<sha256>.json|npy
      runs/r000001/{invocation.json, events.jsonl, outcome.json}
"""

from __future__ import annotations

import json
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from algofinder.trace.artifacts import ArtifactStore
from algofinder.trace.model import Invocation, TraceConfig, TraceSummary
from algofinder.trace.serialize import strict_dumps

SESSIONS_ROOT = Path(__file__).resolve().parents[2] / "private" / "sessions"


def _code_identity() -> dict[str, Any]:
    try:
        import subprocess

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).resolve().parents[2],
        )
        head = commit.stdout.strip() if commit.returncode == 0 else None
    except Exception:
        head = None
    return {"commit": head}


def _runtime_identity() -> dict[str, Any]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for module_name in ("numpy", "sklearn", "networkx", "joblib"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "?")
        except ImportError:
            continue
    return versions


class DevSession:
    """A dev-mode session; use as a context manager (never globally active)."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        profile: str = "full",
        event_limit: int | None = None,
        byte_limit: int | None = None,
        fail_on_limit: bool = True,
        label: str = "session",
    ) -> None:
        self.profile = profile
        self.event_limit = event_limit
        self.byte_limit = byte_limit
        self.fail_on_limit = fail_on_limit
        base = Path(root) if root is not None else SESSIONS_ROOT
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.dir = base / f"{label}-{stamp}-{secrets.token_hex(3)}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.session_id = self.dir.name
        self.store = ArtifactStore(self.dir)
        self._index_fh: Any = None
        self._run_counter = 0
        self._closed = False

    def __enter__(self) -> "DevSession":
        self._index_fh = open(self.dir / "index.jsonl", "ab", buffering=0)
        header = {
            "schema": "algofinder.session/v1",
            "session_id": self.session_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "dev",
            "profile": self.profile,
            "event_limit": self.event_limit,
            "byte_limit": self.byte_limit,
            "fail_on_limit": self.fail_on_limit,
            "code": _code_identity(),
            "runtime": _runtime_identity(),
        }
        (self.dir / "session.json").write_text(
            strict_dumps(header) + "\n", encoding="utf-8"
        )
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def trace_config(self) -> TraceConfig:
        return TraceConfig(
            enabled=True,
            profile=self.profile,
            event_limit=self.event_limit,
            byte_limit=self.byte_limit,
            fail_on_limit=self.fail_on_limit,
        )

    def start_run(
        self,
        *,
        instance: Any,
        instance_sha: str,
        state_sha: str,
        solver_id: str,
        solver_display: str,
        solver_config: dict[str, Any],
        budget_seconds: float | None,
        seed: int | None = None,
        memory_bytes: int | None = None,
        environment_id: str | None = None,
    ) -> Invocation:
        """Reserve a run ID and write its invocation before the worker starts."""
        self._run_counter += 1
        run_id = f"r{self._run_counter:06d}"
        run_dir = self.dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        invocation = Invocation(
            run_id=run_id,
            session_id=self.session_id,
            instance_name=instance.name,
            instance_sha=instance_sha,
            state_sha=state_sha,
            problem=instance.problem,
            subproblem=instance.subproblem,
            family=instance.family,
            split=instance.split,
            solver=solver_id,
            solver_display=solver_display,
            solver_config=solver_config,
            budget_seconds=budget_seconds,
            trace_profile=self.profile,
            seed=seed,
            memory_bytes=memory_bytes,
            environment_id=environment_id,
            created_utc=datetime.now(timezone.utc).isoformat(),
        )
        (run_dir / "invocation.json").write_text(
            strict_dumps(invocation.__dict__) + "\n", encoding="utf-8"
        )
        self._index_row(
            {
                "event": "started",
                "run_id": run_id,
                "instance": instance.name,
                "solver": solver_id,
                "problem": instance.problem,
                "split": instance.split,
                "started_utc": invocation.created_utc,
            }
        )
        return invocation

    def run_dir(self, run_id: str) -> Path:
        return self.dir / "runs" / run_id

    def finish_run(
        self,
        invocation: Invocation,
        *,
        status: str,
        cost_solver: float | None = None,
        cost_harness: float | None = None,
        exact: bool | None = None,
        wall_seconds: float | None = None,
        cpu_seconds: float | None = None,
        peak_rss_bytes: int | None = None,
        seed: int | None = None,
        memory_bytes: int | None = None,
        environment_id: str | None = None,
        error: str | None = None,
        solution: Any = None,
        metadata: dict[str, Any] | None = None,
        trace: TraceSummary | None = None,
    ) -> None:
        """Write the authoritative outcome and the final index row."""
        outcome = {
            "schema": "algofinder.outcome/v1",
            "run_id": invocation.run_id,
            "session_id": self.session_id,
            "status": status,
            "cost_solver": cost_solver,
            "cost_harness": cost_harness,
            "exact": exact,
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "peak_rss_bytes": peak_rss_bytes,
            "seed": seed,
            "memory_bytes": memory_bytes,
            "environment_id": environment_id,
            "error": error,
            "solution": solution,
            "solver_metadata": metadata or {},
            "trace": trace.to_mapping() if trace is not None else None,
        }
        (self.run_dir(invocation.run_id) / "outcome.json").write_text(
            strict_dumps(outcome) + "\n", encoding="utf-8"
        )
        self._index_row(
            {
                "event": "finished",
                "run_id": invocation.run_id,
                "status": status,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "trace_events": trace.event_count if trace else 0,
                "trace_complete": bool(trace and trace.complete),
            }
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._index_fh is not None:
            self._index_fh.flush()
            self._index_fh.close()
            self._index_fh = None
        summary_path = self.dir / "session.json"
        if summary_path.exists():
            record = json.loads(summary_path.read_text(encoding="utf-8"))
            record["runs"] = self._run_counter
            record["closed_utc"] = datetime.now(timezone.utc).isoformat()
            summary_path.write_text(strict_dumps(record) + "\n", encoding="utf-8")

    def _index_row(self, record: dict[str, Any]) -> None:
        if self._index_fh is None:
            return
        self._index_fh.write(strict_dumps(record).encode("utf-8") + b"\n")


__all__ = ["DevSession", "SESSIONS_ROOT"]
