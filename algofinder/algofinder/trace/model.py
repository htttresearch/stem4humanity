"""Session, invocation, trace, and outcome data models.

These models are shared by the session coordinator (parent process) and
the per-cell recorder (worker process); all fields are picklable and
JSON-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TraceProfile = Literal["full", "decisions", "summary"]
CoverageScope = Literal["algorithm", "boundary", "lifecycle"]
Status = Literal["ok", "skipped", "error", "invalid", "timeout", "interrupted"]


@dataclass(frozen=True)
class TraceConfig:
    """Per-cell tracing configuration, passed explicitly to workers.

    The harness builds this in the parent; the worker constructs its own
    recorder from it. ``enabled=False`` is the production path: the null
    recorder performs no I/O and no serialization.
    """

    enabled: bool
    profile: TraceProfile = "full"
    event_limit: int | None = None
    byte_limit: int | None = None
    artifact_threshold: int = 4096
    fail_on_limit: bool = True

    @property
    def full(self) -> bool:
        return self.profile == "full"

    @property
    def decisions(self) -> bool:
        return self.profile in ("full", "decisions")


@dataclass
class Invocation:
    """Immutable identity and inputs of one (instance, solver) cell."""

    run_id: str
    session_id: str | None
    instance_name: str
    instance_sha: str
    state_sha: str
    problem: str
    subproblem: str
    family: str
    split: str
    solver: str
    solver_display: str
    solver_config: dict[str, Any] = field(default_factory=dict)
    budget_seconds: float | None = None
    trace_profile: str = "off"
    seed: int | None = None
    memory_bytes: int | None = None
    environment_id: str | None = None
    created_utc: str = ""


@dataclass
class TraceSummary:
    """Completeness report for one run's event stream."""

    requested_profile: str = "off"
    achieved_profile: str = "off"
    coverage_scope: str = "lifecycle"
    complete: bool = False
    event_count: int = 0
    bytes: int = 0
    last_seq: int = -1
    dropped_events: int = 0
    opaque_components: list[str] = field(default_factory=list)
    termination: str = ""
    error: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "requested_profile": self.requested_profile,
            "achieved_profile": self.achieved_profile,
            "coverage_scope": self.coverage_scope,
            "complete": self.complete,
            "event_count": self.event_count,
            "bytes": self.bytes,
            "last_seq": self.last_seq,
            "dropped_events": self.dropped_events,
            "opaque_components": self.opaque_components,
            "termination": self.termination,
            "error": self.error,
        }


__all__ = [
    "CoverageScope",
    "Invocation",
    "Status",
    "TraceConfig",
    "TraceProfile",
    "TraceSummary",
]
