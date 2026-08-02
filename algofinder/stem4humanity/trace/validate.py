"""Structural trace validation: is a run's event stream honest?

Checks that a run directory contains a strict-JSON invocation, a
readable event stream with contiguous sequence numbers and a valid
state-reference chain, and an authoritative outcome. Used by tests and
as a post-hoc audit tool; the parent outcome remains authoritative.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stem4humanity.trace.recorder import CATEGORIES, ENVELOPE_SCHEMA


class TraceValidationError(Exception):
    """The run's trace failed a structural check."""


def read_events(events_path: Path) -> list[dict[str, Any]]:
    """Parse complete JSONL lines; a trailing partial line is tolerated."""
    events: list[dict[str, Any]] = []
    with open(events_path, "rb") as handle:
        for raw in handle:
            if not raw.endswith(b"\n"):
                continue
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError:
                raise TraceValidationError(
                    f"invalid JSON line in {events_path.name}"
                )
    return events


def validate_run(run_dir: Path) -> dict[str, Any]:
    """Validate one run directory; raises TraceValidationError on failure."""
    invocation_path = run_dir / "invocation.json"
    if not invocation_path.exists():
        raise TraceValidationError("missing invocation.json")
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    if not isinstance(invocation, dict) or "run_id" not in invocation:
        raise TraceValidationError("invocation.json is malformed")

    events_path = run_dir / "events.jsonl"
    partial = not events_path.exists()
    if partial:
        events_path = run_dir / "events.jsonl.partial"
    if not events_path.exists():
        raise TraceValidationError("missing event stream")

    events = read_events(events_path)
    known_refs: set[str] = set()
    for index, event in enumerate(events):
        if event.get("schema") != ENVELOPE_SCHEMA:
            raise TraceValidationError(
                f"event {index} has wrong schema: {event.get('schema')!r}"
            )
        if event.get("seq") != index:
            raise TraceValidationError(
                f"event {index}: seq {event.get('seq')} is not contiguous"
            )
        if event.get("category") not in CATEGORIES:
            raise TraceValidationError(
                f"event {index}: unknown category {event.get('category')!r}"
            )
        before = event.get("state_before")
        after = event.get("state_after")
        if (
            before is not None
            and before != after
            and before not in known_refs
        ):
            raise TraceValidationError(
                f"event {index}: state_before {before} was never created"
            )
        if after is not None:
            known_refs.add(after)

    outcome_path = run_dir / "outcome.json"
    if not outcome_path.exists():
        raise TraceValidationError("missing outcome.json")
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    if not isinstance(outcome, dict) or "status" not in outcome:
        raise TraceValidationError("outcome.json is malformed")

    return {
        "run_id": invocation["run_id"],
        "valid": True,
        "partial": partial,
        "event_count": len(events),
        "last_seq": events[-1]["seq"] if events else -1,
        "status": outcome["status"],
        "trace_complete": bool(
            outcome.get("trace") and outcome["trace"].get("complete")
        ),
        "state_refs": len(known_refs),
    }


__all__ = ["TraceValidationError", "read_events", "validate_run"]
