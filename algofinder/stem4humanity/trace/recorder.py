"""Event recorder: the universal envelope around solver observations.

The recorder owns identity (run ID, session ID), ordering (contiguous
sequence numbers, not timestamps), timing (elapsed monotonic ns), state
references, artifact hashes, and storage. Solvers emit semantic events
and never touch files.

The envelope on every line:

    {
      "schema": "stem4humanity.trace-event/v1",
      "session_id": "...", "run_id": "r000001",
      "seq": 37, "elapsed_ns": 1834201,
      "category": "transition", "name": "uc.dp-hour/v1",
      "component": "solve", "phase": null,
      "state_before": "sha256", "state_after": "sha256",
      "action": {...}, "outcome": {...}, "metrics": {...}, "delta": [...]
    }

``seq`` is authoritative for order. Category is a small common
vocabulary for cross-solver analysis; the namespaced ``name`` carries
precise semantics. ``state_before``/``state_after`` reference
content-addressed state artifacts so a reader can reconstruct logical
state at every event boundary.

Events stream to ``events.jsonl.partial`` in unbuffered binary mode and
are renamed to ``events.jsonl`` only on a normal, flushed, fsynced close.
A killed worker therefore leaves a readable complete-line prefix marked
partial; the parent writes the authoritative outcome.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from stem4humanity.trace.artifacts import ArtifactRef, ArtifactStore
from stem4humanity.trace.model import TraceConfig, TraceSummary
from stem4humanity.trace.serialize import sha256_hex, strict_dumps

ENVELOPE_SCHEMA = "stem4humanity.trace-event/v1"

CATEGORIES = (
    "lifecycle",
    "phase",
    "evaluation",
    "decision",
    "transition",
    "bound",
    "prune",
    "incumbent",
    "prediction",
    "checkpoint",
    "diagnostic",
)


class RecorderError(RuntimeError):
    """Trace limits or serialization failed; the run must not be complete."""


class StateRef:
    """A content-addressed state artifact referenced from events."""

    __slots__ = ("ref",)

    def __init__(self, ref: ArtifactRef) -> None:
        self.ref = ref

    @property
    def id(self) -> str:
        return self.ref.sha256

    def to_mapping(self) -> dict[str, Any]:
        return self.ref.to_mapping()


class _RecorderState:
    """Mutable stream state shared between a recorder and its children.

    Children are views of the same stream with a different
    component/phase; they must share the file handle, sequence counter,
    byte counter, and closure flags instead of copying them.
    """

    __slots__ = (
        "fh",
        "seq",
        "bytes",
        "dropped",
        "closed",
        "first_error",
        "start_ns",
    )

    def __init__(self, fh: Any) -> None:
        self.fh = fh
        self.seq = 0
        self.bytes = 0
        self.dropped = 0
        self.closed = False
        self.first_error: str | None = None
        self.start_ns = perf_counter_ns()


class TraceRecorder:
    """Writes one run's ordered event stream."""

    def __init__(
        self,
        events_path: Path,
        *,
        session_id: str | None,
        run_id: str,
        config: TraceConfig,
        store: ArtifactStore,
        component: str = "solve",
        phase: str | None = None,
    ) -> None:
        if not config.enabled:
            raise ValueError("TraceRecorder requires an enabled TraceConfig")
        self.events_path = Path(events_path)
        self.session_id = session_id
        self.run_id = run_id
        self.config = config
        self.store = store
        self.component = component
        self.phase = phase
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = _RecorderState(open(self.events_path, "wb", buffering=0))

    # ------------------------------------------------------------------ state

    @property
    def enabled(self) -> bool:
        return True

    @property
    def full(self) -> bool:
        return self.config.full

    @property
    def decisions(self) -> bool:
        return self.config.decisions

    @property
    def seq(self) -> int:
        return self._state.seq

    def child(self, component: str, phase: str | None = None) -> "TraceRecorder":
        """A view of the same stream with a different component/phase.

        Nested helpers, fallback solvers, and shared kernels use this so
        their events stay in the same ordered run. The child shares the
        file handle and the sequence/byte counters, so it never reopens
        or truncates the stream and sequence numbers stay contiguous.
        """
        child = object.__new__(TraceRecorder)
        child.events_path = self.events_path
        child.session_id = self.session_id
        child.run_id = self.run_id
        child.config = self.config
        child.store = self.store
        child.component = component
        child.phase = phase
        child._state = self._state
        return child

    # ------------------------------------------------------------- emit API

    def span(self, name: str, **attributes: Any) -> None:
        """A lifecycle boundary (run, construction, solve, verification)."""
        self.event("lifecycle", name, outcome=attributes or None)

    def event(
        self,
        category: str,
        name: str,
        *,
        before: StateRef | None = None,
        after: StateRef | None = None,
        action: Any = None,
        outcome: Any = None,
        metrics: Any = None,
        delta: Any = None,
        phase: str | None = None,
        **fields: Any,
    ) -> None:
        """Emit one envelope event (see module docstring)."""
        if category not in CATEGORIES:
            raise RecorderError(f"unknown event category: {category!r}")
        envelope: dict[str, Any] = {
            "schema": ENVELOPE_SCHEMA,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "seq": self._state.seq,
            "elapsed_ns": perf_counter_ns() - self._state.start_ns,
            "category": category,
            "name": name,
            "component": self.component,
            "phase": phase if phase is not None else self.phase,
            "state_before": before.id if before is not None else None,
            "state_after": after.id if after is not None else None,
            "action": action,
            "outcome": outcome,
            "metrics": metrics,
            "delta": delta,
        }
        envelope.update(fields)
        line = strict_dumps(envelope).encode("utf-8") + b"\n"
        if self.config.event_limit is not None and self._state.seq >= self.config.event_limit:
            self._on_limit_reached(f"event limit {self.config.event_limit} reached")
            return
        if self.config.byte_limit is not None and self._state.bytes + len(line) > self.config.byte_limit:
            self._on_limit_reached(f"byte limit {self.config.byte_limit} reached")
            return
        try:
            self._state.fh.write(line)
        except Exception as exc:
            self._fail(f"event write failed: {exc}")
            return
        self._state.seq += 1
        self._state.bytes += len(line)

    def checkpoint(
        self, state: Any, **attributes: Any
    ) -> StateRef:
        """Store a full reconstructable logical state and reference it."""
        ref = self.store.publish_json("artifacts", state)
        state_ref = StateRef(ref)
        self.event(
            "checkpoint",
            "trace.checkpoint/v1",
            before=state_ref,
            after=state_ref,
            metrics={"bytes": ref.bytes},
            outcome=attributes or None,
        )
        return state_ref

    def observe(self, name: str, *, state: StateRef, **fields: Any) -> None:
        """A non-mutating evaluation of the current state."""
        self.event("evaluation", name, before=state, after=state, **fields)

    def transition(
        self,
        name: str,
        *,
        before: StateRef,
        after: StateRef | None = None,
        action: Any = None,
        outcome: Any = None,
        delta: Any = None,
        metrics: Any = None,
        phase: str | None = None,
    ) -> StateRef:
        """A state-changing step; returns the resulting state reference."""
        self.event(
            "transition",
            name,
            before=before,
            after=after,
            action=action,
            outcome=outcome,
            delta=delta,
            metrics=metrics,
            phase=phase,
        )
        return after if after is not None else before

    def artifact(self, value: Any, **metadata: Any) -> ArtifactRef:
        """Store a large value as a content-addressed artifact."""
        ref = self.store.publish_json("artifacts", value)
        self.event(
            "diagnostic",
            "trace.artifact/v1",
            outcome={"ref": ref.to_mapping(), "metadata": metadata},
        )
        return ref

    # ------------------------------------------------------------- lifecycle

    def close(self, *, complete: bool = True) -> TraceSummary:
        """Flush, fsync, and finalize the stream.

        ``complete=False`` keeps the ``.partial`` suffix and records the
        first error; ``complete=True`` renames to ``events.jsonl``.
        """
        if self._state.closed:
            return self.summary(complete=complete)
        self._state.closed = True
        self._state.fh.flush()
        os.fsync(self._state.fh.fileno())
        self._state.fh.close()
        if complete and self._state.first_error is None:
            final = self.events_path.with_name("events.jsonl")
            os.replace(self.events_path, final)
            termination = "solver-returned"
        else:
            final = self.events_path
            termination = "incomplete"
        return TraceSummary(
            requested_profile=self.config.profile,
            achieved_profile=self.config.profile if complete else "partial",
            complete=complete and self._state.first_error is None,
            event_count=self._state.seq,
            bytes=self._state.bytes,
            last_seq=self._state.seq - 1,
            dropped_events=self._state.dropped,
            termination=termination,
            error=self._state.first_error,
        )

    def summary(self, complete: bool = True) -> TraceSummary:
        return TraceSummary(
            requested_profile=self.config.profile,
            achieved_profile=self.config.profile,
            complete=complete and self._state.first_error is None,
            event_count=self._state.seq,
            bytes=self._state.bytes,
            last_seq=self._state.seq - 1,
            dropped_events=self._state.dropped,
            termination="solver-returned" if complete else "incomplete",
            error=self._state.first_error,
        )

    # ------------------------------------------------------------------ misc

    def _on_limit_reached(self, reason: str) -> None:
        self._state.dropped += 1
        if self.config.fail_on_limit:
            self._fail(reason)
        else:
            try:
                self._state.fh.write(
                    strict_dumps(
                        {
                            "schema": ENVELOPE_SCHEMA,
                            "session_id": self.session_id,
                            "run_id": self.run_id,
                            "seq": self._state.seq,
                            "elapsed_ns": perf_counter_ns() - self._state.start_ns,
                            "category": "diagnostic",
                            "name": "trace.limit-reached/v1",
                            "component": self.component,
                            "phase": self.phase,
                            "outcome": {"reason": reason},
                        }
                    ).encode("utf-8")
                    + b"\n"
                )
                self._state.seq += 1
            except Exception:
                pass

    def _fail(self, reason: str) -> None:
        if self._state.first_error is None:
            self._state.first_error = reason


class NullTraceRecorder:
    """Production recorder: enabled flag False, zero allocation and I/O.

    Solvers must guard payload construction with ``trace.enabled`` (or
    hoist ``trace.full``), so the disabled branch does no work at all.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.full = False
        self.decisions = False
        self.seq = -1
        self.component = "solve"
        self.phase = None

    def child(self, component: str, phase: str | None = None) -> "NullTraceRecorder":
        return self

    def span(self, name: str, **attributes: Any) -> None:
        return None

    def event(self, category: str, name: str, **fields: Any) -> None:
        return None

    def checkpoint(self, state: Any, **attributes: Any) -> None:
        return None

    def observe(self, name: str, *, state: Any, **fields: Any) -> None:
        return None

    def transition(self, name: str, **fields: Any) -> None:
        return None

    def artifact(self, value: Any, **metadata: Any) -> None:
        return None


__all__ = [
    "CATEGORIES",
    "ENVELOPE_SCHEMA",
    "NullTraceRecorder",
    "RecorderError",
    "StateRef",
    "TraceRecorder",
]
