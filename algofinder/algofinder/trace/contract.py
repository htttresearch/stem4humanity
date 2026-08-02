"""Solver trace contracts and solve context.

A solver declares what its atomic semantic steps are, which state values
are recorded versus deterministically derived, which components are
opaque, and what coverage scope it claims. The recorder itself is
storage-agnostic; the contract makes "full" an auditable property.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from algofinder.trace.model import TraceConfig

CoverageScope = Literal["algorithm", "boundary", "lifecycle"]


@dataclass(frozen=True)
class TraceContract:
    """Versioned declaration of one solver's observable decision process."""

    id: str
    version: int = 1
    state_schema: str = ""
    atomic_steps: dict[str, str] = field(default_factory=dict)
    supported_profiles: frozenset[str] = frozenset(
        {"full", "decisions", "summary"}
    )
    derived_state: dict[str, str] = field(default_factory=dict)
    opaque_components: tuple[str, ...] = ()
    coverage_scope: CoverageScope = "lifecycle"

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "state_schema": self.state_schema,
            "atomic_steps": self.atomic_steps,
            "supported_profiles": sorted(self.supported_profiles),
            "derived_state": self.derived_state,
            "opaque_components": list(self.opaque_components),
            "coverage_scope": self.coverage_scope,
        }


def trace_contract(id: str | None = None, **fields: Any):
    """Class decorator attaching a TraceContract to a solver class."""

    def decorate(cls: type) -> type:
        contract_id = id if id is not None else getattr(cls, "id", "")
        cls.trace_contract = TraceContract(id=contract_id, **fields)
        return cls

    return decorate


class SolveContext:
    """Explicit per-solve context passed to solvers by the harness.

    ``trace`` is either a :class:`TraceRecorder` (dev) or the null
    recorder (prod). ``budget`` is the cooperative time budget; solvers
    may use it instead of private perf_counter deadlines.
    """

    __slots__ = ("run_id", "trace", "budget_seconds", "_component")

    def __init__(
        self,
        *,
        run_id: str | None = None,
        trace: Any = None,
        budget_seconds: float | None = None,
    ) -> None:
        from algofinder.trace.recorder import NullTraceRecorder

        self.run_id = run_id
        self.trace = trace if trace is not None else NullTraceRecorder()
        self.budget_seconds = budget_seconds
        self._component = "solve"

    def child(self, component: str) -> "SolveContext":
        child = SolveContext(
            run_id=self.run_id,
            trace=self.trace.child(component),
            budget_seconds=self.budget_seconds,
        )
        child._component = component
        return child

    @property
    def component(self) -> str:
        return self._component


__all__ = ["SolveContext", "TraceContract", "trace_contract"]
