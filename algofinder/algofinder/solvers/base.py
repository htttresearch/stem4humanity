"""Shared solver contracts and the solver registry."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from algofinder.problems.base import Problem, ProblemState


class InapplicableError(Exception):
    """A solver cannot handle this instance (size, family, or configuration).

    The harness records such runs as ``skipped`` rather than ``error``.
    """


class UnsupportedError(InapplicableError):
    """A solver cannot run here (missing binary, platform, or capability).

    Recorded as ``unsupported`` (spec 6.6 status taxonomy) rather than
    ``error``: the row exists but carries no solution.
    """


@dataclass
class SolverResult:
    """A feasible solution returned by a solver."""

    solution: Any
    cost: float
    exact: bool
    wall_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None


@dataclass(frozen=True)
class SolverCapabilities:
    """Public adapter capabilities (benchmark spec section 8).

    Capabilities describe what an adapter can do, not config details:
    the harness owns resource caps and final scoring; web adapters
    translate, launch, capture outputs, and expose incumbent events.
    """

    roles: tuple[str, ...] = ("upper_bound",)
    objective_specs: tuple[str, ...] = ()
    interruptible: bool = False
    streams_incumbents: bool = False
    seed_control: str = "external"  # deterministic | external | none
    memory_control: bool = False
    native_work_counters: bool = False
    proof_artifacts: bool = False
    max_dimension: int | None = None
    binary_name: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "roles": list(self.roles),
            "objective_specs": list(self.objective_specs),
            "interruptible": self.interruptible,
            "streams_incumbents": self.streams_incumbents,
            "seed_control": self.seed_control,
            "memory_control": self.memory_control,
            "native_work_counters": self.native_work_counters,
            "proof_artifacts": self.proof_artifacts,
            "max_dimension": self.max_dimension,
            "binary_name": self.binary_name,
        }


class Solver(ABC):
    """A solver for one or more problem:subproblem pairs."""

    id: ClassVar[str] = ""
    display: ClassVar[str] = ""
    tags: ClassVar[frozenset[str]] = frozenset()  # exact | heuristic | ml | control
    applies_to: ClassVar[frozenset[str]] = frozenset()  # "tsp:euclidean", ...

    @abstractmethod
    def solve(
        self,
        state: ProblemState,
        *,
        budget_seconds: float | None = None,
        context: Any | None = None,
    ) -> SolverResult:
        """Solve one instance state within the optional time budget.

        ``context`` is an optional :class:`SolveContext` (dev mode).
        Direct calls that omit it receive no observability; the harness
        always passes one explicitly.
        """

    def config(self) -> dict[str, Any]:
        """Actual constructor parameter values for reproducibility.

        Non-JSON-safe values (e.g. loaded ML models) are excluded; they
        are identified by the trace contract instead.
        """
        from algofinder.trace.serialize import to_json_value

        result: dict[str, Any] = {}
        try:
            signature = inspect.signature(type(self).__init__)
            for name in signature.parameters:
                if name == "self" or name == "args" or name == "kwargs":
                    continue
                if hasattr(self, name):
                    value = getattr(self, name)
                    try:
                        to_json_value(value)
                    except TypeError:
                        continue
                    result[name] = value
        except (TypeError, ValueError):
            pass
        return result

    def describe(self) -> dict[str, Any]:
        """Reproducibility metadata: identifier, config, and capabilities."""
        return {
            "id": self.id,
            "tags": sorted(self.tags),
            "config": self.config(),
            "capabilities": self.capabilities().to_mapping(),
        }

    def capabilities(self) -> SolverCapabilities:
        """Declared adapter capabilities (spec section 8).

        Default inference: a solver that stores a constructor ``seed``
        is deterministic (its seed is fixed by config); otherwise the
        harness treats it as uncontrolled and never passes a run seed.
        """
        tags = self.tags
        roles: list[str] = []
        if "exact" in tags:
            roles.append("exact")
        if "heuristic" in tags:
            roles.append("upper_bound")
        if "ml" in tags:
            roles.append("selector")
        if "control" in tags:
            roles.append("constructor")
        seed_control = "deterministic" if hasattr(self, "seed") else "none"
        return SolverCapabilities(
            roles=tuple(roles) or ("upper_bound",),
            interruptible=True,
            streams_incumbents=False,
            seed_control=seed_control,
        )


_SOLVERS: dict[str, type[Solver]] = {}


def register_solver(cls: type[Solver]) -> type[Solver]:
    if not cls.id or cls.id in _SOLVERS:
        raise ValueError(f"duplicate or empty solver id: {cls.id!r}")
    _SOLVERS[cls.id] = cls
    return cls


def all_solvers() -> dict[str, type[Solver]]:
    return dict(_SOLVERS)


def solvers_for(problem_id: str, subproblem: str) -> list[type[Solver]]:
    key = f"{problem_id}:{subproblem}"
    return sorted(
        (cls for cls in _SOLVERS.values() if key in cls.applies_to),
        key=lambda cls: cls.id,
    )
