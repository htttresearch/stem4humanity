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


@dataclass
class SolverResult:
    """A feasible solution returned by a solver."""

    solution: Any
    cost: float
    exact: bool
    wall_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


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
        """Reproducibility metadata: identifier, config, and version."""
        return {
            "id": self.id,
            "tags": sorted(self.tags),
            "config": self.config(),
        }


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
