"""Shared solver contracts and result records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from src.problems.base_problem import BaseProblem

ProblemT = TypeVar("ProblemT", bound=BaseProblem)


@dataclass(frozen=True)
class SolverResult:
    """A feasible solution returned by a solver."""

    tour: tuple[int, ...]
    cost: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSolver(ABC, Generic[ProblemT]):
    """A solver that returns a feasible minimization result."""

    @abstractmethod
    def solve(self, problem: ProblemT) -> SolverResult:
        """Solve one problem instance."""
