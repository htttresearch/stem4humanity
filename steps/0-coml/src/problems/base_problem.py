"""Shared contracts for optimization problem instances."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class BaseProblem(ABC):
    """A problem instance that can validate and score candidate solutions."""

    name: str

    @abstractmethod
    def verify(self, solution: Sequence[int]) -> bool:
        """Return whether ``solution`` satisfies all hard constraints."""

    @abstractmethod
    def objective_value(self, solution: Sequence[int]) -> float:
        """Return the minimization objective value of a valid solution."""
